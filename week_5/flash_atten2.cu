#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <cuda_runtime.h>

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

// --- Configuration ---
#define BLOCK_SIZE_M 16 
#define BLOCK_SIZE_N 32  

// --- Helper for Atomic Add ---
__device__ inline void atomicAddFloat(float* address, float val) {
    atomicAdd(address, val);
}

// 1. FORWARD KERNEL
__global__ void flash_attention_kernel_v2(
    const float* Q, const float* K, const float* V, float* O,
    float* L, 
    int N, int d_head, int batch_size, int num_heads, float scale) 
{
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int block_m_idx = blockIdx.x;
    int start_m = block_m_idx * BLOCK_SIZE_M;

    // Pointers to the specific batch/head
    const float* q_ptr = Q + (batch_idx * num_heads + head_idx) * N * d_head;
    const float* k_ptr = K + (batch_idx * num_heads + head_idx) * N * d_head;
    const float* v_ptr = V + (batch_idx * num_heads + head_idx) * N * d_head;
    float* o_ptr = O + (batch_idx * num_heads + head_idx) * N * d_head;
    float* l_ptr = L + (batch_idx * num_heads + head_idx) * N;

    __shared__ float K_tile[BLOCK_SIZE_N][64];
    __shared__ float V_tile[BLOCK_SIZE_N][64];

    int row_q_local = threadIdx.y;
    int row_q_global = start_m + row_q_local;

    if (row_q_global >= N) return;

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc_o[64] = {0.0f};

    // Outer Loop: Iterate over K/V blocks
    for (int block_n_idx = 0; block_n_idx < (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; ++block_n_idx) {
        int start_n = block_n_idx * BLOCK_SIZE_N;
        int row_kv_local = threadIdx.y;
        
        // Load K and V tiles into Shared Memory
        for(int i = row_kv_local; i < BLOCK_SIZE_N; i += blockDim.y) {
            int row_kv_global = start_n + i;
            if(row_kv_global < N){
                for (int k = threadIdx.x; k < d_head; k += blockDim.x) {
                    K_tile[i][k] = k_ptr[row_kv_global * d_head + k];
                    V_tile[i][k] = v_ptr[row_kv_global * d_head + k];
                }
            }
        }
        __syncthreads();

        // Compute Attention Scores
        float S_ij_row[BLOCK_SIZE_N]; 
        for (int j = 0; j < BLOCK_SIZE_N; ++j) {
            float sum = 0.0f;
            if (start_n + j < N) {
                for (int k = 0; k < d_head; ++k) {
                    sum += q_ptr[row_q_global * d_head + k] * K_tile[j][k];
                }
            } else {
                sum = -FLT_MAX;
            }
            S_ij_row[j] = sum * scale;
        }

        // Online Softmax logic
        float m_ij = -FLT_MAX;
        for (int j = 0; j < BLOCK_SIZE_N; ++j) m_ij = fmaxf(m_ij, S_ij_row[j]);
        
        float m_i_new = fmaxf(m_i, m_ij);
        float rescale_factor = expf(m_i - m_i_new);
        
        l_i *= rescale_factor;
        for(int k=0; k<d_head; ++k) acc_o[k] *= rescale_factor;

        float p_ij_sum = 0.0f;
        for (int j = 0; j < BLOCK_SIZE_N; ++j) {
             if (start_n + j < N) {
                float p_val = expf(S_ij_row[j] - m_i_new);
                p_ij_sum += p_val;
                for(int k=0; k<d_head; ++k) {
                    acc_o[k] += p_val * V_tile[j][k];
                }
             }
        }
        
        l_i += p_ij_sum;
        m_i = m_i_new;

        __syncthreads();
    }

    // Write O and L to HBM
    if (l_i > 0) {
        for (int k = 0; k < d_head; ++k) {
            o_ptr[row_q_global * d_head + k] = acc_o[k] / l_i;
        }
    }
    l_ptr[row_q_global] = m_i + logf(l_i);
}


// 2. BACKWARD KERNEL
__global__ void flash_attention_backward_kernel(
    const float* Q, const float* K, const float* V, const float* O, const float* dO,
    const float* L, 
    float* dQ, float* dK, float* dV,
    int N, int d_head, int batch_size, int num_heads, float scale) 
{
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int block_m_idx = blockIdx.x;
    
    int offset_block = (batch_idx * num_heads + head_idx) * N * d_head;
    int offset_L = (batch_idx * num_heads + head_idx) * N;

    const float* q_ptr = Q + offset_block;
    const float* k_ptr = K + offset_block;
    const float* v_ptr = V + offset_block;
    const float* o_ptr = O + offset_block;
    const float* do_ptr = dO + offset_block;
    const float* l_ptr = L + offset_L;

    float* dq_ptr = dQ + offset_block;
    float* dk_ptr = dK + offset_block;
    float* dv_ptr = dV + offset_block;

    __shared__ float K_tile[BLOCK_SIZE_N][64];
    __shared__ float V_tile[BLOCK_SIZE_N][64];

    int start_m = block_m_idx * BLOCK_SIZE_M;
    int row_q_local = threadIdx.y;
    int row_q_global = start_m + row_q_local;

    if (row_q_global >= N) return;

    // 1. Compute Di = rowsum(dO * O) [Required for dSoftmax]
    float D_i = 0.0f;
    for (int k = 0; k < d_head; ++k) {
        D_i += do_ptr[row_q_global * d_head + k] * o_ptr[row_q_global * d_head + k];
    }

    // 2. Load Q and dO into registers
    float q_reg[64];
    float do_reg[64];
    for (int k = 0; k < d_head; ++k) {
        q_reg[k] = q_ptr[row_q_global * d_head + k];
        do_reg[k] = do_ptr[row_q_global * d_head + k];
    }
    float l_i = l_ptr[row_q_global];
    float dq_acc[64] = {0.0f};

    // Iterate over K/V blocks to recompute Attention and gradients
    for (int block_n_idx = 0; block_n_idx < (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; ++block_n_idx) {
        int start_n = block_n_idx * BLOCK_SIZE_N;
        
        // Cooperative load K/V to Shared Memory
        for (int i = threadIdx.y; i < BLOCK_SIZE_N; i += blockDim.y) {
            int row_kv_global = start_n + i;
            if (row_kv_global < N) {
                for (int k = threadIdx.x; k < d_head; k += blockDim.x) {
                    K_tile[i][k] = k_ptr[row_kv_global * d_head + k];
                    V_tile[i][k] = v_ptr[row_kv_global * d_head + k];
                }
            }
        }
        __syncthreads();

        // Compute Gradients
        for (int j = 0; j < BLOCK_SIZE_N; ++j) {
            int row_kv_global = start_n + j;
            if (row_kv_global >= N) continue;

            // Recompute S_ij
            float s_ij = 0.0f;
            for (int k = 0; k < d_head; ++k) {
                s_ij += q_reg[k] * K_tile[j][k];
            }
            s_ij *= scale;

            // Recompute P_ij
            float p_ij = expf(s_ij - l_i);

            // Compute dV contribution (Requires Atomic Add)
            for (int k = 0; k < d_head; ++k) {
                float val = p_ij * do_reg[k];
                atomicAddFloat(&dv_ptr[row_kv_global * d_head + k], val);
            }

            // Compute dS_ij
            float dP_val = 0.0f;
            for (int k = 0; k < d_head; ++k) {
                dP_val += do_reg[k] * V_tile[j][k];
            }
            float ds_ij = p_ij * (dP_val - D_i) * scale;

            // Accumulate dQ
            for (int k = 0; k < d_head; ++k) {
                dq_acc[k] += ds_ij * K_tile[j][k];
            }

            // Compute dK contribution (Requires Atomic Add)
            for (int k = 0; k < d_head; ++k) {
                float val = ds_ij * q_reg[k];
                atomicAddFloat(&dk_ptr[row_kv_global * d_head + k], val);
            }
        }
        __syncthreads();
    }

    // Write dQ to HBM
    for (int k = 0; k < d_head; ++k) {
        dq_ptr[row_q_global * d_head + k] = dq_acc[k];
    }
}


// 3. CPU BASELINE 
void attention_cpu(const float* Q, const float* K, const float* V, float* O, int N, int d_head, int batch_size, int num_heads, float scale) {
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            const float* q_ptr = Q + (b * num_heads + h) * N * d_head;
            const float* k_ptr = K + (b * num_heads + h) * N * d_head;
            const float* v_ptr = V + (b * num_heads + h) * N * d_head;
            float* o_ptr = O + (b * num_heads + h) * N * d_head;
            
            float* S = (float*)malloc(N * N * sizeof(float));
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < d_head; ++k) {
                        sum += q_ptr[i * d_head + k] * k_ptr[j * d_head + k];
                    }
                    S[i * N + j] = sum * scale;
                }
            }
            for (int i = 0; i < N; ++i) {
                float max_val = -FLT_MAX;
                for (int j = 0; j < N; ++j) max_val = fmaxf(max_val, S[i * N + j]);
                float sum_exp = 0.0f;
                for (int j = 0; j < N; ++j) {
                    S[i * N + j] = expf(S[i * N + j] - max_val);
                    sum_exp += S[i * N + j];
                }
                for (int j = 0; j < N; ++j) {
                    if (sum_exp > 1e-6) S[i * N + j] /= sum_exp; else S[i * N + j] = 0;
                }
            }
            for (int i = 0; i < N; ++i) {
                for (int k = 0; k < d_head; ++k) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j) {
                        sum += S[i * N + j] * v_ptr[j * d_head + k];
                    }
                    o_ptr[i * d_head + k] = sum;
                }
            }
            free(S);
        }
    }
}

void verify_results(const float* O_gpu, const float* O_cpu, long long total_elements) {
    printf("Verifying Forward Pass:\n");
    double max_error = 0.0;
    for (long long i = 0; i < total_elements; ++i) {
        max_error = fmax(max_error, fabs(O_gpu[i] - O_cpu[i]));
    }
    printf("Max absolute error: %f\n", max_error);
    if (max_error < 1e-4) printf(">> SUCCESS: Forward Pass matches CPU.\n");
    else printf(">> FAILURE: Forward Pass mismatch.\n");
}

// 4. MAIN EXECUTION
int main() {
    // Parameters
    int N = 1024;
    int d_head = 64;
    int batch_size = 2;
    int num_heads = 4;
    float scale = 1.0f / sqrtf(d_head);
    
    printf("FlashAttention-2\n");
    printf("N=%d, d=%d, B=%d, H=%d\n", N, d_head, batch_size, num_heads);

    // Size calculation
    long long total_elements = (long long)batch_size * num_heads * N * d_head;
    size_t total_size = total_elements * sizeof(float);
    size_t l_size = (long long)batch_size * num_heads * N * sizeof(float);

    // Host Allocation
    float *h_Q = (float*)malloc(total_size);
    float *h_K = (float*)malloc(total_size);
    float *h_V = (float*)malloc(total_size);
    float *h_O_gpu = (float*)malloc(total_size);
    float *h_O_cpu = (float*)malloc(total_size);
    // Gradients (Host)
    float *h_dO = (float*)malloc(total_size);

    // Initialization
    srand(time(NULL));
    for (long long i = 0; i < total_elements; ++i) {
        h_Q[i] = ((float)rand() / RAND_MAX);
        h_K[i] = ((float)rand() / RAND_MAX);
        h_V[i] = ((float)rand() / RAND_MAX);
        h_dO[i] = ((float)rand() / RAND_MAX); // Random upstream gradient
    }

    // Device Allocation
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    HANDLE_ERROR(cudaMalloc(&d_Q, total_size));
    HANDLE_ERROR(cudaMalloc(&d_K, total_size));
    HANDLE_ERROR(cudaMalloc(&d_V, total_size));
    HANDLE_ERROR(cudaMalloc(&d_O, total_size));
    HANDLE_ERROR(cudaMalloc(&d_L, l_size));
    // Gradient Buffers
    float *d_dO, *d_dQ, *d_dK, *d_dV;
    HANDLE_ERROR(cudaMalloc(&d_dO, total_size));
    HANDLE_ERROR(cudaMalloc(&d_dQ, total_size));
    HANDLE_ERROR(cudaMalloc(&d_dK, total_size));
    HANDLE_ERROR(cudaMalloc(&d_dV, total_size));

    // Copy Inputs to Device
    HANDLE_ERROR(cudaMemcpy(d_Q, h_Q, total_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_K, h_K, total_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_V, h_V, total_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dO, h_dO, total_size, cudaMemcpyHostToDevice));

    // --- RUN FORWARD ---
    dim3 gridDim((N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, num_heads, batch_size);
    dim3 blockDim(32, BLOCK_SIZE_M, 1);
    
    printf("\nLaunching Forward Kernel\n");
    flash_attention_kernel_v2<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, d_L, N, d_head, batch_size, num_heads, scale);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Verify Forward
    HANDLE_ERROR(cudaMemcpy(h_O_gpu, d_O, total_size, cudaMemcpyDeviceToHost));
    attention_cpu(h_Q, h_K, h_V, h_O_cpu, N, d_head, batch_size, num_heads, scale);
    verify_results(h_O_gpu, h_O_cpu, total_elements);

    // --- RUN BACKWARD ---
    printf("\nLaunching Backward Kernel\n");

    // Zero out dK and dV because we use atomicAdd
    HANDLE_ERROR(cudaMemset(d_dK, 0, total_size));
    HANDLE_ERROR(cudaMemset(d_dV, 0, total_size));
    
    // Use same grid/block dims as forward (parallelize over Q rows)
    flash_attention_backward_kernel<<<gridDim, blockDim>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_L,
        d_dQ, d_dK, d_dV,
        N, d_head, batch_size, num_heads, scale
    );
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    printf("Backward Pass Completed Successfully.\n");

    // Cleanup
    free(h_Q); free(h_K); free(h_V); free(h_O_gpu); free(h_O_cpu); free(h_dO);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    cudaFree(d_dO); cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);

    return 0;
}