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

// Kernel Configuration
#define BLOCK_SIZE_M 16 
#define BLOCK_SIZE_N 32  

// FlashAttention2 CUDA Kernel 
__global__ void flash_attention_kernel_v2(
    const float* Q, const float* K, const float* V, float* O,
    float* L, // Store the logsumexp statistics for the backward pass
    int N, int d_head, int batch_size, int num_heads, float scale) 
{
    // Parallelism & work partitioning
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int block_m_idx = blockIdx.x;
    int start_m = block_m_idx * BLOCK_SIZE_M;

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

    for (int block_n_idx = 0; block_n_idx < (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; ++block_n_idx) {
        int start_n = block_n_idx * BLOCK_SIZE_N;
        int row_kv_local = threadIdx.y;
        
        // load K and V tiles
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

        // ONLINE SOFTMAX 
        float m_ij = -FLT_MAX;
        for (int j = 0; j < BLOCK_SIZE_N; ++j) m_ij = fmaxf(m_ij, S_ij_row[j]);
        
        float m_i_new = fmaxf(m_i, m_ij);
        float rescale_factor = expf(m_i - m_i_new);
        
        l_i *= rescale_factor;
        for(int k=0; k<d_head; ++k) acc_o[k] *= rescale_factor;

        // Calculate and add current block's contribution, using the new max for scaling
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

    if (l_i > 0) {
        for (int k = 0; k < d_head; ++k) {
            o_ptr[row_q_global * d_head + k] = acc_o[k] / l_i;
        }
    }
    l_ptr[row_q_global] = m_i + logf(l_i);
}


// CPU Baseline for Verification
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

// Verification Function
void verify_results(const float* O_gpu, const float* O_cpu, long long total_elements) {
    printf("Verifying results:\n");
    double max_error = 0.0;
    for (long long i = 0; i < total_elements; ++i) {
        max_error = fmax(max_error, fabs(O_gpu[i] - O_cpu[i]));
    }

    const double tolerance = 1e-4;
    printf("Max absolute error: %f (Tolerance: %f)\n", max_error, tolerance);
    if (max_error < tolerance) {
        printf("SUCCESS: FlashAttention-2 kernel matches CPU baseline.\n");
    } else {
        printf("FAILURE: Results do not match.\n");
    }
}

int main() {
    // model parameters
    int N = 1024;
    int d_head = 64;
    int batch_size = 2;
    int num_heads = 4;
    float scale = 1.0f / sqrtf(d_head);
    
    printf("Running FlashAttention2\n");
    printf("SeqLen N=%d, d_head=%d, Batch=%d, Heads=%d\n", N, d_head, batch_size, num_heads);

    // Memory Allocation
    long long total_elements = (long long)batch_size * num_heads * N * d_head;
    long long l_elements = (long long)batch_size * num_heads * N;
    size_t total_size = total_elements * sizeof(float);
    size_t l_size = l_elements * sizeof(float);

    float *h_Q, *h_K, *h_V, *h_O_gpu, *h_O_cpu;
    h_Q = (float*)malloc(total_size);
    h_K = (float*)malloc(total_size);
    h_V = (float*)malloc(total_size);
    h_O_gpu = (float*)malloc(total_size);
    h_O_cpu = (float*)malloc(total_size);

    // Data Initialization
    srand(time(NULL));
    for (long long i = 0; i < total_elements; ++i) {
        h_Q[i] = ((float)rand() / RAND_MAX);
        h_K[i] = ((float)rand() / RAND_MAX);
        h_V[i] = ((float)rand() / RAND_MAX);
    }

    // Device Memory
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    HANDLE_ERROR(cudaMalloc(&d_Q, total_size));
    HANDLE_ERROR(cudaMalloc(&d_K, total_size));
    HANDLE_ERROR(cudaMalloc(&d_V, total_size));
    HANDLE_ERROR(cudaMalloc(&d_O, total_size));
    HANDLE_ERROR(cudaMalloc(&d_L, l_size));

    // Copy to Device
    HANDLE_ERROR(cudaMemcpy(d_Q, h_Q, total_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_K, h_K, total_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_V, h_V, total_size, cudaMemcpyHostToDevice));

    // Kernel Launch
    dim3 gridDim((N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, num_heads, batch_size);
    dim3 blockDim(32, BLOCK_SIZE_M, 1);
    
    printf("Launching kernel with Grid: (%d, %d, %d), Block: (%d, %d, %d)\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    flash_attention_kernel_v2<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, d_L, N, d_head, batch_size, num_heads, scale);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Copy Result Back
    HANDLE_ERROR(cudaMemcpy(h_O_gpu, d_O, total_size, cudaMemcpyDeviceToHost));

    // CPU Verification
    printf("Running CPU baseline for verification.\n");
    attention_cpu(h_Q, h_K, h_V, h_O_cpu, N, d_head, batch_size, num_heads, scale);

    verify_results(h_O_gpu, h_O_cpu, total_elements);

    // --- Cleanup ---
    free(h_Q); free(h_K); free(h_V); free(h_O_gpu); free(h_O_cpu);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    
    return 0;
}

