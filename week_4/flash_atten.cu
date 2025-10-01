#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <cuda_runtime.h>

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

// Kernel Configuration
#define BLOCK_SIZE_M 32 // Br: Block size for rows of Q
#define BLOCK_SIZE_N 32 // Bc: Block size for columns of K/V

// Implements Algorithm 2 from the FlashAttention paper.
__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d_head, float scale) 
{
    // Each thread block processes one block of rows from Q.
    int block_m_idx = blockIdx.x; 
    int start_m = block_m_idx * BLOCK_SIZE_M;

    // Each thread within the block handles one row of the Q block.
    int thread_m_idx = threadIdx.y;
    int row_q = start_m + thread_m_idx;
    
    // Tiles for K and V blocks are stored in shared memory to be reused across Q.
    __shared__ float K_tile[BLOCK_SIZE_N][64]; 
    __shared__ float V_tile[BLOCK_SIZE_N][64]; 

    float m_i = -FLT_MAX; // Running max for the current row
    float l_i = 0.0f;     // Running sum of exponents for the current row
    float acc_o[64] = {0.0f}; // Accumulator for the output vector O

    // Iterate over blocks of K and V 
    for (int block_n_idx = 0; block_n_idx < (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; ++block_n_idx) {
        int start_n = block_n_idx * BLOCK_SIZE_N;

        // Threads in the block collaborate to load K_j and V_j.
        int row_kv = start_n + threadIdx.y;
        if (row_kv < N && threadIdx.y < BLOCK_SIZE_N) {
             for (int k = threadIdx.x; k < d_head; k += blockDim.x) {
                K_tile[threadIdx.y][k] = K[row_kv * d_head + k];
                V_tile[threadIdx.y][k] = V[row_kv * d_head + k];
            }
        }
        __syncthreads(); // Ensure K and V tiles are fully loaded.

        // Each thread computes one row of the S_ij score block.
        float S_ij_row[BLOCK_SIZE_N]; 
        if (row_q < N) {
            for (int j = 0; j < BLOCK_SIZE_N; ++j) {
                float sum = 0.0f;
                int current_k_row = start_n + j;
                if (current_k_row < N) {
                    for (int k = 0; k < d_head; ++k) {
                        sum += Q[row_q * d_head + k] * K_tile[j][k];
                    }
                } else {
                    sum = -FLT_MAX; // Mask out range elements
                }
                S_ij_row[j] = sum * scale;
            }
        }

        // Online Softmax
        float m_ij = -FLT_MAX;
        if (row_q < N) {
            for (int j = 0; j < BLOCK_SIZE_N; ++j) {
                 m_ij = fmaxf(m_ij, S_ij_row[j]);
            }
        }
        
        float p_ij_sum = 0.0f;
        float P_ij_row[BLOCK_SIZE_N];
        if (row_q < N) {
            for (int j = 0; j < BLOCK_SIZE_N; ++j) {
                float val = expf(S_ij_row[j] - m_ij);
                P_ij_row[j] = val;
                p_ij_sum += val;
            }
        }

        float m_i_new = fmaxf(m_i, m_ij);
        float l_i_new = expf(m_i - m_i_new) * l_i + expf(m_ij - m_i_new) * p_ij_sum;

        // Rescale previous accumulator and add new value
        if (row_q < N) {
            float rescale_factor = l_i / l_i_new * expf(m_i - m_i_new);
            for(int k=0; k<d_head; ++k) {
                acc_o[k] *= rescale_factor;
            }

            for (int j = 0; j < BLOCK_SIZE_N; ++j) {
                 if (start_n + j < N) {
                    for(int k=0; k<d_head; ++k) {
                        acc_o[k] += (P_ij_row[j] / l_i_new * expf(m_ij - m_i_new)) * V_tile[j][k];
                    }
                 }
            }
        }

        // Update running stats for the next iteration
        m_i = m_i_new;
        l_i = l_i_new;

        __syncthreads(); // Ensure all threads finish before loading the next K/V tile
    }

    // Write to Global Memory 
    if (row_q < N) {
        for (int k = 0; k < d_head; ++k) {
            O[row_q * d_head + k] = acc_o[k];
        }
    }
}


// CPU Baseline for Verification
void attention_cpu(const float* Q, const float* K, const float* V, float* O, int N, int d_head, float scale) {
    float* S = (float*)malloc(N * N * sizeof(float));
    if (!S) { fprintf(stderr, "CPU malloc failed for S\n"); return; }

    // 1. S = Q * K^T
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_head; ++k) {
                sum += Q[i * d_head + k] * K[j * d_head + k];
            }
            S[i * N + j] = sum * scale;
        }
    }

    // 2. P = softmax(S)
    for (int i = 0; i < N; ++i) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < N; ++j) {
            max_val = fmaxf(max_val, S[i * N + j]);
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            S[i * N + j] = expf(S[i * N + j] - max_val);
            sum_exp += S[i * N + j];
        }
        for (int j = 0; j < N; ++j) {
            if (sum_exp > 0) S[i * N + j] /= sum_exp;
        }
    }

    // 3. O = P * V
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < d_head; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                sum += S[i * N + j] * V[j * d_head + k];
            }
            O[i * d_head + k] = sum;
        }
    }
    free(S);
}

// Verification Function
void verify_results(const float* O_gpu, const float* O_cpu, int N, int d_head) {
    printf("Verifying results:\n");
    double max_error = 0.0;
    for (int i = 0; i < N * d_head; ++i) {
        max_error = fmax(max_error, fabs(O_gpu[i] - O_cpu[i]));
    }

    const double tolerance = 1e-4;
    printf("Max absolute error: %f (Tolerance: %f)\n", max_error, tolerance);
    if (max_error < tolerance) {
        printf("SUCCESS: FlashAttention kernel matches CPU baseline.\n");
    } else {
        printf("FAILURE: Results do not match.\n");
    }
}

int main() {
    int N = 1024;       // Sequence Length
    int d_head = 64;  // Head Dimension
    float scale = 1.0f / sqrtf(d_head);
    
    printf("Running FlashAttention\n");
    printf("Sequence Length (N): %d, Head Dimension (d_head): %d\n", N, d_head);

    // Memory Allocation
    size_t qkv_size = N * d_head * sizeof(float);
    float *h_Q, *h_K, *h_V, *h_O_gpu, *h_O_cpu;
    h_Q = (float*)malloc(qkv_size);
    h_K = (float*)malloc(qkv_size);
    h_V = (float*)malloc(qkv_size);
    h_O_gpu = (float*)malloc(qkv_size);
    h_O_cpu = (float*)malloc(qkv_size);

    // Data Initialization
    srand(time(NULL));
    for (int i = 0; i < N * d_head; ++i) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Device Memory
    float *d_Q, *d_K, *d_V, *d_O;
    HANDLE_ERROR(cudaMalloc(&d_Q, qkv_size));
    HANDLE_ERROR(cudaMalloc(&d_K, qkv_size));
    HANDLE_ERROR(cudaMalloc(&d_V, qkv_size));
    HANDLE_ERROR(cudaMalloc(&d_O, qkv_size));

    // Copy to Device 
    HANDLE_ERROR(cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice));

    // Kernel Launch
    dim3 gridDim((N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, 1, 1);
    dim3 blockDim(16, BLOCK_SIZE_M, 1);
    
    printf("Launching kernel with Grid: (%d, %d, %d), Block: (%d, %d, %d)\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    flash_attention_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, N, d_head, scale);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Copy Result Back
    HANDLE_ERROR(cudaMemcpy(h_O_gpu, d_O, qkv_size, cudaMemcpyDeviceToHost));

    // Run CPU Verification
    printf("Running CPU baseline for verification:\n");
    attention_cpu(h_Q, h_K, h_V, h_O_cpu, N, d_head, scale);

    // Verify 
    verify_results(h_O_gpu, h_O_cpu, N, d_head);

    // Cleanup 
    free(h_Q); free(h_K); free(h_V); free(h_O_gpu); free(h_O_cpu);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    
    return 0;
}

