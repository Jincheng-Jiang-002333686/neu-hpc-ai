#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <vector>
#include <cuda_runtime.h>

// CUDA Error Handling
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

// Single-GPU FlashAttention-2 Kernel
__global__ void flash_attention_kernel_v2_single_gpu(
    const float* Q, const float* K, const float* V, float* O, float* L,
    int N, int d_head, float scale) 
{
    int block_m_idx = blockIdx.x;
    int start_m = block_m_idx * BLOCK_SIZE_M;

    // Shared memory for K and V tiles
    __shared__ float K_tile[BLOCK_SIZE_N][64];
    __shared__ float V_tile[BLOCK_SIZE_N][64];

    // ThreadIdx.y handles the row within a Q-block
    int row_q_local = threadIdx.y;
    int row_q_global = start_m + row_q_local;

    if (row_q_global >= N) return; // Guard for out-of-bounds rows

    // Running statistics for online softmax
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc_o[64] = {0.0f}; // Accumulator for the output vector

    // Iterate over blocks of K and V
    for (int block_n_idx = 0; block_n_idx < (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; ++block_n_idx) {
        int start_n = block_n_idx * BLOCK_SIZE_N;
        
        // Load K and V tiles into shared memory
        // Each thread loads multiple elements (striding)
        for(int i = threadIdx.y; i < BLOCK_SIZE_N; i += blockDim.y) {
            int row_kv_global = start_n + i;
            if(row_kv_global < N){
                for (int k = threadIdx.x; k < d_head; k += blockDim.x) {
                    K_tile[i][k] = K[row_kv_global * d_head + k];
                    V_tile[i][k] = V[row_kv_global * d_head + k];
                }
            }
        }
        __syncthreads(); // Ensure K_tile and V_tile are fully loaded

        //Compute S_ij = Q_i * K_j^T 
        float S_ij_row[BLOCK_SIZE_N]; 
        for (int j = 0; j < BLOCK_SIZE_N; ++j) {
            float sum = 0.0f;
            if (start_n + j < N) {
                // Dot product for one Q row with one K row
                for (int k = 0; k < d_head; ++k) sum += Q[row_q_global * d_head + k] * K_tile[j][k];
            } else { 
                sum = -FLT_MAX; // Mask out-of-bounds K values
            }
            S_ij_row[j] = sum * scale;
        }

        //Online Softmax Update 
        float m_ij = -FLT_MAX;
        for (int j = 0; j < BLOCK_SIZE_N; ++j) m_ij = fmaxf(m_ij, S_ij_row[j]);
        
        float m_i_new = fmaxf(m_i, m_ij);
        float rescale_factor = expf(m_i - m_i_new);
        
        l_i *= rescale_factor; // Rescale old sum
        for(int k=0; k<d_head; ++k) acc_o[k] *= rescale_factor; // Rescale old accumulator

        // Add contribution from current block
        float p_ij_sum = 0.0f;
        for (int j = 0; j < BLOCK_SIZE_N; ++j) {
             if (start_n + j < N) {
                float p_val = expf(S_ij_row[j] - m_i_new);
                p_ij_sum += p_val;
                for(int k=0; k<d_head; ++k) acc_o[k] += p_val * V_tile[j][k];
             }
        }
        l_i += p_ij_sum;
        m_i = m_i_new; // Update running max
        __syncthreads(); // Ensure all threads are done before next tile load
    }

    //Write Output
    if (l_i > 1e-6) {
        for (int k = 0; k < d_head; ++k) O[row_q_global * d_head + k] = acc_o[k] / l_i;
    }
    L[row_q_global] = m_i + logf(l_i); // Save logsumexp for backward pass (if needed)
}


// CPU Baseline for Verification
void attention_cpu(const float* Q, const float* K, const float* V, float* O, int N, int d_head, float scale) {
    // 1. S = Q * K^T
    float* S = (float*)malloc((long long)N * N * sizeof(float));
    if (!S) { fprintf(stderr, "CPU S malloc failed\n"); return; }
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_head; ++k) sum += Q[i * d_head + k] * K[j * d_head + k];
            S[i * N + j] = sum * scale;
        }
    }

    // 2. P = softmax(S)
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

    // 3. O = P * V
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < d_head; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) sum += S[i * N + j] * V[j * d_head + k];
            O[i * d_head + k] = sum;
        }
    }
    free(S);
}

// Main Host Function
int main(int argc, char **argv) {
    
    // Set parameters
    int N = 1024; // Sequence Length
    int d_head = 64;
    float scale = 1.0f / sqrtf(d_head);
    
    printf("Running Single-GPU FlashAttention Test\n");
    printf("SeqLen N=%d, d_head=%d\n", N, d_head);

    // Host memory allocation
    size_t total_size = (long long)N * d_head * sizeof(float);
    float *h_Q, *h_K, *h_V, *h_O_gpu, *h_O_cpu;
    h_Q = (float*)malloc(total_size);
    h_K = (float*)malloc(total_size);
    h_V = (float*)malloc(total_size);
    h_O_gpu = (float*)malloc(total_size); // For GPU result
    h_O_cpu = (float*)malloc(total_size); // For CPU baseline result

    // Initialize host data
    srand(time(NULL));
    for (long long i = 0; i < (long long)N * d_head; ++i) {
        h_Q[i] = ((float)rand() / RAND_MAX);
        h_K[i] = ((float)rand() / RAND_MAX);
        h_V[i] = ((float)rand() / RAND_MAX);
    }
    
    // Device memory allocation
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    size_t l_size = (long long)N * sizeof(float);

    HANDLE_ERROR(cudaSetDevice(0)); // Use GPU 0
    HANDLE_ERROR(cudaMalloc(&d_Q, total_size));
    HANDLE_ERROR(cudaMalloc(&d_K, total_size));
    HANDLE_ERROR(cudaMalloc(&d_V, total_size));
    HANDLE_ERROR(cudaMalloc(&d_O, total_size));
    HANDLE_ERROR(cudaMalloc(&d_L, l_size));

    // Copy data from Host to Device
    HANDLE_ERROR(cudaMemcpy(d_Q, h_Q, total_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_K, h_K, total_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_V, h_V, total_size, cudaMemcpyHostToDevice));

    // Launch Kernel
    printf("Launching GPU kernel...\n");
    dim3 gridDim((N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, 1, 1);
    dim3 blockDim(32, BLOCK_SIZE_M, 1);
    flash_attention_kernel_v2_single_gpu<<<gridDim, blockDim>>>(
        d_Q, d_K, d_V, d_O, d_L, N, d_head, scale
    );
    HANDLE_ERROR(cudaDeviceSynchronize());
    printf("GPU kernel finished.\n");

    // Copy result from Device to Host
    HANDLE_ERROR(cudaMemcpy(h_O_gpu, d_O, total_size, cudaMemcpyDeviceToHost));
    
    // Run CPU baseline for verification
    printf("\nRunning CPU baseline for verification...\n");
    attention_cpu(h_Q, h_K, h_V, h_O_cpu, N, d_head, scale);
    printf("CPU baseline finished.\n");

    // Verify results
    double max_error = 0.0;
    for (long long i = 0; i < (long long)N * d_head; ++i) {
        max_error = fmax(max_error, fabs(h_O_gpu[i] - h_O_cpu[i]));
    }
    
    printf("\nVerification Result\n");
    printf("Max absolute error: %f (Tolerance: 1e-4)\n", max_error);
    if (max_error < 1e-4) {
        printf("SUCCESS: Single-GPU kernel matches CPU baseline.\n");
    } else {
        printf("FAILURE: Results do not match.\n");
    }

    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    free(h_Q); free(h_K); free(h_V); free(h_O_gpu); free(h_O_cpu);
    
    return 0;
}