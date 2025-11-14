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

// Distributed Ring Attention Kernel (Works for N >= 1)
__global__ void ring_flash_attention_kernel(
    const float** all_Q, const float** all_K, const float** all_V, 
    float* O, float* L, 
    int N_total, int d_head, float scale,
    int gpu_id, int num_gpus) 
{
    // Each GPU processes a shard of the total sequence
    const int N_per_gpu = N_total / num_gpus;
    const float* q_local_ptr = all_Q[gpu_id];
    
    // Standard block/thread indexing for the local Q shard
    int block_m_idx = blockIdx.x;
    int start_m = block_m_idx * BLOCK_SIZE_M;

    __shared__ float K_tile[BLOCK_SIZE_N][64];
    __shared__ float V_tile[BLOCK_SIZE_N][64];

    int row_q_local_idx = threadIdx.y;
    int row_q_this_gpu = start_m + row_q_local_idx;
    if (row_q_this_gpu >= N_per_gpu) return; // Guard for local Q shard size

    // Running stats for online softmax
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc_o[64] = {0.0f};

    //Ring Attention Loop
    // If num_gpus=1, it runs once, attending to its own K/V.
    // If num_gpus=2, it runs twice, attending to its own and then the other GPU's K/V.
    for (int step = 0; step < num_gpus; ++step) {
        // Calculate which GPU's K/V data to use in this step
        int src_gpu_id = (gpu_id - step + num_gpus) % num_gpus;
        const float* k_remote_ptr = all_K[src_gpu_id];
        const float* v_remote_ptr = all_V[src_gpu_id];

        //Standard FlashAttention logic
        // Iterate over the K/V shard from the source GPU
        for (int block_n_idx = 0; block_n_idx < (N_per_gpu + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; ++block_n_idx) {
            int start_n = block_n_idx * BLOCK_SIZE_N;
            
            // Load K/V tile into shared memory
            for(int i = threadIdx.y; i < BLOCK_SIZE_N; i += blockDim.y) {
                int row_kv_global = start_n + i;
                if(row_kv_global < N_per_gpu){
                    for (int k = threadIdx.x; k < d_head; k += blockDim.x) {
                        K_tile[i][k] = k_remote_ptr[row_kv_global * d_head + k];
                        V_tile[i][k] = v_remote_ptr[row_kv_global * d_head + k];
                    }
                }
            }
            __syncthreads();

            // Compute S_ij = Q_i * K_j^T
            float S_ij_row[BLOCK_SIZE_N]; 
            for (int j = 0; j < BLOCK_SIZE_N; ++j) {
                float sum = 0.0f;
                if (start_n + j < N_per_gpu) {
                    for (int k = 0; k < d_head; ++k) sum += q_local_ptr[row_q_this_gpu * d_head + k] * K_tile[j][k];
                } else { sum = -FLT_MAX; }
                S_ij_row[j] = sum * scale;
            }

            //Online Softmax Update
            float m_ij = -FLT_MAX;
            for (int j = 0; j < BLOCK_SIZE_N; ++j) m_ij = fmaxf(m_ij, S_ij_row[j]);
            float m_i_new = fmaxf(m_i, m_ij);
            float rescale_factor = expf(m_i - m_i_new);
            l_i *= rescale_factor;
            for(int k=0; k<d_head; ++k) acc_o[k] *= rescale_factor;

            float p_ij_sum = 0.0f;
            for (int j = 0; j < BLOCK_SIZE_N; ++j) {
                 if (start_n + j < N_per_gpu) {
                    float p_val = expf(S_ij_row[j] - m_i_new);
                    p_ij_sum += p_val;
                    for(int k=0; k<d_head; ++k) acc_o[k] += p_val * V_tile[j][k];
                 }
            }
            l_i += p_ij_sum;
            m_i = m_i_new;
            __syncthreads();
        }
    }

    // Write Final Output
    // This happens only once, after all 'step's are accumulated.
    if (l_i > 1e-6) {
        for (int k = 0; k < d_head; ++k) O[row_q_this_gpu * d_head + k] = acc_o[k] / l_i;
    }
    L[row_q_this_gpu] = m_i + logf(l_i);
}


// CPU Baseline for Verification
void attention_cpu(const float* Q, const float* K, const float* V, float* O, int N, int d_head, float scale) {
    float* S = (float*)malloc((long long)N * N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_head; ++k) sum += Q[i * d_head + k] * K[j * d_head + k];
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
            for (int j = 0; j < N; ++j) sum += S[i * N + j] * V[j * d_head + k];
            O[i * d_head + k] = sum;
        }
    }
    free(S);
}

// Main Host Function
int main(int argc, char **argv) {
    
    // Get number of available GPUs (N >= 1)
    int num_gpus = 0;
    HANDLE_ERROR(cudaGetDeviceCount(&num_gpus));
    if (num_gpus < 1) {
        fprintf(stderr, "This application requires at least 1 GPU. Found %d.\n", num_gpus);
        return 1;
    }
    printf("Found %d CUDA device(s).\n", num_gpus);

    int N_total = 4096; // Total sequence length
    int d_head = 64;
    float scale = 1.0f / sqrtf(d_head);
    
    printf("Running Scalable FlashAttention (Ring Method)\n");
    printf("Total SeqLen N=%d, d_head=%d, GPUs=%d\n", N_total, d_head, num_gpus);

    if (N_total % num_gpus != 0) {
        fprintf(stderr, "Total sequence length N must be divisible by the number of GPUs.\n");
        return 1;
    }
    int N_per_gpu = N_total / num_gpus;

    // Host memory for all data (for init and verification)
    size_t total_size = (long long)N_total * d_head * sizeof(float);
    float *h_Q, *h_K, *h_V, *h_O_gpu, *h_O_cpu;
    h_Q = (float*)malloc(total_size);
    h_K = (float*)malloc(total_size);
    h_V = (float*)malloc(total_size);
    h_O_gpu = (float*)malloc(total_size); // To gather GPU results
    h_O_cpu = (float*)malloc(total_size); // For CPU baseline

    // Init data
    srand(time(NULL));
    for (long long i = 0; i < (long long)N_total * d_head; ++i) {
        h_Q[i] = ((float)rand() / RAND_MAX);
        h_K[i] = ((float)rand() / RAND_MAX);
        h_V[i] = ((float)rand() / RAND_MAX);
    }
    
    // Allocate memory on each GPU
    std::vector<float*> d_Q_gpus(num_gpus), d_K_gpus(num_gpus), d_V_gpus(num_gpus), d_O_gpus(num_gpus), d_L_gpus(num_gpus);
    size_t chunk_size = N_per_gpu * d_head * sizeof(float);
    size_t l_chunk_size = N_per_gpu * sizeof(float);

    for (int i = 0; i < num_gpus; ++i) {
        HANDLE_ERROR(cudaSetDevice(i));
        HANDLE_ERROR(cudaMalloc(&d_Q_gpus[i], chunk_size));
        HANDLE_ERROR(cudaMalloc(&d_K_gpus[i], chunk_size));
        HANDLE_ERROR(cudaMalloc(&d_V_gpus[i], chunk_size));
        HANDLE_ERROR(cudaMalloc(&d_O_gpus[i], chunk_size));
        HANDLE_ERROR(cudaMalloc(&d_L_gpus[i], l_chunk_size));
        // Enable P2P access (no-op if num_gpus=1)
        for (int j = 0; j < num_gpus; ++j) {
            if (i != j) HANDLE_ERROR(cudaDeviceEnablePeerAccess(j, 0));
        }
    }
    
    // Copy data shards to each GPU
    for (int i = 0; i < num_gpus; ++i) {
        HANDLE_ERROR(cudaSetDevice(i));
        HANDLE_ERROR(cudaMemcpy(d_Q_gpus[i], h_Q + i * N_per_gpu * d_head, chunk_size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_K_gpus[i], h_K + i * N_per_gpu * d_head, chunk_size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_V_gpus[i], h_V + i * N_per_gpu * d_head, chunk_size, cudaMemcpyHostToDevice));
    }

    // Create device-side pointer arrays (for P2P access)
    float **d_all_Q, **d_all_K, **d_all_V;
    HANDLE_ERROR(cudaSetDevice(0)); // Use GPU 0 for this meta-pointer
    HANDLE_ERROR(cudaMalloc(&d_all_Q, num_gpus * sizeof(float*)));
    HANDLE_ERROR(cudaMalloc(&d_all_K, num_gpus * sizeof(float*)));
    HANDLE_ERROR(cudaMalloc(&d_all_V, num_gpus * sizeof(float*)));
    HANDLE_ERROR(cudaMemcpy(d_all_Q, d_Q_gpus.data(), num_gpus * sizeof(float*), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_all_K, d_K_gpus.data(), num_gpus * sizeof(float*), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_all_V, d_V_gpus.data(), num_gpus * sizeof(float*), cudaMemcpyHostToDevice));
    
    // Run GPU Kernel(s)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("\nLaunching GPU kernel(s)...\n");
    cudaEventRecord(start);
    
    // Launch one kernel on each GPU
    for (int i = 0; i < num_gpus; ++i) {
        HANDLE_ERROR(cudaSetDevice(i));
        dim3 gridDim((N_per_gpu + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, 1, 1);
        dim3 blockDim(32, BLOCK_SIZE_M, 1);
        ring_flash_attention_kernel<<<gridDim, blockDim>>>(
            (const float**)d_all_Q, (const float**)d_all_K, (const float**)d_all_V, 
            d_O_gpus[i], d_L_gpus[i], N_total, d_head, scale, i, num_gpus);
    }
    
    // Synchronize all GPUs
    for (int i = 0; i < num_gpus; ++i) {
        HANDLE_ERROR(cudaSetDevice(i));
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    
    printf("GPU execution time: %f ms\n", gpu_ms);

    // Copy results from all GPUs back to host
    for (int i = 0; i < num_gpus; ++i) {
        HANDLE_ERROR(cudaSetDevice(i));
        HANDLE_ERROR(cudaMemcpy(h_O_gpu + i * N_per_gpu * d_head, d_O_gpus[i], chunk_size, cudaMemcpyDeviceToHost));
    }
    
    // Run CPU Baseline
    printf("\nRunning CPU baseline for verification...\n");
    attention_cpu(h_Q, h_K, h_V, h_O_cpu, N_total, d_head, scale);
    printf("CPU baseline finished.\n");

    // Verify
    double max_error = 0.0;
    for (long long i = 0; i < (long long)N_total * d_head; ++i) {
        max_error = fmax(max_error, fabs(h_O_gpu[i] - h_O_cpu[i]));
    }
    
    printf("\nVerification Result\n");
    printf("Max absolute error: %f (Tolerance: 1e-4)\n", max_error);
    if (max_error < 1e-4) {
        printf("SUCCESS: Scalable kernel matches CPU baseline.\n");
    } else {
        printf("FAILURE: Results do not match.\n");
    }

    // Cleanup
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaFree(d_Q_gpus[i]); cudaFree(d_K_gpus[i]); cudaFree(d_V_gpus[i]);
        cudaFree(d_O_gpus[i]); cudaFree(d_L_gpus[i]);
    }
    cudaSetDevice(0);
    cudaFree(d_all_Q); cudaFree(d_all_K); cudaFree(d_all_V);
    free(h_Q); free(h_K); free(h_V); free(h_O_gpu); free(h_O_cpu);
    
    return 0;
}