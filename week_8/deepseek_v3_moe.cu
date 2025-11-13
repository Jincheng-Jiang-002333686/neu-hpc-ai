#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h> // For FLT_MAX

// Simple CUDA error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Simple cuBLAS error checking macro
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d: %s\n", __FILE__, __LINE__, cublasGetStatusString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// Configuration

// Config based on a small test model
const int BATCH_SIZE = 1;
const int SEQ_LEN = 16;
const int HIDDEN_DIM = 512;        // config.hidden_size
const int MOE_INTERMEDIATE_SIZE = 128; // config.moe_intermediate_size
const int N_ROUTED_EXPERTS = 16;       // config.n_routed_experts
const int N_SHARED_EXPERTS = 2;        // config.n_shared_experts
const int TOP_K = 4;                   // config.num_experts_per_tok
const int N_GROUP = 4;                 // config.n_group (Must divide N_ROUTED_EXPERTS)
const int TOPK_GROUP = 2;              // config.topk_group

// Routing config
const bool NORM_TOPK_PROB = true;      // config.norm_topk_prob
const float ROUTED_SCALING_FACTOR = 1.0f; // config.routed_scaling_factor

// Calculated dims
const int NUM_TOKENS = BATCH_SIZE * SEQ_LEN;
const int SHARED_INTERMEDIATE_SIZE = MOE_INTERMEDIATE_SIZE * N_SHARED_EXPERTS;


// Helper Kernels (Element-wise)

/*
 * Fused SiLU (Swish) and Element-wise Multiplication Kernel
 * Implements: output = silu(gate_proj_output) * up_proj_output
 */
__global__ void silu_mul_kernel(float* gate_out, float* up_out, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float gate_val = gate_out[idx];
        float up_val = up_out[idx];
        
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
        float silu_val = gate_val * sigmoid_gate;
        
        output[idx] = silu_val * up_val;
    }
}

/*
 * Kernel to add two tensors (residual connection)
 * Implements: output = input_a + input_b
 */
__global__ void add_residuals_kernel(float* input_a, float* input_b, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = input_a[idx] + input_b[idx];
    }
}


// Host-side MLP Implementation (using cuBLAS)

/*
 * Implements a DeepseekV3MLP:
 * output = down_proj( silu(gate_proj(x)) * up_proj(x) )
 *
 * This is a *host* function that orchestrates cuBLAS calls and kernel launches.
 * This is used for the Shared Experts.
 */
void DeepseekV3MLP_forward(cublasHandle_t handle,
                           float* hidden_states,  // Input: [num_tokens, hidden_dim]
                           float* gate_proj_w,    // Weights: [hidden_dim, intermediate_size]
                           float* up_proj_w,      // Weights: [hidden_dim, intermediate_size]
                           float* down_proj_w,    // Weights: [intermediate_size, hidden_dim]
                           float* output,         // Output: [num_tokens, hidden_dim]
                           float* workspace1,     // Workspace: [num_tokens, intermediate_size]
                           float* workspace2,     // Workspace: [num_tokens, intermediate_size]
                           float* workspace3,     // Workspace: [num_tokens, intermediate_size]
                           int num_tokens,
                           int hidden_dim,
                           int intermediate_size) {
    
    float alpha = 1.0f;
    float beta = 0.0f;
    int M = num_tokens;
    int N_intermediate = intermediate_size;
    int K_hidden = hidden_dim;

    // 1. gate_proj(x)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N_intermediate, K_hidden,
                             &alpha,
                             hidden_states, M,
                             gate_proj_w, K_hidden,
                             &beta,
                             workspace1, M)); // workspace1 = gate_proj(x)

    // 2. up_proj(x)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N_intermediate, K_hidden,
                             &alpha,
                             hidden_states, M,
                             up_proj_w, K_hidden,
                             &beta,
                             workspace2, M)); // workspace2 = up_proj(x)

    // 3. silu(gate_proj(x)) * up_proj(x)
    int num_elements = num_tokens * intermediate_size;
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    silu_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        workspace1,      // gate_proj_output
        workspace2,      // up_proj_output
        workspace3,      // output of kernel
        num_elements
    );
    CHECK_CUDA(cudaGetLastError());

    // 4. down_proj(...)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, K_hidden, N_intermediate,
                             &alpha,
                             workspace3, M,
                             down_proj_w, N_intermediate,
                             &beta,
                             output, M)); // output = down_proj(...)
}


// Gating (DeepseekTopKRouter) - Host Function

/*
 * Host-side function for the Gate. This is just one Linear layer.
 * output = gate(hidden_states)
 */
void gate_forward(cublasHandle_t handle,
                  float* hidden_states, // Input: [num_tokens, hidden_dim]
                  float* gate_w,        // Weights: [hidden_dim, n_routed_experts]
                  float* router_logits  // Output: [num_tokens, n_routed_experts]
) {
    float alpha = 1.0f;
    float beta = 0.0f;
    int M = NUM_TOKENS;
    int N = N_ROUTED_EXPERTS;
    int K = HIDDEN_DIM;

    // router_logits = hidden_states * gate_w
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             hidden_states, M,
                             gate_w, K,
                             &beta,
                             router_logits, M));
}


// Routing Logic - 5 Kernel Chain (Replaces route_tokens_to_experts)

/*
 * KERNEL 1: Fused Sigmoid + Bias
 * Implements:
 * 1. sigmoid_scores = sigmoid(router_logits)
 * 2. biased_scores = sigmoid_scores + e_score_correction_bias
 */
__global__ void sigmoid_and_bias_kernel(
    float* router_logits,       // Input: [num_tokens, n_routed_experts]
    float* bias,                // Input: [n_routed_experts]
    float* sigmoid_scores,      // Output: [num_tokens, n_routed_experts]
    float* biased_scores,       // Output: [num_tokens, n_routed_experts]
    int num_tokens,
    int n_routed_experts
) {
    // Launch with 2D grid
    int t = blockIdx.x; // Token index
    int e = blockIdx.y * blockDim.x + threadIdx.y; // Expert index (strided)

    if (t < num_tokens && e < n_routed_experts) {
        int idx = t * n_routed_experts + e;
        
        float logit = router_logits[idx];
        float sig_score = 1.0f / (1.0f + expf(-logit));
        float biased_val = sig_score + bias[e];

        sigmoid_scores[idx] = sig_score;
        biased_scores[idx] = biased_val;
    }
}

/*
 * KERNEL 2: Grouped Top-2 Sum
 *
 * Implements:
 * 1. Logically view biased_scores as [token, group, experts_per_group]
 * 2. Find the Top-2 scores within each group
 * 3. Sum the Top-2 scores
 * 4. Write one score per group to d_group_scores
 *
 * Grid: dim3(NUM_TOKENS, N_GROUP)
 * Block: dim3(256)
 */
__global__ void grouped_top2_sum_kernel(
    float* biased_scores,  // Input: [num_tokens, n_routed_experts]
    float* group_scores,   // Output: [num_tokens, n_group]
    int num_tokens,
    int n_group,
    int n_routed_experts
) {
    // --- Shared memory for the block-wide reduction ---
    __shared__ float s_top1s[256];
    __shared__ float s_top2s[256];

    // --- 1. Identify work ---
    int t = blockIdx.x; // This block handles token 't'
    int g = blockIdx.y; // This block handles group 'g'
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int experts_per_group = n_routed_experts / n_group;
    int group_start_idx = g * experts_per_group;
    int group_end_idx = (g + 1) * experts_per_group;

    // Pointer to the start of this token's expert scores
    float* token_scores = biased_scores + t * n_routed_experts;

    // --- 2. Each thread finds its local Top-2 ---
    float my_top1 = -FLT_MAX;
    float my_top2 = -FLT_MAX;

    // Use a grid-stride loop for each thread to scan its portion of the group
    for (int e = group_start_idx + tid; e < group_end_idx; e += block_size) {
        float val = token_scores[e];

        if (val > my_top1) {
            my_top2 = my_top1;
            my_top1 = val;
        } else if (val > my_top2) {
            my_top2 = val;
        }
    }

    // --- 3. Store local Top-2s in shared memory ---
    s_top1s[tid] = my_top1;
    s_top2s[tid] = my_top2;
    __syncthreads();

    // --- 4. Perform block-wide Top-2 reduction ---
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float my1 = s_top1s[tid];
            float my2 = s_top2s[tid];
            float other1 = s_top1s[tid + s];
            float other2 = s_top2s[tid + s];

            if (my1 > other1) {
                s_top2s[tid] = max(my2, other1);
            } else {
                s_top1s[tid] = other1;
                s_top2s[tid] = max(my1, other2);
            }
        }
        __syncthreads();
    }

    // --- 5. Thread 0 writes the final sum ---
    if (tid == 0) {
        float final_top1 = s_top1s[0];
        float final_top2 = s_top2s[0];
        int output_idx = t * n_group + g;
        
        if (final_top1 == -FLT_MAX) final_top1 = 0.0f;
        if (final_top2 == -FLT_MAX) final_top2 = 0.0f;
        
        group_scores[output_idx] = final_top1 + final_top2;
    }
}

/*
 * KERNEL 3: Group Top-K Mask
 *
 * Implements:
 * 1. For each token, find the indices of the Top-K (TOPK_GROUP) group scores.
 * 2. Write a '1.0f' to d_group_mask for those indices, and '0.0f' otherwise.
 *
 * Grid: dim3(NUM_TOKENS)
 * Block: dim3(256) (Must be power of 2)
 */
__global__ void group_topk_mask_kernel(
    float* group_scores,   // Input: [num_tokens, n_group]
    float* group_mask,     // Output: [num_tokens, n_group]
    int num_tokens,
    int n_group,
    int topk_group
) {
    extern __shared__ float s_data[];
    float* s_scores = s_data;                      // Size: [n_group]
    int* s_indices = (int*)&s_scores[n_group];     // Size: [n_group]

    volatile float* s_reduction_vals = (float*)&s_indices[n_group]; // Size: [blockDim.x]
    volatile int* s_reduction_idxs = (int*)&s_reduction_vals[blockDim.x]; // Size: [blockDim.x]

    int t = blockIdx.x; // This block handles token 't'
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    float* my_scores = group_scores + t * n_group;
    float* my_mask = group_mask + t * n_group;

    // Load token's group scores and initialize mask
    for (int g = tid; g < n_group; g += block_size) {
        s_scores[g] = my_scores[g];
        s_indices[g] = g; 
        my_mask[g] = 0.0f;
    }
    __syncthreads();

    // Serial loop: repeat TOPK_GROUP times
    for (int k = 0; k < topk_group; k++) {
        
        // Block-wide (Arg)Max Reduction
        float my_max_val = -FLT_MAX;
        int my_max_idx = -1;
        for (int g = tid; g < n_group; g += block_size) {
            if (s_scores[g] > my_max_val) {
                my_max_val = s_scores[g];
                my_max_idx = s_indices[g];
            }
        }
        s_reduction_vals[tid] = my_max_val;
        s_reduction_idxs[tid] = my_max_idx;
        __syncthreads();

        for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_reduction_vals[tid + s] > s_reduction_vals[tid]) {
                    s_reduction_vals[tid] = s_reduction_vals[tid + s];
                    s_reduction_idxs[tid] = s_reduction_idxs[tid + s];
                }
            }
            __syncthreads();
        }

        // Thread 0 writes the mask and "consumes" the score
        if (tid == 0) {
            int winning_group_idx = s_reduction_idxs[0];
            if (winning_group_idx != -1) {
                my_mask[winning_group_idx] = 1.0f;
                for(int g = 0; g < n_group; g++) {
                    if (s_indices[g] == winning_group_idx) {
                        s_scores[g] = -FLT_MAX;
                        break;
                    }
                }
            }
        }
        __syncthreads(); 
    }
}

/*
 * KERNEL 4: Masked Final Top-K
 *
 * Implements:
 * 1. "Broadcast" the group_mask to the full expert score list.
 * 2. Apply this mask to the biased_scores.
 * 3. Find the Top-K (config.top_k) expert indices from the masked list.
 * 4. Write final indices to d_topk_indices.
 *
 * Grid: dim3(NUM_TOKENS)
 * Block: dim3(256)
 */
__global__ void masked_final_topk_kernel(
    float* biased_scores,   // Input: [num_tokens, n_routed_experts]
    float* group_mask,      // Input: [num_tokens, n_group]
    int* topk_indices,      // Output: [num_tokens, top_k]
    int num_tokens,
    int n_group,
    int n_routed_experts,
    int top_k
) {
    extern __shared__ float s_data[];
    float* s_scores = s_data;                           // Size: [n_routed_experts]
    int* s_indices = (int*)&s_scores[n_routed_experts]; // Size: [n_routed_experts]
    float* s_group_mask = (float*)&s_indices[n_routed_experts]; // Size: [n_group]
    volatile float* s_reduction_vals = &s_group_mask[n_group];
    volatile int* s_reduction_idxs = (int*)&s_reduction_vals[blockDim.x];
    
    int t = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int experts_per_group = n_routed_experts / n_group;

    float* my_scores_in = biased_scores + t * n_routed_experts;
    float* my_mask_in = group_mask + t * n_group;
    int* my_indices_out = topk_indices + t * top_k;

    // Load group mask
    for (int g = tid; g < n_group; g += block_size) {
        s_group_mask[g] = my_mask_in[g];
    }
    
    // Load biased scores
    for (int e = tid; e < n_routed_experts; e += block_size) {
        s_scores[e] = my_scores_in[e];
        s_indices[e] = e;
    }
    __syncthreads();

    // Apply mask in shared memory
    for (int e = tid; e < n_routed_experts; e += block_size) {
        int g = e / experts_per_group;
        if (s_group_mask[g] == 0.0f) {
            s_scores[e] = -FLT_MAX;
        }
    }
    __syncthreads();

    // Serial loop: repeat TOP_K times (Selection Sort)
    for (int k = 0; k < top_k; k++) {
        
        float my_max_val = -FLT_MAX;
        int my_max_idx = -1;
        for (int e = tid; e < n_routed_experts; e += block_size) {
            if (s_scores[e] > my_max_val) {
                my_max_val = s_scores[e];
                my_max_idx = s_indices[e];
            }
        }
        s_reduction_vals[tid] = my_max_val;
        s_reduction_idxs[tid] = my_max_idx;
        __syncthreads();

        for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_reduction_vals[tid + s] > s_reduction_vals[tid]) {
                    s_reduction_vals[tid] = s_reduction_vals[tid + s];
                    s_reduction_idxs[tid] = s_reduction_idxs[tid + s];
                }
            }
            __syncthreads();
        }

        // Thread 0 writes the winning index
        if (tid == 0) {
            int winning_expert_idx = s_reduction_idxs[0];
            my_indices_out[k] = winning_expert_idx; 

            if (winning_expert_idx != -1) {
                for (int e = 0; e < n_routed_experts; e++) {
                    if (s_indices[e] == winning_expert_idx) {
                        s_scores[e] = -FLT_MAX;
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }
}

/*
 * KERNEL 5: Gather, Normalize, & Scale
 *
 * Implements:
 * 1. Gather the original sigmoid scores using the final top-k indices.
 * 2. Sum the gathered scores for each token.
 * 3. (Optional) Normalize the weights by the sum.
 * 4. Scale the weights by the routed_scaling_factor.
 * 5. Write the final weights to d_topk_weights.
 *
 * Grid: (NUM_TOKENS + 255) / 256
 * Block: 256
 */
__global__ void gather_norm_scale_kernel(
    float* sigmoid_scores,      // Input: [num_tokens, n_routed_experts]
    int* topk_indices,          // Input: [num_tokens, top_k]
    float* topk_weights,        // Output: [num_tokens, top_k]
    int num_tokens,
    int n_routed_experts,
    int top_k,
    bool norm_topk_prob,
    float routed_scaling_factor
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) {
        return;
    }

    // Use a small, on-stack array for the local weights
    float local_weights[TOP_K];
    float sum = 0.0f;

    int* my_indices = topk_indices + t * top_k;
    float* my_scores = sigmoid_scores + t * n_routed_experts;

    for (int k = 0; k < top_k; k++) {
        int expert_idx = my_indices[k];
        
        float weight = 0.0f;
        if (expert_idx != -1) { // Handle -1 index from K4
             weight = my_scores[expert_idx];
        }

        local_weights[k] = weight;
        sum += weight;
    }

    // Calculate final scale factor
    float final_scale = routed_scaling_factor;
    if (norm_topk_prob) {
        final_scale /= (sum + 1e-20f);
    }

    // Apply scale and write to global memory
    float* my_weights_out = topk_weights + t * top_k;
    for (int k = 0; k < top_k; k++) {
        my_weights_out[k] = local_weights[k] * final_scale;
    }
}


// Routed Experts (DeepseekV3NaiveMoe)

/*
 * DEVICE FUNCTION: Naive MLP (Block-wide)
 *
 * This function is called by a block of threads, where
 * blockDim.x = MOE_INTERMEDIATE_SIZE.
 *
 * It performs:
 * output = down_proj( silu(gate_proj(x)) * up_proj(x) )
 *
 * This involves 3 block-wide GEMV operations.
 */
__device__ void naive_routed_expert_mlp(
    int expert_id,
    float* token_input,                 // Input: [H] (in shared mem)
    float* token_output,                // Output: [H] (in shared mem)
    float* all_experts_gate_proj_w,     // Global weights: [E, H, I]
    float* all_experts_up_proj_w,       // Global weights: [E, H, I]
    float* all_experts_down_proj_w,     // Global weights: [E, I, H]
    float* intermediate_workspace_1,    // Shared mem workspace: [I]
    float* intermediate_workspace_2     // Shared mem workspace: [I]
) {
    const int H = HIDDEN_DIM;
    const int I = MOE_INTERMEDIATE_SIZE;
    const int tid = threadIdx.x; // Thread ID, 0 to I-1
    const int block_size = blockDim.x; // Should be == I

    // --- 1. Compute gate_proj(x) --- (1, H) * (H, I) -> (1, I)
    float* gate_w = all_experts_gate_proj_w + (long long)expert_id * H * I;
    float sum = 0.0f;
    for (int h = 0; h < H; h++) {
        sum += token_input[h] * gate_w[h * I + tid]; // gate_w[h][i]
    }
    intermediate_workspace_1[tid] = sum;

    // --- 2. Compute up_proj(x) --- (1, H) * (H, I) -> (1, I)
    float* up_w = all_experts_up_proj_w + (long long)expert_id * H * I;
    sum = 0.0f;
    for (int h = 0; h < H; h++) {
        sum += token_input[h] * up_w[h * I + tid]; // up_w[h][i]
    }
    intermediate_workspace_2[tid] = sum;
    __syncthreads(); 

    // --- 3. Compute silu(gate_proj) * up_proj ---
    float gate_val = intermediate_workspace_1[tid];
    float up_val = intermediate_workspace_2[tid];
    float sig_val = 1.0f / (1.0f + expf(-gate_val));
    intermediate_workspace_1[tid] = (gate_val * sig_val) * up_val;
    __syncthreads(); 

    // --- 4. Compute down_proj(...) --- (1, I) * (I, H) -> (1, H)
    float* down_w = all_experts_down_proj_w + (long long)expert_id * I * H;
    
    // Grid-stride loop: each thread (i) computes multiple outputs (h)
    for (int h = tid; h < H; h += block_size) {
        float down_sum = 0.0f;
        for (int i_loop = 0; i_loop < I; i_loop++) {
            down_sum += intermediate_workspace_1[i_loop] * down_w[i_loop * H + h]; // down_w[i][h]
        }
        token_output[h] = down_sum;
    }
    __syncthreads();
}


/*
 * KERNEL: Naive Routed Experts Forward (UPDATED)
 *
 * We launch one CUDA block per token.
 * LAUNCH CONFIG:
 * Grid:  dim3(NUM_TOKENS)
 * Block: dim3(MOE_INTERMEDIATE_SIZE) (must be <= 1024)
 */
__global__ void routed_experts_naive_kernel(
    float* hidden_states,        // Input: [num_tokens, hidden_dim]
    int* topk_indices,           // Input: [num_tokens, top_k]
    float* topk_weights,         // Input: [num_tokens, top_k]
    float* all_experts_gate_proj_w, // All weights: [E, H, I]
    float* all_experts_up_proj_w,   // All weights: [E, H, I]
    float* all_experts_down_proj_w, // All weights: [E, I, H]
    float* final_output          // Output: [num_tokens, hidden_dim]
) {
    extern __shared__ float shmem[];
    float* sh_token_input = shmem;                                // Size: [H]
    float* sh_token_output_acc = &shmem[HIDDEN_DIM];              // Size: [H]
    float* sh_expert_output = &shmem[HIDDEN_DIM * 2];             // Size: [H]
    float* sh_mlp_workspace_1 = &shmem[HIDDEN_DIM * 3];           // Size: [I]
    float* sh_mlp_workspace_2 = &shmem[HIDDEN_DIM * 3 + MOE_INTERMEDIATE_SIZE]; // Size: [I]

    int t = blockIdx.x; // This block handles token 't'
    int i = threadIdx.x; // This thread handles dim 'i' (0 to I-1)
    int block_size = blockDim.x; // == MOE_INTERMEDIATE_SIZE

    // --- 1. Load token input and initialize accumulators ---
    for (int h = i; h < HIDDEN_DIM; h += block_size) {
        sh_token_input[h] = hidden_states[t * HIDDEN_DIM + h];
        sh_token_output_acc[h] = 0.0f;
    }
    __syncthreads();

    // --- 2. Loop over the top-k experts for this token ---
    for (int k = 0; k < TOP_K; k++) {
        int expert_id = topk_indices[t * TOP_K + k];
        float expert_weight = topk_weights[t * TOP_K + k];
        
        if (expert_id == -1) continue; // Skip if K4 returned invalid index

        // 3. Call the device-level MLP function
        naive_routed_expert_mlp(
            expert_id,
            sh_token_input,
            sh_expert_output, 
            all_experts_gate_proj_w,
            all_experts_up_proj_w,
            all_experts_down_proj_w,
            sh_mlp_workspace_1,
            sh_mlp_workspace_2
        );
        __syncthreads();

        // --- 4. Weight and accumulate the result ---
        for (int h = i; h < HIDDEN_DIM; h += block_size) {
            sh_token_output_acc[h] += sh_expert_output[h] * expert_weight;
        }
        __syncthreads(); 
    }

    // --- 5. Write final accumulated output to global memory ---
    for (int h = i; h < HIDDEN_DIM; h += block_size) {
        final_output[t * HIDDEN_DIM + h] = sh_token_output_acc[h];
    }
}


// Top-Level MoE Forward Function (Host)

// This is a simple struct to hold all the device pointers for weights
struct MoEWeights {
    // Gate
    float* d_gate_w; // [H, E]
    float* d_e_score_correction_bias; // [E]
    
    // Shared Experts
    float* d_shared_gate_proj_w; // [H, I_shared]
    float* d_shared_up_proj_w;   // [H, I_shared]
    float* d_shared_down_proj_w; // [I_shared, H]
    
    // Routed Experts
    float* d_routed_gate_proj_w; // [E, H, I_moe]
    float* d_routed_up_proj_w;   // [E, H, I_moe]
    float* d_routed_down_proj_w; // [E, I_moe, H]
};

// This struct holds all the intermediate activation/workspace buffers
struct MoEWorkspace {
    float* d_residuals;
    float* d_router_logits;
    int* d_topk_indices;
    float* d_topk_weights;
    float* d_routed_expert_output;
    float* d_shared_expert_output;
    
    // Workspaces for the MLPs
    float* d_shared_mlp_ws1;
    float* d_shared_mlp_ws2;
    float* d_shared_mlp_ws3;
    
    // Workspaces for the 5-kernel routing chain
    float* d_sigmoid_scores;
    float* d_biased_scores;
    float* d_group_scores;
    float* d_group_mask;
};


/*
 * Top-level host function to execute the full DeepseekV3MoE forward pass.
 * This function orchestrates all the kernel launches and cuBLAS calls.
 */
void deepseek_v3_moe_forward(cublasHandle_t handle,
                             float* d_hidden_states, // Input
                             MoEWeights& weights,
                             MoEWorkspace& workspace,
                             float* d_output         // Output
) {
    // 0. Copy hidden_states to residuals
    CHECK_CUDA(cudaMemcpy(workspace.d_residuals, d_hidden_states,
                          NUM_TOKENS * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToDevice));

    // --- Path A: Routed Experts ---
    // 1. Gate (DeepseekTopKRouter)
    gate_forward(handle, d_hidden_states, weights.d_gate_w, workspace.d_router_logits);

    // 2. Route Tokens (The 5-Kernel Chain)
    // 2.1. K1: Sigmoid + Bias
    dim3 k1_grid(NUM_TOKENS);
    dim3 k1_block((N_ROUTED_EXPERTS + 255) / 256 * 256); // 1D block
    if (N_ROUTED_EXPERTS < 256) k1_block.x = N_ROUTED_EXPERTS;
    else k1_block.x = 256;
    k1_grid.y = (N_ROUTED_EXPERTS + k1_block.x - 1) / k1_block.x; // 2D Grid
    k1_block.x = 1; // Swap grid/block for striding
    
    sigmoid_and_bias_kernel<<<k1_grid, k1_block>>>(
        workspace.d_router_logits,
        weights.d_e_score_correction_bias,
        workspace.d_sigmoid_scores,
        workspace.d_biased_scores,
        NUM_TOKENS,
        N_ROUTED_EXPERTS
    );
    CHECK_CUDA(cudaGetLastError());

    // 2.2. K2: Grouped Top-2 Sum
    dim3 k2_grid(NUM_TOKENS, N_GROUP);
    dim3 k2_block(256);
    grouped_top2_sum_kernel<<<k2_grid, k2_block>>>(
        workspace.d_biased_scores,
        workspace.d_group_scores,
        NUM_TOKENS,
        N_GROUP,
        N_ROUTED_EXPERTS
    );
    CHECK_CUDA(cudaGetLastError());

    // 2.3. K3: Group Top-K Mask
    dim3 k3_grid(NUM_TOKENS);
    dim3 k3_block(256);
    size_t k3_shmem = (N_GROUP * sizeof(float)) + (N_GROUP * sizeof(int)) +
                      (256 * sizeof(float)) + (256 * sizeof(int));
    group_topk_mask_kernel<<<k3_grid, k3_block, k3_shmem>>>(
        workspace.d_group_scores,
        workspace.d_group_mask,
        NUM_TOKENS,
        N_GROUP,
        TOPK_GROUP
    );
    CHECK_CUDA(cudaGetLastError());
    
    // 2.4. K4: Masked Final Top-K
    dim3 k4_grid(NUM_TOKENS);
    dim3 k4_block(256);
    size_t k4_shmem = (N_ROUTED_EXPERTS * sizeof(float)) + (N_ROUTED_EXPERTS * sizeof(int)) +
                      (N_GROUP * sizeof(float)) +
                      (256 * sizeof(float)) + (256 * sizeof(int));
    masked_final_topk_kernel<<<k4_grid, k4_block, k4_shmem>>>(
        workspace.d_biased_scores,
        workspace.d_group_mask,
        workspace.d_topk_indices,
        NUM_TOKENS,
        N_GROUP,
        N_ROUTED_EXPERTS,
        TOP_K
    );
    CHECK_CUDA(cudaGetLastError());
    
    // 2.5. K5: Gather, Normalize, & Scale
    dim3 k5_grid((NUM_TOKENS + 255) / 256);
    dim3 k5_block(256);
    gather_norm_scale_kernel<<<k5_grid, k5_block>>>(
        workspace.d_sigmoid_scores,
        workspace.d_topk_indices,
        workspace.d_topk_weights,
        NUM_TOKENS,
        N_ROUTED_EXPERTS,
        TOP_K,
        NORM_TOPK_PROB,
        ROUTED_SCALING_FACTOR
    );
    CHECK_CUDA(cudaGetLastError());

    // 3. Routed Experts (DeepseekV3NaiveMoe)
    size_t routed_shmem = (HIDDEN_DIM * 3 + MOE_INTERMEDIATE_SIZE * 2) * sizeof(float);
    if (routed_shmem > 48 * 1024) { 
        printf("Warning: Routed kernel shared memory size (%zu bytes) exceeds 48KB limit.\n", routed_shmem);
    }
    dim3 routed_grid(NUM_TOKENS);
    dim3 routed_block(MOE_INTERMEDIATE_SIZE);
    
    routed_experts_naive_kernel<<<routed_grid, routed_block, routed_shmem>>>(
        d_hidden_states,
        workspace.d_topk_indices,
        workspace.d_topk_weights,
        weights.d_routed_gate_proj_w,
        weights.d_routed_up_proj_w,
        weights.d_routed_down_proj_w,
        workspace.d_routed_expert_output
    );
    CHECK_CUDA(cudaGetLastError());

    // --- Path B: Shared Experts ---
    // 4. Shared Experts (DeepseekV3MLP)
    DeepseekV3MLP_forward(
        handle,
        workspace.d_residuals, // Shared experts run on the original residuals
        weights.d_shared_gate_proj_w,
        weights.d_shared_up_proj_w,
        weights.d_shared_down_proj_w,
        workspace.d_shared_expert_output,
        workspace.d_shared_mlp_ws1,
        workspace.d_shared_mlp_ws2,
        workspace.d_shared_mlp_ws3,
        NUM_TOKENS,
        HIDDEN_DIM,
        SHARED_INTERMEDIATE_SIZE
    );

    // 5. Add outputs: hidden_states = routed_output + shared_output
    int total_elements = NUM_TOKENS * HIDDEN_DIM;
    int add_threads = 256;
    int add_blocks = (total_elements + add_threads - 1) / add_threads;
    
    add_residuals_kernel<<<add_blocks, add_threads>>>(
        workspace.d_routed_expert_output,
        workspace.d_shared_expert_output,
        d_output,
        total_elements
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}


// Simple Test 

// Helper to allocate and initialize a tensor with random data
void init_tensor(float** h_ptr, float** d_ptr, long long num_elements) {
    *h_ptr = (float*)malloc(num_elements * sizeof(float));
    for (long long i = 0; i < num_elements; i++) {
        (*h_ptr)[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f; 
    }
    CHECK_CUDA(cudaMalloc(d_ptr, num_elements * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(*d_ptr, *h_ptr, num_elements * sizeof(float), cudaMemcpyHostToDevice));
}

// Helper to free host and device memory
void free_tensor(float* h_ptr, float* d_ptr) {
    if (h_ptr) free(h_ptr);
    if (d_ptr) CHECK_CUDA(cudaFree(d_ptr));
}
void free_tensor_dev(void* d_ptr) {
    if (d_ptr) CHECK_CUDA(cudaFree(d_ptr));
}


int main() {
    printf("--- DeepseekV3MoE CUDA Full Implementation Test ---\n");
    printf("Config: B=%d, S=%d, H=%d, E_routed=%d, E_shared=%d, K=%d\n",
           BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, N_ROUTED_EXPERTS, N_SHARED_EXPERTS, TOP_K);
    printf("Routing: N_Group=%d, TopK_Group=%d\n", N_GROUP, TOPK_GROUP);

    if (HIDDEN_DIM > 1024) {
        fprintf(stderr, "Error: HIDDEN_DIM (%d) > 1024. Shared mem may be an issue.\n", HIDDEN_DIM);
    }
    if (MOE_INTERMEDIATE_SIZE > 1024) {
        fprintf(stderr, "Error: MOE_INTERMEDIATE_SIZE (%d) > 1024. The routed kernel blockDim will fail.\n", MOE_INTERMEDIATE_SIZE);
        return 1;
    }
    if (N_ROUTED_EXPERTS % N_GROUP != 0) {
        fprintf(stderr, "Error: N_ROUTED_EXPERTS (%d) must be divisible by N_GROUP (%d).\n", N_ROUTED_EXPERTS, N_GROUP);
        return 1;
    }

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    // --- Allocate Host and Device Memory ---
    MoEWeights weights;
    MoEWorkspace workspace;
    float *h_input, *d_input, *h_output, *d_output;
    float* h_dummy; // Host dummy for init
    
    // Inputs/Outputs
    init_tensor(&h_input, &d_input, NUM_TOKENS * HIDDEN_DIM);
    h_output = (float*)malloc(NUM_TOKENS * HIDDEN_DIM * sizeof(float));
    CHECK_CUDA(cudaMalloc(&d_output, NUM_TOKENS * HIDDEN_DIM * sizeof(float)));

    // Weights
    init_tensor(&h_dummy, &weights.d_gate_w, (long long)HIDDEN_DIM * N_ROUTED_EXPERTS);
    init_tensor(&h_dummy, &weights.d_e_score_correction_bias, N_ROUTED_EXPERTS);
    init_tensor(&h_dummy, &weights.d_shared_gate_proj_w, (long long)HIDDEN_DIM * SHARED_INTERMEDIATE_SIZE);
    init_tensor(&h_dummy, &weights.d_shared_up_proj_w,   (long long)HIDDEN_DIM * SHARED_INTERMEDIATE_SIZE);
    init_tensor(&h_dummy, &weights.d_shared_down_proj_w, (long long)SHARED_INTERMEDIATE_SIZE * HIDDEN_DIM);
    init_tensor(&h_dummy, &weights.d_routed_gate_proj_w, (long long)N_ROUTED_EXPERTS * HIDDEN_DIM * MOE_INTERMEDIATE_SIZE);
    init_tensor(&h_dummy, &weights.d_routed_up_proj_w,   (long long)N_ROUTED_EXPERTS * HIDDEN_DIM * MOE_INTERMEDIATE_SIZE);
    init_tensor(&h_dummy, &weights.d_routed_down_proj_w, (long long)N_ROUTED_EXPERTS * MOE_INTERMEDIATE_SIZE * HIDDEN_DIM);
    free(h_dummy); // We only needed one dummy

    // Workspace
    CHECK_CUDA(cudaMalloc(&workspace.d_residuals, (long long)NUM_TOKENS * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&workspace.d_router_logits, (long long)NUM_TOKENS * N_ROUTED_EXPERTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&workspace.d_topk_indices, (long long)NUM_TOKENS * TOP_K * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&workspace.d_topk_weights, (long long)NUM_TOKENS * TOP_K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&workspace.d_routed_expert_output, (long long)NUM_TOKENS * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&workspace.d_shared_expert_output, (long long)NUM_TOKENS * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&workspace.d_shared_mlp_ws1, (long long)NUM_TOKENS * SHARED_INTERMEDIATE_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&workspace.d_shared_mlp_ws2, (long long)NUM_TOKENS * SHARED_INTERMEDIATE_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&workspace.d_shared_mlp_ws3, (long long)NUM_TOKENS * SHARED_INTERMEDIATE_SIZE * sizeof(float)));
    
    // Routing Workspaces
    CHECK_CUDA(cudaMalloc(&workspace.d_sigmoid_scores, (long long)NUM_TOKENS * N_ROUTED_EXPERTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&workspace.d_biased_scores,  (long long)NUM_TOKENS * N_ROUTED_EXPERTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&workspace.d_group_scores, (long long)NUM_TOKENS * N_GROUP * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&workspace.d_group_mask,   (long long)NUM_TOKENS * N_GROUP * sizeof(float)));
    
    printf("Memory allocated. Running forward pass...\n");

    // --- Run the MoE Forward Pass ---
    deepseek_v3_moe_forward(cublas_handle, d_input, weights, workspace, d_output);
    
    printf("Forward pass complete.\n");

    // --- Copy results back and print ---
    CHECK_CUDA(cudaMemcpy(h_output, d_output, NUM_TOKENS * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n--- Output (first 5 values of token 0) ---\n");
    for (int i = 0; i < 5 && i < HIDDEN_DIM; i++) {
        printf("h_output[0][%d] = %f\n", i, h_output[i]);
    }
    printf("\n--- Input (first 5 values of token 0) ---\n");
    for (int i = 0; i < 5 && i < HIDDEN_DIM; i++) {
        printf("h_input[0][%d] = %f\n", i, h_input[i]);
    }

    // --- Cleanup ---
    printf("\nCleaning up memory...\n");
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    free_tensor(h_input, d_input);
    free(h_output);
    free_tensor_dev(d_output);

    free_tensor_dev(weights.d_gate_w);
    free_tensor_dev(weights.d_e_score_correction_bias);
    free_tensor_dev(weights.d_shared_gate_proj_w);
    free_tensor_dev(weights.d_shared_up_proj_w);
    free_tensor_dev(weights.d_shared_down_proj_w);
    free_tensor_dev(weights.d_routed_gate_proj_w);
    free_tensor_dev(weights.d_routed_up_proj_w);
    free_tensor_dev(weights.d_routed_down_proj_w);
    
    free_tensor_dev(workspace.d_residuals);
    free_tensor_dev(workspace.d_router_logits);
    free_tensor_dev(workspace.d_topk_indices);
    free_tensor_dev(workspace.d_topk_weights);
    free_tensor_dev(workspace.d_routed_expert_output);
    free_tensor_dev(workspace.d_shared_expert_output);
    free_tensor_dev(workspace.d_shared_mlp_ws1);
    free_tensor_dev(workspace.d_shared_mlp_ws2);
    free_tensor_dev(workspace.d_shared_mlp_ws3);
    free_tensor_dev(workspace.d_sigmoid_scores);
    free_tensor_dev(workspace.d_biased_scores);
    free_tensor_dev(workspace.d_group_scores);
    free_tensor_dev(workspace.d_group_mask);
    
    printf("Done.\n");

    return 0;
}