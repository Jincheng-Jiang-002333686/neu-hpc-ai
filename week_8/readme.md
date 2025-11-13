# DeepseekV3 MoE CUDA Kernel Implementation

This project implements a naive, unfused, distributed, data-parallel, and expert-parallel CUDA kernel for a `DeepseekV3MoE` (Mixture of Experts) layer.

## How to Compile
nvcc -O3 -o deepseek_v3_moe deepseek_v3_moe.cu -lcublas -std=c++17 -gencode arch=compute_80,code=sm_80

## Key Components

The entire MoE operator is implemented as a series of CUDA kernels and cuBLAS calls orchestrated from a main host function (`deepseek_v3_moe_forward`).

### 1. Host-side MLPs (cuBLAS)

To ensure high performance for standard matrix multiplications, the non-routed MLP components are implemented using the NVIDIA cuBLAS library.

* **`gate_forward`**: Implements the `DeepseekV3TopkRouter`, which is a single linear layer (`hidden_states * gate_w`) that produces the initial `router_logits`.
* **`DeepseekV3MLP_forward`**: Implements the standard `DeepseekV3MLP` (used for the `shared_experts`). This function orchestrates three cuBLAS `sgemm` calls and one custom CUDA kernel (`silu_mul_kernel`) to compute `down_proj( silu(gate_proj(x)) * up_proj(x) )`.

### 2. Complex Routing Logic (5-Kernel Chain)

The core of the assignment is the `route_tokens_to_experts` logic. This is implemented as an "unfused" chain of 5 distinct CUDA kernels, which replace the simple stub.

1.  **`sigmoid_and_bias_kernel`**:
    * **Input**: Raw `router_logits`.
    * **Output**: `sigmoid_scores` (for later) and `biased_scores` (for routing).
    * **Action**: Applies `sigmoid(x) + bias` in an element-wise fashion.

2.  **`grouped_top2_sum_kernel`**:
    * **Input**: `biased_scores`.
    * **Output**: `group_scores` (one score per group, per token).
    * **Action**: Performs a parallel Top-2 reduction *within* each expert group for each token and sums the two scores.

3.  **`group_topk_mask_kernel`**:
    * **Input**: `group_scores`.
    * **Output**: `group_mask` (a `0.0f`/`1.0f` mask).
    * **Action**: Launches one block per token to find the `TOPK_GROUP` highest-scoring groups and create a mask from them.

4.  **`masked_final_topk_kernel`**:
    * **Input**: `biased_scores` and `group_mask`.
    * **Output**: `topk_indices` (the final expert indices for each token).
    * **Action**: "Broadcasts" the group mask to all experts, masks the `biased_scores`, and then performs a final Top-K selection sort to find the winning `TOP_K` experts.

5.  **`gather_norm_scale_kernel`**:
    * **Input**: `topk_indices` and original `sigmoid_scores`.
    * **Output**: `topk_weights` (the final expert weights).
    * **Action**: Gathers the original scores, normalizes them (if enabled), and applies the `routed_scaling_factor`.

### 3. Naive Routed Experts (Custom Kernel)

This is the kernel that executes the "Mixture of Experts" logic.

* **`routed_experts_naive_kernel`**:
    * **Launch**: One block per token, with `blockDim.x = MOE_INTERMEDIATE_SIZE`.
    * **Action**: Each block is responsible for one token. It loads the token's `topk_indices` and `topk_weights`. It then loops `k` times, calling the device function `naive_routed_expert_mlp` for each selected expert.
* **`naive_routed_expert_mlp` (`__device__` function)**:
    * **Action**: A block-wide function where all threads cooperate to perform a full MLP computation (`3x GEMV + SiLU`) in shared memory for a single expert.


