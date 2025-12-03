/*
 * Assignment: Fast Distributed MoE - Single Kernel Implementation
 * Goal: Implement Symmetric Tensor Layout and verify write-write conflict-free property.
 */

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cstdio>
#include <iostream>

// Step 1: Define Layout Dimensions
struct MoEConfig {
    int P; 
    int R; 
    int B; 
    int E; 
    int C; 
    int H;
};

// Step 5: Optional Task Abstraction
enum TaskType { GEMM0, GEMM1, Combine };

struct __align__(16) Task {
    const float* aData;      // Input tile
    float* cData;            // Output tile
    int tileIdx;
    int expertIdx;
    int sourceDevice;        // Metadata M
    TaskType type;
    char padding[64];        // Padding for alignment
};

// Step 2: Symmetric Tensor Indexing
__device__ size_t get_tensor_offset(int source_gpu_id, int round, int buffer, 
                                    int expert_id, int token_idx, const MoEConfig& cfg) {

    
    // Calculate Strides (Row-Major)
    // Note: H_stride is 1, but we return the offset to the start of the token (vector of size H).
    
    size_t C_stride = cfg.H;                // Stride for one Token (Capacity slot)
    size_t E_stride = cfg.C * C_stride;     // Stride for one Expert
    size_t B_stride = cfg.E * E_stride;     // Stride for one Buffer
    size_t R_stride = cfg.B * B_stride;     // Stride for one Round
    size_t P_stride = cfg.R * R_stride;     // Stride for one Source Process

    size_t offset = 0;
    offset += source_gpu_id * P_stride;
    offset += round * R_stride;
    offset += buffer * B_stride;
    offset += expert_id * E_stride;
    offset += token_idx * C_stride;
    
    return offset;
}

// Step 3: Host Allocation
void allocate_symmetric_tensor(MoEConfig cfg, float** d_symmetric_tensor) {
    // Total elements = P * R * B * E * C * H
    size_t total_elements = (size_t)cfg.P * cfg.R * cfg.B * cfg.E * cfg.C * cfg.H;
    size_t size_bytes = total_elements * sizeof(float);
    
    // nvshmem_malloc ensures this memory is mapped and accessible by other GPUs
    *d_symmetric_tensor = (float*)nvshmem_malloc(size_bytes);
    
    if (*d_symmetric_tensor == nullptr) {
        printf("CRITICAL ERROR: Failed to allocate NVSHMEM symmetric tensor.\n");
        exit(1);
    }

    // Initialize to zero
    cudaMemset(*d_symmetric_tensor, 0, size_bytes);
    cudaDeviceSynchronize();
}

// Step 4: Verification Kernel 
__global__ void verify_layout_kernel(float* symmetric_tensor, MoEConfig cfg, int my_pe, int n_pes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > 0) return; // Single thread verification

    // PHASE 1: WRITE
    for (int target_pe = 0; target_pe < n_pes; ++target_pe) {
        // Skip self to strictly test RDMA logic
        if (target_pe == my_pe) continue; 

        int round = 0;
        int buffer = 0;
        int expert = 0;
        int token_idx = 0; 

        // Use 'my_pe' as source to calculate MY reserved slot on TARGET
        size_t offset = get_tensor_offset(my_pe, round, buffer, expert, token_idx, cfg);
        
        float signature = (float)(my_pe * 1000.0f + 123.0f);

        // One-sided Put
        nvshmem_float_p(&symmetric_tensor[offset], signature, target_pe);
    }

    // PHASE 2: BARRIER
    nvshmem_barrier_all();

    // PHASE 3: READ & VERIFY
    for (int source_pe = 0; source_pe < n_pes; ++source_pe) {
        if (source_pe == my_pe) continue;

        size_t offset = get_tensor_offset(source_pe, 0, 0, 0, 0, cfg);
        
        float expected = (float)(source_pe * 1000.0f + 123.0f);
        float actual = symmetric_tensor[offset];

        if (actual != expected) {
            printf("[FAIL] GPU %d: Slot for Source %d has WRONG value. Expected %.1f, Got %.1f\n", 
                   my_pe, source_pe, expected, actual);
        } else {
            printf("[PASS] GPU %d: Received correct RDMA from GPU %d\n", my_pe, source_pe);
        }
    }
}

// Main Driver 
int main(int argc, char **argv) {
    // 1. Initialize NVSHMEM
    nvshmem_init();

    int my_rank = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    if (my_rank == 0) {
        printf("FlashMoE Assignment: Single Kernel Implementation\n");
        printf("World Size (P): %d GPUs\n", n_pes);
    }

    // 2. Setup Config
    MoEConfig cfg;
    cfg.P = n_pes;
    cfg.R = 2;   
    cfg.B = 2;   
    cfg.E = 4;   
    cfg.C = 128; 
    cfg.H = 64;  

    // 3. Allocate
    float* d_symmetric_tensor = nullptr;
    allocate_symmetric_tensor(cfg, &d_symmetric_tensor);

    // 4. Launch Verification
    if (my_rank == 0) printf("Launching Verification Kernel...\n");
    verify_layout_kernel<<<1, 1>>>(d_symmetric_tensor, cfg, my_rank, n_pes);

    // 5. Cleanup
    cudaDeviceSynchronize();
    nvshmem_barrier_all();
    
    nvshmem_free(d_symmetric_tensor);
    nvshmem_finalize();

    if (my_rank == 0) printf("Test Completed Successfully.\n");
    return 0;
}