File: flash_attention.cu

The standard attention mechanism requires a large N x N attention matrix, which becomes a bottleneck for long sequences (N). FlashAttention avoids this by being IO-aware.

Tiling: The input matrices (Q, K, V) are split into blocks. The kernel processes these blocks, loading them from slow HBM into fast on-chip SRAM. The algorithm's memory usage scales O(N)instead of O(N^2).

Online Softmax: Compute the softmax normalization factor in a single pass. As the kernel iterates through blocks of K and V, it maintains running statistics (the max value and the exponential sum) for each row of the output. When a new block is processed, these statistics are updated.

Verification: The CUDA kernel's output is compared with standard attention implementation running on the CPU to ensure correctness.

Compile:
nvcc -o flash_attention flash_attention.cu

Run:
./flash_attention