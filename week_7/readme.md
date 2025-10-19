Distributed FlashAttention

File: distributed_flash_atten.cu

This project scaling the FlashAttention-2 algorithm to run on multiple GPUs within a single node. 

Technical Implementation

The implementation uses Sequence Parallelism based on the "Ring Attention" strategy.

Sequence Partitioning: The input sequence (Q, K, V) is split evenly across all available GPUs. GPU 0 holds the first chunk, GPU 1 the second, and so on.

Ring Communication: The computation proceeds in steps, forming a logical "ring". In each step, every GPU computes attention for its local Q chunk against a K/V chunk. 

Compile: nvcc -o distributed_flash_atten distributed_flash_atten.cu
run: ./distributed_flash_atten