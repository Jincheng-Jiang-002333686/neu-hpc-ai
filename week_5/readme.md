FlashAttention-2 Implementation

Improved Parallelism: The workload is now parallelized across the sequence length dimension in addition to the batch and head dimensions. 

Work Partitioning: The work within a thread block is partitioned more efficiently. Instead of splitting the K matrix across warps, the Q matrix is split. This makes each warp's computation independent, reducing shared memory traffic.

Compile: nvcc -o flash_atten2 flash_atten2.cu

Run: ./flash_atten2 

Example output:
(base) jincheng@jincheng-desktop:~/NEU/7375HPC/neu-hpc-for-ai/week_5$ nvcc -o flash_atten2 flash_atten2.cu
(base) jincheng@jincheng-desktop:~/NEU/7375HPC/neu-hpc-for-ai/week_5$ ./flash_atten2 
Running FlashAttention2
SeqLen N=1024, d_head=64, Batch=2, Heads=4
Launching kernel with Grid: (64, 4, 2), Block: (32, 16, 1)
Running CPU baseline for verification.
Verifying results:
Max absolute error: 0.000002 (Tolerance: 0.000100)
SUCCESS: FlashAttention-2 kernel matches CPU baseline.