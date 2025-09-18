// Exercise 1

// No. THis only adds complexity and shared memory usage.


// Exercise 3

// Forgetting either `__syncthreads()` would lead to race conditions
// and produce incorrect results.

// - If the FIRST `__syncthreads()` is removed (line 21):
//   Faster threads would start computing using the shared memory tile before 
//   slower threads have finished writing their data into it. 

// - If the SECOND `__syncthreads()` is removed (line 26):
//   Faster threads could loop around and start loading the *next* tile's data, 
//   overwriting the shared memory values that slower threads are still trying to read. 


// Exercise 4


// If a thread loaded a value from global memory into its own private register,
// no other thread could access it. To achieve data reuse across threads
// data must be placed in a memory space that shared by those threads.



// Exercise 5: 

// For a 32x32 tile, the reduction is by a factor of 32.


// Exercise 6


// Total threads = 1000 blocks * 512 threads/block = 512,000 
// 512,000 versions of the variable will be created.


// Exercise 7

// A separate version is created for every thread block.
// 1,000 versions of the variable will be created.


// Exercise 8

// a. There is no tiling?
// N times. Each of calculations is done by a different thread, and 
// each thread will load M[i][k] from global memory.


// b. Tiles of size TxT are used?
// 1 time. An input element is part of a single tile. 
// All accesses to that element are from fast shared memory.


// Exercise 9

// a. Peak FLOPS = 200 GFLOPS, Peak BW = 100 GB/s
//     Machine Balance = 200 / 100 = 2.0 FLOP/B.
//     Kernel AI (1.286) < Machine Balance (2.0).
//    Memory-bound.

// b. Peak FLOPS = 300 GFLOPS, Peak BW = 250 GB/s
//     Machine Balance = 300 / 250 = 1.2 FLOP/B.
//     Kernel AI (1.286) > Machine Balance (1.2).
//    Compute-bound.



// Exercise 10


// a. For what values of BLOCK_SIZE will this kernel function execute correctly?
// It is incorrect for all values.
// While the block spans multiple warps that are scheduled independently, a
// race condition will happen. 

// b. What is the root cause of this incorrect execution behavior? Suggest a fix.


// Add a barrier synchronization to ensure all threads have finished.


// Exercise 11

// Given a kernel launched with N=1024 and 128 threads/block.
// Total blocks = 1024/128 = 8. Total threads = 1024.

// a. How many versions of the variable i are there?
// `i` is an automatic scalar 1024 versions.

// b. How many versions of the array x[] are there?
// `x` is an automatic array 1024 versions.

// c. How many versions of the variable y_s are there?
// `y_s` is a shared variable 8 versions.

// d. How many versions of the array b_s[] are there?
// `b_s` is a shared array 8 versions.

// e. What is the amount of shared memory used per block (in bytes)?
// `y_s` (1 float) + `b_s` (128 floats) = 129 floats.
// 129 floats * 4 bytes/float = **516 bytes**.

// f. What is the floating-point to global memory access ratio of the kernel?

// - Global Reads: 4 reads for `a` inside the loop + 1 read for `b`. Total = 5 reads.
// - Global Writes: 1 write to `b`. Total = 1 write.
// - Total Bytes: (5 reads + 1 write) * 4 bytes/access = 24 bytes.
// - FLOPs: 5 multiplications and 5 additions in the final statement = 10 FLOPs.
// - Ratio = 10 FLOPs / 24 Bytes ≈ 0.417 FLOP/B.


// Exercise 12

// a. 64 threads/block, 27 regs/thread, 4 KB shared/block.
//    - Thread Limit: 2048/64 = 32 blocks needed for full occupancy. OK (<= 32 blocks).
//    - Register Limit: 2048 threads * 27 regs = 55,296 regs. OK (< 65,536).
//    - Shared Mem Limit: 32 blocks * 4KB/block = 128 KB. NOT OK (> 96 KB).
//      Max blocks allowed by shmem = floor(96KB/4KB) = 24 blocks.
//      Max threads = 24 blocks * 64 threads/block = 1536.
//    No. Limiting factor is shared memory per SM.
//
// b. 256 threads/block, 31 regs/thread, 8 KB shared/block.
//    - Thread Limit: 2048/256 = 8 blocks. OK (<= 32 blocks).
//    - Register Limit: 2048 threads * 31 regs = 63,488 regs. OK (< 65,536).
//    - Shared Mem Limit: 8 blocks * 8KB/block = 64 KB. OK (< 96 KB).
//    Yes, can achieve full occupancy.