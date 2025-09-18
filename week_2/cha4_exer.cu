// Exercise 1

// a. What is the number of warps per block?
// 128 threads/block / 32 threads/warp = 4 

// b. What is the number of warps in the grid?
// 8 blocks * 4 warps/block = 32 

// c. For the statement on line 04: if (threadIdx.x < 40 || threadIdx.x >= 104)

//    i. How many warps in the grid are active?
//    3 warps/block * 8 blocks = 24 

//    ii. How many warps in the grid are divergent?
//    2 warps/block * 8 blocks = 16 

//    iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
//    100%. All 32 threads take the same path.

//    iv. What is the SIMD efficiency (in %) of warp 1 of block 0?
//    50%. The threads take two different paths.

//    v. What is the SIMD efficiency (in %) of warp 3 of block 0?
//    50%. The threads take two different paths.

// d. For the statement on line 07: if ((i % 2 == 0)) 

//    i. How many warps in the grid are active?
//    32 warps. Every warp of 32 consecutive threads will contain 16 even and 16 odd global indices.

//    ii. How many warps in the grid are divergent?
//    32 warps. All warps will be divergent.

//    iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
//    50%. Two paths are executed.

// e. For the loop on line 09: for (unsigned int j = 0; j < 5 - (i / 3); ++j)

//    i. How many iterations have no divergence?
//    0. Threads with different `i` values will drop out of the loop at different times.

//    ii. How many iterations have divergence?
//    5. The loop runs for j = 0, 1, 2, 3, 4 Each of these iterations will exhibit divergence.


// Exercise 2
// How many threads will be in the grid?
// 4 blocks * 512 threads/block = 2048 


// Exercise 3: Vector Addition Divergence

// How many warps do you expect to have divergence?
// 1 warp.

// Exercise 4

// What percentage of the threads' total execution time is spent waiting for the barrier?
// (4.1 / 24.0) * 100 = 17.08%.


// Exercise 5

// Do you think this is a good idea? Explain.
// No
// 2. If the code contains any data-dependent `if` statements or loops, threads within the warp will diverge.
// 3. GPU architectures could change the warp size or scheduling logic. Code that assumes a warp size of 32 will break on hardware. 


// Exercise 6

// a. 128 threads/block -> 4 blocks * 128 = 512 
// b. 256 threads/block -> 4 blocks * 256 = 1024 
// c. 512 threads/block -> 1536/512 = 3 blocks. Total threads = 3 * 512 = 1536. 
// d. 1024 threads/block-> 1536/1024 = 1 block (can't fit 2). Total threads = 1 * 1024 = 1024.

// Which block configuration would result in the most number of threads in the SM?
// c. 512 threads per block.


// Exercise 7

// a. 8 blocks, 128 threads/block: 8*128=1024 threads.  Occupancy = 1024/2048 = 50%.
// b. 16 blocks, 64 threads/block: 16*64=1024 threads.  Occupancy = 1024/2048 = 50%.
// c. 32 blocks, 32 threads/block: 32*32=1024 threads.  Occupancy = 1024/2048 = 50%.
// d. 64 blocks, 32 threads/block: 64*32=2048 threads.  Occupancy = 2048/2048 = 100%.
// e. 32 blocks, 64 threads/block: 32*64=2048 threads.  Occupancy = 2048/2048 = 100%.


// Exercise 8

// a. 128 threads/block, 30 registers/thread:
//    - Blocks needed for full occupancy: 2048/128 = 16 blocks. ( < 32)
//    - Registers needed for full occupancy: 2048 * 30 = 61,440 registers. ( < 65,536)
//    Yes

// b. 32 threads/block, 29 registers/thread:
//    - Blocks needed for full occupancy: 2048/32 = 64 blocks. ( > 32)
//    - The max number of blocks (32) is the bottleneck. Max threads = 32*32 = 1024.
//    No. Limiting factor is max blocks per SM.

// c. 256 threads/block, 34 registers/thread:
//    - Blocks needed for full occupancy: 2048/256 = 8 blocks. ( < 32)
//    - Registers needed for full occupancy: 2048 * 34 = 69,632 registers. ( > 65,536)
//    - Max threads = 65536 / 34 = 1927.
//    No. Limiting factor is registers per SM.


// Exercise 9


// What would be your reaction and why?
// The claim is impossible.
// The requested number of threads per block 1024 is greater than themaximum supported by the device 512.
