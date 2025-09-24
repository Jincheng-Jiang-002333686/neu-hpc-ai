1. Tiled GEMM CUDA Kernel

Verification: Includes a host-side (CPU) function to verify the correctness of the GPU computation.

Compile the code:
nvcc -o gemm_kernel gemm_kernel.cu

run the code:
./gemm_kernel

sampel output:
Running Tiled GEMM: C = 1.5 * op_N(A) * op_T(B) + 2.5 * C
Logical dims: m=101, n=113, k=127. Tile width: 16
Allocated A dims: 101 x 127
Allocated B dims: 113 x 127
Launching kernel with grid 8x7 and block 16x16
Verifying result:
Max error: 0.001465 (Tolerance: 0.010000)
SUCCESS: GPU result matches CPU result.

2. Online Normalizer for Softmax

Verification: The program includes an implementation of the standard softmax as a baseline and verifies the online version produces a correct result.

Compile the code:
gcc -o softmax softmax.cu 

Run the executable:
./softmax

sample output:
Vector size: 1024

Input (x)     : [-1.2555, 5.9068, 5.2729, 9.5001, 7.4208, 9.5257, 1.0105, 5.4742, -3.4772, 1.6098]
Online (y)    : [0.0000, 0.0003, 0.0002, 0.0113, 0.0014, 0.0116, 0.0000, 0.0002, 0.0000, 0.0000]
Baseline (y)  : [0.0000, 0.0003, 0.0002, 0.0113, 0.0014, 0.0116, 0.0000, 0.0002, 0.0000, 0.0000]
...

Verifying results...
SUCCESS: Online softmax matches the result.