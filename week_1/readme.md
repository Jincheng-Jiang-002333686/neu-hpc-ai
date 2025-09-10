Navigate to the directory containing matrix_multiply.c in your terminal and run the following command:

gcc matrix_multiply.c -o matrix_multiply

Once the compilation is successful, run the following command:

./matrix_multiply

example ouput:
Running Correctness Tests
Testing [1 x 1] * [1 x 1]... PASS
Testing [1 x 1] * [1 x 5]... PASS
Testing [2 x 1] * [1 x 3]... PASS
Testing [2 x 2] * [2 x 2]... PASS
Testing [10 x 5] * [5 x 20]... PASS
Testing [50 x 50] * [50 x 50]... PASS
Testing [3 x 7] * [5 x 2]... SKIPPED (Invalid Dimensions).

Running Performance Measurement
Matrix dimensions: 2048 x 2048

Running with 1 thread(s)...
Elapsed time: 85.462820 seconds

Running with 4 thread(s)...
Elapsed time: 20.305326 seconds
Speedup vs 1 thread: 4.21x

Running with 16 thread(s)...
Elapsed time: 7.439244 seconds
Speedup vs 1 thread: 11.49x

Running with 32 thread(s)...
Elapsed time: 6.068929 seconds
Speedup vs 1 thread: 14.08x

Running with 64 thread(s)...
Elapsed time: 5.497256 seconds
Speedup vs 1 thread: 15.55x

Running with 128 thread(s)...
Elapsed time: 5.504838 seconds
Speedup vs 1 thread: 15.53x