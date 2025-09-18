#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h> 

// Helper macro for checking CUDA errors
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}


// Exercise 1

// a. Kernel with each thread producing one output matrix row
__global__ void MatrixMul_RowPerThread(float* M, float* N, float* P, int Width, int Height) {
    // Each thread computes one row of P
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check for rows
    if (row < Height) {
        // Loop over each column in the output row to calculate each element
        for (int col = 0; col < Width; ++col) {
            float pValue = 0.0f;
            // Dot product calculation for P[row][col]
            for (int k = 0; k < Width; ++k) {
                pValue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = pValue;
        }
    }
}

// b. Kernel with each thread producing one output matrix column
__global__ void MatrixMul_ColPerThread(float* M, float* N, float* P, int Width, int Height) {
    // Each thread computes one column of P
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check for columns
    if (col < Width) {
        // Loop over each row in the output column to calculate each element
        for (int row = 0; row < Height; ++row) {
            float pValue = 0.0f;
            // Dot product calculation for P[row][col]
            for (int k = 0; k < Width; ++k) {
                pValue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = pValue;
        }
    }
}



// Exercise 2

__global__ void matVecMulKernel(float* A, float* B, float* C, int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Use one thread to calculate one element of the output vector A
    if (i < N) {
        float sum = 0.0f;
        // Perform the dot product of matrix B with vector C
        for (int j = 0; j < N; j++) {
            sum += B[i * N + j] * C[j];
        }
        A[i] = sum;
    }
}

// Host stub function to launch the matrix-vector multiplication kernel
void matrixVectorMul(float* A_d, float* B_d, float* C_d, int N) {
    // Configure a 1D grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching Matrix-Vector Multiplication Kernel\n");
    printf("Grid size: %d blocks, Block size: %d threads\n", blocksPerGrid, threadsPerBlock);

    matVecMulKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);

    HANDLE_ERROR(cudaPeekAtLastError());

    HANDLE_ERROR(cudaDeviceSynchronize());

    printf("Kernel execution finished.\n");
}


// Exercise 3
// a. What is the number of threads per block?
// 16 * 32 = 512 threads per block.

// b. What is the number of threads in the grid?
// (Total Blocks) * (Threads per Block) = (19 * 5) * 512 = 95 * 512 = 48,640 threads.

// c. What is the number of blocks in the grid?
// 19 * 5 = 95 blocks.

// d. What is the number of threads that execute the code on line 05?
// the number of threads is equal to the number of data elements.
// M * N = 150 * 300 = 45,000 threads.


// Exercise 4: 2D Linearization

// Matrix dimensions: width = 400, height = 500
// Element location: row = 20, column = 10

// a. If the matrix is stored in row-major order:
// index = row * width + column = 20 * 400 + 10 = 8000 + 10 = 8010


// b. If the matrix is stored in column-major order:
// index = column * height + row = 10 * 500 + 20 = 5000 + 20 = 5020


// Exercise 5: 3D Linearization

// Tensor dimensions: width = 400 (x), height = 500 (y), depth = 300 (z)
// Element location: x = 10, y = 20, z = 5

// index = z * height * width + y * width + x = 5 * 500 * 400 + 20 * 400 + 10 = 1008010


int main() {

    printf("Testing Exercise 2: Matrix-Vector Multiplication\n");
    int N = 256; 

    // Calculate sizes
    size_t vec_size = N * sizeof(float);
    size_t mat_size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(vec_size);
    h_B = (float*)malloc(mat_size);
    h_C = (float*)malloc(vec_size);

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_C[i] = (float)(i % 10);
        for (int j = 0; j < N; ++j) {
            h_B[i * N + j] = (float)((i + j) % 20);
        }
    }
    printf("Host data initialized.\n");

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HANDLE_ERROR( cudaMalloc((void**)&d_A, vec_size) );
    HANDLE_ERROR( cudaMalloc((void**)&d_B, mat_size) );
    HANDLE_ERROR( cudaMalloc((void**)&d_C, vec_size) );
    printf("Device memory allocated.\n");

    // Copy data from host to device
    HANDLE_ERROR( cudaMemcpy(d_B, h_B, mat_size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_C, h_C, vec_size, cudaMemcpyHostToDevice) );
    printf("Data copied from host to device.\n");

    // Call the host stub function to run the kernel
    matrixVectorMul(d_A, d_B, d_C, N);

    // Copy result from device to host
    HANDLE_ERROR( cudaMemcpy(h_A, d_A, vec_size, cudaMemcpyDeviceToHost) );
    printf("Result copied from device to host.\n");

    // Verification
    printf("\nVerification (first 5 elements):\n");
    for(int i = 0; i < 5; ++i) {
        printf("A[%d] = %f\n", i, h_A[i]);
    }
    printf("\n");

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}