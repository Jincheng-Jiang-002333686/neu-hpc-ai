#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Tile width for shared memory
#define TILE_WIDTH 16

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

// Uses a tiled algorithm with shared memory to reduce global memory traffic.
__global__ void gemmKernel(const float* A, const float* B, const float* C, float* D,
                           float alpha, float beta, int m, int n, int k) {

    // Shared memory tiles for matrices A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Thread's local indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Thread's global indices for the output matrix D
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Accumulator register for the dot product, private to each thread
    float pValue = 0.0f;

    // Loop over the tiles of A and B required to compute the final dot product
    for (int ph = 0; ph < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        
        // Load a tile of A into shared memory (As)
        int a_row_idx = row;
        int a_col_idx = ph * TILE_WIDTH + tx;
        if (a_row_idx < m && a_col_idx < k) {
            As[ty][tx] = A[a_row_idx * k + a_col_idx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load a tile of B into shared memory (Bs)
        int b_row_idx = ph * TILE_WIDTH + ty;
        int b_col_idx = col;
        if (b_row_idx < k && b_col_idx < n) {
            Bs[ty][tx] = B[b_row_idx * n + b_col_idx];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded before consumption
        __syncthreads();

        // Multiply tiles from shared memory and accumulate result
        for (int i = 0; i < TILE_WIDTH; ++i) {
            pValue += As[ty][i] * Bs[i][tx];
        }

        // Synchronize to make sure all threads are done with the current tiles
        __syncthreads();
    }

    // Write the final result to matrix D if within bounds
    if (row < m && col < n) {
        int c_idx = row * n + col;
        D[c_idx] = alpha * pValue + beta * C[c_idx];
    }
}

// Host function to verify the result from the GPU
void verifyResult(const float* D_gpu, const float* A, const float* B, const float* C,
                  float alpha, float beta, int m, int n, int k) {
    printf("Verifying result\n");
    float* D_cpu = (float*)malloc(m * n * sizeof(float));

    // Perform GEMM on the CPU
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float pValue = 0.0f;
            for (int l = 0; l < k; ++l) {
                pValue += A[i * k + l] * B[l * n + j];
            }
            D_cpu[i * n + j] = alpha * pValue + beta * C[i * n + j];
        }
    }

    float max_error = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        max_error = fmax(max_error, fabs(D_gpu[i] - D_cpu[i]));
    }

    printf("Max error: %f\n", max_error);
    if (max_error < 1e-3) {
        printf("SUCCESS: GPU result matches CPU result.\n");
    } else {
        printf("FAILURE: GPU result does not match CPU result.\n");
    }

    free(D_cpu);
}

int main() {
    // Set matrix dimensions 
    int m = 100;
    int n = 110;
    int k = 120;

    // Set scalar values
    float alpha = 1.5f;
    float beta = 2.5f;

    printf("Running GEMM: D = %.1f * A * B + %.1f * C\n", alpha, beta);
    printf("Matrix dimensions: A(%d x %d), B(%d x %d), C(%d x %d)\n", m, k, k, n, m, n);

    // Calculate matrix sizes in bytes
    size_t a_size = m * k * sizeof(float);
    size_t b_size = k * n * sizeof(float);
    size_t c_size = m * n * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(a_size);
    float *h_B = (float*)malloc(b_size);
    float *h_C = (float*)malloc(c_size);
    float *h_D = (float*)malloc(c_size);

    // Initialize host matrices
    for (int i = 0; i < m * k; ++i) h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < k * n; ++i) h_B[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < m * n; ++i) h_C[i] = (float)(rand() % 100) / 10.0f;

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    HANDLE_ERROR(cudaMalloc((void**)&d_A, a_size));
    HANDLE_ERROR(cudaMalloc((void**)&d_B, b_size));
    HANDLE_ERROR(cudaMalloc((void**)&d_C, c_size));
    HANDLE_ERROR(cudaMalloc((void**)&d_D, c_size));

    // Copy input matrices from host to device
    HANDLE_ERROR(cudaMemcpy(d_A, h_A, a_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_B, h_B, b_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_C, h_C, c_size, cudaMemcpyHostToDevice));

    // Configure the kernel launch
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, 
                       (m + TILE_WIDTH - 1) / TILE_WIDTH);

    printf("Launching kernel with grid %dx%d and block %dx%d\n", 
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

    // Launch the GEMM kernel
    gemmKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, alpha, beta, m, n, k);

    // Check for kernel launch errors
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Copy output matrix from device to host
    HANDLE_ERROR(cudaMemcpy(h_D, d_D, c_size, cudaMemcpyDeviceToHost));
    
    // Verify the result
    verifyResult(h_D, h_A, h_B, h_C, alpha, beta, m, n, k);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return 0;
}