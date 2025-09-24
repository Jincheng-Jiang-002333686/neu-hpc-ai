#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

// Kernel to perform tiled matrix multiplication C = alpha * op(A) * op(B) + beta * C
__global__ void tiledGemmKernel(char transa, char transb,
                                const float* A, const float* B, float* C,
                                float alpha, float beta, 
                                int m, int n, int k,
                                int lda, int ldb, int ldc) {

    // Shared memory tiles for matrices A and B.
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Thread's local indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Thread's global indices for the output matrix C
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Accumulator register for the dot product
    float pValue = 0.0f;

    // Loop over the tiles of A and B in phases
    for (int ph = 0; ph < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        
        // Load a tile of A into As
        int a_row_for_op = row;
        int a_col_for_op = ph * TILE_WIDTH + tx;
        if (transa == 'N') {
            if (a_row_for_op < m && a_col_for_op < k) {
                As[ty][tx] = A[a_row_for_op * lda + a_col_for_op];
            } else {
                As[ty][tx] = 0.0f;
            }
        } else { // Transposed A
            if (a_col_for_op < k && a_row_for_op < m) {
                As[ty][tx] = A[a_col_for_op * lda + a_row_for_op];
            } else {
                As[ty][tx] = 0.0f;
            }
        }

        // Load a tile of B into Bs
        int b_row_for_op = ph * TILE_WIDTH + ty;
        int b_col_for_op = col;
        if (transb == 'N') {
            if (b_row_for_op < k && b_col_for_op < n) {
                Bs[ty][tx] = B[b_row_for_op * ldb + b_col_for_op];
            } else {
                Bs[ty][tx] = 0.0f;
            }
        } else { // Transposed B
            if (b_col_for_op < n && b_row_for_op < k) {
                Bs[ty][tx] = B[b_col_for_op * ldb + b_row_for_op];
            } else {
                Bs[ty][tx] = 0.0f;
            }
        }

        // Synchronize to ensure all threads in the block have finished loading tiles.
        __syncthreads();

        // Compute partial dot product from shared memory
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            pValue += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Write final result to matrix C
    if (row < m && col < n) {
        int c_idx = row * ldc + col;
        if (beta == 0.0f) {
            C[c_idx] = alpha * pValue;
        } else {
            C[c_idx] = alpha * pValue + beta * C[c_idx];
        }
    }
}


// verify the result from the GPU
void verifyResult(const float* C_gpu, const float* A, const float* B, const float* C_initial,
                  char transa, char transb, float alpha, float beta, 
                  int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Verifying result:\n");
    float* C_cpu = (float*)malloc(m * n * sizeof(float));

    // Perform GEMM on the CPU
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float pValue = 0.0f;
            for (int l = 0; l < k; ++l) {
                float a_val, b_val;
                if (transa == 'N') a_val = A[i * lda + l]; else a_val = A[l * lda + i];
                if (transb == 'N') b_val = B[l * ldb + j]; else b_val = B[j * ldb + l];
                pValue += a_val * b_val;
            }
            if (beta == 0.0f) {
                C_cpu[i * ldc + j] = alpha * pValue;
            } else {
                C_cpu[i * ldc + j] = alpha * pValue + beta * C_initial[i * ldc + j];
            }
        }
    }

    float max_error = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        max_error = fmax(max_error, fabs(C_gpu[i] - C_cpu[i]));
    }

    const float tolerance = 1e-2; 
    if (max_error < tolerance) {
        printf("SUCCESS: GPU result matches CPU result.\n");
    } else {
        printf("FAILURE: GPU result does not match CPU result.\n");
    }

    free(C_cpu);
}

int main() {
    // Configuration 
    char transa = 'N';
    char transb = 'T';

    // Set matrix dimensions 
    int m = 101;
    int n = 113;
    int k = 127;

    float alpha = 1.5f;
    float beta = 2.5f;

    int rows_a = (transa == 'N') ? m : k;
    int cols_a = (transa == 'N') ? k : m;
    int rows_b = (transb == 'N') ? k : n;
    int cols_b = (transb == 'N') ? n : k;
    
    int lda = cols_a, ldb = cols_b, ldc = n;

    printf("Running Tiled GEMM: C = %.1f * op_%c(A) * op_%c(B) + %.1f * C\n", alpha, transa, transb, beta);
    printf("Logical dims: m=%d, n=%d, k=%d. Tile width: %d\n", m, n, k, TILE_WIDTH);
    printf("Allocated A dims: %d x %d\n", rows_a, cols_a);
    printf("Allocated B dims: %d x %d\n", rows_b, cols_b);

    size_t a_size = rows_a * cols_a * sizeof(float);
    size_t b_size = rows_b * cols_b * sizeof(float);
    size_t c_size = m * n * sizeof(float);

    float *h_A, *h_B, *h_C, *h_C_initial;
    h_A = (float*)malloc(a_size);
    h_B = (float*)malloc(b_size);
    h_C = (float*)malloc(c_size);
    h_C_initial = (float*)malloc(c_size);

    for (int i = 0; i < rows_a * cols_a; ++i) h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < rows_b * cols_b; ++i) h_B[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < m * n; ++i) {
        h_C[i] = (float)(rand() % 100) / 10.0f;
        h_C_initial[i] = h_C[i];
    }

    float *d_A, *d_B, *d_C;
    HANDLE_ERROR(cudaMalloc((void**)&d_A, a_size));
    HANDLE_ERROR(cudaMalloc((void**)&d_B, b_size));
    HANDLE_ERROR(cudaMalloc((void**)&d_C, c_size));

    HANDLE_ERROR(cudaMemcpy(d_A, h_A, a_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_B, h_B, b_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_C, h_C, c_size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, 
                       (m + TILE_WIDTH - 1) / TILE_WIDTH);

    printf("Launching kernel with grid %dx%d and block %dx%d\n", 
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

    tiledGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(transa, transb, d_A, d_B, d_C, 
                                                       alpha, beta, m, n, k, lda, ldb, ldc);

    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(h_C, d_C, c_size, cudaMemcpyDeviceToHost));
    
    verifyResult(h_C, h_A, h_B, h_C_initial, transa, transb, alpha, beta, m, n, k, lda, ldb, ldc);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_initial);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

