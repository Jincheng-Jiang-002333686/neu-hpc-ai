#define _POSIX_C_SOURCE 200809L 

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> 
#include <time.h>    
#include <assert.h>  
#include <math.h>    


// hold matrix dimensions and its data (flattened 1D array)
typedef struct {
    int rows;
    int cols;
    float* data;
} Matrix;

// pass all necessary data to each thread
typedef struct {
    int thread_id;     
    const Matrix* a;   
    const Matrix* b;    
    Matrix* c;          
    int start_row;      
    int end_row;        
} ThreadData;

// allocates memory for a matrix.  initialize all elements to 0.0f
Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float*)calloc(rows * cols, sizeof(float));
    assert(m.data != NULL);
    return m;
}

// frees the memory 
void free_matrix(Matrix* m) {
    free(m->data);
    m->data = NULL;
}

// fills a matrix with random float values between 0.0 and 1.0
void fill_matrix_random(Matrix* m) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = (float)rand() / (float)RAND_MAX;
    }
}

// test dimensions and values of two matrices for equality(single-threaded vs multi-threaded results)
int compare_matrices(const Matrix* m1, const Matrix* m2) {
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        return 0; // Different dimensions
    }
    for (int i = 0; i < m1->rows * m1->cols; i++) {
        if (fabs(m1->data[i] - m2->data[i]) > 1e-5) {
            return 0; // value mismatch
        }
    }
    return 1; 
}


// Single-Threaded Matrix Multiplication 
void multiply_single_thread(const Matrix* a, const Matrix* b, Matrix* c) {
    assert(a->cols == b->rows);
    assert(c->rows == a->rows);
    assert(c->cols == b->cols);

    for (int r = 0; r < a->rows; r++) {
        for (int c_col = 0; c_col < b->cols; c_col++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[r * a->cols + k] * b->data[k * b->cols + c_col];
            }
            c->data[r * c->cols + c_col] = sum;
        }
    }
}


// Multi-Threaded Matrix Multiplication 
void* multiply_worker(void* args) {
    ThreadData* data = (ThreadData*)args;
    const Matrix* a = data->a;
    const Matrix* b = data->b;
    Matrix* c = data->c;

    for (int r = data->start_row; r < data->end_row; r++) {
        for (int c_col = 0; c_col < b->cols; c_col++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[r * a->cols + k] * b->data[k * b->cols + c_col];
            }
            c->data[r * c->cols + c_col] = sum;
        }
    }
    pthread_exit(NULL);
}

// creates, manages, and cleans up the threads.
void multiply_multi_thread(const Matrix* a, const Matrix* b, Matrix* c, int num_threads) {
    assert(a->cols == b->rows);
    assert(c->rows == a->rows);
    assert(c->cols == b->cols);

    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadData* thread_data = (ThreadData*)malloc(num_threads * sizeof(ThreadData));
    assert(threads != NULL && thread_data != NULL);

    int rows_per_thread = a->rows / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].a = a;
        thread_data[i].b = b;
        thread_data[i].c = c;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? a->rows : (i + 1) * rows_per_thread;

        pthread_create(&threads[i], NULL, multiply_worker, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(thread_data);
}


// Runs a test case, comparing single-threaded vs multi-threaded results.
void run_test(int rA, int cA, int rB, int cB) {
    printf("Testing [%d x %d] * [%d x %d]... ", rA, cA, rB, cB);

    if (cA != rB) {
        printf("SKIPPED (Invalid Dimensions).\n");
        return;
    }

    Matrix a = create_matrix(rA, cA);
    Matrix b = create_matrix(rB, cB);
    fill_matrix_random(&a);
    fill_matrix_random(&b);

    Matrix c_single = create_matrix(rA, cB);
    Matrix c_multi = create_matrix(rA, cB);
    
    multiply_single_thread(&a, &b, &c_single);

    multiply_multi_thread(&a, &b, &c_multi, 4);

    // Compare the results
    if (compare_matrices(&c_single, &c_multi)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&c_single);
    free_matrix(&c_multi);
}

int main(void) {
    srand(time(NULL)); 

    // Correctness Tests
    printf("Running Correctness Tests\n");
    run_test(1, 1, 1, 1);
    run_test(1, 1, 1, 5);
    run_test(2, 1, 1, 3);
    run_test(2, 2, 2, 2);
    run_test(10, 5, 5, 20);
    run_test(50, 50, 50, 50);
    run_test(3, 7, 5, 2); // Invalid dimensions test
    printf("\n");

    // Performance Measurement 
    printf("Running Performance Measurement\n");
    const int PERF_SIZE = 2048; // Use large matrices for measurable speedup
    printf("Matrix dimensions: %d x %d\n", PERF_SIZE, PERF_SIZE);

    Matrix a_perf = create_matrix(PERF_SIZE, PERF_SIZE);
    Matrix b_perf = create_matrix(PERF_SIZE, PERF_SIZE);
    Matrix c_perf = create_matrix(PERF_SIZE, PERF_SIZE);

    fill_matrix_random(&a_perf);
    fill_matrix_random(&b_perf);
    
    int thread_counts[] = {1, 4, 16, 32, 64, 128};
    int num_tests = sizeof(thread_counts) / sizeof(int);
    double baseline_time = 0.0;

    for (int i = 0; i < num_tests; i++) {
        int num_threads = thread_counts[i];
        if (num_threads > PERF_SIZE) {
            printf("\nSkipping %d threads (more threads than rows).\n", num_threads);
            continue;
        }
        printf("\nRunning with %d thread(s)...\n", num_threads);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        multiply_multi_thread(&a_perf, &b_perf, &c_perf, num_threads);

        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Elapsed time: %.6f seconds\n", elapsed_time);
        
        if (i == 0) {
            baseline_time = elapsed_time;
        } else {
            double speedup = baseline_time / elapsed_time;
            printf("Speedup vs 1 thread: %.2fx\n", speedup);
        }
    }

    free_matrix(&a_perf);
    free_matrix(&b_perf);
    free_matrix(&c_perf);

    return 0;
}