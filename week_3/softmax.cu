#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

void online_softmax(const float* x, float* y, int V) {
    // Initialize
    float m = -FLT_MAX;
    float d = 0.0f;

    for (int j = 0; j < V; j++) {
        float old_m = m;
        // Update the running max
        m = fmaxf(m, x[j]);
        
        // Rescales the previous sum 'd' to be relative to the new max 'm'
        d = d * expf(old_m - m) + expf(x[j] - m);
    }

    // calculation of softmax values
    for (int i = 0; i < V; i++) {
        y[i] = expf(x[i] - m) / d;
    }
}

void safe_softmax_baseline(const float* x, float* y, int V) {
    // Find the maximum value in the input vector
    float m = -FLT_MAX;
    for (int k = 0; k < V; k++) {
        m = fmaxf(m, x[k]);
    }

    // Calculate the normalization term
    float d = 0.0f;
    for (int j = 0; j < V; j++) {
        d += expf(x[j] - m);
    }

    // Calculate the final softmax values
    for (int i = 0; i < V; i++) {
        y[i] = expf(x[i] - m) / d;
    }
}
// Function to verify that two vectors are approximately equal
void verify_results(const float* y_online, const float* y_baseline, int V) {
    printf("Verifying results...\n");
    float max_error = 0.0f;
    for (int i = 0; i < V; i++) {
        max_error = fmaxf(max_error, fabsf(y_online[i] - y_baseline[i]));
    }

    const float tolerance = 1e-6;
    if (max_error < tolerance) {
        printf("SUCCESS: Online softmax matches the result.\n");
    } else {
        printf("FAILURE: Results do not match.\n");
    }
}

void print_vector(const char* name, const float* vec, int V) {
    printf("%s: [", name);
    for(int i = 0; i < V; i++) {
        printf("%.4f%s", vec[i], (i == V - 1) ? "" : ", ");
    }
    printf("]\n");
}


int main() {
    // Define the size of the vector
    const int VECTOR_SIZE = 1024;
    const int PRINT_SIZE = 10; // print a few elements

    printf("Vector size: %d\n\n", VECTOR_SIZE);

    // Allocate memory for host vectors
    float* x = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* y_online = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* y_baseline = (float*)malloc(VECTOR_SIZE * sizeof(float));

    if (!x || !y_online || !y_baseline) {
        fprintf(stderr, "Failed to allocate memory.\n");
        return 1;
    }

    // Initialize input vector
    srand(time(NULL));
    for (int i = 0; i < VECTOR_SIZE; i++) {
        x[i] = ((float)rand() / (float)(RAND_MAX)) * 20.0f - 10.0f; // Random floats between -10 and 10
    }

    // Run the online softmax implementation
    online_softmax(x, y_online, VECTOR_SIZE);

    // Run the baseline safe softmax implementation
    safe_softmax_baseline(x, y_baseline, VECTOR_SIZE);
    
    // Print the first few elements to see the results
    print_vector("Input (x)     ", x, PRINT_SIZE);
    print_vector("Online (y)    ", y_online, PRINT_SIZE);
    print_vector("Baseline (y)  ", y_baseline, PRINT_SIZE);
    printf("...\n\n");

    // Verify that the results are identical
    verify_results(y_online, y_baseline, VECTOR_SIZE);

    // Free allocated memory
    free(x);
    free(y_online);
    free(y_baseline);

    return 0;
}
