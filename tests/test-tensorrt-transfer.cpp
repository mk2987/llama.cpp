#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-tensorrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main() {
    printf("Testing TensorRT-RTX tensor data transfer...\n\n");

    // Initialize backend
    printf("Initializing backend...\n");
    ggml_backend_t backend = ggml_backend_tensorrt_init(0);
    if (!backend) {
        printf("✗ ERROR: Failed to initialize backend\n");
        return 1;
    }
    printf("✓ Backend initialized\n\n");

    // Create context
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,  // We'll allocate manually
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("✗ ERROR: Failed to create GGML context\n");
        ggml_backend_free(backend);
        return 1;
    }

    // Allocate buffer
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 64 * 1024 * 1024);
    if (!buffer) {
        printf("✗ ERROR: Failed to allocate buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // Test 1: Small transfer (1024 elements)
    printf("Test 1: Small transfer (1024 float32 elements, 4 KB)...\n");
    const int small_size = 1024;
    struct ggml_tensor * small_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, small_size);

    // Allocate at buffer base
    void * buffer_base = ggml_backend_buffer_get_base(buffer);
    ggml_backend_tensor_alloc(buffer, small_tensor, buffer_base);

    float * small_host = (float *)malloc(small_size * sizeof(float));
    for (int i = 0; i < small_size; i++) {
        small_host[i] = (float)i;
    }

    ggml_backend_tensor_set_async(backend, small_tensor, small_host, 0, small_size * sizeof(float));
    ggml_backend_synchronize(backend);
    printf("✓ Transferred to device\n");

    float * small_readback = (float *)malloc(small_size * sizeof(float));
    memset(small_readback, 0, small_size * sizeof(float));
    ggml_backend_tensor_get_async(backend, small_tensor, small_readback, 0, small_size * sizeof(float));
    ggml_backend_synchronize(backend);
    printf("✓ Read back from device\n");

    int small_errors = 0;
    for (int i = 0; i < small_size; i++) {
        if (small_host[i] != small_readback[i]) {
            if (small_errors < 5) {
                printf("✗ Mismatch at index %d: %.2f != %.2f\n", i, small_host[i], small_readback[i]);
            }
            small_errors++;
        }
    }

    if (small_errors == 0) {
        printf("✓ Verification PASSED (all %d values match)\n\n", small_size);
    } else {
        printf("✗ Verification FAILED (%d mismatches)\n\n", small_errors);
        free(small_readback);
        free(small_host);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    free(small_readback);
    free(small_host);

    // Test 2: Larger transfer (1M elements, 4 MB)
    printf("Test 2: Larger transfer (1M float32 elements, 4 MB)...\n");
    const int large_size = 1024 * 1024;
    struct ggml_tensor * large_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, large_size);

    // Allocate after small tensor (with alignment)
    size_t small_alloc_size = ggml_backend_buffer_get_alloc_size(buffer, small_tensor);
    size_t alignment = 256; // TensorRT alignment requirement
    size_t offset = (small_alloc_size + alignment - 1) / alignment * alignment;
    void * large_addr = (char *)buffer_base + offset;
    ggml_backend_tensor_alloc(buffer, large_tensor, large_addr);

    float * large_host = (float *)malloc(large_size * sizeof(float));
    srand(42);
    for (int i = 0; i < large_size; i++) {
        large_host[i] = (float)rand() / RAND_MAX;
    }

    // Measure transfer time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    ggml_backend_tensor_set_async(backend, large_tensor, large_host, 0, large_size * sizeof(float));
    ggml_backend_synchronize(backend);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double write_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double write_bw = (large_size * sizeof(float)) / (write_time * 1024.0 * 1024.0 * 1024.0);
    printf("✓ Transferred to device: %.2f ms (%.2f GB/s)\n", write_time * 1000, write_bw);

    float * large_readback = (float *)malloc(large_size * sizeof(float));

    clock_gettime(CLOCK_MONOTONIC, &start);

    ggml_backend_tensor_get_async(backend, large_tensor, large_readback, 0, large_size * sizeof(float));
    ggml_backend_synchronize(backend);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double read_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double read_bw = (large_size * sizeof(float)) / (read_time * 1024.0 * 1024.0 * 1024.0);
    printf("✓ Read back from device: %.2f ms (%.2f GB/s)\n", read_time * 1000, read_bw);

    int large_errors = 0;
    for (int i = 0; i < large_size; i++) {
        if (large_host[i] != large_readback[i]) {
            large_errors++;
        }
    }

    if (large_errors == 0) {
        printf("✓ Verification PASSED (all %d values match)\n\n", large_size);
    } else {
        printf("✗ Verification FAILED (%d mismatches)\n\n", large_errors);
        free(large_readback);
        free(large_host);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    free(large_readback);
    free(large_host);

    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("✓ Tensor transfer test PASSED\n");
    return 0;
}
