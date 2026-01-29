#include "ggml-tensorrt.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Testing TensorRT-RTX buffer management...\n\n");

    // Initialize backend
    printf("Initializing backend...\n");
    ggml_backend_t backend = ggml_backend_tensorrt_init(0);
    if (!backend) {
        printf("✗ ERROR: Failed to initialize backend\n");
        return 1;
    }
    printf("✓ Backend initialized: %s\n", ggml_backend_name(backend));

    // Get buffer type
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    printf("✓ Got default buffer type: %s\n\n", ggml_backend_buft_name(buft));

    // Test 1: Small buffer (1 MB)
    printf("Test 1: Allocating small buffer (1 MB)...\n");
    size_t small_size = 1 * 1024 * 1024;
    ggml_backend_buffer_t small_buffer = ggml_backend_buft_alloc_buffer(buft, small_size);
    if (!small_buffer) {
        printf("✗ ERROR: Failed to allocate small buffer\n");
        ggml_backend_free(backend);
        return 1;
    }
    printf("✓ Allocated buffer: %.2f MB\n", small_size / (1024.0 * 1024.0));
    void * small_base = ggml_backend_buffer_get_base(small_buffer);
    printf("✓ Buffer base pointer: %p\n\n", small_base);
    ggml_backend_buffer_free(small_buffer);

    // Test 2: Larger buffer (100 MB)
    printf("Test 2: Allocating larger buffer (100 MB)...\n");
    size_t large_size = 100 * 1024 * 1024;
    ggml_backend_buffer_t large_buffer = ggml_backend_buft_alloc_buffer(buft, large_size);
    if (!large_buffer) {
        printf("✗ ERROR: Failed to allocate large buffer\n");
        ggml_backend_free(backend);
        return 1;
    }
    printf("✓ Allocated buffer: %.2f MB\n\n", large_size / (1024.0 * 1024.0));
    ggml_backend_buffer_free(large_buffer);

    // Test 3: Host buffer (pinned memory)
    printf("Test 3: Allocating host buffer (pinned, 10 MB)...\n");
    ggml_backend_buffer_type_t host_buft = ggml_backend_tensorrt_host_buffer_type();
    size_t host_size = 10 * 1024 * 1024;
    ggml_backend_buffer_t host_buffer = ggml_backend_buft_alloc_buffer(host_buft, host_size);
    if (!host_buffer) {
        printf("✗ ERROR: Failed to allocate host buffer\n");
        ggml_backend_free(backend);
        return 1;
    }
    printf("✓ Allocated host buffer: %.2f MB\n", host_size / (1024.0 * 1024.0));
    void * host_base = ggml_backend_buffer_get_base(host_buffer);
    printf("✓ Host buffer base: %p\n", host_base);
    printf("✓ Host buffer is pinned memory (faster PCIe transfers)\n\n");
    ggml_backend_buffer_free(host_buffer);

    // Test 4: Multiple allocations
    printf("Test 4: Multiple simultaneous allocations...\n");
    const int num_buffers = 5;
    ggml_backend_buffer_t buffers[num_buffers];
    for (int i = 0; i < num_buffers; i++) {
        buffers[i] = ggml_backend_buft_alloc_buffer(buft, 5 * 1024 * 1024);
        if (!buffers[i]) {
            printf("✗ ERROR: Failed to allocate buffer %d\n", i);
            // Cleanup already allocated
            for (int j = 0; j < i; j++) {
                ggml_backend_buffer_free(buffers[j]);
            }
            ggml_backend_free(backend);
            return 1;
        }
        printf("✓ Allocated buffer %d: 5 MB\n", i + 1);
    }
    printf("✓ All %d buffers allocated successfully\n\n", num_buffers);

    // Cleanup
    printf("Cleaning up...\n");
    for (int i = 0; i < num_buffers; i++) {
        ggml_backend_buffer_free(buffers[i]);
    }
    ggml_backend_free(backend);
    printf("✓ All buffers freed\n");

    printf("\n✓ Buffer management test PASSED\n");
    return 0;
}
