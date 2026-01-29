#include "ggml-tensorrt.h"
#include <stdio.h>

int main() {
    printf("Testing TensorRT-RTX device enumeration...\n\n");

    int device_count = ggml_backend_tensorrt_get_device_count();
    printf("Device count: %d\n\n", device_count);

    if (device_count == 0) {
        printf("✗ ERROR: No devices found!\n");
        printf("  This usually means:\n");
        printf("  1. No NVIDIA GPU available\n");
        printf("  2. CUDA driver not installed\n");
        printf("  3. Docker not running with --gpus all\n");
        return 1;
    }

    for (int i = 0; i < device_count; i++) {
        printf("Device %d:\n", i);

        // Get description
        char description[256];
        ggml_backend_tensorrt_get_device_description(i, description, sizeof(description));
        printf("  Description: %s\n", description);

        // Get memory
        size_t free_mem, total_mem;
        ggml_backend_tensorrt_get_device_memory(i, &free_mem, &total_mem);
        printf("  Total memory: %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
        printf("  Free memory:  %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
        printf("  Used memory:  %.2f GB\n", (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0));
        printf("  Utilization:  %.1f%%\n\n", 100.0 * (total_mem - free_mem) / total_mem);
    }

    printf("✓ Device enumeration test PASSED\n");
    return 0;
}
