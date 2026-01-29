#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-tensorrt.h"
#include <stdio.h>
#include <string.h>

int main() {
    printf("Testing TensorRT-RTX backend registration...\n\n");

    // Get backend registry
    size_t backend_count = ggml_backend_reg_count();
    printf("Total backends registered: %zu\n", backend_count);

    // Find TensorRT backend
    bool found = false;
    for (size_t i = 0; i < backend_count; i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        const char * name = ggml_backend_reg_name(reg);
        printf("  Backend %zu: %s\n", i, name);

        if (strcmp(name, "TensorRT-RTX") == 0) {
            found = true;
            printf("    ✓ TensorRT-RTX backend found!\n");

            // Get device count
            size_t dev_count = ggml_backend_reg_dev_count(reg);
            printf("    ✓ Device count: %zu\n", dev_count);

            // List devices
            for (size_t j = 0; j < dev_count; j++) {
                ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, j);
                printf("    ✓ Device %zu: %s\n", j, ggml_backend_dev_name(dev));
            }
        }
    }

    if (!found) {
        printf("\n✗ ERROR: TensorRT-RTX backend NOT found!\n");
        printf("  This usually means:\n");
        printf("  1. The backend was not compiled (check GGML_TENSORRT=ON)\n");
        printf("  2. The library failed to load (check LD_LIBRARY_PATH)\n");
        return 1;
    }

    printf("\n✓ Backend registration test PASSED\n");
    return 0;
}
