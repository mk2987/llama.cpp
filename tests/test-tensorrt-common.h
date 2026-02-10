// Shared helpers for TensorRT backend tests
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-tensorrt.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#define ASSERT_TRUE(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "ASSERTION FAILED: %s at %s:%d\n", #x, __FILE__, __LINE__); \
            return false; \
        } \
    } while (0)

// Helper function to compare tensors with tolerance
static bool tensors_are_close(const ggml_tensor* a, const ggml_tensor* b, float rtol = 1e-3f, float atol = 1e-5f) {
    if (ggml_nelements(a) != ggml_nelements(b)) {
        return false;
    }

    const float* a_data = (const float*)a->data;
    const float* b_data = (const float*)b->data;

    size_t n = ggml_nelements(a);
    for (size_t i = 0; i < n; ++i) {
        float diff = fabs(a_data[i] - b_data[i]);
        float threshold = atol + rtol * fabs(b_data[i]);
        if (diff > threshold) {
            fprintf(stderr, "Mismatch at index %zu: %.6f vs %.6f (diff: %.6f, threshold: %.6f)\n",
                    i, a_data[i], b_data[i], diff, threshold);
            return false;
        }
    }

    return true;
}

// CPU reference: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
static float ref_silu(float x) {
    return x / (1.0f + expf(-x));
}

// CPU reference: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
static float ref_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}
