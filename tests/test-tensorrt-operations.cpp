// Test TensorRT-RTX backend operations (Milestone 2)
// This test validates the core operations implemented for the TensorRT backend

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

// Test matrix multiplication (MUL_MAT)
static bool test_mul_mat() {
    printf("Testing MUL_MAT operation...\n");

    // Create context
    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    // Get TensorRT backend
    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    // Create test tensors: C = A @ B
    // A: [K, M] = [4, 3]
    // B: [K, N] = [4, 2]
    // C: [M, N] = [3, 2]
    const int64_t K = 4, M = 3, N = 2;

    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);

    // Fill with test data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;

    for (int64_t i = 0; i < K * M; ++i) {
        a_data[i] = (float)(i % 10);
    }

    for (int64_t i = 0; i < K * N; ++i) {
        b_data[i] = (float)((i % 5) + 1);
    }

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    // Allocate buffers
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    // Copy input data to backend
    ggml_backend_tensor_set(a, a_data, 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data, 0, ggml_nbytes(b));

    // Execute
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<float> c_result(M * N);
    ggml_backend_tensor_get(c, c_result.data(), 0, ggml_nbytes(c));

    // Verify dimensions
    ASSERT_TRUE(c->ne[0] == N);
    ASSERT_TRUE(c->ne[1] == M);

    printf("MUL_MAT test passed!\n");

    // Cleanup
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);

    return true;
}

// Test elementwise ADD operation
static bool test_add() {
    printf("Testing ADD operation...\n");

    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    // Create test tensors
    const int64_t n = 100;
    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    // Fill with test data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;

    for (int64_t i = 0; i < n; ++i) {
        a_data[i] = (float)i;
        b_data[i] = (float)(i * 2);
    }

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* c = ggml_add(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(a, a_data, 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data, 0, ggml_nbytes(b));

    ggml_backend_graph_compute(backend_trt, gf);

    // Verify result
    std::vector<float> c_result(n);
    ggml_backend_tensor_get(c, c_result.data(), 0, ggml_nbytes(c));

    for (int64_t i = 0; i < n; ++i) {
        float expected = a_data[i] + b_data[i];
        ASSERT_TRUE(fabs(c_result[i] - expected) < 1e-5f);
    }

    printf("ADD test passed!\n");

    // Cleanup
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);

    return true;
}

// Test RMS_NORM operation
static bool test_rms_norm() {
    printf("Testing RMS_NORM operation...\n");

    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    // Create test tensor
    const int64_t n = 64;
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    // Fill with test data
    float* x_data = (float*)x->data;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int64_t i = 0; i < n; ++i) {
        x_data[i] = dist(rng);
    }

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_rms_norm(ctx, x, 1e-6f);
    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data, 0, ggml_nbytes(x));

    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<float> y_result(n);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    // Verify: RMS should be approximately 1.0 after normalization
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum_sq += y_result[i] * y_result[i];
    }
    float rms = sqrtf(sum_sq / n);

    ASSERT_TRUE(fabs(rms - 1.0f) < 0.1f); // RMS should be close to 1

    printf("RMS_NORM test passed!\n");

    // Cleanup
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);

    return true;
}

int main(int argc, char** argv) {
    printf("=== TensorRT-RTX Backend Operations Test (Milestone 2) ===\n\n");

    bool all_passed = true;

    // Run tests
    all_passed &= test_mul_mat();
    all_passed &= test_add();
    all_passed &= test_rms_norm();

    printf("\n=== Test Summary ===\n");
    if (all_passed) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
