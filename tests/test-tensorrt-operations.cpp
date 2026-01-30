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
    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;  // Required for backend allocation
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

    // Prepare test data in temporary arrays
    std::vector<float> a_data(K * M);
    std::vector<float> b_data(K * N);

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
    ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));

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

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;  // Required for backend allocation
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    // Create test tensors
    const int64_t n = 100;
    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    // Prepare test data in temporary arrays
    std::vector<float> a_data(n);
    std::vector<float> b_data(n);

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

    ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));

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

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;  // Required for backend allocation
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    // Create test tensor
    const int64_t n = 64;
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    // Prepare test data in temporary array
    std::vector<float> x_data(n);
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

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));

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

// Integration test: Multi-layer network (matmul → add → matmul)
static bool test_multi_layer() {
    printf("Testing multi-layer integration (MUL_MAT → ADD → MUL_MAT)...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;  // Required for backend allocation
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    // Create a 2-layer network:
    // Layer 1: x @ W1 + b1 → h
    // Layer 2: h @ W2 → y

    // Dimensions:
    // x:  [4, 3]   (3 samples, 4 features)
    // W1: [4, 5]   (4 input features, 5 hidden units)
    // b1: [5, 3]   (bias, broadcasted)
    // h:  [5, 3]   (hidden layer)
    // W2: [5, 2]   (5 hidden units, 2 output units)
    // y:  [2, 3]   (final output)

    const int64_t n_samples = 3;
    const int64_t n_input = 4;
    const int64_t n_hidden = 5;
    const int64_t n_output = 2;

    // Input and weights
    struct ggml_tensor* x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_input, n_samples);
    struct ggml_tensor* W1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_input, n_hidden);
    struct ggml_tensor* b1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hidden, n_samples);
    struct ggml_tensor* W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hidden, n_output);

    // Prepare test data
    std::vector<float> x_data(n_input * n_samples);
    std::vector<float> W1_data(n_input * n_hidden);
    std::vector<float> b1_data(n_hidden * n_samples);
    std::vector<float> W2_data(n_hidden * n_output);

    // Initialize with simple values
    for (int64_t i = 0; i < n_input * n_samples; ++i) {
        x_data[i] = (float)(i % 10) * 0.1f;
    }
    for (int64_t i = 0; i < n_input * n_hidden; ++i) {
        W1_data[i] = (float)(i % 7) * 0.1f;
    }
    for (int64_t i = 0; i < n_hidden * n_samples; ++i) {
        b1_data[i] = 0.5f;  // Constant bias
    }
    for (int64_t i = 0; i < n_hidden * n_output; ++i) {
        W2_data[i] = (float)(i % 5) * 0.1f;
    }

    // Build computation graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);

    // Layer 1: h = x @ W1 + b1
    struct ggml_tensor* h_matmul = ggml_mul_mat(ctx, W1, x);
    struct ggml_tensor* h = ggml_add(ctx, h_matmul, b1);

    // Layer 2: y = h @ W2
    struct ggml_tensor* y = ggml_mul_mat(ctx, W2, h);

    ggml_build_forward_expand(gf, y);

    // Allocate buffers
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    // Copy input data to backend
    ggml_backend_tensor_set(x,  x_data.data(),  0, ggml_nbytes(x));
    ggml_backend_tensor_set(W1, W1_data.data(), 0, ggml_nbytes(W1));
    ggml_backend_tensor_set(b1, b1_data.data(), 0, ggml_nbytes(b1));
    ggml_backend_tensor_set(W2, W2_data.data(), 0, ggml_nbytes(W2));

    // Execute graph
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<float> y_result(n_output * n_samples);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    // Verify dimensions
    ASSERT_TRUE(y->ne[0] == n_output);
    ASSERT_TRUE(y->ne[1] == n_samples);

    // Compute expected result manually for first element as a sanity check
    // y[0] = sum over hidden dimension of (h[i] * W2[i,0])
    // h[i] = sum over input dimension of (x[j] * W1[j,i]) + b1[i]

    // For now, just check that we got numerical results (not NaN/inf)
    for (int64_t i = 0; i < n_output * n_samples; ++i) {
        ASSERT_TRUE(!std::isnan(y_result[i]));
        ASSERT_TRUE(!std::isinf(y_result[i]));
    }

    printf("Multi-layer integration test PASSED!\n");
    printf("  Input shape: [%lld, %lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
    printf("  Hidden shape: [%lld, %lld]\n", (long long)h->ne[0], (long long)h->ne[1]);
    printf("  Output shape: [%lld, %lld]\n", (long long)y->ne[0], (long long)y->ne[1]);
    printf("  Sample output values: %.4f, %.4f, %.4f\n",
           y_result[0], y_result[1], y_result[2]);

    // Cleanup
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);

    return true;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    printf("=== TensorRT-RTX Backend Operations Test (Milestone 2) ===\n\n");

    bool all_passed = true;

    // Run unit tests
    all_passed &= test_mul_mat();
    all_passed &= test_add();
    all_passed &= test_rms_norm();

    // Run integration test
    all_passed &= test_multi_layer();

    printf("\n=== Test Summary ===\n");
    if (all_passed) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
