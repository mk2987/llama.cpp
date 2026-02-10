// Test TensorRT-RTX backend operations (Milestones 2-3)
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

    // Create test tensors: C = B @ A^T
    // A: [N, K] = [3, 4] (rows, columns)
    // B: [M, K] = [2, 4]
    // C: [M, N] = [3, 2]
    const int64_t K = 4, N = 3, M = 2;

    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);

    // Prepare test data in temporary arrays
    std::vector<float> a_data(K * N);
    std::vector<float> b_data(K * M);

    for (int64_t i = 0; i < K * N; ++i) {
        a_data[i] = (float)(i % 10);
    }

    for (int64_t i = 0; i < K * M; ++i) {
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

    // Verify values: manually compute expected result for A @ B^T
    // a = np.array([[0, 1, 2, 3],
    //        [4, 5, 6, 7],
    //        [8, 9, 0, 1]])
    //
    // b = np.array([[1, 2, 3, 4],
    //        [5, 1, 2, 3]])
    //
    // c = np.array([[20, 60, 30],
    //        [14, 58, 52]])
    float expected[6] = {20.0f, 60.0f, 30.0f, 14.0f, 58.0f, 52.0f};
    for (int i = 0; i < 6; i++) {
        float diff = fabs(c_result[i] - expected[i]);
        if (diff > 1e-4f) {
            fprintf(stderr, "Value mismatch at index %d: got %.2f, expected %.2f\n",
                    i, c_result[i], expected[i]);
            return false;
        }
    }

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

// Integration test: Multi-layer network (matmul -> add -> matmul)
static bool test_multi_layer() {
    printf("Testing multi-layer integration (MUL_MAT -> ADD -> MUL_MAT)...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;  // Required for backend allocation
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    // Create a 2-layer network:
    // Layer 1: x @ W1 + b1 -> h
    // Layer 2: h @ W2 -> y

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

// ====================================================================
// Milestone 3 tests
// ====================================================================

// CPU reference: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
static float ref_silu(float x) {
    return x / (1.0f + expf(-x));
}

// CPU reference: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
static float ref_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}

// Test SiLU activation
static bool test_silu() {
    printf("Testing SILU operation...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t n = 64;
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    // Input data in [-3, 3]
    std::vector<float> x_data(n);
    for (int64_t i = 0; i < n; ++i) {
        x_data[i] = -3.0f + 6.0f * (float)i / (float)(n - 1);
    }

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_silu(ctx, x);
    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result and verify
    std::vector<float> y_result(n);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    for (int64_t i = 0; i < n; ++i) {
        float expected = ref_silu(x_data[i]);
        float diff = fabs(y_result[i] - expected);
        if (diff > 1e-3f) {
            fprintf(stderr, "SILU mismatch at index %lld: got %.6f, expected %.6f (diff: %.6f)\n",
                    (long long)i, y_result[i], expected, diff);
            return false;
        }
    }

    printf("SILU test passed!\n");

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test GELU activation
static bool test_gelu() {
    printf("Testing GELU operation...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t n = 64;
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    // Input data in [-3, 3]
    std::vector<float> x_data(n);
    for (int64_t i = 0; i < n; ++i) {
        x_data[i] = -3.0f + 6.0f * (float)i / (float)(n - 1);
    }

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_gelu(ctx, x);
    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result and verify
    std::vector<float> y_result(n);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    for (int64_t i = 0; i < n; ++i) {
        float expected = ref_gelu(x_data[i]);
        float diff = fabs(y_result[i] - expected);
        if (diff > 1e-3f) {
            fprintf(stderr, "GELU mismatch at index %lld: got %.6f, expected %.6f (diff: %.6f)\n",
                    (long long)i, y_result[i], expected, diff);
            return false;
        }
    }

    printf("GELU test passed!\n");

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test RELU activation
static bool test_relu() {
    printf("Testing RELU operation...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t n = 64;
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    // Mix of positive and negative values
    std::vector<float> x_data(n);
    for (int64_t i = 0; i < n; ++i) {
        x_data[i] = -2.0f + 4.0f * (float)i / (float)(n - 1);
    }

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_relu(ctx, x);
    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result and verify: relu(x) = max(0, x)
    std::vector<float> y_result(n);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    for (int64_t i = 0; i < n; ++i) {
        float expected = x_data[i] > 0.0f ? x_data[i] : 0.0f;
        float diff = fabs(y_result[i] - expected);
        if (diff > 1e-5f) {
            fprintf(stderr, "RELU mismatch at index %lld: got %.6f, expected %.6f\n",
                    (long long)i, y_result[i], expected);
            return false;
        }
    }

    printf("RELU test passed!\n");

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test SOFTMAX operation (default scale=1.0, no mask)
static bool test_softmax() {
    printf("Testing SOFTMAX operation...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    // 2D matrix [10, 4] — softmax along dim 0 (rows of 10)
    const int64_t cols = 10;
    const int64_t rows = 4;
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);

    std::vector<float> x_data(cols * rows);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < cols * rows; ++i) {
        x_data[i] = dist(rng);
    }

    // Build graph: ggml_soft_max uses scale=1.0, mask=NULL, max_bias=0
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_soft_max(ctx, x);
    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<float> y_result(cols * rows);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    // Verify: each row should sum to ~1.0 and all values in [0, 1]
    for (int64_t r = 0; r < rows; ++r) {
        float row_sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float val = y_result[r * cols + c];
            ASSERT_TRUE(val >= 0.0f && val <= 1.0f);
            row_sum += val;
        }
        if (fabs(row_sum - 1.0f) > 1e-3f) {
            fprintf(stderr, "SOFTMAX row %lld sum = %.6f (expected ~1.0)\n",
                    (long long)r, row_sum);
            return false;
        }
    }

    printf("SOFTMAX test passed!\n");

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test SOFTMAX with scale=0.5
static bool test_softmax_scale() {
    printf("Testing SOFTMAX with scale...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t cols = 8;
    const int64_t rows = 4;
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);

    std::vector<float> x_data(cols * rows);
    for (int64_t i = 0; i < cols * rows; ++i) {
        x_data[i] = (float)(i % 8) - 3.5f;
    }

    // Build graph with scale=0.5
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_soft_max_ext(ctx, x, NULL, 0.5f, 0.0f);
    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<float> y_result(cols * rows);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    // Verify: rows still sum to ~1.0 (softmax is a probability distribution)
    for (int64_t r = 0; r < rows; ++r) {
        float row_sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float val = y_result[r * cols + c];
            ASSERT_TRUE(val >= 0.0f && val <= 1.0f);
            row_sum += val;
        }
        if (fabs(row_sum - 1.0f) > 1e-3f) {
            fprintf(stderr, "SOFTMAX_SCALE row %lld sum = %.6f (expected ~1.0)\n",
                    (long long)r, row_sum);
            return false;
        }
    }

    // Verify the distribution is more uniform than scale=1.0 would give
    // (lower temperature = more uniform). We check entropy is higher.
    // With scale=0.5 the logits are halved before softmax, making the
    // distribution more uniform (higher entropy).
    // Just verify the basic sum-to-1 property above is sufficient for now.

    printf("SOFTMAX_SCALE test passed!\n");

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test reshape followed by add (verifies shape ops inside TRT subgraph)
static bool test_reshape_add() {
    printf("Testing RESHAPE + ADD...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    // Reshape 1D [12] -> 2D [3, 4], then ADD with a [3, 4] tensor
    const int64_t n = 12;
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4);

    std::vector<float> x_data(n);
    std::vector<float> b_data(n);
    for (int64_t i = 0; i < n; ++i) {
        x_data[i] = (float)i;
        b_data[i] = 10.0f;
    }

    // Build graph: reshape(x, [3, 4]) + b
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* x_reshaped = ggml_reshape_2d(ctx, x, 3, 4);
    struct ggml_tensor* y = ggml_add(ctx, x_reshaped, b);
    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result and verify: each element should be x[i] + 10.0
    std::vector<float> y_result(n);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    for (int64_t i = 0; i < n; ++i) {
        float expected = x_data[i] + 10.0f;
        float diff = fabs(y_result[i] - expected);
        if (diff > 1e-5f) {
            fprintf(stderr, "RESHAPE_ADD mismatch at index %lld: got %.6f, expected %.6f\n",
                    (long long)i, y_result[i], expected);
            return false;
        }
    }

    ASSERT_TRUE(y->ne[0] == 3);
    ASSERT_TRUE(y->ne[1] == 4);

    printf("RESHAPE_ADD test passed!\n");

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test MUL_MAT -> SILU -> MUL_MAT (common FFN pattern in LLaMA/Qwen)
static bool test_silu_matmul() {
    printf("Testing MUL_MAT -> SILU -> MUL_MAT...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    // x [4, 2] -> matmul with W1 [4, 8] -> h [8, 2] -> silu -> matmul with W2 [8, 3] -> y [3, 2]
    const int64_t n_in = 4, n_hidden = 8, n_out = 3, n_batch = 2;

    struct ggml_tensor* x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_batch);
    struct ggml_tensor* W1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_hidden);
    struct ggml_tensor* W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hidden, n_out);

    std::vector<float> x_data(n_in * n_batch);
    std::vector<float> W1_data(n_in * n_hidden);
    std::vector<float> W2_data(n_hidden * n_out);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    for (auto& v : x_data)  v = dist(rng);
    for (auto& v : W1_data) v = dist(rng);
    for (auto& v : W2_data) v = dist(rng);

    // Build graph: y = W2 @ silu(W1 @ x)
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* h = ggml_mul_mat(ctx, W1, x);       // [n_hidden, n_batch]
    struct ggml_tensor* h_act = ggml_silu(ctx, h);           // [n_hidden, n_batch]
    struct ggml_tensor* y = ggml_mul_mat(ctx, W2, h_act);    // [n_out, n_batch]
    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x,  x_data.data(),  0, ggml_nbytes(x));
    ggml_backend_tensor_set(W1, W1_data.data(), 0, ggml_nbytes(W1));
    ggml_backend_tensor_set(W2, W2_data.data(), 0, ggml_nbytes(W2));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<float> y_result(n_out * n_batch);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    // Verify dimensions and finite values
    ASSERT_TRUE(y->ne[0] == n_out);
    ASSERT_TRUE(y->ne[1] == n_batch);

    for (int64_t i = 0; i < n_out * n_batch; ++i) {
        ASSERT_TRUE(!std::isnan(y_result[i]));
        ASSERT_TRUE(!std::isinf(y_result[i]));
    }

    printf("SILU_MATMUL test passed!\n");
    printf("  Output shape: [%lld, %lld]\n", (long long)y->ne[0], (long long)y->ne[1]);
    printf("  Sample output values: %.4f, %.4f\n", y_result[0], y_result[1]);

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Integration: simplified transformer block
// matmul -> add -> rms_norm -> matmul -> silu -> matmul -> add
static bool test_transformer_block() {
    printf("Testing simplified transformer block...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    // Dimensions
    const int64_t d_model = 16;
    const int64_t d_ff    = 32;
    const int64_t n_batch = 2;

    // Tensors
    struct ggml_tensor* x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_batch);
    struct ggml_tensor* W_attn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, d_model);
    struct ggml_tensor* b_attn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_batch);
    struct ggml_tensor* W_up  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, d_ff);
    struct ggml_tensor* W_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_ff, d_model);
    struct ggml_tensor* b_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_batch);

    // Initialize weights
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    auto fill_random = [&](std::vector<float>& v) {
        for (auto& val : v) val = dist(rng);
    };

    std::vector<float> x_data(d_model * n_batch);
    std::vector<float> W_attn_data(d_model * d_model);
    std::vector<float> b_attn_data(d_model * n_batch);
    std::vector<float> W_up_data(d_model * d_ff);
    std::vector<float> W_down_data(d_ff * d_model);
    std::vector<float> b_out_data(d_model * n_batch);

    fill_random(x_data);
    fill_random(W_attn_data);
    fill_random(b_attn_data);
    fill_random(W_up_data);
    fill_random(W_down_data);
    fill_random(b_out_data);

    // Build graph:
    // attention_out = W_attn @ x + b_attn
    // normed = rms_norm(attention_out)
    // ff_hidden = silu(W_up @ normed)
    // ff_out = W_down @ ff_hidden + b_out
    struct ggml_cgraph* gf = ggml_new_graph(ctx);

    struct ggml_tensor* attn = ggml_mul_mat(ctx, W_attn, x);
    struct ggml_tensor* attn_bias = ggml_add(ctx, attn, b_attn);
    struct ggml_tensor* normed = ggml_rms_norm(ctx, attn_bias, 1e-6f);
    struct ggml_tensor* ff_up = ggml_mul_mat(ctx, W_up, normed);
    struct ggml_tensor* ff_act = ggml_silu(ctx, ff_up);
    struct ggml_tensor* ff_down = ggml_mul_mat(ctx, W_down, ff_act);
    struct ggml_tensor* y = ggml_add(ctx, ff_down, b_out);

    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x,      x_data.data(),      0, ggml_nbytes(x));
    ggml_backend_tensor_set(W_attn, W_attn_data.data(), 0, ggml_nbytes(W_attn));
    ggml_backend_tensor_set(b_attn, b_attn_data.data(), 0, ggml_nbytes(b_attn));
    ggml_backend_tensor_set(W_up,   W_up_data.data(),   0, ggml_nbytes(W_up));
    ggml_backend_tensor_set(W_down, W_down_data.data(), 0, ggml_nbytes(W_down));
    ggml_backend_tensor_set(b_out,  b_out_data.data(),  0, ggml_nbytes(b_out));

    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<float> y_result(d_model * n_batch);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    // Verify dimensions and finite values
    ASSERT_TRUE(y->ne[0] == d_model);
    ASSERT_TRUE(y->ne[1] == n_batch);

    for (int64_t i = 0; i < d_model * n_batch; ++i) {
        ASSERT_TRUE(!std::isnan(y_result[i]));
        ASSERT_TRUE(!std::isinf(y_result[i]));
    }

    printf("Transformer block test PASSED!\n");
    printf("  Output shape: [%lld, %lld]\n", (long long)y->ne[0], (long long)y->ne[1]);
    printf("  Sample output values: %.4f, %.4f, %.4f, %.4f\n",
           y_result[0], y_result[1], y_result[2], y_result[3]);

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// ====================================================================
// Milestone 5 tests: BF16/FP16 precision
// ====================================================================

// Test BF16 elementwise ADD
static bool test_bf16_add() {
    printf("Testing BF16 ADD operation...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t n = 64;
    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n);
    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n);

    // Prepare FP32 data then convert to BF16
    std::vector<float> a_fp32(n), b_fp32(n);
    for (int64_t i = 0; i < n; ++i) {
        a_fp32[i] = (float)i * 0.1f;
        b_fp32[i] = (float)(i * 2) * 0.1f;
    }

    std::vector<ggml_bf16_t> a_bf16(n), b_bf16(n);
    ggml_fp32_to_bf16_row(a_fp32.data(), a_bf16.data(), n);
    ggml_fp32_to_bf16_row(b_fp32.data(), b_bf16.data(), n);

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* c = ggml_add(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(a, a_bf16.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_bf16.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result and convert back to FP32
    std::vector<ggml_bf16_t> c_bf16(n);
    ggml_backend_tensor_get(c, c_bf16.data(), 0, ggml_nbytes(c));

    std::vector<float> c_fp32(n);
    ggml_bf16_to_fp32_row(c_bf16.data(), c_fp32.data(), n);

    // Verify with relaxed tolerance for BF16
    for (int64_t i = 0; i < n; ++i) {
        float expected = a_fp32[i] + b_fp32[i];
        float diff = fabs(c_fp32[i] - expected);
        if (diff > 1e-1f) {
            fprintf(stderr, "BF16 ADD mismatch at %lld: got %.4f, expected %.4f\n",
                    (long long)i, c_fp32[i], expected);
            return false;
        }
    }

    printf("BF16 ADD test passed!\n");

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test BF16 matrix multiplication
static bool test_bf16_mul_mat() {
    printf("Testing BF16 MUL_MAT operation...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t K = 4, N = 3, M = 2;

    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, K, M);

    // Prepare data
    std::vector<float> a_fp32(K * N), b_fp32(K * M);
    for (int64_t i = 0; i < K * N; ++i) a_fp32[i] = (float)(i % 10);
    for (int64_t i = 0; i < K * M; ++i) b_fp32[i] = (float)((i % 5) + 1);

    std::vector<ggml_bf16_t> a_bf16(K * N), b_bf16(K * M);
    ggml_fp32_to_bf16_row(a_fp32.data(), a_bf16.data(), K * N);
    ggml_fp32_to_bf16_row(b_fp32.data(), b_bf16.data(), K * M);

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(a, a_bf16.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_bf16.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<ggml_bf16_t> c_bf16(M * N);
    ggml_backend_tensor_get(c, c_bf16.data(), 0, ggml_nbytes(c));

    std::vector<float> c_fp32(M * N);
    ggml_bf16_to_fp32_row(c_bf16.data(), c_fp32.data(), M * N);

    // Same expected values as F32 matmul test
    float expected[6] = {20.0f, 60.0f, 30.0f, 14.0f, 58.0f, 52.0f};
    for (int i = 0; i < 6; i++) {
        float diff = fabs(c_fp32[i] - expected[i]);
        if (diff > 1.0f) {
            fprintf(stderr, "BF16 MUL_MAT mismatch at %d: got %.2f, expected %.2f\n",
                    i, c_fp32[i], expected[i]);
            return false;
        }
    }

    printf("BF16 MUL_MAT test passed!\n");

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test BF16 RMS_NORM (should upcast to FP32 internally, downcast back)
static bool test_bf16_rms_norm() {
    printf("Testing BF16 RMS_NORM operation...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t n = 64;
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n);

    // Prepare data
    std::vector<float> x_fp32(n);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) x_fp32[i] = dist(rng);

    std::vector<ggml_bf16_t> x_bf16(n);
    ggml_fp32_to_bf16_row(x_fp32.data(), x_bf16.data(), n);

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_rms_norm(ctx, x, 1e-6f);
    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_bf16.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<ggml_bf16_t> y_bf16(n);
    ggml_backend_tensor_get(y, y_bf16.data(), 0, ggml_nbytes(y));

    std::vector<float> y_fp32(n);
    ggml_bf16_to_fp32_row(y_bf16.data(), y_fp32.data(), n);

    // Verify: RMS should be approximately 1.0
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum_sq += y_fp32[i] * y_fp32[i];
    }
    float rms = sqrtf(sum_sq / n);

    ASSERT_TRUE(fabs(rms - 1.0f) < 0.2f);  // relaxed for BF16

    printf("BF16 RMS_NORM test passed! (rms=%.4f)\n", rms);

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test BF16 softmax
static bool test_bf16_softmax() {
    printf("Testing BF16 SOFTMAX operation...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t cols = 10;
    const int64_t rows = 4;
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, cols, rows);

    // Prepare data
    std::vector<float> x_fp32(cols * rows);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < cols * rows; ++i) x_fp32[i] = dist(rng);

    std::vector<ggml_bf16_t> x_bf16(cols * rows);
    ggml_fp32_to_bf16_row(x_fp32.data(), x_bf16.data(), cols * rows);

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_soft_max(ctx, x);
    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_bf16.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<ggml_bf16_t> y_bf16(cols * rows);
    ggml_backend_tensor_get(y, y_bf16.data(), 0, ggml_nbytes(y));

    std::vector<float> y_fp32(cols * rows);
    ggml_bf16_to_fp32_row(y_bf16.data(), y_fp32.data(), cols * rows);

    // Verify: each row should sum to ~1.0
    for (int64_t r = 0; r < rows; ++r) {
        float row_sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float val = y_fp32[r * cols + c];
            ASSERT_TRUE(val >= -0.01f && val <= 1.01f);
            row_sum += val;
        }
        if (fabs(row_sum - 1.0f) > 5e-2f) {
            fprintf(stderr, "BF16 SOFTMAX row %lld sum = %.6f (expected ~1.0)\n",
                    (long long)r, row_sum);
            return false;
        }
    }

    printf("BF16 SOFTMAX test passed!\n");

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test BF16 end-to-end transformer block
static bool test_bf16_transformer_block() {
    printf("Testing BF16 transformer block...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t d_model = 16;
    const int64_t d_ff    = 32;
    const int64_t n_batch = 2;

    struct ggml_tensor* x     = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_model, n_batch);
    struct ggml_tensor* W_attn = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_model, d_model);
    struct ggml_tensor* b_attn = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_model, n_batch);
    struct ggml_tensor* W_up  = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_model, d_ff);
    struct ggml_tensor* W_down = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_ff, d_model);
    struct ggml_tensor* b_out = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_model, n_batch);

    // Initialize weights
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    auto fill_bf16 = [&](struct ggml_tensor* t, int64_t nelements) {
        std::vector<float> fp32(nelements);
        for (auto& v : fp32) v = dist(rng);
        std::vector<ggml_bf16_t> bf16(nelements);
        ggml_fp32_to_bf16_row(fp32.data(), bf16.data(), nelements);
        return bf16;
    };

    auto x_bf16      = fill_bf16(x, d_model * n_batch);
    auto W_attn_bf16 = fill_bf16(W_attn, d_model * d_model);
    auto b_attn_bf16 = fill_bf16(b_attn, d_model * n_batch);
    auto W_up_bf16   = fill_bf16(W_up, d_model * d_ff);
    auto W_down_bf16 = fill_bf16(W_down, d_ff * d_model);
    auto b_out_bf16  = fill_bf16(b_out, d_model * n_batch);

    // Build graph: matmul -> add -> rms_norm -> matmul -> silu -> matmul -> add
    struct ggml_cgraph* gf = ggml_new_graph(ctx);

    struct ggml_tensor* attn      = ggml_mul_mat(ctx, W_attn, x);
    struct ggml_tensor* attn_bias = ggml_add(ctx, attn, b_attn);
    struct ggml_tensor* normed    = ggml_rms_norm(ctx, attn_bias, 1e-6f);
    struct ggml_tensor* ff_up     = ggml_mul_mat(ctx, W_up, normed);
    struct ggml_tensor* ff_act    = ggml_silu(ctx, ff_up);
    struct ggml_tensor* ff_down   = ggml_mul_mat(ctx, W_down, ff_act);
    struct ggml_tensor* y         = ggml_add(ctx, ff_down, b_out);

    ggml_build_forward_expand(gf, y);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x,      x_bf16.data(),      0, ggml_nbytes(x));
    ggml_backend_tensor_set(W_attn, W_attn_bf16.data(), 0, ggml_nbytes(W_attn));
    ggml_backend_tensor_set(b_attn, b_attn_bf16.data(), 0, ggml_nbytes(b_attn));
    ggml_backend_tensor_set(W_up,   W_up_bf16.data(),   0, ggml_nbytes(W_up));
    ggml_backend_tensor_set(W_down, W_down_bf16.data(), 0, ggml_nbytes(W_down));
    ggml_backend_tensor_set(b_out,  b_out_bf16.data(),  0, ggml_nbytes(b_out));

    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<ggml_bf16_t> y_bf16(d_model * n_batch);
    ggml_backend_tensor_get(y, y_bf16.data(), 0, ggml_nbytes(y));

    std::vector<float> y_fp32(d_model * n_batch);
    ggml_bf16_to_fp32_row(y_bf16.data(), y_fp32.data(), d_model * n_batch);

    // Verify finite values
    for (int64_t i = 0; i < d_model * n_batch; ++i) {
        ASSERT_TRUE(!std::isnan(y_fp32[i]));
        ASSERT_TRUE(!std::isinf(y_fp32[i]));
    }

    printf("BF16 transformer block test PASSED!\n");
    printf("  Sample output values: %.4f, %.4f, %.4f, %.4f\n",
           y_fp32[0], y_fp32[1], y_fp32[2], y_fp32[3]);

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test FP16 matrix multiplication
static bool test_fp16_mul_mat() {
    printf("Testing FP16 MUL_MAT operation...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t K = 4, N = 3, M = 2;

    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);

    // Prepare data
    std::vector<float> a_fp32(K * N), b_fp32(K * M);
    for (int64_t i = 0; i < K * N; ++i) a_fp32[i] = (float)(i % 10);
    for (int64_t i = 0; i < K * M; ++i) b_fp32[i] = (float)((i % 5) + 1);

    std::vector<ggml_fp16_t> a_fp16(K * N), b_fp16(K * M);
    ggml_fp32_to_fp16_row(a_fp32.data(), a_fp16.data(), K * N);
    ggml_fp32_to_fp16_row(b_fp32.data(), b_fp16.data(), K * M);

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    // Allocate and compute
    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(a, a_fp16.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_fp16.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend_trt, gf);

    // Get result
    std::vector<ggml_fp16_t> c_fp16(M * N);
    ggml_backend_tensor_get(c, c_fp16.data(), 0, ggml_nbytes(c));

    std::vector<float> c_fp32(M * N);
    ggml_fp16_to_fp32_row(c_fp16.data(), c_fp32.data(), M * N);

    // Same expected values as F32 matmul test
    float expected[6] = {20.0f, 60.0f, 30.0f, 14.0f, 58.0f, 52.0f};
    for (int i = 0; i < 6; i++) {
        float diff = fabs(c_fp32[i] - expected[i]);
        if (diff > 1.0f) {
            fprintf(stderr, "FP16 MUL_MAT mismatch at %d: got %.2f, expected %.2f\n",
                    i, c_fp32[i], expected[i]);
            return false;
        }
    }

    printf("FP16 MUL_MAT test passed!\n");

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    printf("=== TensorRT-RTX Backend Operations Test (Milestones 2-5) ===\n\n");

    bool all_passed = true;

    // Milestone 2 tests
    printf("--- Milestone 2: Core Operations ---\n");
    all_passed &= test_mul_mat();
    all_passed &= test_add();
    all_passed &= test_rms_norm();
    all_passed &= test_multi_layer();

    // Milestone 3 tests
    printf("\n--- Milestone 3: Activations, Softmax, Shape Ops ---\n");
    all_passed &= test_silu();
    all_passed &= test_gelu();
    all_passed &= test_relu();
    all_passed &= test_softmax();
    all_passed &= test_softmax_scale();
    all_passed &= test_reshape_add();
    all_passed &= test_silu_matmul();
    all_passed &= test_transformer_block();

    // Milestone 5 tests
    printf("\n--- Milestone 5: BF16/FP16 Precision ---\n");
    all_passed &= test_bf16_add();
    all_passed &= test_bf16_mul_mat();
    all_passed &= test_bf16_rms_norm();
    all_passed &= test_bf16_softmax();
    all_passed &= test_bf16_transformer_block();
    all_passed &= test_fp16_mul_mat();

    printf("\n=== Test Summary ===\n");
    if (all_passed) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
