// TensorRT backend tests — Milestone 2: Core Operations
// MUL_MAT, ADD, RMS_NORM, multi-layer integration

#include "test-tensorrt-common.h"

// Test matrix multiplication (MUL_MAT)
static bool test_mul_mat() {
    printf("Testing MUL_MAT operation...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t K = 4, N = 3, M = 2;

    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);

    std::vector<float> a_data(K * N);
    std::vector<float> b_data(K * M);

    for (int64_t i = 0; i < K * N; ++i) a_data[i] = (float)(i % 10);
    for (int64_t i = 0; i < K * M; ++i) b_data[i] = (float)((i % 5) + 1);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> c_result(M * N);
    ggml_backend_tensor_get(c, c_result.data(), 0, ggml_nbytes(c));

    ASSERT_TRUE(c->ne[0] == N);
    ASSERT_TRUE(c->ne[1] == M);

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
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t n = 100;
    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    std::vector<float> a_data(n);
    std::vector<float> b_data(n);
    for (int64_t i = 0; i < n; ++i) {
        a_data[i] = (float)i;
        b_data[i] = (float)(i * 2);
    }

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* c = ggml_add(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> c_result(n);
    ggml_backend_tensor_get(c, c_result.data(), 0, ggml_nbytes(c));

    for (int64_t i = 0; i < n; ++i) {
        float expected = a_data[i] + b_data[i];
        ASSERT_TRUE(fabs(c_result[i] - expected) < 1e-5f);
    }

    printf("ADD test passed!\n");

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
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t n = 64;
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    std::vector<float> x_data(n);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) x_data[i] = dist(rng);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_rms_norm(ctx, x, 1e-6f);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_result(n);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum_sq += y_result[i] * y_result[i];
    float rms = sqrtf(sum_sq / n);
    ASSERT_TRUE(fabs(rms - 1.0f) < 0.1f);

    printf("RMS_NORM test passed!\n");

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
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t n_samples = 3, n_input = 4, n_hidden = 5, n_output = 2;

    struct ggml_tensor* x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_input, n_samples);
    struct ggml_tensor* W1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_input, n_hidden);
    struct ggml_tensor* b1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hidden, n_samples);
    struct ggml_tensor* W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hidden, n_output);

    std::vector<float> x_data(n_input * n_samples);
    std::vector<float> W1_data(n_input * n_hidden);
    std::vector<float> b1_data(n_hidden * n_samples);
    std::vector<float> W2_data(n_hidden * n_output);

    for (int64_t i = 0; i < n_input * n_samples; ++i) x_data[i] = (float)(i % 10) * 0.1f;
    for (int64_t i = 0; i < n_input * n_hidden; ++i)  W1_data[i] = (float)(i % 7) * 0.1f;
    for (int64_t i = 0; i < n_hidden * n_samples; ++i) b1_data[i] = 0.5f;
    for (int64_t i = 0; i < n_hidden * n_output; ++i)  W2_data[i] = (float)(i % 5) * 0.1f;

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* h_matmul = ggml_mul_mat(ctx, W1, x);
    struct ggml_tensor* h = ggml_add(ctx, h_matmul, b1);
    struct ggml_tensor* y = ggml_mul_mat(ctx, W2, h);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x,  x_data.data(),  0, ggml_nbytes(x));
    ggml_backend_tensor_set(W1, W1_data.data(), 0, ggml_nbytes(W1));
    ggml_backend_tensor_set(b1, b1_data.data(), 0, ggml_nbytes(b1));
    ggml_backend_tensor_set(W2, W2_data.data(), 0, ggml_nbytes(W2));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_result(n_output * n_samples);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    ASSERT_TRUE(y->ne[0] == n_output);
    ASSERT_TRUE(y->ne[1] == n_samples);

    for (int64_t i = 0; i < n_output * n_samples; ++i) {
        ASSERT_TRUE(!std::isnan(y_result[i]));
        ASSERT_TRUE(!std::isinf(y_result[i]));
    }

    printf("Multi-layer integration test PASSED!\n");

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

int main() {
    printf("=== TensorRT M2: Core Operations ===\n\n");
    bool all_passed = true;
    all_passed &= test_mul_mat();
    all_passed &= test_add();
    all_passed &= test_rms_norm();
    all_passed &= test_multi_layer();
    printf("\n%s\n", all_passed ? "All M2 tests PASSED!" : "Some M2 tests FAILED!");
    return all_passed ? 0 : 1;
}
