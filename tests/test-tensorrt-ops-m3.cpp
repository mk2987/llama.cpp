// TensorRT backend tests — Milestone 3: Activations, Softmax, Shape Ops
// SILU, GELU, RELU, SOFTMAX, RESHAPE+ADD, SILU+MATMUL, transformer block

#include "test-tensorrt-common.h"

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

    std::vector<float> x_data(n);
    for (int64_t i = 0; i < n; ++i)
        x_data[i] = -3.0f + 6.0f * (float)i / (float)(n - 1);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_silu(ctx, x);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_result(n);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    for (int64_t i = 0; i < n; ++i) {
        float expected = ref_silu(x_data[i]);
        float diff = fabs(y_result[i] - expected);
        if (diff > 1e-3f) {
            fprintf(stderr, "SILU mismatch at %lld: got %.6f, expected %.6f\n",
                    (long long)i, y_result[i], expected);
            return false;
        }
    }

    printf("SILU test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

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

    std::vector<float> x_data(n);
    for (int64_t i = 0; i < n; ++i)
        x_data[i] = -3.0f + 6.0f * (float)i / (float)(n - 1);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_gelu(ctx, x);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_result(n);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    for (int64_t i = 0; i < n; ++i) {
        float expected = ref_gelu(x_data[i]);
        float diff = fabs(y_result[i] - expected);
        if (diff > 1e-3f) {
            fprintf(stderr, "GELU mismatch at %lld: got %.6f, expected %.6f\n",
                    (long long)i, y_result[i], expected);
            return false;
        }
    }

    printf("GELU test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

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

    std::vector<float> x_data(n);
    for (int64_t i = 0; i < n; ++i)
        x_data[i] = -2.0f + 4.0f * (float)i / (float)(n - 1);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_relu(ctx, x);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_result(n);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    for (int64_t i = 0; i < n; ++i) {
        float expected = x_data[i] > 0.0f ? x_data[i] : 0.0f;
        ASSERT_TRUE(fabs(y_result[i] - expected) < 1e-5f);
    }

    printf("RELU test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

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

    const int64_t cols = 10, rows = 4;
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);

    std::vector<float> x_data(cols * rows);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < cols * rows; ++i) x_data[i] = dist(rng);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_soft_max(ctx, x);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_result(cols * rows);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    for (int64_t r = 0; r < rows; ++r) {
        float row_sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float val = y_result[r * cols + c];
            ASSERT_TRUE(val >= 0.0f && val <= 1.0f);
            row_sum += val;
        }
        if (fabs(row_sum - 1.0f) > 1e-3f) {
            fprintf(stderr, "SOFTMAX row %lld sum = %.6f\n", (long long)r, row_sum);
            return false;
        }
    }

    printf("SOFTMAX test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

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

    const int64_t cols = 8, rows = 4;
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);

    std::vector<float> x_data(cols * rows);
    for (int64_t i = 0; i < cols * rows; ++i) x_data[i] = (float)(i % 8) - 3.5f;

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_soft_max_ext(ctx, x, NULL, 0.5f, 0.0f);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_result(cols * rows);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    for (int64_t r = 0; r < rows; ++r) {
        float row_sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float val = y_result[r * cols + c];
            ASSERT_TRUE(val >= 0.0f && val <= 1.0f);
            row_sum += val;
        }
        if (fabs(row_sum - 1.0f) > 1e-3f) {
            fprintf(stderr, "SOFTMAX_SCALE row %lld sum = %.6f\n", (long long)r, row_sum);
            return false;
        }
    }

    printf("SOFTMAX_SCALE test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

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

    const int64_t n = 12;
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4);

    std::vector<float> x_data(n), b_data(n);
    for (int64_t i = 0; i < n; ++i) { x_data[i] = (float)i; b_data[i] = 10.0f; }

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* x_reshaped = ggml_reshape_2d(ctx, x, 3, 4);
    struct ggml_tensor* y = ggml_add(ctx, x_reshaped, b);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_result(n);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    for (int64_t i = 0; i < n; ++i) {
        ASSERT_TRUE(fabs(y_result[i] - (x_data[i] + 10.0f)) < 1e-5f);
    }
    ASSERT_TRUE(y->ne[0] == 3);
    ASSERT_TRUE(y->ne[1] == 4);

    printf("RESHAPE_ADD test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

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

    const int64_t n_in = 4, n_hidden = 8, n_out = 3, n_batch = 2;

    struct ggml_tensor* x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_batch);
    struct ggml_tensor* W1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_hidden);
    struct ggml_tensor* W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hidden, n_out);

    std::vector<float> x_data(n_in * n_batch), W1_data(n_in * n_hidden), W2_data(n_hidden * n_out);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    for (auto& v : x_data)  v = dist(rng);
    for (auto& v : W1_data) v = dist(rng);
    for (auto& v : W2_data) v = dist(rng);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* h     = ggml_mul_mat(ctx, W1, x);
    struct ggml_tensor* h_act = ggml_silu(ctx, h);
    struct ggml_tensor* y     = ggml_mul_mat(ctx, W2, h_act);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x,  x_data.data(),  0, ggml_nbytes(x));
    ggml_backend_tensor_set(W1, W1_data.data(), 0, ggml_nbytes(W1));
    ggml_backend_tensor_set(W2, W2_data.data(), 0, ggml_nbytes(W2));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_result(n_out * n_batch);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    ASSERT_TRUE(y->ne[0] == n_out);
    ASSERT_TRUE(y->ne[1] == n_batch);
    for (int64_t i = 0; i < n_out * n_batch; ++i) {
        ASSERT_TRUE(!std::isnan(y_result[i]));
        ASSERT_TRUE(!std::isinf(y_result[i]));
    }

    printf("SILU_MATMUL test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

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

    const int64_t d_model = 16, d_ff = 32, n_batch = 2;

    struct ggml_tensor* x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_batch);
    struct ggml_tensor* W_attn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, d_model);
    struct ggml_tensor* b_attn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_batch);
    struct ggml_tensor* W_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, d_ff);
    struct ggml_tensor* W_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_ff, d_model);
    struct ggml_tensor* b_out  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_batch);

    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    auto fill = [&](int64_t n) { std::vector<float> v(n); for (auto& x : v) x = dist(rng); return v; };

    auto x_data      = fill(d_model * n_batch);
    auto W_attn_data = fill(d_model * d_model);
    auto b_attn_data = fill(d_model * n_batch);
    auto W_up_data   = fill(d_model * d_ff);
    auto W_down_data = fill(d_ff * d_model);
    auto b_out_data  = fill(d_model * n_batch);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* attn      = ggml_mul_mat(ctx, W_attn, x);
    struct ggml_tensor* attn_bias = ggml_add(ctx, attn, b_attn);
    struct ggml_tensor* normed    = ggml_rms_norm(ctx, attn_bias, 1e-6f);
    struct ggml_tensor* ff_up     = ggml_mul_mat(ctx, W_up, normed);
    struct ggml_tensor* ff_act    = ggml_silu(ctx, ff_up);
    struct ggml_tensor* ff_down   = ggml_mul_mat(ctx, W_down, ff_act);
    struct ggml_tensor* y         = ggml_add(ctx, ff_down, b_out);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x,      x_data.data(),      0, ggml_nbytes(x));
    ggml_backend_tensor_set(W_attn, W_attn_data.data(), 0, ggml_nbytes(W_attn));
    ggml_backend_tensor_set(b_attn, b_attn_data.data(), 0, ggml_nbytes(b_attn));
    ggml_backend_tensor_set(W_up,   W_up_data.data(),   0, ggml_nbytes(W_up));
    ggml_backend_tensor_set(W_down, W_down_data.data(), 0, ggml_nbytes(W_down));
    ggml_backend_tensor_set(b_out,  b_out_data.data(),  0, ggml_nbytes(b_out));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_result(d_model * n_batch);
    ggml_backend_tensor_get(y, y_result.data(), 0, ggml_nbytes(y));

    ASSERT_TRUE(y->ne[0] == d_model);
    ASSERT_TRUE(y->ne[1] == n_batch);
    for (int64_t i = 0; i < d_model * n_batch; ++i) {
        ASSERT_TRUE(!std::isnan(y_result[i]));
        ASSERT_TRUE(!std::isinf(y_result[i]));
    }

    printf("Transformer block test PASSED!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

int main() {
    printf("=== TensorRT M3: Activations, Softmax, Shape Ops ===\n\n");
    bool all_passed = true;
    all_passed &= test_silu();
    all_passed &= test_gelu();
    all_passed &= test_relu();
    all_passed &= test_softmax();
    all_passed &= test_softmax_scale();
    all_passed &= test_reshape_add();
    all_passed &= test_silu_matmul();
    all_passed &= test_transformer_block();
    printf("\n%s\n", all_passed ? "All M3 tests PASSED!" : "Some M3 tests FAILED!");
    return all_passed ? 0 : 1;
}
