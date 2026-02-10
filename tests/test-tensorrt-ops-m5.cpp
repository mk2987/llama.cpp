// TensorRT backend tests — Milestone 5: BF16/FP16 Precision
// BF16 add, mul_mat, rms_norm, softmax, transformer block; FP16 mul_mat

#include "test-tensorrt-common.h"

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

    std::vector<float> a_fp32(n), b_fp32(n);
    for (int64_t i = 0; i < n; ++i) {
        a_fp32[i] = (float)i * 0.1f;
        b_fp32[i] = (float)(i * 2) * 0.1f;
    }

    std::vector<ggml_bf16_t> a_bf16(n), b_bf16(n);
    ggml_fp32_to_bf16_row(a_fp32.data(), a_bf16.data(), n);
    ggml_fp32_to_bf16_row(b_fp32.data(), b_bf16.data(), n);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* c = ggml_add(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(a, a_bf16.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_bf16.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<ggml_bf16_t> c_bf16(n);
    ggml_backend_tensor_get(c, c_bf16.data(), 0, ggml_nbytes(c));

    std::vector<float> c_fp32(n);
    ggml_bf16_to_fp32_row(c_bf16.data(), c_fp32.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        float expected = a_fp32[i] + b_fp32[i];
        if (fabs(c_fp32[i] - expected) > 1e-1f) {
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

    std::vector<float> a_fp32(K * N), b_fp32(K * M);
    for (int64_t i = 0; i < K * N; ++i) a_fp32[i] = (float)(i % 10);
    for (int64_t i = 0; i < K * M; ++i) b_fp32[i] = (float)((i % 5) + 1);

    std::vector<ggml_bf16_t> a_bf16(K * N), b_bf16(K * M);
    ggml_fp32_to_bf16_row(a_fp32.data(), a_bf16.data(), K * N);
    ggml_fp32_to_bf16_row(b_fp32.data(), b_bf16.data(), K * M);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(a, a_bf16.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_bf16.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<ggml_bf16_t> c_bf16(M * N);
    ggml_backend_tensor_get(c, c_bf16.data(), 0, ggml_nbytes(c));

    std::vector<float> c_fp32(M * N);
    ggml_bf16_to_fp32_row(c_bf16.data(), c_fp32.data(), M * N);

    float expected[6] = {20.0f, 60.0f, 30.0f, 14.0f, 58.0f, 52.0f};
    for (int i = 0; i < 6; i++) {
        if (fabs(c_fp32[i] - expected[i]) > 1.0f) {
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

    std::vector<float> x_fp32(n);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) x_fp32[i] = dist(rng);

    std::vector<ggml_bf16_t> x_bf16(n);
    ggml_fp32_to_bf16_row(x_fp32.data(), x_bf16.data(), n);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_rms_norm(ctx, x, 1e-6f);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_bf16.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<ggml_bf16_t> y_bf16(n);
    ggml_backend_tensor_get(y, y_bf16.data(), 0, ggml_nbytes(y));

    std::vector<float> y_fp32(n);
    ggml_bf16_to_fp32_row(y_bf16.data(), y_fp32.data(), n);

    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum_sq += y_fp32[i] * y_fp32[i];
    float rms = sqrtf(sum_sq / n);
    ASSERT_TRUE(fabs(rms - 1.0f) < 0.2f);

    printf("BF16 RMS_NORM test passed! (rms=%.4f)\n", rms);
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

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

    const int64_t cols = 10, rows = 4;
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, cols, rows);

    std::vector<float> x_fp32(cols * rows);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < cols * rows; ++i) x_fp32[i] = dist(rng);

    std::vector<ggml_bf16_t> x_bf16(cols * rows);
    ggml_fp32_to_bf16_row(x_fp32.data(), x_bf16.data(), cols * rows);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_soft_max(ctx, x);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_bf16.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<ggml_bf16_t> y_bf16(cols * rows);
    ggml_backend_tensor_get(y, y_bf16.data(), 0, ggml_nbytes(y));

    std::vector<float> y_fp32(cols * rows);
    ggml_bf16_to_fp32_row(y_bf16.data(), y_fp32.data(), cols * rows);

    for (int64_t r = 0; r < rows; ++r) {
        float row_sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float val = y_fp32[r * cols + c];
            ASSERT_TRUE(val >= -0.01f && val <= 1.01f);
            row_sum += val;
        }
        if (fabs(row_sum - 1.0f) > 5e-2f) {
            fprintf(stderr, "BF16 SOFTMAX row %lld sum = %.6f\n", (long long)r, row_sum);
            return false;
        }
    }

    printf("BF16 SOFTMAX test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

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

    const int64_t d_model = 16, d_ff = 32, n_batch = 2;

    struct ggml_tensor* x      = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_model, n_batch);
    struct ggml_tensor* W_attn = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_model, d_model);
    struct ggml_tensor* b_attn = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_model, n_batch);
    struct ggml_tensor* W_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_model, d_ff);
    struct ggml_tensor* W_down = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_ff, d_model);
    struct ggml_tensor* b_out  = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d_model, n_batch);

    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    auto fill_bf16 = [&](int64_t nelements) {
        std::vector<float> fp32(nelements);
        for (auto& v : fp32) v = dist(rng);
        std::vector<ggml_bf16_t> bf16(nelements);
        ggml_fp32_to_bf16_row(fp32.data(), bf16.data(), nelements);
        return bf16;
    };

    auto x_bf16      = fill_bf16(d_model * n_batch);
    auto W_attn_bf16 = fill_bf16(d_model * d_model);
    auto b_attn_bf16 = fill_bf16(d_model * n_batch);
    auto W_up_bf16   = fill_bf16(d_model * d_ff);
    auto W_down_bf16 = fill_bf16(d_ff * d_model);
    auto b_out_bf16  = fill_bf16(d_model * n_batch);

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

    ggml_backend_tensor_set(x,      x_bf16.data(),      0, ggml_nbytes(x));
    ggml_backend_tensor_set(W_attn, W_attn_bf16.data(), 0, ggml_nbytes(W_attn));
    ggml_backend_tensor_set(b_attn, b_attn_bf16.data(), 0, ggml_nbytes(b_attn));
    ggml_backend_tensor_set(W_up,   W_up_bf16.data(),   0, ggml_nbytes(W_up));
    ggml_backend_tensor_set(W_down, W_down_bf16.data(), 0, ggml_nbytes(W_down));
    ggml_backend_tensor_set(b_out,  b_out_bf16.data(),  0, ggml_nbytes(b_out));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<ggml_bf16_t> y_bf16(d_model * n_batch);
    ggml_backend_tensor_get(y, y_bf16.data(), 0, ggml_nbytes(y));

    std::vector<float> y_fp32(d_model * n_batch);
    ggml_bf16_to_fp32_row(y_bf16.data(), y_fp32.data(), d_model * n_batch);

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

    std::vector<float> a_fp32(K * N), b_fp32(K * M);
    for (int64_t i = 0; i < K * N; ++i) a_fp32[i] = (float)(i % 10);
    for (int64_t i = 0; i < K * M; ++i) b_fp32[i] = (float)((i % 5) + 1);

    std::vector<ggml_fp16_t> a_fp16(K * N), b_fp16(K * M);
    ggml_fp32_to_fp16_row(a_fp32.data(), a_fp16.data(), K * N);
    ggml_fp32_to_fp16_row(b_fp32.data(), b_fp16.data(), K * M);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(a, a_fp16.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_fp16.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<ggml_fp16_t> c_fp16(M * N);
    ggml_backend_tensor_get(c, c_fp16.data(), 0, ggml_nbytes(c));

    std::vector<float> c_fp32(M * N);
    ggml_fp16_to_fp32_row(c_fp16.data(), c_fp32.data(), M * N);

    float expected[6] = {20.0f, 60.0f, 30.0f, 14.0f, 58.0f, 52.0f};
    for (int i = 0; i < 6; i++) {
        if (fabs(c_fp32[i] - expected[i]) > 1.0f) {
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

int main() {
    printf("=== TensorRT M5: BF16/FP16 Precision ===\n\n");
    bool all_passed = true;
    all_passed &= test_bf16_add();
    all_passed &= test_bf16_mul_mat();
    all_passed &= test_bf16_rms_norm();
    all_passed &= test_bf16_softmax();
    all_passed &= test_bf16_transformer_block();
    all_passed &= test_fp16_mul_mat();
    printf("\n%s\n", all_passed ? "All M5 tests PASSED!" : "Some M5 tests FAILED!");
    return all_passed ? 0 : 1;
}
