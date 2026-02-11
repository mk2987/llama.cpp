// TensorRT backend tests — Milestone 5b: CUDA Graph Capture
// Tests verify CUDA graph mode doesn't break correctness.
// CUDA graphs are enabled by default, so these tests exercise the graph path
// automatically without needing to set env vars.
// These tests also serve as regression tests for execution context reuse
// with address rebinding across repeated graph_compute calls.

#include "test-tensorrt-common.h"

#include <cstring>

// Helper: read output tensor into FP32 vector, handling any output type
static std::vector<float> read_output_as_fp32(struct ggml_tensor* t) {
    int64_t n = ggml_nelements(t);
    std::vector<float> result(n);

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        ggml_bf16_to_fp32_row(buf.data(), result.data(), n);
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        ggml_fp16_to_fp32_row(buf.data(), result.data(), n);
    }

    return result;
}

// Test 1: MUL_MAT with CUDA graphs (enabled by default)
static bool test_cuda_graph_mul_mat() {
    printf("Testing MUL_MAT with CUDA graph capture...\n");

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

    std::vector<float> a_data(K * N), b_data(K * M);
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

    float expected[6] = {20.0f, 60.0f, 30.0f, 14.0f, 58.0f, 52.0f};
    for (int i = 0; i < 6; i++) {
        if (fabs(c_result[i] - expected[i]) > 1e-4f) {
            fprintf(stderr, "CUDA graph MUL_MAT mismatch at %d: got %.2f, expected %.2f\n",
                    i, c_result[i], expected[i]);
            return false;
        }
    }

    printf("CUDA graph MUL_MAT test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 2: Full transformer block with CUDA graphs
static bool test_cuda_graph_transformer_block() {
    printf("Testing transformer block with CUDA graph capture...\n");

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
    auto fill = [&](int64_t n) { std::vector<float> v(n); for (auto& val : v) val = dist(rng); return v; };

    auto x_data      = fill(d_model * n_batch);
    auto W_attn_data = fill(d_model * d_model);
    auto b_attn_data = fill(d_model * n_batch);
    auto W_up_data   = fill(d_model * d_ff);
    auto W_down_data = fill(d_ff * d_model);
    auto b_out_data  = fill(d_model * n_batch);

    // Graph: matmul -> add -> rms_norm -> matmul -> silu -> matmul -> add
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

    for (int64_t i = 0; i < d_model * n_batch; ++i) {
        ASSERT_TRUE(!std::isnan(y_result[i]));
        ASSERT_TRUE(!std::isinf(y_result[i]));
    }

    printf("CUDA graph transformer block test PASSED!\n");
    printf("  Sample output: %.4f, %.4f, %.4f, %.4f\n",
           y_result[0], y_result[1], y_result[2], y_result[3]);

    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 3: Repeated execution (first call captures graph, subsequent calls replay)
static bool test_cuda_graph_repeated_execution() {
    printf("Testing repeated execution with CUDA graph capture...\n");

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

    std::vector<float> a_data(K * N), b_data(K * M);
    for (int64_t i = 0; i < K * N; ++i) a_data[i] = (float)(i % 10);
    for (int64_t i = 0; i < K * M; ++i) b_data[i] = (float)((i % 5) + 1);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));

    float expected[6] = {20.0f, 60.0f, 30.0f, 14.0f, 58.0f, 52.0f};

    // Run the same graph 5 times — first call captures, subsequent calls replay
    for (int run = 0; run < 5; ++run) {
        ggml_backend_graph_compute(backend_trt, gf);

        std::vector<float> c_result(M * N);
        ggml_backend_tensor_get(c, c_result.data(), 0, ggml_nbytes(c));

        for (int i = 0; i < 6; i++) {
            if (fabs(c_result[i] - expected[i]) > 1e-4f) {
                fprintf(stderr, "Run %d: CUDA graph repeated MUL_MAT mismatch at %d: got %.2f, expected %.2f\n",
                        run, i, c_result[i], expected[i]);
                return false;
            }
        }
    }

    printf("CUDA graph repeated execution test passed! (5 runs, all identical)\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 4: BF16 MUL_MAT with CUDA graphs
static bool test_cuda_graph_bf16() {
    printf("Testing BF16 MUL_MAT with CUDA graph capture...\n");

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

    // ggml_mul_mat always outputs F32
    std::vector<float> c_fp32 = read_output_as_fp32(c);

    float expected[6] = {20.0f, 60.0f, 30.0f, 14.0f, 58.0f, 52.0f};
    for (int i = 0; i < 6; i++) {
        if (fabs(c_fp32[i] - expected[i]) > 1.0f) {
            fprintf(stderr, "CUDA graph BF16 MUL_MAT mismatch at %d: got %.2f, expected %.2f\n",
                    i, c_fp32[i], expected[i]);
            return false;
        }
    }

    printf("CUDA graph BF16 MUL_MAT test passed! (output type: %s)\n", ggml_type_name(c->type));
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 5: CUDA graphs disabled — verify fallback path still works
static bool test_cuda_graph_disabled() {
    printf("Testing with CUDA graph capture disabled...\n");

    // Disable CUDA graphs via env var
    setenv("GGML_TENSORRT_CUDA_GRAPHS", "0", 1);

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

    std::vector<float> a_data(K * N), b_data(K * M);
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

    float expected[6] = {20.0f, 60.0f, 30.0f, 14.0f, 58.0f, 52.0f};
    for (int i = 0; i < 6; i++) {
        if (fabs(c_result[i] - expected[i]) > 1e-4f) {
            fprintf(stderr, "Disabled CUDA graph MUL_MAT mismatch at %d: got %.2f, expected %.2f\n",
                    i, c_result[i], expected[i]);
            // Restore env before returning
            unsetenv("GGML_TENSORRT_CUDA_GRAPHS");
            return false;
        }
    }

    // Restore env var for subsequent tests
    unsetenv("GGML_TENSORRT_CUDA_GRAPHS");

    printf("CUDA graph disabled test passed! (fallback path works correctly)\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

int main() {
    printf("=== TensorRT M5b: CUDA Graph Capture ===\n\n");
    bool all_passed = true;
    all_passed &= test_cuda_graph_mul_mat();
    all_passed &= test_cuda_graph_transformer_block();
    all_passed &= test_cuda_graph_repeated_execution();
    all_passed &= test_cuda_graph_bf16();
    all_passed &= test_cuda_graph_disabled();
    printf("\n%s\n", all_passed ? "All M5b tests PASSED!" : "Some M5b tests FAILED!");
    return all_passed ? 0 : 1;
}
