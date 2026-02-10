// TensorRT backend tests — Milestone 6: SCALE and GET_ROWS ops
// SCALE: elementwise multiply + optional bias (used for attention scaling, embedding scaling)
// GET_ROWS: row gather / embedding lookup (always outputs F32)

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

// Test 1: F32 SCALE by 2.5
static bool test_scale_f32() {
    printf("Testing F32 SCALE (x * 2.5)...\n");

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
    for (int64_t i = 0; i < n; ++i) {
        x_data[i] = (float)i * 0.1f;
    }

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_scale(ctx, x, 2.5f);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_data = read_output_as_fp32(y);

    for (int64_t i = 0; i < n; ++i) {
        float expected = x_data[i] * 2.5f;
        float tol = 1e-4f * fabs(expected) + 1e-5f;
        if (fabs(y_data[i] - expected) > tol) {
            fprintf(stderr, "SCALE F32 mismatch at %lld: got %.6f, expected %.6f\n",
                    (long long)i, y_data[i], expected);
            return false;
        }
    }

    printf("F32 SCALE test passed! (output type: %s)\n", ggml_type_name(y->type));
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 2: F32 SCALE with bias (x * 3.0 + 1.5)
static bool test_scale_bias() {
    printf("Testing F32 SCALE with bias (x * 3.0 + 1.5)...\n");

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
    for (int64_t i = 0; i < n; ++i) {
        x_data[i] = (float)i * 0.1f;
    }

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_scale_bias(ctx, x, 3.0f, 1.5f);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> y_data = read_output_as_fp32(y);

    for (int64_t i = 0; i < n; ++i) {
        float expected = x_data[i] * 3.0f + 1.5f;
        float tol = 1e-4f * fabs(expected) + 1e-5f;
        if (fabs(y_data[i] - expected) > tol) {
            fprintf(stderr, "SCALE BIAS mismatch at %lld: got %.6f, expected %.6f\n",
                    (long long)i, y_data[i], expected);
            return false;
        }
    }

    printf("F32 SCALE+BIAS test passed! (output type: %s)\n", ggml_type_name(y->type));
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 3: BF16 SCALE by 0.5
static bool test_scale_bf16() {
    printf("Testing BF16 SCALE (x * 0.5)...\n");

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
    for (int64_t i = 0; i < n; ++i) {
        x_fp32[i] = (float)(i + 1) * 0.5f;
    }
    std::vector<ggml_bf16_t> x_bf16(n);
    ggml_fp32_to_bf16_row(x_fp32.data(), x_bf16.data(), n);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* y = ggml_scale(ctx, x, 0.5f);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(x, x_bf16.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    // ggml_scale inherits type from src[0], so output is BF16
    std::vector<float> y_fp32 = read_output_as_fp32(y);

    for (int64_t i = 0; i < n; ++i) {
        float expected = x_fp32[i] * 0.5f;
        float tol = 0.02f * fabs(expected) + 0.05f;  // relaxed for BF16
        if (fabs(y_fp32[i] - expected) > tol) {
            fprintf(stderr, "BF16 SCALE mismatch at %lld: got %.4f, expected %.4f\n",
                    (long long)i, y_fp32[i], expected);
            return false;
        }
    }

    printf("BF16 SCALE test passed! (output type: %s)\n", ggml_type_name(y->type));
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 4: GET_ROWS F32 — gather 3 rows from 8-row matrix
static bool test_get_rows_f32() {
    printf("Testing F32 GET_ROWS (gather 3 rows from 8x4 matrix)...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t cols = 4;
    const int64_t rows = 8;
    const int64_t n_indices = 3;

    // Data matrix: 8 rows x 4 cols (F32)
    struct ggml_tensor* data = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    // Indices: 3 I32 values
    struct ggml_tensor* indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_indices);

    // Fill data: row i has values [i*10, i*10+1, i*10+2, i*10+3]
    std::vector<float> data_values(cols * rows);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            data_values[r * cols + c] = (float)(r * 10 + c);
        }
    }

    // Gather rows 2, 5, 7
    std::vector<int32_t> idx_values = {2, 5, 7};

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* result = ggml_get_rows(ctx, data, indices);
    ggml_build_forward_expand(gf, result);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(data, data_values.data(), 0, ggml_nbytes(data));
    ggml_backend_tensor_set(indices, idx_values.data(), 0, ggml_nbytes(indices));
    ggml_backend_graph_compute(backend_trt, gf);

    // Output: 3 rows x 4 cols, always F32
    ASSERT_TRUE(result->type == GGML_TYPE_F32);
    std::vector<float> out(cols * n_indices);
    ggml_backend_tensor_get(result, out.data(), 0, ggml_nbytes(result));

    // Verify: row 0 of output = row 2 of data, etc.
    for (int64_t i = 0; i < n_indices; ++i) {
        int32_t src_row = idx_values[i];
        for (int64_t c = 0; c < cols; ++c) {
            float expected = (float)(src_row * 10 + c);
            float got = out[i * cols + c];
            if (fabs(got - expected) > 1e-4f) {
                fprintf(stderr, "GET_ROWS F32 mismatch at [%lld,%lld]: got %.4f, expected %.4f\n",
                        (long long)i, (long long)c, got, expected);
                return false;
            }
        }
    }

    printf("F32 GET_ROWS test passed! (output type: %s)\n", ggml_type_name(result->type));
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 5: GET_ROWS F16 — gather from F16 matrix, output is F32
static bool test_get_rows_f16() {
    printf("Testing F16 GET_ROWS (gather from F16 matrix, output F32)...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t cols = 4;
    const int64_t rows = 8;
    const int64_t n_indices = 3;

    struct ggml_tensor* data = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cols, rows);
    struct ggml_tensor* indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_indices);

    // Fill data
    std::vector<float> data_fp32(cols * rows);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            data_fp32[r * cols + c] = (float)(r * 10 + c);
        }
    }
    std::vector<ggml_fp16_t> data_fp16(cols * rows);
    ggml_fp32_to_fp16_row(data_fp32.data(), data_fp16.data(), cols * rows);

    // Gather rows 1, 3, 6
    std::vector<int32_t> idx_values = {1, 3, 6};

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* result = ggml_get_rows(ctx, data, indices);
    ggml_build_forward_expand(gf, result);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(data, data_fp16.data(), 0, ggml_nbytes(data));
    ggml_backend_tensor_set(indices, idx_values.data(), 0, ggml_nbytes(indices));
    ggml_backend_graph_compute(backend_trt, gf);

    // Output is always F32
    ASSERT_TRUE(result->type == GGML_TYPE_F32);
    std::vector<float> out(cols * n_indices);
    ggml_backend_tensor_get(result, out.data(), 0, ggml_nbytes(result));

    for (int64_t i = 0; i < n_indices; ++i) {
        int32_t src_row = idx_values[i];
        for (int64_t c = 0; c < cols; ++c) {
            float expected = (float)(src_row * 10 + c);
            float tol = 0.1f;  // FP16 precision
            float got = out[i * cols + c];
            if (fabs(got - expected) > tol) {
                fprintf(stderr, "GET_ROWS F16 mismatch at [%lld,%lld]: got %.4f, expected %.4f\n",
                        (long long)i, (long long)c, got, expected);
                return false;
            }
        }
    }

    printf("F16 GET_ROWS test passed! (output type: %s)\n", ggml_type_name(result->type));
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 6: Integration — mul_mat followed by scale (mimics Gemma attention query scaling)
static bool test_scale_matmul_integration() {
    printf("Testing MUL_MAT -> SCALE integration (attention query scaling)...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t K = 8, N = 4, M = 2;
    const float scale = 1.0f / sqrtf((float)K);  // 1/sqrt(d_k)

    struct ggml_tensor* W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);

    std::vector<float> w_data(K * N), x_data(K * M);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    for (auto& v : w_data) v = dist(rng);
    for (auto& v : x_data) v = dist(rng);

    // Graph: mul_mat -> scale
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* mm = ggml_mul_mat(ctx, W, x);
    struct ggml_tensor* scaled = ggml_scale(ctx, mm, scale);
    ggml_build_forward_expand(gf, scaled);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(W, w_data.data(), 0, ggml_nbytes(W));
    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend_trt, gf);

    // mul_mat outputs F32, scale inherits F32
    ASSERT_TRUE(scaled->type == GGML_TYPE_F32);
    std::vector<float> out = read_output_as_fp32(scaled);

    // CPU reference: mul_mat [N,M] then scale by 1/sqrt(K)
    // ggml_mul_mat(W[K,N], x[K,M]) -> out[N,M]
    // out[n,m] = sum_k(W[k,n] * x[k,m])
    std::vector<float> ref(N * M, 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += w_data[n * K + k] * x_data[m * K + k];
            }
            ref[m * N + n] = sum * scale;
        }
    }

    for (int64_t i = 0; i < N * M; ++i) {
        float tol = 0.01f * fabs(ref[i]) + 1e-4f;
        if (fabs(out[i] - ref[i]) > tol) {
            fprintf(stderr, "SCALE+MATMUL mismatch at %lld: got %.6f, expected %.6f\n",
                    (long long)i, out[i], ref[i]);
            return false;
        }
    }

    printf("MUL_MAT -> SCALE integration test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

int main() {
    printf("=== TensorRT M6: SCALE and GET_ROWS ===\n\n");
    bool all_passed = true;
    all_passed &= test_scale_f32();
    all_passed &= test_scale_bias();
    all_passed &= test_scale_bf16();
    all_passed &= test_get_rows_f32();
    all_passed &= test_get_rows_f16();
    all_passed &= test_scale_matmul_integration();
    printf("\n%s\n", all_passed ? "All M6 tests PASSED!" : "Some M6 tests FAILED!");
    return all_passed ? 0 : 1;
}
