// TensorRT backend tests — Milestone 6a: SET_ROWS op
// SET_ROWS: scatter F32 source rows into destination (KV cache) at indexed positions
// Supports type conversion: F32→F32, F32→F16, F32→BF16

#include "test-tensorrt-common.h"
#include <cstring>

// Helper: read output tensor into FP32 vector, handling any output type
static std::vector<float> read_as_fp32(struct ggml_tensor * t) {
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

// Test 1: F32→F32 SET_ROWS — scatter 3 rows into 8-row F32 matrix
static bool test_set_rows_f32() {
    printf("Testing F32 SET_ROWS (scatter 3 F32 rows into 8x4 F32 matrix)...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context * ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t cols = 4;
    const int64_t dst_rows = 8;
    const int64_t n_src_rows = 3;

    // Destination: 8x4 F32 (simulates KV cache)
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, dst_rows);
    // Source: 3x4 F32 rows to scatter
    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, n_src_rows);
    // Indices: 3 I32 values — scatter to rows 1, 4, 6
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_src_rows);

    // Fill destination with -1.0 so we can verify untouched rows
    std::vector<float> dst_data(cols * dst_rows, -1.0f);

    // Fill source rows: row i has values [100+i*10, 100+i*10+1, ...]
    std::vector<float> src_data(cols * n_src_rows);
    for (int64_t r = 0; r < n_src_rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            src_data[r * cols + c] = 100.0f + (float)(r * 10 + c);
        }
    }

    std::vector<int32_t> idx_data = {1, 4, 6};

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * result = ggml_set_rows(ctx, dst, src, idx);
    ggml_build_forward_expand(gf, result);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    // Initialize tensors
    ggml_backend_tensor_set(dst, dst_data.data(), 0, ggml_nbytes(dst));
    ggml_backend_tensor_set(src, src_data.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_set(idx, idx_data.data(), 0, ggml_nbytes(idx));

    ggml_backend_graph_compute(backend_trt, gf);

    // Read back destination (result shares data pointer with dst)
    std::vector<float> out = read_as_fp32(dst);

    // Verify: scattered rows should contain source data
    for (int64_t i = 0; i < n_src_rows; ++i) {
        int32_t dst_row = idx_data[i];
        for (int64_t c = 0; c < cols; ++c) {
            float expected = src_data[i * cols + c];
            float got = out[dst_row * cols + c];
            if (fabs(got - expected) > 1e-4f) {
                fprintf(stderr, "SET_ROWS F32 mismatch at dst_row=%d col=%lld: got %.4f, expected %.4f\n",
                        dst_row, (long long)c, got, expected);
                return false;
            }
        }
    }

    // Verify: untouched rows should still be -1.0
    std::vector<bool> touched(dst_rows, false);
    for (int32_t r : idx_data) { touched[r] = true; }
    for (int64_t r = 0; r < dst_rows; ++r) {
        if (touched[r]) continue;
        for (int64_t c = 0; c < cols; ++c) {
            float got = out[r * cols + c];
            if (fabs(got - (-1.0f)) > 1e-4f) {
                fprintf(stderr, "SET_ROWS F32: untouched row %lld col %lld changed: got %.4f, expected -1.0\n",
                        (long long)r, (long long)c, got);
                return false;
            }
        }
    }

    printf("F32 SET_ROWS test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 2: F32→F16 SET_ROWS — scatter F32 rows into F16 destination
static bool test_set_rows_f16() {
    printf("Testing F32→F16 SET_ROWS (scatter 3 F32 rows into 8x4 F16 matrix)...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context * ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t cols = 4;
    const int64_t dst_rows = 8;
    const int64_t n_src_rows = 3;

    // Destination: 8x4 F16
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cols, dst_rows);
    // Source: 3x4 F32 rows
    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, n_src_rows);
    // Indices: I32
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_src_rows);

    // Fill destination F16 with zeros
    std::vector<ggml_fp16_t> dst_data(cols * dst_rows);
    {
        std::vector<float> zeros(cols * dst_rows, 0.0f);
        ggml_fp32_to_fp16_row(zeros.data(), dst_data.data(), cols * dst_rows);
    }

    // Fill source rows
    std::vector<float> src_data(cols * n_src_rows);
    for (int64_t r = 0; r < n_src_rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            src_data[r * cols + c] = 10.0f + (float)(r * 10 + c);
        }
    }

    std::vector<int32_t> idx_data = {0, 3, 7};

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * result = ggml_set_rows(ctx, dst, src, idx);
    ggml_build_forward_expand(gf, result);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(dst, dst_data.data(), 0, ggml_nbytes(dst));
    ggml_backend_tensor_set(src, src_data.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_set(idx, idx_data.data(), 0, ggml_nbytes(idx));

    ggml_backend_graph_compute(backend_trt, gf);

    // Read back as FP32 (converts from F16)
    std::vector<float> out = read_as_fp32(dst);

    // Verify scattered rows
    for (int64_t i = 0; i < n_src_rows; ++i) {
        int32_t dst_row = idx_data[i];
        for (int64_t c = 0; c < cols; ++c) {
            float expected = src_data[i * cols + c];
            float got = out[dst_row * cols + c];
            float tol = 0.1f;  // FP16 precision
            if (fabs(got - expected) > tol) {
                fprintf(stderr, "SET_ROWS F16 mismatch at dst_row=%d col=%lld: got %.4f, expected %.4f\n",
                        dst_row, (long long)c, got, expected);
                return false;
            }
        }
    }

    // Verify untouched rows are still ~0
    std::vector<bool> touched(dst_rows, false);
    for (int32_t r : idx_data) { touched[r] = true; }
    for (int64_t r = 0; r < dst_rows; ++r) {
        if (touched[r]) continue;
        for (int64_t c = 0; c < cols; ++c) {
            float got = out[r * cols + c];
            if (fabs(got) > 0.01f) {
                fprintf(stderr, "SET_ROWS F16: untouched row %lld col %lld changed: got %.4f\n",
                        (long long)r, (long long)c, got);
                return false;
            }
        }
    }

    printf("F32→F16 SET_ROWS test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 3: F32→BF16 SET_ROWS — scatter F32 rows into BF16 destination
static bool test_set_rows_bf16() {
    printf("Testing F32→BF16 SET_ROWS (scatter 3 F32 rows into 8x4 BF16 matrix)...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context * ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t cols = 4;
    const int64_t dst_rows = 8;
    const int64_t n_src_rows = 3;

    // Destination: 8x4 BF16
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, cols, dst_rows);
    // Source: 3x4 F32 rows
    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, n_src_rows);
    // Indices: I32
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_src_rows);

    // Fill destination BF16 with zeros
    std::vector<ggml_bf16_t> dst_data(cols * dst_rows);
    {
        std::vector<float> zeros(cols * dst_rows, 0.0f);
        ggml_fp32_to_bf16_row(zeros.data(), dst_data.data(), cols * dst_rows);
    }

    // Fill source rows
    std::vector<float> src_data(cols * n_src_rows);
    for (int64_t r = 0; r < n_src_rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            src_data[r * cols + c] = 5.0f + (float)(r * 10 + c);
        }
    }

    std::vector<int32_t> idx_data = {2, 5, 6};

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * result = ggml_set_rows(ctx, dst, src, idx);
    ggml_build_forward_expand(gf, result);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(dst, dst_data.data(), 0, ggml_nbytes(dst));
    ggml_backend_tensor_set(src, src_data.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_set(idx, idx_data.data(), 0, ggml_nbytes(idx));

    ggml_backend_graph_compute(backend_trt, gf);

    // Read back as FP32 (converts from BF16)
    std::vector<float> out = read_as_fp32(dst);

    // Verify scattered rows
    for (int64_t i = 0; i < n_src_rows; ++i) {
        int32_t dst_row = idx_data[i];
        for (int64_t c = 0; c < cols; ++c) {
            float expected = src_data[i * cols + c];
            float got = out[dst_row * cols + c];
            float tol = 0.05f;  // BF16 precision
            if (fabs(got - expected) > tol) {
                fprintf(stderr, "SET_ROWS BF16 mismatch at dst_row=%d col=%lld: got %.4f, expected %.4f\n",
                        dst_row, (long long)c, got, expected);
                return false;
            }
        }
    }

    // Verify untouched rows are still ~0
    std::vector<bool> touched(dst_rows, false);
    for (int32_t r : idx_data) { touched[r] = true; }
    for (int64_t r = 0; r < dst_rows; ++r) {
        if (touched[r]) continue;
        for (int64_t c = 0; c < cols; ++c) {
            float got = out[r * cols + c];
            if (fabs(got) > 0.01f) {
                fprintf(stderr, "SET_ROWS BF16: untouched row %lld col %lld changed: got %.4f\n",
                        (long long)r, (long long)c, got);
                return false;
            }
        }
    }

    printf("F32→BF16 SET_ROWS test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

// Test 4: F32→F32 SET_ROWS with I64 indices
static bool test_set_rows_i64_indices() {
    printf("Testing F32 SET_ROWS with I64 indices...\n");

    struct ggml_init_params params;
    params.mem_size   = 128 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context * ctx = ggml_init(params);
    ASSERT_TRUE(ctx != NULL);

    ggml_backend_t backend_trt = ggml_backend_tensorrt_init(0);
    ASSERT_TRUE(backend_trt != NULL);

    const int64_t cols = 4;
    const int64_t dst_rows = 8;
    const int64_t n_src_rows = 2;

    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, dst_rows);
    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, n_src_rows);
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_src_rows);

    std::vector<float> dst_data(cols * dst_rows, -1.0f);
    std::vector<float> src_data(cols * n_src_rows);
    for (int64_t r = 0; r < n_src_rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            src_data[r * cols + c] = 50.0f + (float)(r * 10 + c);
        }
    }

    std::vector<int64_t> idx_data = {3, 7};

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * result = ggml_set_rows(ctx, dst, src, idx);
    ggml_build_forward_expand(gf, result);

    ggml_backend_buffer_t buffer_trt = ggml_backend_alloc_ctx_tensors(ctx, backend_trt);
    ASSERT_TRUE(buffer_trt != NULL);

    ggml_backend_tensor_set(dst, dst_data.data(), 0, ggml_nbytes(dst));
    ggml_backend_tensor_set(src, src_data.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_set(idx, idx_data.data(), 0, ggml_nbytes(idx));

    ggml_backend_graph_compute(backend_trt, gf);

    std::vector<float> out = read_as_fp32(dst);

    // Verify scattered rows
    for (int64_t i = 0; i < n_src_rows; ++i) {
        int64_t dst_row = idx_data[i];
        for (int64_t c = 0; c < cols; ++c) {
            float expected = src_data[i * cols + c];
            float got = out[dst_row * cols + c];
            if (fabs(got - expected) > 1e-4f) {
                fprintf(stderr, "SET_ROWS I64 mismatch at dst_row=%lld col=%lld: got %.4f, expected %.4f\n",
                        (long long)dst_row, (long long)c, got, expected);
                return false;
            }
        }
    }

    // Verify untouched rows
    std::vector<bool> touched(dst_rows, false);
    for (int64_t r : idx_data) { touched[r] = true; }
    for (int64_t r = 0; r < dst_rows; ++r) {
        if (touched[r]) continue;
        for (int64_t c = 0; c < cols; ++c) {
            float got = out[r * cols + c];
            if (fabs(got - (-1.0f)) > 1e-4f) {
                fprintf(stderr, "SET_ROWS I64: untouched row %lld col %lld changed: got %.4f\n",
                        (long long)r, (long long)c, got);
                return false;
            }
        }
    }

    printf("F32 SET_ROWS with I64 indices test passed!\n");
    ggml_backend_buffer_free(buffer_trt);
    ggml_backend_free(backend_trt);
    ggml_free(ctx);
    return true;
}

int main() {
    printf("=== TensorRT M6a: SET_ROWS ===\n\n");
    bool all_passed = true;
    all_passed &= test_set_rows_f32();
    all_passed &= test_set_rows_f16();
    all_passed &= test_set_rows_bf16();
    all_passed &= test_set_rows_i64_indices();
    printf("\n%s\n", all_passed ? "All M6a tests PASSED!" : "Some M6a tests FAILED!");
    return all_passed ? 0 : 1;
}
