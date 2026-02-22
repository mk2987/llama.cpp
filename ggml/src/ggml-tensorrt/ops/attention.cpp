#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "../utils/type-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>
#include <cmath>
#include <cinttypes>
#include <cstring>
#include <string>

namespace ggml_tensorrt {

// ---------------------------------------------------------------------------
// handle_set_rows — Map GGML SET_ROWS to TRT IKVCacheUpdateLayer
// ---------------------------------------------------------------------------
//
// GGML SET_ROWS scatters rows into a KV cache buffer:
//   cache[indices[i]] = values[i]  for each token i
//
// TRT IKVCacheUpdateLayer writes contiguous rows:
//   output[b, :, writeIndices[b]+s, :] = update[b, :, s, :]
//
// Equivalent when indices are contiguous from a start position (always true
// for standard prompt fill and autoregressive decode).
//
// Memory layout trick: we use numHeads=1, headSize=n_embd_gqa.
//   GGML 2D cache [n_embd_gqa, kv_size] → row-major [kv_size, n_embd_gqa]
//   TRT 4D cache  [1, 1, kv_size, n_embd_gqa]
// Same flat layout — no transpose needed.
//
// The cache tensor is added as a DIRECT 4D network input by build_trt_engine
// (IKVCacheUpdateLayer requires cache to be a network input, not reshaped).
//
// GGML tensor layout:
//   node          = view of cache (shares data pointer)
//   node->src[0]  = values  [n_embd_gqa, n_tokens] (GGML column-major)
//   node->src[1]  = indices  [n_tokens] (I64)
//   node->src[2]  = cache    [n_embd_gqa, kv_size] (GGML column-major)
//
nvinfer1::ITensor* handle_set_rows(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr && node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_SET_ROWS);

    const ggml_tensor* values_ggml  = node->src[0];
    const ggml_tensor* indices_ggml = node->src[1];
    const ggml_tensor* cache_ggml   = node->src[2];

    GGML_ASSERT(values_ggml && indices_ggml && cache_ggml);

    const int64_t n_embd_gqa = cache_ggml->ne[0];
    const int64_t kv_size    = cache_ggml->ne[1];

    // -- Get TRT tensors --
    // cache_4d was added as [1, 1, kv_size, n_embd_gqa] by build_trt_engine.
    nvinfer1::ITensor* values     = builder->get_tensor(values_ggml);
    nvinfer1::ITensor* indices    = builder->get_tensor(indices_ggml);
    nvinfer1::ITensor* cache_4d   = builder->get_tensor(cache_ggml);

    if (!values || !indices || !cache_4d) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    auto* network = builder->get_network();

    // -- Reshape values: TRT [n_tokens, n_embd_gqa] → [1, 1, n_tokens, n_embd_gqa] --
    nvinfer1::Dims values_4d;
    values_4d.nbDims = 4;
    values_4d.d[0] = 1;
    values_4d.d[1] = 1;
    values_4d.d[2] = -1;                    // n_tokens (dynamic)
    values_4d.d[3] = (int32_t)n_embd_gqa;

    auto* val_reshape = network->addShuffle(*values);
    val_reshape->setReshapeDimensions(values_4d);
    val_reshape->setName(("kv_val_reshape_" + std::to_string(reinterpret_cast<uintptr_t>(node))).c_str());
    nvinfer1::ITensor* values_4d_t = val_reshape->getOutput(0);

    // Cast values to match cache type (TRT computes in F32, cache is F16/BF16)
    if (values_4d_t->getType() != cache_4d->getType()) {
        auto* cast = network->addCast(*values_4d_t, cache_4d->getType());
        cast->setName(("kv_val_cast_" + std::to_string(reinterpret_cast<uintptr_t>(node))).c_str());
        values_4d_t = cast->getOutput(0);
    }

    // -- Extract writeIndices[0] → [1] (batchSize=1) --
    nvinfer1::Dims s0, s1, ss;
    s0.nbDims = 1; s0.d[0] = 0;
    s1.nbDims = 1; s1.d[0] = 1;
    ss.nbDims = 1; ss.d[0] = 1;

    auto* idx_slice = network->addSlice(*indices, s0, s1, ss);
    idx_slice->setName(("kv_write_idx_" + std::to_string(reinterpret_cast<uintptr_t>(node))).c_str());
    nvinfer1::ITensor* write_idx = idx_slice->getOutput(0);

    // Cast I64 → I32 if needed
    if (write_idx->getType() == nvinfer1::DataType::kINT64) {
        auto* cast = network->addCast(*write_idx, nvinfer1::DataType::kINT32);
        cast->setName(("kv_idx_cast_" + std::to_string(reinterpret_cast<uintptr_t>(node))).c_str());
        write_idx = cast->getOutput(0);
    }

    // -- IKVCacheUpdateLayer --
    auto* kv_layer = network->addKVCacheUpdate(
        *cache_4d, *values_4d_t, *write_idx,
        nvinfer1::KVCacheMode::kLINEAR);
    if (!kv_layer) {
        GGML_LOG_ERROR("%s: addKVCacheUpdate failed\n", __func__);
        return nullptr;
    }
    kv_layer->setName(("kv_update_" + std::to_string(reinterpret_cast<uintptr_t>(node))).c_str());

    nvinfer1::ITensor* kv_out = kv_layer->getOutput(0);  // [1, 1, kv_size, n_embd_gqa]

    // TRT requires IKVCacheUpdateLayer output to be a network output —
    // in-place aliased layers must write to externally-bound memory.
    // Use kv_inplace_N naming (N = count of existing kv_inplace outputs).
    {
        int kv_idx = 0;
        for (int i = 0; i < network->getNbOutputs(); i++) {
            const char * oname = network->getOutput(i)->getName();
            if (oname && strncmp(oname, "kv_inplace_", 11) == 0)
                kv_idx++;
        }
        char kv_name[64];
        snprintf(kv_name, sizeof(kv_name), "kv_inplace_%d", kv_idx);
        network->markOutput(*kv_out);
        kv_out->setName(kv_name);
    }

    // -- Reshape output back to 2D [kv_size, n_embd_gqa] --
    nvinfer1::Dims out_2d;
    out_2d.nbDims = 2;
    out_2d.d[0] = (int32_t)kv_size;
    out_2d.d[1] = (int32_t)n_embd_gqa;

    auto* out_reshape = network->addShuffle(*kv_out);
    out_reshape->setReshapeDimensions(out_2d);
    out_reshape->setName(("kv_out_reshape_" + std::to_string(reinterpret_cast<uintptr_t>(node))).c_str());

    nvinfer1::ITensor* result = out_reshape->getOutput(0);

    // Update tensor_map: downstream VIEWs reading from the cache tensor
    // should see the updated 2D output (not the original 4D input).
    builder->set_tensor(cache_ggml, result);

    GGML_LOG_DEBUG("%s: KVCacheUpdate [1,1,%" PRId64 ",%" PRId64 "] → %s\n",
                   __func__, kv_size, n_embd_gqa,
                   dims_to_string(result->getDimensions()).c_str());

    return result;
}

// ---------------------------------------------------------------------------
// handle_flash_attn_ext — Map GGML FLASH_ATTN_EXT to TRT IAttention
// ---------------------------------------------------------------------------
//
// GGML FLASH_ATTN_EXT computes scaled dot-product attention:
//   output = softmax(Q @ K^T * scale + mask) @ V
//
// GGML tensor shapes (ne[0..3]):
//   Q:      (head_dim, n_heads_q, n_tokens, batch)
//   K:      (head_dim, n_heads_kv, n_kv,    batch)
//   V:      (head_dim, n_heads_kv, n_kv,    batch)
//   mask:   (n_kv, n_tokens, mask_heads, mask_batch)  [F16/F32 additive]
//   result: (head_dim_v, n_tokens, n_heads_q, batch)  [always F32]
//
// After ggml_tensor_to_dims reversal:
//   Q:      (batch, n_tokens, n_heads_q, head_dim)   = [B, S_q, H_q, D]
//   K:      (batch, n_kv, n_heads_kv, head_dim)      = [B, S_kv, H_kv, D]
//   V:      (batch, n_kv, n_heads_kv, head_dim)      = [B, S_kv, H_kv, D]
//   mask:   (mask_batch, mask_heads, n_tokens, n_kv)  = [B, H, S_q, S_kv]
//   result: (batch, n_heads_q, n_tokens, head_dim_v)  = [B, H_q, S_q, D_v]
//
// TRT addAttention expects Q/K/V as [B, H, S, D].
// After reversal we have [B, S, H, D] → permute axes 1↔2.
//
// TRT output: [B, H_q, S_q, D_v] — matches GGML expected reversed shape.
//
// Op params: [0]=scale(float), [1]=max_bias(float), [2]=logit_softcap(float)
//
nvinfer1::ITensor* handle_flash_attn_ext(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr && node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor* q_ggml    = node->src[0];
    const ggml_tensor* k_ggml    = node->src[1];
    const ggml_tensor* v_ggml    = node->src[2];
    const ggml_tensor* mask_ggml = node->src[3];  // may be nullptr

    GGML_ASSERT(q_ggml && k_ggml && v_ggml);

    nvinfer1::ITensor* q_t    = builder->get_tensor(q_ggml);
    nvinfer1::ITensor* k_t    = builder->get_tensor(k_ggml);
    nvinfer1::ITensor* v_t    = builder->get_tensor(v_ggml);
    nvinfer1::ITensor* mask_t = mask_ggml ? builder->get_tensor(mask_ggml) : nullptr;

    if (!q_t || !k_t || !v_t) {
        GGML_LOG_ERROR("%s: Q/K/V tensor not found in network\n", __func__);
        return nullptr;
    }
    if (mask_ggml && !mask_t) {
        GGML_LOG_ERROR("%s: mask tensor not found in network\n", __func__);
        return nullptr;
    }

    auto* network = builder->get_network();

    // Extract scale from op_params
    float scale = 0.0f;
    memcpy(&scale, (const float *)node->op_params, sizeof(float));

    const int64_t head_dim = q_ggml->ne[0];

    std::string pfx = "flash_attn_" + std::to_string(reinterpret_cast<uintptr_t>(node));

    // ── Cast Q/K/V to common type ──
    // TRT IAttention requires Q/K/V to have the same type (kFLOAT, kHALF, or kBF16).
    // Use F32 if any input is F32, otherwise keep the narrowest common type.
    nvinfer1::DataType compute_type = q_t->getType();
    {
        nvinfer1::DataType types[3] = { q_t->getType(), k_t->getType(), v_t->getType() };
        bool has_f32 = false;
        bool has_bf16 = false;
        for (int i = 0; i < 3; i++) {
            if (types[i] == nvinfer1::DataType::kFLOAT) has_f32 = true;
            if (types[i] == nvinfer1::DataType::kBF16)  has_bf16 = true;
        }
        if (has_f32) {
            compute_type = nvinfer1::DataType::kFLOAT;
        } else if (has_bf16) {
            compute_type = nvinfer1::DataType::kBF16;
        } else {
            compute_type = nvinfer1::DataType::kHALF;
        }
    }

    q_t = builder->maybe_cast(q_t, compute_type);
    k_t = builder->maybe_cast(k_t, compute_type);
    v_t = builder->maybe_cast(v_t, compute_type);

    // ── Pad to 4D ──
    // After ggml_tensor_to_dims: Q/K/V are [S, H, D] (3D, batch=1 stripped).
    // Pad to [1, S, H, D] = [B, S, H, D].
    q_t = builder->pad_to_ndims(q_t, 4);
    k_t = builder->pad_to_ndims(k_t, 4);
    v_t = builder->pad_to_ndims(v_t, 4);

    // ── Permute Q/K/V from [B, S, H, D] to [B, H, S, D] ──
    auto* q_perm = network->addShuffle(*q_t);
    if (!q_perm) { GGML_LOG_ERROR("%s: Q permute failed\n", __func__); return nullptr; }
    q_perm->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    q_perm->setName((pfx + "_q_perm").c_str());
    nvinfer1::ITensor* q_bhsd = q_perm->getOutput(0);

    auto* k_perm = network->addShuffle(*k_t);
    if (!k_perm) { GGML_LOG_ERROR("%s: K permute failed\n", __func__); return nullptr; }
    k_perm->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    k_perm->setName((pfx + "_k_perm").c_str());
    nvinfer1::ITensor* k_bhsd = k_perm->getOutput(0);

    auto* v_perm = network->addShuffle(*v_t);
    if (!v_perm) { GGML_LOG_ERROR("%s: V permute failed\n", __func__); return nullptr; }
    v_perm->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    v_perm->setName((pfx + "_v_perm").c_str());
    nvinfer1::ITensor* v_bhsd = v_perm->getOutput(0);

    // ── Scale adjustment ──
    // TRT auto-scales by 1/sqrt(dimHead).  If GGML scale differs, pre-scale Q.
    float trt_default_scale = 1.0f / sqrtf((float)head_dim);
    float scale_ratio = scale / trt_default_scale;  // usually 1.0
    if (fabsf(scale_ratio - 1.0f) > 1e-6f) {
        nvinfer1::ITensor* scale_const = builder->create_typed_scalar(scale_ratio, q_bhsd);
        if (!scale_const) {
            GGML_LOG_ERROR("%s: failed to create scale constant\n", __func__);
            return nullptr;
        }
        auto* scale_layer = network->addElementWise(
            *q_bhsd, *scale_const, nvinfer1::ElementWiseOperation::kPROD);
        if (!scale_layer) {
            GGML_LOG_ERROR("%s: failed to create Q scale layer\n", __func__);
            return nullptr;
        }
        scale_layer->setName((pfx + "_q_scale").c_str());
        q_bhsd = scale_layer->getOutput(0);
    }

    // ── addAttention ──
    // causal=false when mask is provided (mask encodes causality).
    // causal=true only when no mask (TRT handles it internally).
    bool has_mask = (mask_t != nullptr);
    auto* attn = network->addAttention(
        *q_bhsd, *k_bhsd, *v_bhsd,
        nvinfer1::AttentionNormalizationOp::kSOFTMAX,
        /*causal=*/ !has_mask);
    if (!attn) {
        GGML_LOG_ERROR("%s: addAttention failed\n", __func__);
        return nullptr;
    }
    attn->setName((pfx + "_attn").c_str());

    // ── Set mask if present ──
    // Mask shape after reversal: (mask_batch, mask_heads, n_tokens, n_kv)
    //   = [B, H, S_q, S_kv] — exactly what TRT expects.
    // GGML mask is additive (numeric type, not kBOOL).
    if (has_mask) {
        // Pad mask to 4D if needed (usually already 4D from GGML)
        mask_t = builder->maybe_cast(mask_t, compute_type);
        mask_t = builder->pad_to_ndims(mask_t, 4);
        attn->setMask(*mask_t);
    }

    // Allow decomposition on GPUs without fused attention kernel support
    attn->setDecomposable(true);

    nvinfer1::ITensor* attn_out = attn->getOutput(0);
    // attn_out shape: [B, H_q, S_q, D_v]

    // ── Reshape output to match GGML expected TRT-reversed shape ──
    // GGML result ne = (head_dim_v, n_tokens, n_heads_q, batch)
    // TRT reversed = (batch, n_heads_q, n_tokens, head_dim_v) = [B, H_q, S_q, D_v]
    // attn_out is already [B, H_q, S_q, D_v] — matches!
    // Just need to strip batch dim if batch=1 (to match ggml_tensor_to_dims output rank).
    int expected_ndims = ggml_n_dims(node);
    nvinfer1::ITensor* output = attn_out;

    if (attn_out->getDimensions().nbDims > expected_ndims) {
        // Strip leading batch dim: [1, H_q, S_q, D_v] → [H_q, S_q, D_v]
        nvinfer1::Dims out_dims = ggml_tensor_to_dims(node);
        // Handle dynamic S_q dim — position 1 in out_dims (n_tokens)
        // The GGML node result has concrete shapes, but in TRT the token dim is dynamic.
        // We need to figure out which dim is dynamic.
        // GGML result shape: (head_dim_v, n_tokens, n_heads_q[, batch])
        // TRT reversed (3D): (n_heads_q, n_tokens, head_dim_v)
        // n_tokens = out_dims.d[1] is dynamic
        nvinfer1::Dims strip_dims;
        strip_dims.nbDims = expected_ndims;
        for (int i = 0; i < expected_ndims; i++) {
            strip_dims.d[i] = out_dims.d[i];
        }
        // Mark n_tokens dim as dynamic (position 1 in 3D TRT layout)
        // n_tokens is GGML ne[1] of result, which maps to TRT position expected_ndims-2
        // For 3D: [n_heads_q, n_tokens, head_dim_v], position 1 is n_tokens
        if (expected_ndims >= 2) {
            strip_dims.d[expected_ndims - 2] = -1;  // n_tokens is dynamic
        }

        auto* strip = network->addShuffle(*attn_out);
        if (!strip) {
            GGML_LOG_ERROR("%s: failed to strip batch dim\n", __func__);
            return nullptr;
        }
        strip->setReshapeDimensions(strip_dims);
        strip->setName((pfx + "_strip_batch").c_str());
        output = strip->getOutput(0);
    }

    // ── Cast to F32 (FLASH_ATTN_EXT always outputs F32) ──
    output = builder->maybe_cast(output, nvinfer1::DataType::kFLOAT);

    GGML_LOG_DEBUG("%s: head_dim=%" PRId64 " scale=%.4f mask=%s output shape %s\n",
        __func__, head_dim, scale,
        has_mask ? "yes" : "no",
        dims_to_string(output->getDimensions()).c_str());

    return output;
}

// Register handlers (only when native attention is enabled)
static void __attribute__((constructor)) register_attention_handlers() {
    const char* env = getenv("GGML_TENSORRT_NATIVE_ATTN");
    if (env == nullptr || atoi(env) != 0) {
        NetworkBuilder::register_op_handler(GGML_OP_SET_ROWS, handle_set_rows);
        NetworkBuilder::register_op_handler(GGML_OP_FLASH_ATTN_EXT, handle_flash_attn_ext);
    }
}

} // namespace ggml_tensorrt
