#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "../utils/type-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>
#include <cinttypes>
#include <cstring>

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

// Register handler (only when native attention is enabled)
static void __attribute__((constructor)) register_attention_handlers() {
    const char* env = getenv("GGML_TENSORRT_NATIVE_ATTN");
    if (env != nullptr && atoi(env) != 0) {
        NetworkBuilder::register_op_handler(GGML_OP_SET_ROWS, handle_set_rows);
    }
}

} // namespace ggml_tensorrt
