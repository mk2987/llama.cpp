#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "../utils/type-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>

namespace ggml_tensorrt {

// Handle CPY: cast to destination type + optional reshape.
// In GGML, ggml_cpy(a, b) copies data from a (src[0]) to b (src[1]).
// src[1] is the destination template — it shares the data pointer with the
// output node and must NOT be treated as a TRT input.
static nvinfer1::ITensor* handle_cpy(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_CPY || node->op == GGML_OP_DUP);

    const ggml_tensor* src = node->src[0];
    GGML_ASSERT(src != nullptr);

    nvinfer1::ITensor* trt_src = builder->get_tensor(src);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    // Cast to destination type if needed (e.g. F32 → F16 for KV cache)
    nvinfer1::DataType dst_type = ggml_type_to_tensorrt(node->type);
    nvinfer1::ITensor* output = builder->maybe_cast(trt_src, dst_type);

    // Reshape if output dimensions differ from source
    nvinfer1::Dims dst_dims = ggml_tensor_to_dims(node);
    nvinfer1::Dims safe_dims = NetworkBuilder::make_dynamic_reshape_dims(output, dst_dims);
    nvinfer1::Dims out_dims = output->getDimensions();

    bool dims_match = (out_dims.nbDims == safe_dims.nbDims);
    if (dims_match) {
        for (int i = 0; i < out_dims.nbDims; i++) {
            if (safe_dims.d[i] == 0) continue;  // copy-through always matches
            if (out_dims.d[i] != safe_dims.d[i]) {
                dims_match = false;
                break;
            }
        }
    }

    if (!dims_match) {
        auto* network = builder->get_network();
        auto* shuffle = network->addShuffle(*output);
        if (shuffle == nullptr) {
            GGML_LOG_ERROR("%s: failed to create shuffle layer for CPY reshape\n", __func__);
            return nullptr;
        }
        shuffle->setReshapeDimensions(safe_dims);

        std::string layer_name = "cpy_reshape_" + std::to_string(reinterpret_cast<uintptr_t>(node));
        shuffle->setName(layer_name.c_str());
        output = shuffle->getOutput(0);
    }

    GGML_LOG_DEBUG("%s: CPY %s -> %s\n", __func__,
        ggml_type_name(src->type), ggml_type_name(node->type));

    return output;
}

// Handle CONT: make non-contiguous data contiguous.
// In TRT, all tensors are logically contiguous — IShuffleLayer (PERMUTE)
// produces a reordered but contiguous output.  CONT after PERMUTE is just
// a reshape to the output dimensions.
static nvinfer1::ITensor* handle_cont(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_CONT);

    const ggml_tensor* src = node->src[0];
    GGML_ASSERT(src != nullptr);

    nvinfer1::ITensor* trt_src = builder->get_tensor(src);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    nvinfer1::Dims dst_dims = ggml_tensor_to_dims(node);
    nvinfer1::Dims safe_dims = NetworkBuilder::make_dynamic_reshape_dims(trt_src, dst_dims);
    nvinfer1::Dims src_dims = trt_src->getDimensions();

    // If dims already match, pass through (no-op)
    bool dims_match = (src_dims.nbDims == safe_dims.nbDims);
    if (dims_match) {
        for (int i = 0; i < src_dims.nbDims; i++) {
            if (safe_dims.d[i] == 0) continue;  // copy-through always matches
            if (src_dims.d[i] != safe_dims.d[i]) {
                dims_match = false;
                break;
            }
        }
    }
    if (dims_match) {
        return trt_src;
    }

    // Reshape to output dimensions
    auto* network = builder->get_network();
    auto* shuffle = network->addShuffle(*trt_src);
    if (shuffle == nullptr) {
        GGML_LOG_ERROR("%s: failed to create shuffle layer for CONT\n", __func__);
        return nullptr;
    }

    shuffle->setReshapeDimensions(safe_dims);

    std::string layer_name = "cont_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    shuffle->setName(layer_name.c_str());

    nvinfer1::ITensor* output = shuffle->getOutput(0);

    GGML_LOG_DEBUG("%s: CONT reshape %s -> %s\n", __func__,
        dims_to_string(src_dims).c_str(), dims_to_string(dst_dims).c_str());

    return output;
}

// Register copy operation handlers
static void __attribute__((constructor)) register_copy_handlers() {
    NetworkBuilder::register_op_handler(GGML_OP_CPY,  handle_cpy);
    NetworkBuilder::register_op_handler(GGML_OP_DUP,  handle_cpy);  // DUP = CPY with same type
    NetworkBuilder::register_op_handler(GGML_OP_CONT, handle_cont);
}

} // namespace ggml_tensorrt
