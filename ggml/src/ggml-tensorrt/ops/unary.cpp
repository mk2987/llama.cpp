#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>

namespace ggml_tensorrt {

// Handle GGML_OP_UNARY — dispatches by ggml_get_unary_op(node)
nvinfer1::ITensor* handle_unary(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_UNARY);

    const ggml_tensor* src = node->src[0];
    GGML_ASSERT(src != nullptr);

    nvinfer1::ITensor* trt_src = builder->get_tensor(src);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    auto* network = builder->get_network();
    enum ggml_unary_op uop = ggml_get_unary_op(node);

    std::string name_prefix = "unary_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    nvinfer1::ITensor* output = apply_activation(network, builder, trt_src, uop, name_prefix.c_str());

    if (output == nullptr) {
        GGML_LOG_ERROR("%s: unsupported unary op %d\n", __func__, (int)uop);
        return nullptr;
    }

    GGML_LOG_DEBUG("%s: created unary op %d, output shape %s\n",
        __func__, (int)uop, dims_to_string(output->getDimensions()).c_str());

    return output;
}

// Register the unary operation handler
static void __attribute__((constructor)) register_unary_handler() {
    NetworkBuilder::register_op_handler(GGML_OP_UNARY, handle_unary);
}

} // namespace ggml_tensorrt
