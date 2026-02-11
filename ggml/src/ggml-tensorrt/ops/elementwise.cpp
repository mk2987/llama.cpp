#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "../utils/type-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>

namespace ggml_tensorrt {

// Generic handler for elementwise binary operations
static nvinfer1::ITensor* handle_elementwise_binary(
    NetworkBuilder* builder,
    const ggml_tensor* node,
    nvinfer1::ElementWiseOperation op,
    const char* op_name
) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);

    // Get input tensors
    const ggml_tensor* src0 = node->src[0];
    const ggml_tensor* src1 = node->src[1];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);

    // Get or create TensorRT tensors for inputs
    nvinfer1::ITensor* trt_src0 = builder->get_tensor(src0);
    nvinfer1::ITensor* trt_src1 = builder->get_tensor(src1);

    if (trt_src0 == nullptr || trt_src1 == nullptr) {
        GGML_LOG_ERROR("%s: input tensors not found in network for %s\n", __func__, op_name);
        return nullptr;
    }

    // Get dimensions
    nvinfer1::Dims dims0 = trt_src0->getDimensions();
    nvinfer1::Dims dims1 = trt_src1->getDimensions();

    GGML_LOG_DEBUG("%s: %s with src0 shape %s, src1 shape %s\n",
        __func__, op_name,
        dims_to_string(dims0).c_str(),
        dims_to_string(dims1).c_str());

    // Equalize ranks by padding the lower-rank tensor with leading 1-dims.
    // TRT's addElementWise handles broadcasting natively (dims of 1 expand
    // to match the other operand), but requires both operands to have the
    // same number of dimensions.
    if (dims0.nbDims != dims1.nbDims) {
        int max_ndims = std::max(dims0.nbDims, dims1.nbDims);
        trt_src0 = builder->pad_to_ndims(trt_src0, max_ndims);
        trt_src1 = builder->pad_to_ndims(trt_src1, max_ndims);

        if (trt_src0 == nullptr || trt_src1 == nullptr) {
            GGML_LOG_ERROR("%s: failed to pad tensors for %s\n", __func__, op_name);
            return nullptr;
        }
    }

    // Add ElementWise layer
    auto* layer = builder->get_network()->addElementWise(*trt_src0, *trt_src1, op);
    if (layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create ElementWise layer for %s\n", __func__, op_name);
        return nullptr;
    }

    // Set layer name for debugging
    std::string layer_name = std::string(op_name) + "_" +
                            std::to_string(reinterpret_cast<uintptr_t>(node));
    layer->setName(layer_name.c_str());

    // Get output
    nvinfer1::ITensor* output = layer->getOutput(0);
    if (output == nullptr) {
        GGML_LOG_ERROR("%s: failed to get output from ElementWise layer for %s\n",
                      __func__, op_name);
        return nullptr;
    }

    GGML_LOG_DEBUG("%s: created %s layer, output shape %s\n",
        __func__, op_name,
        dims_to_string(output->getDimensions()).c_str());

    return output;
}

// ADD operation handler
nvinfer1::ITensor* handle_add(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(node->op == GGML_OP_ADD);
    return handle_elementwise_binary(builder, node,
                                     nvinfer1::ElementWiseOperation::kSUM,
                                     "add");
}

// MUL operation handler
nvinfer1::ITensor* handle_mul(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(node->op == GGML_OP_MUL);
    return handle_elementwise_binary(builder, node,
                                     nvinfer1::ElementWiseOperation::kPROD,
                                     "mul");
}

// SUB operation handler
nvinfer1::ITensor* handle_sub(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(node->op == GGML_OP_SUB);
    return handle_elementwise_binary(builder, node,
                                     nvinfer1::ElementWiseOperation::kSUB,
                                     "sub");
}

// DIV operation handler
nvinfer1::ITensor* handle_div(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(node->op == GGML_OP_DIV);
    return handle_elementwise_binary(builder, node,
                                     nvinfer1::ElementWiseOperation::kDIV,
                                     "div");
}

// Register operation handlers
static void __attribute__((constructor)) register_elementwise_handlers() {
    NetworkBuilder::register_op_handler(GGML_OP_ADD, handle_add);
    NetworkBuilder::register_op_handler(GGML_OP_MUL, handle_mul);
    NetworkBuilder::register_op_handler(GGML_OP_SUB, handle_sub);
    NetworkBuilder::register_op_handler(GGML_OP_DIV, handle_div);
}

} // namespace ggml_tensorrt
