#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>
#include <cstring>

namespace ggml_tensorrt {

// Handle SOFT_MAX: softmax along GGML dim 0 (TRT last dim), with optional scale
nvinfer1::ITensor* handle_soft_max(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_SOFT_MAX);

    const ggml_tensor* src = node->src[0];
    GGML_ASSERT(src != nullptr);

    // Reject mask (src[1]) and ALiBi (max_bias != 0)
    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    &node->op_params[0], sizeof(float));
    memcpy(&max_bias, &node->op_params[1], sizeof(float));

    if (node->src[1] != nullptr) {
        GGML_LOG_ERROR("%s: masked softmax not supported\n", __func__);
        return nullptr;
    }
    if (max_bias != 0.0f) {
        GGML_LOG_ERROR("%s: ALiBi softmax not supported\n", __func__);
        return nullptr;
    }

    nvinfer1::ITensor* trt_src = builder->get_tensor(src);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    auto* network = builder->get_network();

    // Save original type and upcast to FP32 for numerical stability
    nvinfer1::DataType input_type = trt_src->getType();
    trt_src = builder->maybe_cast(trt_src, nvinfer1::DataType::kFLOAT);

    nvinfer1::Dims dims = trt_src->getDimensions();
    nvinfer1::ITensor* input = trt_src;

    // Apply scale if not 1.0
    if (scale != 1.0f) {
        // Scalar must have same number of dims as input for TRT broadcasting
        nvinfer1::Dims scalar_dims;
        scalar_dims.nbDims = dims.nbDims;
        for (int i = 0; i < dims.nbDims; i++) {
            scalar_dims.d[i] = 1;
        }

        nvinfer1::ITensor* scale_tensor = builder->create_constant_tensor(
            &scale, scalar_dims, nvinfer1::DataType::kFLOAT);
        if (scale_tensor == nullptr) {
            GGML_LOG_ERROR("%s: failed to create scale constant\n", __func__);
            return nullptr;
        }

        auto* scale_layer = network->addElementWise(
            *input, *scale_tensor, nvinfer1::ElementWiseOperation::kPROD);
        if (scale_layer == nullptr) {
            GGML_LOG_ERROR("%s: failed to create scale layer\n", __func__);
            return nullptr;
        }
        input = scale_layer->getOutput(0);
    }

    // Softmax along GGML dim 0 = TRT last dim
    uint32_t softmax_axes = 1U << (dims.nbDims - 1);

    auto* softmax_layer = network->addSoftMax(*input);
    if (softmax_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create softmax layer\n", __func__);
        return nullptr;
    }
    softmax_layer->setAxes(softmax_axes);

    std::string layer_name = "soft_max_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    softmax_layer->setName(layer_name.c_str());

    nvinfer1::ITensor* output = softmax_layer->getOutput(0);

    // Cast back to original type if needed
    output = builder->maybe_cast(output, input_type);

    GGML_LOG_DEBUG("%s: softmax on axes=0x%x, scale=%.4f, output shape %s\n",
        __func__, softmax_axes, scale,
        dims_to_string(output->getDimensions()).c_str());

    return output;
}

// Register the softmax handler
static void __attribute__((constructor)) register_softmax_handler() {
    NetworkBuilder::register_op_handler(GGML_OP_SOFT_MAX, handle_soft_max);
}

} // namespace ggml_tensorrt
