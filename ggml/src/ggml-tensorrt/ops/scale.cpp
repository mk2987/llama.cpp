#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>
#include <cstring>

namespace ggml_tensorrt {

// Handle SCALE: y[i] = a[i] * s + b
// op_params layout: float[0] = scale, float[1] = bias
nvinfer1::ITensor* handle_scale(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_SCALE);

    const ggml_tensor* src = node->src[0];
    GGML_ASSERT(src != nullptr);

    float s = 1.0f;
    float b = 0.0f;
    memcpy(&s, &node->op_params[0], sizeof(float));
    memcpy(&b, &node->op_params[1], sizeof(float));

    nvinfer1::ITensor* trt_src = builder->get_tensor(src);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    auto* network = builder->get_network();
    nvinfer1::ITensor* result = trt_src;

    // Apply scale: result = input * s
    if (s != 1.0f) {
        nvinfer1::ITensor* scale_tensor = builder->create_typed_scalar(s, result);
        if (scale_tensor == nullptr) {
            GGML_LOG_ERROR("%s: failed to create scale constant\n", __func__);
            return nullptr;
        }

        auto* scale_layer = network->addElementWise(
            *result, *scale_tensor, nvinfer1::ElementWiseOperation::kPROD);
        if (scale_layer == nullptr) {
            GGML_LOG_ERROR("%s: failed to create scale multiply layer\n", __func__);
            return nullptr;
        }

        std::string layer_name = "scale_mul_" + std::to_string(reinterpret_cast<uintptr_t>(node));
        scale_layer->setName(layer_name.c_str());
        result = scale_layer->getOutput(0);
    }

    // Apply bias: result = result + b
    if (b != 0.0f) {
        nvinfer1::ITensor* bias_tensor = builder->create_typed_scalar(b, result);
        if (bias_tensor == nullptr) {
            GGML_LOG_ERROR("%s: failed to create bias constant\n", __func__);
            return nullptr;
        }

        auto* bias_layer = network->addElementWise(
            *result, *bias_tensor, nvinfer1::ElementWiseOperation::kSUM);
        if (bias_layer == nullptr) {
            GGML_LOG_ERROR("%s: failed to create scale bias layer\n", __func__);
            return nullptr;
        }

        std::string layer_name = "scale_add_" + std::to_string(reinterpret_cast<uintptr_t>(node));
        bias_layer->setName(layer_name.c_str());
        result = bias_layer->getOutput(0);
    }

    GGML_LOG_DEBUG("%s: scale=%.4f, bias=%.4f, output shape %s\n",
        __func__, s, b,
        dims_to_string(result->getDimensions()).c_str());

    return result;
}

// Register the scale handler
static void __attribute__((constructor)) register_scale_handler() {
    NetworkBuilder::register_op_handler(GGML_OP_SCALE, handle_scale);
}

} // namespace ggml_tensorrt
