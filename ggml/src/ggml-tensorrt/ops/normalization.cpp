#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "../utils/type-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>
#include <cmath>
#include <vector>

namespace ggml_tensorrt {

// Handle RMS normalization
// RMS_NORM: y = x / sqrt(mean(x^2) + eps) * scale
nvinfer1::ITensor* handle_rms_norm(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_RMS_NORM);

    // Get input tensor
    const ggml_tensor* src = node->src[0];
    GGML_ASSERT(src != nullptr);

    // Get TensorRT tensor for input
    nvinfer1::ITensor* trt_src = builder->get_tensor(src);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    auto* network = builder->get_network();
    nvinfer1::Dims dims = trt_src->getDimensions();

    // Get epsilon from operation parameters
    // RMS norm typically uses eps = 1e-6 or 1e-5
    float eps = 1e-6f;
    if (node->op_params[0] != 0) {
        memcpy(&eps, &node->op_params[0], sizeof(float));
    }

    GGML_LOG_DEBUG("%s: input shape %s, eps=%e\n",
        __func__, dims_to_string(dims).c_str(), eps);

    // RMS normalization: y = x / sqrt(mean(x^2) + eps)
    // In TensorRT, we can implement this as:
    // 1. x^2 (power)
    // 2. mean(x^2) (reduce mean)
    // 3. sqrt(mean(x^2) + eps) (unary ops)
    // 4. x / sqrt(...) (elementwise divide)

    // Step 1: x^2
    // Using power operation with exponent 2
    auto* square_layer = network->addElementWise(
        *trt_src, *trt_src, nvinfer1::ElementWiseOperation::kPROD);
    if (square_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create square layer\n", __func__);
        return nullptr;
    }
    nvinfer1::ITensor* x_squared = square_layer->getOutput(0);

    // Step 2: mean(x^2) along the last dimension
    // For RMS norm, we typically normalize along the feature dimension (last dim)
    uint32_t reduce_axes = 1U << (dims.nbDims - 1); // Reduce along last dimension

    auto* mean_layer = network->addReduce(
        *x_squared,
        nvinfer1::ReduceOperation::kAVG,
        reduce_axes,
        true // keep dimensions
    );
    if (mean_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create mean layer\n", __func__);
        return nullptr;
    }
    nvinfer1::ITensor* mean_x_squared = mean_layer->getOutput(0);

    // Step 3: Add epsilon as a constant
    nvinfer1::Dims scalar_dims;
    scalar_dims.nbDims = 1;
    scalar_dims.d[0] = 1;

    auto* eps_tensor = create_constant_tensor(
        network, &eps, scalar_dims, nvinfer1::DataType::kFLOAT);
    if (eps_tensor == nullptr) {
        GGML_LOG_ERROR("%s: failed to create epsilon tensor\n", __func__);
        return nullptr;
    }

    // Add epsilon to mean(x^2)
    auto* add_eps_layer = network->addElementWise(
        *mean_x_squared, *eps_tensor, nvinfer1::ElementWiseOperation::kSUM);
    if (add_eps_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create add epsilon layer\n", __func__);
        return nullptr;
    }
    nvinfer1::ITensor* variance_plus_eps = add_eps_layer->getOutput(0);

    // Step 4: sqrt(mean(x^2) + eps)
    auto* sqrt_layer = network->addUnary(
        *variance_plus_eps, nvinfer1::UnaryOperation::kSQRT);
    if (sqrt_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create sqrt layer\n", __func__);
        return nullptr;
    }
    nvinfer1::ITensor* rms = sqrt_layer->getOutput(0);

    // Step 5: x / sqrt(mean(x^2) + eps)
    auto* div_layer = network->addElementWise(
        *trt_src, *rms, nvinfer1::ElementWiseOperation::kDIV);
    if (div_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create divide layer\n", __func__);
        return nullptr;
    }
    nvinfer1::ITensor* output = div_layer->getOutput(0);

    // Set layer name for debugging
    std::string layer_name = "rms_norm_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    div_layer->setName(layer_name.c_str());

    GGML_LOG_DEBUG("%s: created RMS_NORM layer, output shape %s\n",
        __func__, dims_to_string(output->getDimensions()).c_str());

    return output;
}

// Handle GROUP normalization
// GROUP_NORM: Normalize in groups along channel dimension
nvinfer1::ITensor* handle_group_norm(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_GROUP_NORM);

    // Get input tensor
    const ggml_tensor* src = node->src[0];
    GGML_ASSERT(src != nullptr);

    // Get TensorRT tensor for input
    nvinfer1::ITensor* trt_src = builder->get_tensor(src);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    auto* network = builder->get_network();
    nvinfer1::Dims dims = trt_src->getDimensions();

    // Get number of groups and epsilon from operation parameters
    int32_t num_groups = 1;
    float eps = 1e-5f;

    if (node->op_params[0] != 0) {
        memcpy(&num_groups, &node->op_params[0], sizeof(int32_t));
    }
    if (node->op_params[1] != 0) {
        memcpy(&eps, &node->op_params[1], sizeof(float));
    }

    GGML_LOG_DEBUG("%s: input shape %s, num_groups=%d, eps=%e\n",
        __func__, dims_to_string(dims).c_str(), num_groups, eps);

    // TensorRT doesn't have a native GROUP_NORM layer in older versions
    // We need to implement it using available operations
    // For Milestone 2, we'll implement a simplified version
    // Full implementation would require reshape -> normalize -> reshape back

    // For now, fallback to a simpler normalization
    // This will be enhanced in later milestones if needed
    GGML_LOG_WARN("%s: GROUP_NORM implementation is simplified\n", __func__);

    // Use normalization layer as approximation
    uint32_t axes = (1U << (dims.nbDims - 1)); // Normalize along last dimension

    // Create scale (1.0) and bias (0.0) constant tensors
    // TensorRT-RTX requires non-null scale and bias
    int64_t channel_dim = dims.d[dims.nbDims - 1];
    std::vector<float> scale_data(channel_dim, 1.0f);
    std::vector<float> bias_data(channel_dim, 0.0f);

    nvinfer1::Dims scale_dims;
    scale_dims.nbDims = 1;
    scale_dims.d[0] = channel_dim;

    nvinfer1::ITensor* scale_tensor = create_constant_tensor(
        network, scale_data.data(), scale_dims, nvinfer1::DataType::kFLOAT);
    nvinfer1::ITensor* bias_tensor = create_constant_tensor(
        network, bias_data.data(), scale_dims, nvinfer1::DataType::kFLOAT);

    if (scale_tensor == nullptr || bias_tensor == nullptr) {
        GGML_LOG_ERROR("%s: failed to create scale/bias tensors\n", __func__);
        return nullptr;
    }

    // Note: addNormalization is deprecated but still functional
    // Suppress deprecation warning - will migrate to newer API in future milestone
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    auto* norm_layer = network->addNormalization(
        *trt_src,
        *scale_tensor,
        *bias_tensor,
        axes
    );
#pragma GCC diagnostic pop

    if (norm_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create normalization layer\n", __func__);
        return nullptr;
    }

    norm_layer->setEpsilon(eps);

    // Set layer name for debugging
    std::string layer_name = "group_norm_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    norm_layer->setName(layer_name.c_str());

    nvinfer1::ITensor* output = norm_layer->getOutput(0);

    GGML_LOG_DEBUG("%s: created GROUP_NORM layer, output shape %s\n",
        __func__, dims_to_string(output->getDimensions()).c_str());

    return output;
}

// Register operation handlers
static void __attribute__((constructor)) register_normalization_handlers() {
    NetworkBuilder::register_op_handler(GGML_OP_RMS_NORM, handle_rms_norm);
    NetworkBuilder::register_op_handler(GGML_OP_GROUP_NORM, handle_group_norm);
}

} // namespace ggml_tensorrt
