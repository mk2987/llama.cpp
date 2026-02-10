#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>
#include <cmath>

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

    nvinfer1::ITensor* output = nullptr;

    switch (uop) {
        case GGML_UNARY_OP_RELU:
        {
            auto* layer = network->addActivation(*trt_src, nvinfer1::ActivationType::kRELU);
            if (layer == nullptr) {
                GGML_LOG_ERROR("%s: failed to create RELU activation layer\n", __func__);
                return nullptr;
            }
            std::string name = "relu_" + std::to_string(reinterpret_cast<uintptr_t>(node));
            layer->setName(name.c_str());
            output = layer->getOutput(0);
            break;
        }

        case GGML_UNARY_OP_TANH:
        {
            auto* layer = network->addActivation(*trt_src, nvinfer1::ActivationType::kTANH);
            if (layer == nullptr) {
                GGML_LOG_ERROR("%s: failed to create TANH activation layer\n", __func__);
                return nullptr;
            }
            std::string name = "tanh_" + std::to_string(reinterpret_cast<uintptr_t>(node));
            layer->setName(name.c_str());
            output = layer->getOutput(0);
            break;
        }

        case GGML_UNARY_OP_SIGMOID:
        {
            auto* layer = network->addActivation(*trt_src, nvinfer1::ActivationType::kSIGMOID);
            if (layer == nullptr) {
                GGML_LOG_ERROR("%s: failed to create SIGMOID activation layer\n", __func__);
                return nullptr;
            }
            std::string name = "sigmoid_" + std::to_string(reinterpret_cast<uintptr_t>(node));
            layer->setName(name.c_str());
            output = layer->getOutput(0);
            break;
        }

        case GGML_UNARY_OP_SILU:
        {
            // SiLU = x * sigmoid(x)
            // TRT has no native SiLU, decompose into sigmoid + elementwise product
            auto* sigmoid_layer = network->addActivation(*trt_src, nvinfer1::ActivationType::kSIGMOID);
            if (sigmoid_layer == nullptr) {
                GGML_LOG_ERROR("%s: failed to create SIGMOID layer for SILU\n", __func__);
                return nullptr;
            }
            nvinfer1::ITensor* sigmoid_out = sigmoid_layer->getOutput(0);

            auto* mul_layer = network->addElementWise(
                *trt_src, *sigmoid_out, nvinfer1::ElementWiseOperation::kPROD);
            if (mul_layer == nullptr) {
                GGML_LOG_ERROR("%s: failed to create MUL layer for SILU\n", __func__);
                return nullptr;
            }
            std::string name = "silu_" + std::to_string(reinterpret_cast<uintptr_t>(node));
            mul_layer->setName(name.c_str());
            output = mul_layer->getOutput(0);
            break;
        }

        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
        {
            // GELU = x * 0.5 * (1 + erf(x / sqrt(2)))
            // Decompose using erf unary + elementwise ops

            // x / sqrt(2)
            float inv_sqrt2 = 1.0f / sqrtf(2.0f);
            nvinfer1::Dims src_dims = trt_src->getDimensions();
            nvinfer1::Dims scalar_dims;
            scalar_dims.nbDims = src_dims.nbDims;
            for (int i = 0; i < src_dims.nbDims; i++) {
                scalar_dims.d[i] = 1;
            }

            nvinfer1::ITensor* scale_tensor = builder->create_constant_tensor(
                &inv_sqrt2, scalar_dims, nvinfer1::DataType::kFLOAT);
            if (scale_tensor == nullptr) {
                GGML_LOG_ERROR("%s: failed to create scale constant for GELU\n", __func__);
                return nullptr;
            }

            auto* scale_layer = network->addElementWise(
                *trt_src, *scale_tensor, nvinfer1::ElementWiseOperation::kPROD);
            if (scale_layer == nullptr) {
                GGML_LOG_ERROR("%s: failed to create scale layer for GELU\n", __func__);
                return nullptr;
            }
            nvinfer1::ITensor* scaled = scale_layer->getOutput(0);

            // erf(x / sqrt(2))
            auto* erf_layer = network->addUnary(*scaled, nvinfer1::UnaryOperation::kERF);
            if (erf_layer == nullptr) {
                GGML_LOG_ERROR("%s: failed to create ERF layer for GELU\n", __func__);
                return nullptr;
            }
            nvinfer1::ITensor* erf_out = erf_layer->getOutput(0);

            // 1 + erf(...)
            float one = 1.0f;
            nvinfer1::ITensor* one_tensor = builder->create_constant_tensor(
                &one, scalar_dims, nvinfer1::DataType::kFLOAT);
            if (one_tensor == nullptr) {
                GGML_LOG_ERROR("%s: failed to create one constant for GELU\n", __func__);
                return nullptr;
            }

            auto* add_layer = network->addElementWise(
                *erf_out, *one_tensor, nvinfer1::ElementWiseOperation::kSUM);
            if (add_layer == nullptr) {
                GGML_LOG_ERROR("%s: failed to create add layer for GELU\n", __func__);
                return nullptr;
            }
            nvinfer1::ITensor* one_plus_erf = add_layer->getOutput(0);

            // 0.5 * (1 + erf(...))
            float half = 0.5f;
            nvinfer1::ITensor* half_tensor = builder->create_constant_tensor(
                &half, scalar_dims, nvinfer1::DataType::kFLOAT);
            if (half_tensor == nullptr) {
                GGML_LOG_ERROR("%s: failed to create half constant for GELU\n", __func__);
                return nullptr;
            }

            auto* half_layer = network->addElementWise(
                *one_plus_erf, *half_tensor, nvinfer1::ElementWiseOperation::kPROD);
            if (half_layer == nullptr) {
                GGML_LOG_ERROR("%s: failed to create half mul layer for GELU\n", __func__);
                return nullptr;
            }
            nvinfer1::ITensor* cdf = half_layer->getOutput(0);

            // x * cdf
            auto* mul_layer = network->addElementWise(
                *trt_src, *cdf, nvinfer1::ElementWiseOperation::kPROD);
            if (mul_layer == nullptr) {
                GGML_LOG_ERROR("%s: failed to create final mul layer for GELU\n", __func__);
                return nullptr;
            }
            std::string name = "gelu_" + std::to_string(reinterpret_cast<uintptr_t>(node));
            mul_layer->setName(name.c_str());
            output = mul_layer->getOutput(0);
            break;
        }

        case GGML_UNARY_OP_EXP:
        {
            auto* layer = network->addUnary(*trt_src, nvinfer1::UnaryOperation::kEXP);
            if (layer == nullptr) {
                GGML_LOG_ERROR("%s: failed to create EXP unary layer\n", __func__);
                return nullptr;
            }
            std::string name = "exp_" + std::to_string(reinterpret_cast<uintptr_t>(node));
            layer->setName(name.c_str());
            output = layer->getOutput(0);
            break;
        }

        default:
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
