#include "network-builder.hpp"
#include "utils/tensor-utils.hpp"
#include "utils/type-utils.hpp"
#include "ggml-impl.h"

#include <cmath>
#include <cstring>
#include <sstream>

namespace ggml_tensorrt {

// Initialize static member
std::map<ggml_op, OpHandler> NetworkBuilder::op_handlers_;

NetworkBuilder::NetworkBuilder(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ILogger* logger
) : network_(network), logger_(logger) {
    GGML_ASSERT(network_ != nullptr);
    GGML_ASSERT(logger_ != nullptr);
}

NetworkBuilder::~NetworkBuilder() {
    // TensorRT objects are managed by the network
    // We don't need to manually delete ITensor objects
    tensor_map_.clear();
}

nvinfer1::ITensor* NetworkBuilder::add_input(const ggml_tensor* tensor, const std::string& name) {
    GGML_ASSERT(tensor != nullptr);

    // Check if tensor is already added
    if (has_tensor(tensor)) {
        return get_tensor(tensor);
    }

    // Validate tensor
    if (!validate_tensor_for_tensorrt(tensor)) {
        GGML_LOG_ERROR("%s: tensor validation failed for %s\n", __func__, name.c_str());
        return nullptr;
    }

    // Convert dimensions
    nvinfer1::Dims dims = ggml_tensor_to_dims(tensor);

    // Convert data type
    nvinfer1::DataType dtype = ggml_type_to_tensorrt(tensor->type);

    // Add input to network
    nvinfer1::ITensor* trt_tensor = network_->addInput(name.c_str(), dtype, dims);
    if (trt_tensor == nullptr) {
        GGML_LOG_ERROR("%s: failed to add input %s\n", __func__, name.c_str());
        return nullptr;
    }

    // Store in map
    set_tensor(tensor, trt_tensor);

    GGML_LOG_DEBUG("%s: added input %s with shape %s\n",
        __func__, name.c_str(), dims_to_string(dims).c_str());

    return trt_tensor;
}

nvinfer1::ITensor* NetworkBuilder::get_tensor(const ggml_tensor* tensor) {
    GGML_ASSERT(tensor != nullptr);

    auto it = tensor_map_.find(tensor);
    if (it != tensor_map_.end()) {
        return it->second;
    }

    return nullptr;
}

void NetworkBuilder::set_tensor(const ggml_tensor* ggml_tensor, nvinfer1::ITensor* trt_tensor) {
    GGML_ASSERT(ggml_tensor != nullptr);
    GGML_ASSERT(trt_tensor != nullptr);

    tensor_map_[ggml_tensor] = trt_tensor;
}

bool NetworkBuilder::has_tensor(const ggml_tensor* tensor) const {
    GGML_ASSERT(tensor != nullptr);
    return tensor_map_.find(tensor) != tensor_map_.end();
}

nvinfer1::ITensor* NetworkBuilder::add_operation(const ggml_tensor* node) {
    GGML_ASSERT(node != nullptr);

    // Check if output already exists
    if (has_tensor(node)) {
        return get_tensor(node);
    }

    // Get operation type
    ggml_op op = node->op;

    // Check if operation is supported
    if (!is_operation_supported(op)) {
        GGML_LOG_DEBUG("%s: operation %s not supported\n",
            __func__, ggml_op_name(op));
        return nullptr;
    }

    // Get handler
    auto it = op_handlers_.find(op);
    if (it == op_handlers_.end()) {
        GGML_LOG_DEBUG("%s: no handler for operation %s\n",
            __func__, ggml_op_name(op));
        return nullptr;
    }

    // Call handler
    OpHandler handler = it->second;
    nvinfer1::ITensor* output = handler(this, node);

    if (output == nullptr) {
        GGML_LOG_ERROR("%s: handler failed for operation %s\n",
            __func__, ggml_op_name(op));
        return nullptr;
    }

    // Store output
    set_tensor(node, output);

    GGML_LOG_DEBUG("%s: added operation %s\n", __func__, ggml_op_name(op));

    return output;
}

void NetworkBuilder::mark_output(nvinfer1::ITensor* tensor, const std::string& name) {
    GGML_ASSERT(tensor != nullptr);

    network_->markOutput(*tensor);
    tensor->setName(name.c_str());

    GGML_LOG_DEBUG("%s: marked output %s with shape %s\n",
        __func__, name.c_str(), dims_to_string(tensor->getDimensions()).c_str());
}

bool NetworkBuilder::is_operation_supported(ggml_op op) {
    return op_handlers_.find(op) != op_handlers_.end();
}

void NetworkBuilder::register_op_handler(ggml_op op, OpHandler handler) {
    op_handlers_[op] = handler;
}

static size_t tensorrt_dtype_size(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kBF16:  return 2;
        case nvinfer1::DataType::kFP8:   return 1;
        case nvinfer1::DataType::kINT64: return 8;
        default:                         return 4;
    }
}

nvinfer1::ITensor* NetworkBuilder::create_constant_tensor(
    const void* data,
    nvinfer1::Dims dims,
    nvinfer1::DataType dtype
) {
    GGML_ASSERT(data != nullptr);

    // Calculate total number of elements
    int64_t numel = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        numel *= dims.d[i];
    }

    // Copy weight data into persistent storage.
    // TensorRT does not copy weights — the pointer passed via
    // nvinfer1::Weights must remain valid for the lifetime of
    // the INetworkDefinition (i.e. until the engine is built).
    size_t nbytes = static_cast<size_t>(numel) * tensorrt_dtype_size(dtype);
    weight_storage_.emplace_back(nbytes);
    memcpy(weight_storage_.back().data(), data, nbytes);

    nvinfer1::Weights weights{dtype, weight_storage_.back().data(), numel};

    // Add constant layer
    nvinfer1::IConstantLayer* layer = network_->addConstant(dims, weights);
    if (layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create constant layer\n", __func__);
        return nullptr;
    }

    return layer->getOutput(0);
}

nvinfer1::ITensor* NetworkBuilder::create_typed_scalar(
    float value,
    nvinfer1::ITensor* reference_tensor
) {
    GGML_ASSERT(reference_tensor != nullptr);

    nvinfer1::Dims ref_dims = reference_tensor->getDimensions();
    nvinfer1::DataType dtype = reference_tensor->getType();

    // Build scalar dims matching the reference tensor rank (all 1s)
    nvinfer1::Dims scalar_dims;
    scalar_dims.nbDims = ref_dims.nbDims;
    for (int i = 0; i < ref_dims.nbDims; i++) {
        scalar_dims.d[i] = 1;
    }

    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: {
            return create_constant_tensor(&value, scalar_dims, nvinfer1::DataType::kFLOAT);
        }
        case nvinfer1::DataType::kHALF: {
            ggml_fp16_t half_val = ggml_fp32_to_fp16(value);
            return create_constant_tensor(&half_val, scalar_dims, nvinfer1::DataType::kHALF);
        }
        case nvinfer1::DataType::kBF16: {
            ggml_bf16_t bf16_val = ggml_fp32_to_bf16(value);
            return create_constant_tensor(&bf16_val, scalar_dims, nvinfer1::DataType::kBF16);
        }
        default:
            GGML_LOG_ERROR("%s: unsupported dtype for scalar constant\n", __func__);
            return nullptr;
    }
}

nvinfer1::ITensor* NetworkBuilder::maybe_cast(
    nvinfer1::ITensor* tensor,
    nvinfer1::DataType target_type
) {
    GGML_ASSERT(tensor != nullptr);

    if (tensor->getType() == target_type) {
        return tensor;
    }

    nvinfer1::ICastLayer* cast = network_->addCast(*tensor, target_type);
    if (cast == nullptr) {
        GGML_LOG_ERROR("%s: failed to add cast layer\n", __func__);
        return nullptr;
    }

    return cast->getOutput(0);
}

nvinfer1::ITensor* NetworkBuilder::pad_to_ndims(
    nvinfer1::ITensor* tensor,
    int target_ndims
) {
    GGML_ASSERT(tensor != nullptr);

    nvinfer1::Dims current = tensor->getDimensions();
    if (current.nbDims >= target_ndims) {
        return tensor;
    }

    nvinfer1::Dims new_dims;
    new_dims.nbDims = target_ndims;
    int pad = target_ndims - current.nbDims;
    for (int i = 0; i < pad; i++) {
        new_dims.d[i] = 1;
    }
    for (int i = 0; i < current.nbDims; i++) {
        new_dims.d[pad + i] = current.d[i];
    }

    auto* shuffle = network_->addShuffle(*tensor);
    if (shuffle == nullptr) {
        GGML_LOG_ERROR("%s: failed to add shuffle layer for rank padding\n", __func__);
        return nullptr;
    }
    shuffle->setReshapeDimensions(new_dims);
    return shuffle->getOutput(0);
}

nvinfer1::ITensor* apply_activation(
    nvinfer1::INetworkDefinition* network,
    NetworkBuilder* builder,
    nvinfer1::ITensor* input,
    enum ggml_unary_op activation,
    const char* name_prefix
) {
    GGML_ASSERT(network != nullptr);
    GGML_ASSERT(input != nullptr);

    switch (activation) {
        case GGML_UNARY_OP_RELU:
        {
            auto* layer = network->addActivation(*input, nvinfer1::ActivationType::kRELU);
            if (!layer) {
                GGML_LOG_ERROR("%s: failed to create RELU layer\n", __func__);
                return nullptr;
            }
            layer->setName((std::string(name_prefix) + "_relu").c_str());
            return layer->getOutput(0);
        }

        case GGML_UNARY_OP_TANH:
        {
            auto* layer = network->addActivation(*input, nvinfer1::ActivationType::kTANH);
            if (!layer) {
                GGML_LOG_ERROR("%s: failed to create TANH layer\n", __func__);
                return nullptr;
            }
            layer->setName((std::string(name_prefix) + "_tanh").c_str());
            return layer->getOutput(0);
        }

        case GGML_UNARY_OP_SIGMOID:
        {
            auto* layer = network->addActivation(*input, nvinfer1::ActivationType::kSIGMOID);
            if (!layer) {
                GGML_LOG_ERROR("%s: failed to create SIGMOID layer\n", __func__);
                return nullptr;
            }
            layer->setName((std::string(name_prefix) + "_sigmoid").c_str());
            return layer->getOutput(0);
        }

        case GGML_UNARY_OP_SILU:
        {
            // SiLU = x * sigmoid(x)
            auto* sigmoid_layer = network->addActivation(*input, nvinfer1::ActivationType::kSIGMOID);
            if (!sigmoid_layer) {
                GGML_LOG_ERROR("%s: failed to create SIGMOID layer for SILU\n", __func__);
                return nullptr;
            }
            nvinfer1::ITensor* sigmoid_out = sigmoid_layer->getOutput(0);

            auto* mul_layer = network->addElementWise(
                *input, *sigmoid_out, nvinfer1::ElementWiseOperation::kPROD);
            if (!mul_layer) {
                GGML_LOG_ERROR("%s: failed to create MUL layer for SILU\n", __func__);
                return nullptr;
            }
            mul_layer->setName((std::string(name_prefix) + "_silu").c_str());
            return mul_layer->getOutput(0);
        }

        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
        {
            // GELU = x * 0.5 * (1 + erf(x / sqrt(2)))
            float inv_sqrt2 = 1.0f / sqrtf(2.0f);
            nvinfer1::ITensor* scale_tensor = builder->create_typed_scalar(inv_sqrt2, input);
            if (!scale_tensor) {
                GGML_LOG_ERROR("%s: failed to create scale constant for GELU\n", __func__);
                return nullptr;
            }

            auto* scale_layer = network->addElementWise(
                *input, *scale_tensor, nvinfer1::ElementWiseOperation::kPROD);
            if (!scale_layer) {
                GGML_LOG_ERROR("%s: failed to create scale layer for GELU\n", __func__);
                return nullptr;
            }
            nvinfer1::ITensor* scaled = scale_layer->getOutput(0);

            auto* erf_layer = network->addUnary(*scaled, nvinfer1::UnaryOperation::kERF);
            if (!erf_layer) {
                GGML_LOG_ERROR("%s: failed to create ERF layer for GELU\n", __func__);
                return nullptr;
            }
            nvinfer1::ITensor* erf_out = erf_layer->getOutput(0);

            nvinfer1::ITensor* one_tensor = builder->create_typed_scalar(1.0f, input);
            if (!one_tensor) {
                GGML_LOG_ERROR("%s: failed to create one constant for GELU\n", __func__);
                return nullptr;
            }

            auto* add_layer = network->addElementWise(
                *erf_out, *one_tensor, nvinfer1::ElementWiseOperation::kSUM);
            if (!add_layer) {
                GGML_LOG_ERROR("%s: failed to create add layer for GELU\n", __func__);
                return nullptr;
            }
            nvinfer1::ITensor* one_plus_erf = add_layer->getOutput(0);

            nvinfer1::ITensor* half_tensor = builder->create_typed_scalar(0.5f, input);
            if (!half_tensor) {
                GGML_LOG_ERROR("%s: failed to create half constant for GELU\n", __func__);
                return nullptr;
            }

            auto* half_layer = network->addElementWise(
                *one_plus_erf, *half_tensor, nvinfer1::ElementWiseOperation::kPROD);
            if (!half_layer) {
                GGML_LOG_ERROR("%s: failed to create half mul layer for GELU\n", __func__);
                return nullptr;
            }
            nvinfer1::ITensor* cdf = half_layer->getOutput(0);

            auto* mul_layer = network->addElementWise(
                *input, *cdf, nvinfer1::ElementWiseOperation::kPROD);
            if (!mul_layer) {
                GGML_LOG_ERROR("%s: failed to create final mul layer for GELU\n", __func__);
                return nullptr;
            }
            mul_layer->setName((std::string(name_prefix) + "_gelu").c_str());
            return mul_layer->getOutput(0);
        }

        case GGML_UNARY_OP_EXP:
        {
            auto* layer = network->addUnary(*input, nvinfer1::UnaryOperation::kEXP);
            if (!layer) {
                GGML_LOG_ERROR("%s: failed to create EXP layer\n", __func__);
                return nullptr;
            }
            layer->setName((std::string(name_prefix) + "_exp").c_str());
            return layer->getOutput(0);
        }

        default:
            GGML_LOG_ERROR("%s: unsupported activation %d\n", __func__, (int)activation);
            return nullptr;
    }
}

} // namespace ggml_tensorrt
