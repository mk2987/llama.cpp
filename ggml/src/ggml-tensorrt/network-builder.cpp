#include "network-builder.hpp"
#include "utils/tensor-utils.hpp"
#include "utils/type-utils.hpp"
#include "ggml-impl.h"

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

// Utility: Create a constant tensor
nvinfer1::ITensor* create_constant_tensor(
    nvinfer1::INetworkDefinition* network,
    const void* data,
    nvinfer1::Dims dims,
    nvinfer1::DataType dtype
) {
    GGML_ASSERT(network != nullptr);
    GGML_ASSERT(data != nullptr);

    // Calculate total number of elements
    int64_t numel = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        numel *= dims.d[i];
    }

    // Create weights
    nvinfer1::Weights weights{dtype, data, numel};

    // Add constant layer
    nvinfer1::IConstantLayer* layer = network->addConstant(dims, weights);
    if (layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create constant layer\n", __func__);
        return nullptr;
    }

    return layer->getOutput(0);
}

} // namespace ggml_tensorrt
