#pragma once

#include "common.hpp"
#include <NvInfer.h>
#include <map>
#include <string>
#include <memory>

namespace ggml_tensorrt {

// Forward declarations
class NetworkBuilder;

// Operation handler function pointer type
// Returns the output ITensor* or nullptr on failure
using OpHandler = nvinfer1::ITensor* (*)(
    NetworkBuilder* builder,
    const ggml_tensor* node
);

// NetworkBuilder manages the creation of TensorRT networks from GGML graphs
class NetworkBuilder {
public:
    NetworkBuilder(
        nvinfer1::INetworkDefinition* network,
        nvinfer1::ILogger* logger
    );

    ~NetworkBuilder();

    // Add a GGML tensor as an input to the network
    nvinfer1::ITensor* add_input(const ggml_tensor* tensor, const std::string& name);

    // Get or create a TensorRT ITensor for a GGML tensor
    // If the tensor is already converted, returns the cached version
    nvinfer1::ITensor* get_tensor(const ggml_tensor* tensor);

    // Add a TensorRT tensor to the mapping
    void set_tensor(const ggml_tensor* ggml_tensor, nvinfer1::ITensor* trt_tensor);

    // Check if a GGML tensor has been converted to TensorRT
    bool has_tensor(const ggml_tensor* tensor) const;

    // Add an operation to the network
    // Dispatches to the appropriate operation handler
    nvinfer1::ITensor* add_operation(const ggml_tensor* node);

    // Mark a tensor as a network output
    void mark_output(nvinfer1::ITensor* tensor, const std::string& name);

    // Get the underlying network definition
    nvinfer1::INetworkDefinition* get_network() { return network_; }

    // Get the logger
    nvinfer1::ILogger* get_logger() { return logger_; }

    // Check if an operation is supported
    static bool is_operation_supported(ggml_op op);

    // Register an operation handler
    static void register_op_handler(ggml_op op, OpHandler handler);

private:
    nvinfer1::INetworkDefinition* network_;
    nvinfer1::ILogger* logger_;

    // Maps GGML tensor pointers to TensorRT ITensor pointers
    std::map<const ggml_tensor*, nvinfer1::ITensor*> tensor_map_;

    // Static map of operation handlers
    static std::map<ggml_op, OpHandler> op_handlers_;
};

// Operation handler declarations
// These will be implemented in the ops/ files

// Matrix multiplication
nvinfer1::ITensor* handle_mul_mat(NetworkBuilder* builder, const ggml_tensor* node);

// Elementwise operations
nvinfer1::ITensor* handle_add(NetworkBuilder* builder, const ggml_tensor* node);
nvinfer1::ITensor* handle_mul(NetworkBuilder* builder, const ggml_tensor* node);
nvinfer1::ITensor* handle_sub(NetworkBuilder* builder, const ggml_tensor* node);
nvinfer1::ITensor* handle_div(NetworkBuilder* builder, const ggml_tensor* node);

// Normalization operations
nvinfer1::ITensor* handle_rms_norm(NetworkBuilder* builder, const ggml_tensor* node);
nvinfer1::ITensor* handle_group_norm(NetworkBuilder* builder, const ggml_tensor* node);

// Utility: Create a constant tensor
nvinfer1::ITensor* create_constant_tensor(
    nvinfer1::INetworkDefinition* network,
    const void* data,
    nvinfer1::Dims dims,
    nvinfer1::DataType dtype
);

} // namespace ggml_tensorrt
