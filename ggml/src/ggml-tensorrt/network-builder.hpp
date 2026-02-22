#pragma once

#include "common.hpp"
#include <NvInfer.h>
#include <cstdint>
#include <map>
#include <string>
#include <memory>
#include <vector>

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

    // Add a GGML tensor as an input with dynamic shape support.
    // When is_dynamic=true, all dimensions are set to -1 (TRT wildcard).
    // TRT resolves the actual shape at runtime via setInputShape().
    nvinfer1::ITensor* add_input(const ggml_tensor* tensor, const std::string& name, bool is_dynamic);

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

    // Create a constant tensor with persistent weight storage.
    // The data is copied internally so the caller's pointer need not
    // outlive this call.
    nvinfer1::ITensor* create_constant_tensor(
        const void* data,
        nvinfer1::Dims dims,
        nvinfer1::DataType dtype
    );

    // Create a scalar constant matching the type and rank of a reference tensor.
    // Converts the float value to the appropriate format (FP32/FP16/BF16).
    // Dims are all-ones matching the reference tensor's number of dimensions.
    nvinfer1::ITensor* create_typed_scalar(
        float value,
        nvinfer1::ITensor* reference_tensor
    );

    // Insert a cast layer if the tensor type does not match the target type.
    // Returns the original tensor unchanged if types already match.
    nvinfer1::ITensor* maybe_cast(
        nvinfer1::ITensor* tensor,
        nvinfer1::DataType target_type
    );

    // Build a 1D I32 shape tensor for ISliceLayer that copies dims from the
    // input's runtime shape, with one dim overridden to a static value.
    // Used when dynamic input dims make static ISliceLayer sizes impossible.
    // Returns a shape tensor suitable for ISliceLayer::setInput(2, ...).
    nvinfer1::ITensor* make_slice_size(
        nvinfer1::ITensor* input,
        int override_dim,
        int64_t override_value
    );

    // Make reshape dimensions safe for dynamic inputs by replacing concrete
    // dims with `0` (copy from input) where the input has wildcard dims (-1).
    // If the ranks differ or a split/merge prevents direct copy, uses `-1`
    // (infer from total) for exactly one dynamic dim.
    // This is needed because ggml_tensor_to_dims always returns concrete shapes
    // (GGML computes shapes eagerly), but TRT needs special values when the
    // input tensor has wildcard dimensions.
    static nvinfer1::Dims make_dynamic_reshape_dims(
        nvinfer1::ITensor* input,
        nvinfer1::Dims target_dims
    );

    // Pad a tensor with leading 1-dims so it has target_ndims dimensions.
    // E.g. [N, K] with target_ndims=3 becomes [1, N, K].
    // Returns the tensor unchanged if it already has enough dimensions.
    nvinfer1::ITensor* pad_to_ndims(
        nvinfer1::ITensor* tensor,
        int target_ndims
    );

private:
    nvinfer1::INetworkDefinition* network_;
    nvinfer1::ILogger* logger_;

    // Maps GGML tensor pointers to TensorRT ITensor pointers
    std::map<const ggml_tensor*, nvinfer1::ITensor*> tensor_map_;

    // Static map of operation handlers
    static std::map<ggml_op, OpHandler> op_handlers_;

    // Persistent storage for weight data passed to TensorRT constant layers.
    // TensorRT does not copy weight data — pointers must remain valid for the
    // lifetime of the INetworkDefinition.
    std::vector<std::vector<uint8_t>> weight_storage_;
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

// Shape operations
nvinfer1::ITensor* handle_reshape(NetworkBuilder* builder, const ggml_tensor* node);
nvinfer1::ITensor* handle_permute(NetworkBuilder* builder, const ggml_tensor* node);
nvinfer1::ITensor* handle_transpose(NetworkBuilder* builder, const ggml_tensor* node);
nvinfer1::ITensor* handle_view(NetworkBuilder* builder, const ggml_tensor* node);

// Unary activation operations
nvinfer1::ITensor* handle_unary(NetworkBuilder* builder, const ggml_tensor* node);

// Softmax
nvinfer1::ITensor* handle_soft_max(NetworkBuilder* builder, const ggml_tensor* node);

// Scale (multiply + optional bias)
nvinfer1::ITensor* handle_scale(NetworkBuilder* builder, const ggml_tensor* node);

// Gather rows (embedding lookup)
nvinfer1::ITensor* handle_get_rows(NetworkBuilder* builder, const ggml_tensor* node);

// GLU (gated linear unit) operations
nvinfer1::ITensor* handle_glu(NetworkBuilder* builder, const ggml_tensor* node);

// Rotary position embedding (ROPE) — decomposed path
nvinfer1::ITensor* handle_rope(NetworkBuilder* builder, const ggml_tensor* node);

// Rotary position embedding (ROPE) — native IRotaryEmbeddingLayer path
nvinfer1::ITensor* handle_rope_native(NetworkBuilder* builder, const ggml_tensor* node);

// SET_ROWS → IKVCacheUpdateLayer (native attention path)
nvinfer1::ITensor* handle_set_rows(NetworkBuilder* builder, const ggml_tensor* node);

// FLASH_ATTN_EXT → IAttention (native attention path)
nvinfer1::ITensor* handle_flash_attn_ext(NetworkBuilder* builder, const ggml_tensor* node);

// Shared activation helper — applies an activation function to a TRT tensor.
// Used by both unary ops and GLU.  Returns nullptr on failure.
// Supported activations: SILU, GELU, GELU_ERF, RELU, TANH, SIGMOID, EXP, GELU_QUICK
nvinfer1::ITensor* apply_activation(
    nvinfer1::INetworkDefinition* network,
    NetworkBuilder* builder,
    nvinfer1::ITensor* input,
    enum ggml_unary_op activation,
    const char* name_prefix
);

} // namespace ggml_tensorrt
