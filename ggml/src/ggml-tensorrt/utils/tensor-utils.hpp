#pragma once

#include "../common.hpp"
#include <NvInfer.h>
#include <vector>

namespace ggml_tensorrt {

// Convert GGML tensor dimensions to TensorRT Dims
nvinfer1::Dims ggml_tensor_to_dims(const ggml_tensor* tensor);

// Convert TensorRT Dims to GGML shape array
void dims_to_ggml_shape(const nvinfer1::Dims& dims, int64_t* shape);

// Get number of elements in a tensor
int64_t get_tensor_numel(const ggml_tensor* tensor);

// Check if two tensors have compatible shapes for broadcasting
bool are_shapes_broadcastable(const ggml_tensor* a, const ggml_tensor* b);

// Compute output shape after broadcasting
nvinfer1::Dims broadcast_dims(const nvinfer1::Dims& a, const nvinfer1::Dims& b);

// Check if tensor is contiguous in memory
bool is_tensor_contiguous(const ggml_tensor* tensor);

// Get tensor strides in elements (not bytes)
std::vector<int64_t> get_tensor_strides(const ggml_tensor* tensor);

// Convert GGML tensor shape to string for logging
std::string dims_to_string(const nvinfer1::Dims& dims);

// Convert GGML tensor shape to string
std::string ggml_tensor_shape_to_string(const ggml_tensor* tensor);

// Check if reshape is valid
bool is_reshape_valid(const nvinfer1::Dims& src, const nvinfer1::Dims& dst);

// Validate tensor for TensorRT processing
bool validate_tensor_for_tensorrt(const ggml_tensor* tensor);

} // namespace ggml_tensorrt
