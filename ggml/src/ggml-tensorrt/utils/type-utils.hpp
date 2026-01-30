#pragma once

#include "../common.hpp"
#include <NvInferRuntime.h>
#include <cstddef>

namespace ggml_tensorrt {

// Convert GGML type to TensorRT DataType
nvinfer1::DataType ggml_type_to_tensorrt(ggml_type type);

// Check if GGML type is supported by TensorRT-RTX backend
bool is_type_supported(ggml_type type);

// Check if type is a quantized type that needs special handling
bool is_quantized_type(ggml_type type);

// Get the base compute type for a quantized GGML type
// (e.g., Q4_0 would use FP16 for computation)
nvinfer1::DataType get_compute_type_for_quantized(ggml_type type);

// Get type name for logging
const char* get_type_name(ggml_type type);

// Get element size in bytes
size_t get_type_size(ggml_type type);

} // namespace ggml_tensorrt
