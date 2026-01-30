#include "type-utils.hpp"
#include "ggml-impl.h"

#include <NvInferRuntime.h>

namespace ggml_tensorrt {

nvinfer1::DataType ggml_type_to_tensorrt(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return nvinfer1::DataType::kFLOAT;
        case GGML_TYPE_F16:
            return nvinfer1::DataType::kHALF;
        case GGML_TYPE_BF16:
            return nvinfer1::DataType::kBF16;
        case GGML_TYPE_I8:
            return nvinfer1::DataType::kINT8;
        case GGML_TYPE_I32:
            return nvinfer1::DataType::kINT32;
        case GGML_TYPE_I64:
            return nvinfer1::DataType::kINT64;

        // Quantized types - use FP16 as compute type for now
        // Dequantization will happen before TensorRT operations
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_MXFP4:
            return nvinfer1::DataType::kHALF;

        default:
            GGML_ABORT("Unsupported GGML type for TensorRT: %d", type);
    }
}

bool is_type_supported(ggml_type type) {
    switch (type) {
        // Native TensorRT types
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I8:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
            return true;

        // Quantized types - supported with dequantization
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_MXFP4:
            return true;

        // Unsupported types
        case GGML_TYPE_I16:
        case GGML_TYPE_F64:
        default:
            return false;
    }
}

bool is_quantized_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

nvinfer1::DataType get_compute_type_for_quantized(ggml_type type) {
    // For quantized types, we'll use FP16 as the compute type
    // This provides a good balance between performance and accuracy
    if (is_quantized_type(type)) {
        return nvinfer1::DataType::kHALF;
    }
    // For non-quantized types, return the native type
    return ggml_type_to_tensorrt(type);
}

const char* get_type_name(ggml_type type) {
    return ggml_type_name(type);
}

size_t get_type_size(ggml_type type) {
    return ggml_type_size(type);
}

} // namespace ggml_tensorrt
