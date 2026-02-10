#include "tensor-utils.hpp"
#include "type-utils.hpp"
#include "ggml-impl.h"

#include <algorithm>
#include <sstream>
#include <cstring>

namespace ggml_tensorrt {

nvinfer1::Dims ggml_tensor_to_dims(const ggml_tensor* tensor) {
    GGML_ASSERT(tensor != nullptr);

    nvinfer1::Dims dims;
    dims.nbDims = 0;

    // GGML tensors can have up to GGML_MAX_DIMS dimensions
    // TensorRT supports up to 8 dimensions
    // We need to skip dimensions of size 1 at the end (trailing)
    // but keep them in the middle

    int n_dims = ggml_n_dims(tensor);
    GGML_ASSERT(n_dims <= nvinfer1::Dims::MAX_DIMS);

    // Convert dimensions from GGML to TensorRT
    // GGML dimensions are stored in reverse order:
    // GGML: [ne0, ne1, ne2, ne3] where ne0=cols, ne1=rows (fastest to slowest)
    // TensorRT expects: [rows, cols, ...] (standard layout)
    // Therefore we must REVERSE the dimension order
    for (int i = 0; i < n_dims; ++i) {
        dims.d[i] = tensor->ne[n_dims - 1 - i];
    }
    dims.nbDims = n_dims;

    return dims;
}

void dims_to_ggml_shape(const nvinfer1::Dims& dims, int64_t* shape) {
    GGML_ASSERT(shape != nullptr);
    GGML_ASSERT(dims.nbDims >= 0 && dims.nbDims <= nvinfer1::Dims::MAX_DIMS);

    // Reverse dimensions back to GGML convention
    for (int i = 0; i < dims.nbDims; ++i) {
        shape[i] = dims.d[dims.nbDims - 1 - i];
    }
}

int64_t get_tensor_numel(const ggml_tensor* tensor) {
    GGML_ASSERT(tensor != nullptr);
    return ggml_nelements(tensor);
}

bool are_shapes_broadcastable(const ggml_tensor* a, const ggml_tensor* b) {
    GGML_ASSERT(a != nullptr && b != nullptr);

    int n_dims_a = ggml_n_dims(a);
    int n_dims_b = ggml_n_dims(b);

    // Broadcasting rules: dimensions are compatible if:
    // 1. They are equal, or
    // 2. One of them is 1

    int max_dims = std::max(n_dims_a, n_dims_b);

    for (int i = 0; i < max_dims; ++i) {
        int64_t dim_a = (i < n_dims_a) ? a->ne[i] : 1;
        int64_t dim_b = (i < n_dims_b) ? b->ne[i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }

    return true;
}

nvinfer1::Dims broadcast_dims(const nvinfer1::Dims& a, const nvinfer1::Dims& b) {
    nvinfer1::Dims result;
    result.nbDims = std::max(a.nbDims, b.nbDims);

    // Right-align dimensions (NumPy/TRT broadcast rules):
    // [n_tokens, 1536] vs [1536] aligns as:
    //   [n_tokens, 1536]
    //   [       1, 1536]  ← padded with leading 1
    for (int i = 0; i < result.nbDims; ++i) {
        int idx_a = i - (result.nbDims - a.nbDims);
        int idx_b = i - (result.nbDims - b.nbDims);

        int64_t dim_a = (idx_a >= 0) ? a.d[idx_a] : 1;
        int64_t dim_b = (idx_b >= 0) ? b.d[idx_b] : 1;

        GGML_ASSERT(dim_a == dim_b || dim_a == 1 || dim_b == 1);
        result.d[i] = std::max(dim_a, dim_b);
    }

    return result;
}

bool is_tensor_contiguous(const ggml_tensor* tensor) {
    GGML_ASSERT(tensor != nullptr);
    return ggml_is_contiguous(tensor);
}

std::vector<int64_t> get_tensor_strides(const ggml_tensor* tensor) {
    GGML_ASSERT(tensor != nullptr);

    std::vector<int64_t> strides;
    int n_dims = ggml_n_dims(tensor);

    for (int i = 0; i < n_dims; ++i) {
        // GGML stores strides in bytes (nb), convert to element count
        size_t type_size = ggml_type_size(tensor->type);
        GGML_ASSERT(type_size > 0);
        strides.push_back(tensor->nb[i] / type_size);
    }

    return strides;
}

std::string dims_to_string(const nvinfer1::Dims& dims) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) oss << ", ";
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

std::string ggml_tensor_shape_to_string(const ggml_tensor* tensor) {
    GGML_ASSERT(tensor != nullptr);

    std::ostringstream oss;
    oss << "[";
    int n_dims = ggml_n_dims(tensor);
    for (int i = 0; i < n_dims; ++i) {
        if (i > 0) oss << ", ";
        oss << tensor->ne[i];
    }
    oss << "]";
    return oss.str();
}

bool is_reshape_valid(const nvinfer1::Dims& src, const nvinfer1::Dims& dst) {
    // Calculate total number of elements in both shapes
    int64_t src_numel = 1;
    for (int i = 0; i < src.nbDims; ++i) {
        src_numel *= src.d[i];
    }

    int64_t dst_numel = 1;
    for (int i = 0; i < dst.nbDims; ++i) {
        dst_numel *= dst.d[i];
    }

    // Reshape is valid if total number of elements is the same
    return src_numel == dst_numel;
}

bool validate_tensor_for_tensorrt(const ggml_tensor* tensor) {
    if (tensor == nullptr) {
        return false;
    }

    // Check if type is supported
    if (!is_type_supported(tensor->type)) {
        return false;
    }

    // Check dimension count
    int n_dims = ggml_n_dims(tensor);
    if (n_dims > nvinfer1::Dims::MAX_DIMS) {
        return false;
    }

    // Check for valid dimensions (no zero or negative sizes)
    for (int i = 0; i < n_dims; ++i) {
        if (tensor->ne[i] <= 0) {
            return false;
        }
    }

    return true;
}

} // namespace ggml_tensorrt
