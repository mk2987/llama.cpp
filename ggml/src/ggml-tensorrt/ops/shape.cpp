#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>
#include <algorithm>
#include <cstring>

namespace ggml_tensorrt {

// Handle RESHAPE: use IShuffleLayer to change tensor dimensions
nvinfer1::ITensor* handle_reshape(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_RESHAPE);

    const ggml_tensor* src = node->src[0];
    GGML_ASSERT(src != nullptr);

    nvinfer1::ITensor* trt_src = builder->get_tensor(src);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    // Target dimensions from the output tensor shape (always concrete from GGML)
    nvinfer1::Dims target_dims = ggml_tensor_to_dims(node);

    // Make reshape dims safe for dynamic inputs (replace with 0/-1 as needed)
    nvinfer1::Dims safe_dims = NetworkBuilder::make_dynamic_reshape_dims(trt_src, target_dims);

    // Validate element count — skip if any dim is dynamic (0 or -1)
    nvinfer1::Dims src_dims = trt_src->getDimensions();
    {
        bool all_static = true;
        for (int i = 0; i < src_dims.nbDims; i++) {
            if (src_dims.d[i] == -1) { all_static = false; break; }
        }
        for (int i = 0; i < safe_dims.nbDims; i++) {
            if (safe_dims.d[i] <= 0) { all_static = false; break; }
        }
        if (all_static && !is_reshape_valid(src_dims, safe_dims)) {
            GGML_LOG_ERROR("%s: reshape element count mismatch: %s -> %s\n",
                __func__, dims_to_string(src_dims).c_str(),
                dims_to_string(safe_dims).c_str());
            return nullptr;
        }
    }

    auto* network = builder->get_network();
    auto* shuffle = network->addShuffle(*trt_src);
    if (shuffle == nullptr) {
        GGML_LOG_ERROR("%s: failed to create shuffle layer\n", __func__);
        return nullptr;
    }

    shuffle->setReshapeDimensions(safe_dims);

    std::string layer_name = "reshape_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    shuffle->setName(layer_name.c_str());

    nvinfer1::ITensor* output = shuffle->getOutput(0);

    GGML_LOG_DEBUG("%s: reshape %s -> %s\n",
        __func__, dims_to_string(src_dims).c_str(),
        dims_to_string(target_dims).c_str());

    return output;
}

// Handle PERMUTE: use IShuffleLayer with setSecondTranspose
nvinfer1::ITensor* handle_permute(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_PERMUTE);

    const ggml_tensor* src = node->src[0];
    GGML_ASSERT(src != nullptr);

    nvinfer1::ITensor* trt_src = builder->get_tensor(src);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    // Read GGML permutation axes from op_params
    const int32_t* axes = (const int32_t*)node->op_params;
    int ggml_axes[4] = { axes[0] & 0x3, axes[1] & 0x3, axes[2] & 0x3, axes[3] & 0x3 };

    // Use max of source and output n_dims.  A permute like [0,2,1,3] on a
    // 2D source (dims 2,3 are size 1) references axis 2, which is beyond
    // ggml_n_dims(src)=2.  Using only the source n_dims produces negative
    // TRT permutation indices → crash.  The output n_dims accounts for all
    // non-trivial dimensions after the permutation.
    int n_dims = std::max(ggml_n_dims(src), ggml_n_dims(node));

    // Pad the TRT input tensor to n_dims if needed (adds leading 1-dims)
    trt_src = builder->pad_to_ndims(trt_src, n_dims);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: failed to pad input to %d dims\n", __func__, n_dims);
        return nullptr;
    }

    // Convert GGML permutation (innermost-first) to TRT permutation (outermost-first)
    // GGML dim i -> TRT dim (n_dims - 1 - i)
    // If GGML says axis_j = ggml_axes[j], the TRT equivalent is:
    //   trt_perm[n_dims - 1 - j] = n_dims - 1 - ggml_axes[j]
    nvinfer1::Permutation trt_perm;
    for (int j = 0; j < n_dims; j++) {
        trt_perm.order[n_dims - 1 - j] = n_dims - 1 - ggml_axes[j];
    }
    // Fill remaining dims with identity
    for (int j = n_dims; j < nvinfer1::Dims::MAX_DIMS; j++) {
        trt_perm.order[j] = j;
    }

    auto* network = builder->get_network();
    auto* shuffle = network->addShuffle(*trt_src);
    if (shuffle == nullptr) {
        GGML_LOG_ERROR("%s: failed to create shuffle layer\n", __func__);
        return nullptr;
    }

    shuffle->setSecondTranspose(trt_perm);

    std::string layer_name = "permute_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    shuffle->setName(layer_name.c_str());

    nvinfer1::ITensor* output = shuffle->getOutput(0);

    GGML_LOG_DEBUG("%s: permuted with axes [%d,%d,%d,%d]\n",
        __func__, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]);

    return output;
}

// Handle TRANSPOSE: swap dims 0 and 1 (GGML convention)
// This is equivalent to permute(1, 0, 2, 3)
nvinfer1::ITensor* handle_transpose(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_TRANSPOSE);

    const ggml_tensor* src = node->src[0];
    GGML_ASSERT(src != nullptr);

    nvinfer1::ITensor* trt_src = builder->get_tensor(src);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    nvinfer1::Dims dims = trt_src->getDimensions();

    // GGML transpose swaps dims 0 and 1 (innermost two)
    // In TRT, that means swapping the last two dims
    nvinfer1::Permutation trt_perm;
    for (int i = 0; i < dims.nbDims; i++) {
        trt_perm.order[i] = i;
    }
    // Swap last two
    if (dims.nbDims >= 2) {
        trt_perm.order[dims.nbDims - 1] = dims.nbDims - 2;
        trt_perm.order[dims.nbDims - 2] = dims.nbDims - 1;
    }
    // Fill remaining
    for (int i = dims.nbDims; i < nvinfer1::Dims::MAX_DIMS; i++) {
        trt_perm.order[i] = i;
    }

    auto* network = builder->get_network();
    auto* shuffle = network->addShuffle(*trt_src);
    if (shuffle == nullptr) {
        GGML_LOG_ERROR("%s: failed to create shuffle layer\n", __func__);
        return nullptr;
    }

    shuffle->setSecondTranspose(trt_perm);

    std::string layer_name = "transpose_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    shuffle->setName(layer_name.c_str());

    nvinfer1::ITensor* output = shuffle->getOutput(0);

    GGML_LOG_DEBUG("%s: transposed last two dims of %s\n",
        __func__, dims_to_string(dims).c_str());

    return output;
}

// Handle VIEW: reshape source tensor to match view output dimensions.
// A VIEW in GGML reinterprets the data layout (e.g. splitting heads from
// a flat projection).  In TRT this is a zero-copy reshape via IShuffleLayer.
nvinfer1::ITensor* handle_view(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_VIEW);

    const ggml_tensor* src = node->src[0];
    GGML_ASSERT(src != nullptr);

    nvinfer1::ITensor* trt_src = builder->get_tensor(src);
    if (trt_src == nullptr) {
        GGML_LOG_ERROR("%s: input tensor not found in network\n", __func__);
        return nullptr;
    }

    // Target dimensions from the VIEW output shape (always concrete from GGML)
    nvinfer1::Dims target_dims = ggml_tensor_to_dims(node);
    nvinfer1::Dims src_dims = trt_src->getDimensions();

    // Make reshape dims safe for dynamic inputs
    nvinfer1::Dims safe_dims = NetworkBuilder::make_dynamic_reshape_dims(trt_src, target_dims);

    // If dims already match (accounting for 0 = copy-through), just propagate
    bool dims_match = (src_dims.nbDims == safe_dims.nbDims);
    if (dims_match) {
        bool all_equal = true;
        for (int i = 0; i < src_dims.nbDims; i++) {
            // 0 means "copy from input" — always matches
            if (safe_dims.d[i] == 0) continue;
            if (src_dims.d[i] != safe_dims.d[i]) { all_equal = false; break; }
        }
        if (all_equal) {
            return trt_src;
        }
    }

    // Validate element count — skip if any dim is dynamic
    {
        bool all_static = true;
        for (int i = 0; i < src_dims.nbDims; i++) {
            if (src_dims.d[i] == -1) { all_static = false; break; }
        }
        for (int i = 0; i < safe_dims.nbDims; i++) {
            if (safe_dims.d[i] <= 0) { all_static = false; break; }
        }
        if (all_static && !is_reshape_valid(src_dims, safe_dims)) {
            GGML_LOG_ERROR("%s: view element count mismatch: %s -> %s (partial view not supported in TRT)\n",
                __func__, dims_to_string(src_dims).c_str(),
                dims_to_string(safe_dims).c_str());
            return nullptr;
        }
    }

    auto* network = builder->get_network();
    auto* shuffle = network->addShuffle(*trt_src);
    if (shuffle == nullptr) {
        GGML_LOG_ERROR("%s: failed to create shuffle layer for view\n", __func__);
        return nullptr;
    }

    shuffle->setReshapeDimensions(safe_dims);

    std::string layer_name = "view_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    shuffle->setName(layer_name.c_str());

    nvinfer1::ITensor* output = shuffle->getOutput(0);

    GGML_LOG_DEBUG("%s: view reshape %s -> %s\n",
        __func__, dims_to_string(src_dims).c_str(),
        dims_to_string(target_dims).c_str());

    return output;
}

// Register shape operation handlers
static void __attribute__((constructor)) register_shape_handlers() {
    NetworkBuilder::register_op_handler(GGML_OP_RESHAPE,   handle_reshape);
    NetworkBuilder::register_op_handler(GGML_OP_PERMUTE,   handle_permute);
    NetworkBuilder::register_op_handler(GGML_OP_TRANSPOSE, handle_transpose);
    NetworkBuilder::register_op_handler(GGML_OP_VIEW,      handle_view);
}

} // namespace ggml_tensorrt
