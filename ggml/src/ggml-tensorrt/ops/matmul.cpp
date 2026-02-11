#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "../utils/type-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>

namespace ggml_tensorrt {

// Handle matrix multiplication (MUL_MAT)
// GGML MUL_MAT: C = B @ A^T
// Where A is src0 and B is src1
nvinfer1::ITensor* handle_mul_mat(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_MUL_MAT);

    // Get input tensors
    const ggml_tensor* src0 = node->src[0]; // A (left matrix)
    const ggml_tensor* src1 = node->src[1]; // B (right matrix)

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);

    // Get or create TensorRT tensors for inputs
    nvinfer1::ITensor* trt_src0 = builder->get_tensor(src0);
    nvinfer1::ITensor* trt_src1 = builder->get_tensor(src1);

    if (trt_src0 == nullptr || trt_src1 == nullptr) {
        GGML_LOG_ERROR("%s: input tensors not found in network\n", __func__);
        return nullptr;
    }

    // Get network
    auto* network = builder->get_network();

    // TRT strongly-typed mode requires both matmul inputs to have the
    // same type.  In real models the weights may be F16/BF16 while the
    // activation (from a previous F32-output MUL_MAT) is F32.
    // Cast the narrower type up to the wider one.
    nvinfer1::DataType type0 = trt_src0->getType();
    nvinfer1::DataType type1 = trt_src1->getType();
    if (type0 != type1) {
        // Promote to the wider type: F32 > BF16 > F16
        nvinfer1::DataType common = nvinfer1::DataType::kFLOAT;
        if (type0 != nvinfer1::DataType::kFLOAT && type1 != nvinfer1::DataType::kFLOAT) {
            // Neither is F32 — pick BF16 if either is BF16, else stay with what we have
            common = (type0 == nvinfer1::DataType::kBF16 || type1 == nvinfer1::DataType::kBF16)
                   ? nvinfer1::DataType::kBF16
                   : type0;
        }
        trt_src0 = builder->maybe_cast(trt_src0, common);
        trt_src1 = builder->maybe_cast(trt_src1, common);
    }

    // GGML MUL_MAT operation: C = A @ B
    // A has shape [K, N, ...] (K is the innermost dimension)
    // B has shape [K, M, ...] (Needs to be transposed by TensorRT)
    // C has shape [N, M, ...]
    //
    // In GGML:
    // - src0->ne[0] = K, src0->ne[1] = N
    // - src1->ne[0] = K, src1->ne[1] = M
    // - dst->ne[0] = N, dst->ne[1] = M
    //
    // This is GGML's convention for matrix multiply

    // Get dimensions
    nvinfer1::Dims dims0 = trt_src0->getDimensions();
    nvinfer1::Dims dims1 = trt_src1->getDimensions();

    GGML_LOG_DEBUG("%s: src0 shape %s, src1 shape %s\n",
        __func__,
        dims_to_string(dims0).c_str(),
        dims_to_string(dims1).c_str());

    // TRT-RTX requires both matmul inputs to have the same number of
    // dimensions.  In real models, weights are typically 2D while
    // activations carry a batch dimension (3D+).  Pad the lower-rank
    // tensor with leading 1-dims to match.
    if (dims0.nbDims != dims1.nbDims) {
        int max_ndims = std::max(dims0.nbDims, dims1.nbDims);
        trt_src0 = builder->pad_to_ndims(trt_src0, max_ndims);
        trt_src1 = builder->pad_to_ndims(trt_src1, max_ndims);

        if (trt_src0 == nullptr || trt_src1 == nullptr) {
            GGML_LOG_ERROR("%s: failed to pad matmul inputs to matching rank\n", __func__);
            return nullptr;
        }
    }

    // For basic 2D matrix multiplication:
    // GGML specification: A @ B^T (B is transposed internally)
    //
    // From ggml.h:
    // - A: k columns, n rows, stored as [k, n] → represents m×k matrix
    // - B: k columns, m rows, stored as [k, m] → represents n×k matrix
    // - Operation: B @ A^T = (m×k) @ (k×n) = m×n

    // Add MatrixMultiply layer
    // Compute: B @ A^T = transpose(src1) @ src0
    auto* layer = network->addMatrixMultiply(
        *trt_src1,                             // B: [K, M, ...]
        nvinfer1::MatrixOperation::kNONE,
        *trt_src0,                             // B: [K, N, ...]
        nvinfer1::MatrixOperation::kTRANSPOSE  // Transpose to [N, K, ...]
    );

    if (layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create MatrixMultiply layer\n", __func__);
        return nullptr;
    }

    // Set layer name for debugging
    std::string layer_name = "mul_mat_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    layer->setName(layer_name.c_str());

    // Get output
    nvinfer1::ITensor* output = layer->getOutput(0);
    if (output == nullptr) {
        GGML_LOG_ERROR("%s: failed to get output from MatrixMultiply layer\n", __func__);
        return nullptr;
    }

    GGML_LOG_DEBUG("%s: created MatrixMultiply layer, output shape %s\n",
        __func__,
        dims_to_string(output->getDimensions()).c_str());

    return output;
}

// Register the operation handler
static void __attribute__((constructor)) register_mul_mat_handler() {
    NetworkBuilder::register_op_handler(GGML_OP_MUL_MAT, handle_mul_mat);
}

} // namespace ggml_tensorrt
