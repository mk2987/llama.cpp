#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "../utils/type-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>

namespace ggml_tensorrt {

// Handle matrix multiplication (MUL_MAT)
// GGML MUL_MAT: C = A @ B
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

    // GGML MUL_MAT operation: C = A @ B
    // A has shape [K, M, ...] (note: GGML stores dimensions in reverse)
    // B has shape [K, N, ...]
    // C has shape [M, N, ...]
    //
    // In GGML:
    // - src0->ne[0] = K, src0->ne[1] = M
    // - src1->ne[0] = K, src1->ne[1] = N
    // - dst->ne[0] = M, dst->ne[1] = N
    //
    // TensorRT MatrixMultiply expects:
    // - Input0: [..., K, M] or [..., M, K] with transpose
    // - Input1: [..., K, N] or [..., N, K] with transpose
    // - Output: [..., M, N]

    // Get dimensions
    nvinfer1::Dims dims0 = trt_src0->getDimensions();
    nvinfer1::Dims dims1 = trt_src1->getDimensions();

    GGML_LOG_DEBUG("%s: src0 shape %s, src1 shape %s\n",
        __func__,
        dims_to_string(dims0).c_str(),
        dims_to_string(dims1).c_str());

    // For basic 2D matrix multiplication:
    // GGML stores matrices as [K, M] and [K, N]
    // We need to transpose the first matrix to get [M, K]
    // Then multiply [M, K] @ [K, N] = [M, N]

    // Add MatrixMultiply layer
    // op0 controls transposition of first input
    // op1 controls transposition of second input
    auto* layer = network->addMatrixMultiply(
        *trt_src0,
        nvinfer1::MatrixOperation::kTRANSPOSE, // Transpose first matrix
        *trt_src1,
        nvinfer1::MatrixOperation::kNONE       // Don't transpose second matrix
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
