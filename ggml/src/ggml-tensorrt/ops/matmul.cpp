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
    // A has shape [K, M, ...]
    // B has shape [K, N, ...]
    // C has shape [N, M, ...] ← NOTE: Output is [N, M], not [M, N]!
    //
    // In GGML:
    // - src0->ne[0] = K, src0->ne[1] = M
    // - src1->ne[0] = K, src1->ne[1] = N
    // - dst->ne[0] = N, dst->ne[1] = M  ← Swapped compared to mathematical notation!
    //
    // This is GGML's convention for matrix multiply

    // Get dimensions
    nvinfer1::Dims dims0 = trt_src0->getDimensions();
    nvinfer1::Dims dims1 = trt_src1->getDimensions();

    GGML_LOG_DEBUG("%s: src0 shape %s, src1 shape %s\n",
        __func__,
        dims_to_string(dims0).c_str(),
        dims_to_string(dims1).c_str());

    // For basic 2D matrix multiplication:
    // GGML stores matrices as [K, M] and [K, N]
    // GGML's mul_mat(src0, src1) produces output with shape [N, M] (note the order!)
    //
    // To achieve this with TensorRT:
    // We compute: transpose(src1) @ src0
    // - src1 [K, N] transposed becomes [N, K]
    // - src0 [K, M] stays as [K, M]
    // - [N, K] @ [K, M] = [N, M] ✓

    // Add MatrixMultiply layer
    // Note: swapped input order and transpose operations for GGML semantics
    auto* layer = network->addMatrixMultiply(
        *trt_src1,                             // Use src1 first (swapped!)
        nvinfer1::MatrixOperation::kTRANSPOSE, // Transpose src1: [K,N] -> [N,K]
        *trt_src0,                             // Use src0 second (swapped!)
        nvinfer1::MatrixOperation::kNONE       // Keep src0 as [K,M]
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
