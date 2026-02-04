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
    // B has shape [K, N, ...] ← NOTE: Shape is [K, N], not [N, K]!
    // C has shape [N, M, ...]
    //
    // In GGML:
    // - src0->ne[0] = K, src0->ne[1] = M
    // - src1->ne[0] = K, src1->ne[1] = N
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

    // For basic 2D matrix multiplication:
    // GGML specification: A @ B^T (B is transposed internally)
    //
    // From ggml.h:
    // - A: k columns, n rows, stored as [k, m] → represents m×k matrix
    // - B: k columns, m rows, stored as [k, n] → represents n×k matrix
    // - Operation: A @ B^T = (m×k) @ (k×n) = m×n
    //
    // TensorRT interprets [k, n] as a k×n matrix, so:
    // - trt_src0 [k, n] is k×n in TensorRT, but represents n×k in GGML
    // - trt_src1 [k, m] is k×m in TensorRT, but represents m×k in GGML
    //
    // To compute A @ B^T with GGML semantics:
    // - A (n×k) = transpose(trt_src0) where trt_src0 is (k×n)
    // - B^T (k×m) = trt_src1 as-is (k×m)
    // - A @ B^T = transpose(trt_src0) @ trt_src1 = (n×k) @ (k×m) = (n×m)
    //
    // TensorRT will output (n×m) which needs to be stored as [m, n] in GGML.
    // But TensorRT outputs [n, m], so we need to transpose the result!
    //
    // Using (A @ B)^T = B^T @ A^T, we can compute (A @ B^T)^T = B @ A^T:
    // - B @ A^T = transpose(trt_src1) @ trt_src0 = (m×k) @ (k×n) = (m×n)
    // - TensorRT outputs [m, n] ✓ This is what GGML expects!

    // Add MatrixMultiply layer
    // Compute: B @ A^T = transpose(src1) @ src0
    auto* layer = network->addMatrixMultiply(
        *trt_src1,                             // B: [k, m]
        nvinfer1::MatrixOperation::kTRANSPOSE, // Transpose to [m, k]
        *trt_src0,                             // A: [k, n] (used as A^T in math)
        nvinfer1::MatrixOperation::kNONE       // Keep as [k, n]
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
