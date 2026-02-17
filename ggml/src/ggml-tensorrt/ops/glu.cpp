#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>

namespace ggml_tensorrt {

// Map ggml_glu_op to the ggml_unary_op used for the activation half.
// Returns the activation to apply to the "data" half before multiplying
// with the "gate" half.
static enum ggml_unary_op glu_op_to_activation(enum ggml_glu_op glu_op) {
    switch (glu_op) {
        case GGML_GLU_OP_SWIGLU:      return GGML_UNARY_OP_SILU;
        case GGML_GLU_OP_GEGLU:       return GGML_UNARY_OP_GELU;
        case GGML_GLU_OP_GEGLU_ERF:   return GGML_UNARY_OP_GELU_ERF;
        case GGML_GLU_OP_REGLU:       return GGML_UNARY_OP_RELU;
        case GGML_GLU_OP_GEGLU_QUICK: return GGML_UNARY_OP_SIGMOID;  // placeholder, handled specially
        default:
            GGML_LOG_ERROR("%s: unsupported GLU op %d\n", __func__, (int)glu_op);
            return GGML_UNARY_OP_SILU; // fallback
    }
}

// Handle GGML_OP_GLU — Gated Linear Unit
//
// Single-input form (no src[1]):
//   input shape: [n, rows, ...] where n = 2 * nc
//   output shape: [nc, rows, ...]
//   data = input[:nc] or input[nc:] depending on 'swapped'
//   gate = input[nc:] or input[:nc]
//   output = activation(data) * gate
//
// Split form (src[1] present):
//   src[0] = data, src[1] = gate, both [nc, rows, ...]
//   output = activation(data) * gate
nvinfer1::ITensor* handle_glu(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_GLU);

    const ggml_tensor* src0 = node->src[0];
    const ggml_tensor* src1 = node->src[1];
    GGML_ASSERT(src0 != nullptr);

    nvinfer1::ITensor* trt_src0 = builder->get_tensor(src0);
    if (trt_src0 == nullptr) {
        GGML_LOG_ERROR("%s: src0 tensor not found in network\n", __func__);
        return nullptr;
    }

    auto* network = builder->get_network();

    enum ggml_glu_op glu_op = ggml_get_glu_op(node);
    const int32_t swapped = ggml_get_op_params_i32(node, 1);

    nvinfer1::ITensor* data_tensor = nullptr;
    nvinfer1::ITensor* gate_tensor = nullptr;

    if (src1 != nullptr) {
        // Split form: src[0] = data, src[1] = gate
        nvinfer1::ITensor* trt_src1 = builder->get_tensor(src1);
        if (trt_src1 == nullptr) {
            GGML_LOG_ERROR("%s: src1 tensor not found in network\n", __func__);
            return nullptr;
        }
        data_tensor = trt_src0;
        gate_tensor = trt_src1;
    } else {
        // Single-input form: split along dimension 0 (innermost, ne[0] in GGML = last dim in TRT)
        // GGML layout: ne[0] is the contiguous (innermost) dimension.
        // TRT layout: dims are stored in GGML order (ne[0] first) by ggml_tensor_to_dims,
        //             so the split is along the last TRT dimension.
        nvinfer1::Dims src_dims = trt_src0->getDimensions();
        int split_axis = src_dims.nbDims - 1;  // last dimension = ne[0] in GGML
        // Use GGML shape for nc (always concrete, even for dynamic inputs)
        int64_t nc = src0->ne[0] / 2;

        // Build start/size/stride for ISliceLayer
        nvinfer1::Dims start, size, stride;
        start.nbDims = src_dims.nbDims;
        size.nbDims = src_dims.nbDims;
        stride.nbDims = src_dims.nbDims;
        for (int i = 0; i < src_dims.nbDims; i++) {
            start.d[i] = 0;
            size.d[i] = 1;  // placeholder for shape tensor
            stride.d[i] = 1;
        }
        size.d[split_axis] = nc;

        // Shape tensor: copies dynamic dims from input, overrides split axis = nc
        nvinfer1::ITensor* half_size_tensor = builder->make_slice_size(
            trt_src0, split_axis, nc);
        if (half_size_tensor == nullptr) {
            GGML_LOG_ERROR("%s: failed to create half size tensor\n", __func__);
            return nullptr;
        }

        // First half: start at 0
        nvinfer1::Dims start_first = start;
        auto* slice_first = network->addSlice(*trt_src0, start_first, size, stride);
        if (slice_first == nullptr) {
            GGML_LOG_ERROR("%s: failed to create first half slice\n", __func__);
            return nullptr;
        }
        slice_first->setInput(2, *half_size_tensor);
        nvinfer1::ITensor* first_half = slice_first->getOutput(0);

        // Second half: start at nc along split axis
        nvinfer1::Dims start_second = start;
        start_second.d[split_axis] = nc;
        auto* slice_second = network->addSlice(*trt_src0, start_second, size, stride);
        if (slice_second == nullptr) {
            GGML_LOG_ERROR("%s: failed to create second half slice\n", __func__);
            return nullptr;
        }
        slice_second->setInput(2, *half_size_tensor);
        nvinfer1::ITensor* second_half = slice_second->getOutput(0);

        // Default: data = first half ([:nc]), gate = second half ([nc:])
        // Swapped: data = second half ([nc:]), gate = first half ([:nc])
        if (swapped) {
            data_tensor = second_half;
            gate_tensor = first_half;
        } else {
            data_tensor = first_half;
            gate_tensor = second_half;
        }
    }

    // Apply activation to data
    std::string prefix = "glu_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    nvinfer1::ITensor* activated = nullptr;

    if (glu_op == GGML_GLU_OP_GEGLU_QUICK) {
        // GELU_QUICK = x * sigmoid(1.702 * x)
        nvinfer1::ITensor* scale = builder->create_typed_scalar(1.702f, data_tensor);
        if (scale == nullptr) {
            GGML_LOG_ERROR("%s: failed to create scale for GEGLU_QUICK\n", __func__);
            return nullptr;
        }
        auto* scale_layer = network->addElementWise(
            *data_tensor, *scale, nvinfer1::ElementWiseOperation::kPROD);
        if (scale_layer == nullptr) {
            GGML_LOG_ERROR("%s: failed to create scale layer for GEGLU_QUICK\n", __func__);
            return nullptr;
        }
        auto* sigmoid_layer = network->addActivation(
            *scale_layer->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
        if (sigmoid_layer == nullptr) {
            GGML_LOG_ERROR("%s: failed to create sigmoid layer for GEGLU_QUICK\n", __func__);
            return nullptr;
        }
        auto* mul_layer = network->addElementWise(
            *data_tensor, *sigmoid_layer->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
        if (mul_layer == nullptr) {
            GGML_LOG_ERROR("%s: failed to create mul layer for GEGLU_QUICK\n", __func__);
            return nullptr;
        }
        mul_layer->setName((prefix + "_geglu_quick").c_str());
        activated = mul_layer->getOutput(0);
    } else if (glu_op == GGML_GLU_OP_SWIGLU_OAI) {
        // SWIGLU_OAI: same as SiLU activation, extra params (alpha, limit) are
        // applied differently but the TRT decomposition is identical to SILU
        // for the common case.  Full SWIGLU_OAI with clamping is not yet
        // supported — fall through to SiLU.
        activated = apply_activation(network, builder, data_tensor,
                                     GGML_UNARY_OP_SILU, prefix.c_str());
    } else {
        enum ggml_unary_op act = glu_op_to_activation(glu_op);
        activated = apply_activation(network, builder, data_tensor,
                                     act, prefix.c_str());
    }

    if (activated == nullptr) {
        GGML_LOG_ERROR("%s: failed to apply activation for GLU op %d\n",
            __func__, (int)glu_op);
        return nullptr;
    }

    // Multiply activated data with gate
    auto* prod_layer = network->addElementWise(
        *activated, *gate_tensor, nvinfer1::ElementWiseOperation::kPROD);
    if (prod_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create product layer\n", __func__);
        return nullptr;
    }
    prod_layer->setName((prefix + "_prod").c_str());
    nvinfer1::ITensor* output = prod_layer->getOutput(0);

    GGML_LOG_DEBUG("%s: created GLU op %d (swapped=%d), output shape %s\n",
        __func__, (int)glu_op, swapped,
        dims_to_string(output->getDimensions()).c_str());

    return output;
}

// Register the GLU operation handler
static void __attribute__((constructor)) register_glu_handler() {
    NetworkBuilder::register_op_handler(GGML_OP_GLU, handle_glu);
}

} // namespace ggml_tensorrt
