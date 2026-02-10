#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>

namespace ggml_tensorrt {

// Handle GET_ROWS: gather rows from src[0] using I32 indices in src[1].
// GGML output is always F32 (dequantizes F16/BF16).
// Output shape: [src0.ne[0], src1.ne[0], src1.ne[1], src1.ne[2]]
//
// Dimension mapping:
//   GGML [ne0=cols, ne1=rows] -> TRT [rows, cols]  (reversed by ggml_tensor_to_dims)
//   GET_ROWS gathers along GGML ne[1] (rows) -> TRT axis = nbDims - 2  for the data tensor
nvinfer1::ITensor* handle_get_rows(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_GET_ROWS);

    const ggml_tensor* data    = node->src[0];
    const ggml_tensor* indices = node->src[1];
    GGML_ASSERT(data != nullptr);
    GGML_ASSERT(indices != nullptr);

    nvinfer1::ITensor* trt_data = builder->get_tensor(data);
    nvinfer1::ITensor* trt_indices = builder->get_tensor(indices);

    if (trt_data == nullptr || trt_indices == nullptr) {
        GGML_LOG_ERROR("%s: input tensors not found in network\n", __func__);
        return nullptr;
    }

    auto* network = builder->get_network();

    // Gather axis: GGML gathers along ne[1] (rows).
    // ggml_tensor_to_dims reverses dimension order, so GGML ne[1] maps to
    // TRT dimension (nbDims - 2) for the data tensor.
    int gather_axis = trt_data->getDimensions().nbDims - 2;
    GGML_ASSERT(gather_axis >= 0);

    auto* gather_layer = network->addGather(*trt_data, *trt_indices, gather_axis);
    if (gather_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create gather layer\n", __func__);
        return nullptr;
    }

    std::string layer_name = "get_rows_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    gather_layer->setName(layer_name.c_str());

    nvinfer1::ITensor* output = gather_layer->getOutput(0);

    // GET_ROWS always outputs F32 in GGML — cast if needed
    output = builder->maybe_cast(output, nvinfer1::DataType::kFLOAT);

    GGML_LOG_DEBUG("%s: gather axis=%d, output shape %s\n",
        __func__, gather_axis,
        dims_to_string(output->getDimensions()).c_str());

    return output;
}

// Register the GET_ROWS handler
static void __attribute__((constructor)) register_get_rows_handler() {
    NetworkBuilder::register_op_handler(GGML_OP_GET_ROWS, handle_get_rows);
}

} // namespace ggml_tensorrt
