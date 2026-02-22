#include "../common.hpp"
#include "../network-builder.hpp"
#include "../utils/tensor-utils.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>
#include <cmath>
#include <cinttypes>
#include <cstring>
#include <string>

namespace ggml_tensorrt {

// Handle GGML_OP_ROPE — Rotary Position Embedding
//
// Supported modes: NORMAL (interleaved pairs) and NEOX (split halves).
// Rejects: MROPE, VISION, IMROPE, YaRN (ext_factor != 0), freq_factors (src[2]).
//
// Inputs:
//   src[0]: Q/K data — GGML shape (head_dim, n_heads, n_tokens[, batch])
//   src[1]: position IDs — GGML shape (n_tokens[, batch]), type I32
//   src[2]: freq_factors — must be nullptr (rejected in supports_op)
//
// GGML dim layout (ne[0..3]) maps to TRT dims in REVERSED order:
//   ggml_tensor_to_dims reverses: GGML (head_dim, n_heads, n_tokens) → TRT (n_tokens, n_heads, head_dim)
//   TRT last dim = GGML ne[0] = head_dim — the dimension we rotate.
//
// Op params layout (int32_t[]):
//   [0] = n_past (unused), [1] = n_dims, [2] = mode,
//   [4] = n_ctx_orig, [5] = freq_base (float), [6] = freq_scale (float),
//   [7] = ext_factor (float), [8] = attn_factor (float)

nvinfer1::ITensor* handle_rope(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_ROPE);

    const ggml_tensor* src0 = node->src[0];  // Q/K data
    const ggml_tensor* src1 = node->src[1];  // position IDs (I32)
    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);

    nvinfer1::ITensor* input = builder->get_tensor(src0);
    nvinfer1::ITensor* pos   = builder->get_tensor(src1);
    if (input == nullptr || pos == nullptr) {
        GGML_LOG_ERROR("%s: input or position tensor not found in network\n", __func__);
        return nullptr;
    }

    auto* network = builder->get_network();

    // ── Extract op parameters ──
    const int n_dims = ((const int32_t *)node->op_params)[1];
    const int mode   = ((const int32_t *)node->op_params)[2];

    float freq_base  = 0.0f;
    float freq_scale = 0.0f;
    memcpy(&freq_base,  (const float *)node->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (const float *)node->op_params + 6, sizeof(float));

    const bool is_neox = (mode == GGML_ROPE_TYPE_NEOX);
    const int half = n_dims / 2;

    // Save original input type and upcast to F32 for computation
    nvinfer1::DataType input_type = input->getType();
    input = builder->maybe_cast(input, nvinfer1::DataType::kFLOAT);

    nvinfer1::Dims input_dims = input->getDimensions();
    const int ndims = input_dims.nbDims;
    // TRT last dim = head_dim (GGML ne[0])
    const int last_axis = ndims - 1;
    // head_dim from GGML (always concrete, even for dynamic inputs)
    const int64_t head_dim = src0->ne[0];

    // Unique name prefix for layer naming
    std::string pfx = "rope_" + std::to_string(reinterpret_cast<uintptr_t>(node));

    // ── Build frequency constant table ──
    // freq[i] = freq_scale / pow(freq_base, 2.0*i/n_dims) for i = 0..half-1
    // Shape: all-1s except last dim = half (broadcasts with input)
    std::vector<float> freq_data(half);
    for (int i = 0; i < half; i++) {
        freq_data[i] = freq_scale / powf(freq_base, 2.0f * (float)i / (float)n_dims);
    }

    nvinfer1::Dims freq_dims;
    freq_dims.nbDims = ndims;
    for (int i = 0; i < ndims; i++) {
        freq_dims.d[i] = 1;
    }
    freq_dims.d[last_axis] = half;

    nvinfer1::ITensor* freq = builder->create_constant_tensor(
        freq_data.data(), freq_dims, nvinfer1::DataType::kFLOAT);
    if (freq == nullptr) {
        GGML_LOG_ERROR("%s: failed to create frequency constant\n", __func__);
        return nullptr;
    }

    // ── Compute angles: theta = pos_f32 * freq ──
    // pos is I32 with shape (n_tokens,) or (n_tokens, batch) in TRT.
    // We need to reshape it to (n_tokens, 1, ..., 1) to broadcast with input.
    nvinfer1::ITensor* pos_f32 = builder->maybe_cast(pos, nvinfer1::DataType::kFLOAT);

    // Reshape position: add trailing 1-dims to match input rank.
    // pos TRT dims: (n_tokens,) → need (n_tokens, 1, 1, ...) with ndims total dims.
    nvinfer1::Dims pos_dims = pos_f32->getDimensions();
    if (pos_dims.nbDims < ndims) {
        nvinfer1::Dims reshape_dims;
        reshape_dims.nbDims = ndims;
        for (int i = 0; i < ndims; i++) {
            if (i < pos_dims.nbDims) {
                // Use 0 (copy from input) for dynamic dims
                reshape_dims.d[i] = (pos_dims.d[i] == -1) ? 0 : pos_dims.d[i];
            } else {
                reshape_dims.d[i] = 1;
            }
        }
        auto* pos_shuffle = network->addShuffle(*pos_f32);
        if (pos_shuffle == nullptr) {
            GGML_LOG_ERROR("%s: failed to reshape position tensor\n", __func__);
            return nullptr;
        }
        pos_shuffle->setReshapeDimensions(reshape_dims);
        pos_shuffle->setName((pfx + "_pos_reshape").c_str());
        pos_f32 = pos_shuffle->getOutput(0);
    }

    // theta = pos_f32 * freq  (broadcasts: (n_tokens,1,..,1) * (1,1,..,half))
    auto* theta_layer = network->addElementWise(
        *pos_f32, *freq, nvinfer1::ElementWiseOperation::kPROD);
    if (theta_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create theta layer\n", __func__);
        return nullptr;
    }
    theta_layer->setName((pfx + "_theta").c_str());
    nvinfer1::ITensor* theta = theta_layer->getOutput(0);

    // cos(theta) and sin(theta)
    auto* cos_layer = network->addUnary(*theta, nvinfer1::UnaryOperation::kCOS);
    auto* sin_layer = network->addUnary(*theta, nvinfer1::UnaryOperation::kSIN);
    if (cos_layer == nullptr || sin_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create cos/sin layers\n", __func__);
        return nullptr;
    }
    cos_layer->setName((pfx + "_cos").c_str());
    sin_layer->setName((pfx + "_sin").c_str());
    nvinfer1::ITensor* cos_theta = cos_layer->getOutput(0);
    nvinfer1::ITensor* sin_theta = sin_layer->getOutput(0);

    // ── Split input into rotated and passthrough parts ──
    // Build base start/size/stride for slicing along last_axis
    nvinfer1::Dims start_zero, stride_ones;
    start_zero.nbDims = ndims;
    stride_ones.nbDims = ndims;
    for (int i = 0; i < ndims; i++) {
        start_zero.d[i] = 0;
        stride_ones.d[i] = 1;
    }

    nvinfer1::ITensor* x0 = nullptr;
    nvinfer1::ITensor* x1 = nullptr;
    nvinfer1::ITensor* x_pass = nullptr;

    // Extract passthrough portion (if n_dims < head_dim)
    bool has_passthrough = (n_dims < head_dim);
    if (has_passthrough) {
        nvinfer1::Dims pass_start = start_zero;
        pass_start.d[last_axis] = n_dims;

        // Use placeholder size (1s) + shape tensor for dynamic dims
        nvinfer1::Dims pass_size;
        pass_size.nbDims = ndims;
        for (int i = 0; i < ndims; i++) {
            pass_size.d[i] = 1;  // placeholder
        }
        pass_size.d[last_axis] = head_dim - n_dims;

        auto* pass_slice = network->addSlice(*input, pass_start, pass_size, stride_ones);
        if (pass_slice == nullptr) {
            GGML_LOG_ERROR("%s: failed to create passthrough slice\n", __func__);
            return nullptr;
        }
        // Override size with shape tensor — copies dynamic dims from input,
        // overrides last axis with the static passthrough size
        nvinfer1::ITensor* pass_size_tensor = builder->make_slice_size(
            input, last_axis, head_dim - n_dims);
        if (pass_size_tensor == nullptr) {
            GGML_LOG_ERROR("%s: failed to create passthrough size tensor\n", __func__);
            return nullptr;
        }
        pass_slice->setInput(2, *pass_size_tensor);
        pass_slice->setName((pfx + "_pass").c_str());
        x_pass = pass_slice->getOutput(0);
    }

    if (is_neox) {
        // ── NEOX mode: split halves along last dim ──
        // x0 = input[..., :half], x1 = input[..., half:n_dims]
        nvinfer1::Dims half_size;
        half_size.nbDims = ndims;
        for (int i = 0; i < ndims; i++) {
            half_size.d[i] = 1;  // placeholder for shape tensor
        }
        half_size.d[last_axis] = half;

        // Shape tensor: copies dynamic dims from input, override last axis = half
        nvinfer1::ITensor* half_size_tensor = builder->make_slice_size(
            input, last_axis, half);
        if (half_size_tensor == nullptr) {
            GGML_LOG_ERROR("%s: failed to create half size tensor (NEOX)\n", __func__);
            return nullptr;
        }

        auto* slice_x0 = network->addSlice(*input, start_zero, half_size, stride_ones);
        if (slice_x0 == nullptr) {
            GGML_LOG_ERROR("%s: failed to create x0 slice (NEOX)\n", __func__);
            return nullptr;
        }
        slice_x0->setInput(2, *half_size_tensor);
        slice_x0->setName((pfx + "_x0").c_str());
        x0 = slice_x0->getOutput(0);

        nvinfer1::Dims x1_start = start_zero;
        x1_start.d[last_axis] = half;
        auto* slice_x1 = network->addSlice(*input, x1_start, half_size, stride_ones);
        if (slice_x1 == nullptr) {
            GGML_LOG_ERROR("%s: failed to create x1 slice (NEOX)\n", __func__);
            return nullptr;
        }
        slice_x1->setInput(2, *half_size_tensor);
        slice_x1->setName((pfx + "_x1").c_str());
        x1 = slice_x1->getOutput(0);
    } else {
        // ── NORMAL mode: interleaved pairs ──
        // input[..., n_dims] → reshape to [..., half, 2] → split along new last dim
        nvinfer1::Dims rot_size;
        rot_size.nbDims = ndims;
        for (int i = 0; i < ndims; i++) {
            rot_size.d[i] = 1;  // placeholder for shape tensor
        }
        rot_size.d[last_axis] = n_dims;

        // Slice out the first n_dims elements (the rotated portion)
        auto* rot_slice = network->addSlice(*input, start_zero, rot_size, stride_ones);
        if (rot_slice == nullptr) {
            GGML_LOG_ERROR("%s: failed to create rotation slice (NORMAL)\n", __func__);
            return nullptr;
        }
        // Shape tensor: copy dynamic dims from input, override last axis = n_dims
        nvinfer1::ITensor* rot_size_tensor = builder->make_slice_size(
            input, last_axis, n_dims);
        if (rot_size_tensor == nullptr) {
            GGML_LOG_ERROR("%s: failed to create rot size tensor (NORMAL)\n", __func__);
            return nullptr;
        }
        rot_slice->setInput(2, *rot_size_tensor);
        rot_slice->setName((pfx + "_rot_slice").c_str());
        nvinfer1::ITensor* x_rot = rot_slice->getOutput(0);

        // Reshape to (..., half, 2)
        // Use 0 for dynamic dims (batch/tokens at position 0)
        nvinfer1::Dims pairs_dims;
        pairs_dims.nbDims = ndims + 1;
        for (int i = 0; i < last_axis; i++) {
            // Position i in x_rot corresponds to position i in input — use 0 for dynamic
            pairs_dims.d[i] = (input_dims.d[i] == -1) ? 0 : input_dims.d[i];
        }
        pairs_dims.d[last_axis] = half;
        pairs_dims.d[last_axis + 1] = 2;

        auto* pairs_shuffle = network->addShuffle(*x_rot);
        if (pairs_shuffle == nullptr) {
            GGML_LOG_ERROR("%s: failed to reshape to pairs (NORMAL)\n", __func__);
            return nullptr;
        }
        pairs_shuffle->setReshapeDimensions(pairs_dims);
        pairs_shuffle->setName((pfx + "_pairs").c_str());
        nvinfer1::ITensor* x_pairs = pairs_shuffle->getOutput(0);

        // Slice even (x0) and odd (x1) elements along the new last dim (axis = ndims)
        int pair_axis = ndims;  // the new last dim of pairs_dims (index ndims since 0-based, nbDims = ndims+1)
        nvinfer1::Dims elem_start, elem_size, elem_stride;
        elem_start.nbDims = ndims + 1;
        elem_size.nbDims = ndims + 1;
        elem_stride.nbDims = ndims + 1;
        for (int i = 0; i < ndims + 1; i++) {
            elem_start.d[i] = 0;
            elem_size.d[i] = 1;  // placeholder for shape tensor
            elem_stride.d[i] = 1;
        }
        elem_size.d[pair_axis] = 1;  // slice 1 element along pair axis

        // Shape tensor for elem slice: copy all dims from x_pairs, override pair_axis=1
        nvinfer1::ITensor* elem_size_tensor = builder->make_slice_size(
            x_pairs, pair_axis, 1);
        if (elem_size_tensor == nullptr) {
            GGML_LOG_ERROR("%s: failed to create elem size tensor (NORMAL)\n", __func__);
            return nullptr;
        }

        // x0 = even elements (start=0 on pair_axis)
        auto* slice_even = network->addSlice(*x_pairs, elem_start, elem_size, elem_stride);
        if (slice_even == nullptr) {
            GGML_LOG_ERROR("%s: failed to create even slice (NORMAL)\n", __func__);
            return nullptr;
        }
        slice_even->setInput(2, *elem_size_tensor);
        slice_even->setName((pfx + "_even").c_str());
        x0 = slice_even->getOutput(0);

        // x1 = odd elements (start=1 on pair_axis)
        nvinfer1::Dims odd_start = elem_start;
        odd_start.d[pair_axis] = 1;
        auto* slice_odd = network->addSlice(*x_pairs, odd_start, elem_size, elem_stride);
        if (slice_odd == nullptr) {
            GGML_LOG_ERROR("%s: failed to create odd slice (NORMAL)\n", __func__);
            return nullptr;
        }
        slice_odd->setInput(2, *elem_size_tensor);
        slice_odd->setName((pfx + "_odd").c_str());
        x1 = slice_odd->getOutput(0);

        // cos/sin need to be reshaped to (..., half, 1) for NORMAL mode broadcasting
        // theta shape is (n_tokens, 1, ..., half) in TRT
        // We need cos/sin to be (n_tokens, 1, ..., half, 1)
        nvinfer1::Dims trig_dims;
        trig_dims.nbDims = ndims + 1;
        nvinfer1::Dims theta_dims = cos_theta->getDimensions();
        for (int i = 0; i < ndims; i++) {
            // Use 0 (copy from input) for dynamic dims
            trig_dims.d[i] = (theta_dims.d[i] == -1) ? 0 : theta_dims.d[i];
        }
        trig_dims.d[ndims] = 1;  // trailing 1 for pair_axis broadcasting

        auto* cos_shuffle = network->addShuffle(*cos_theta);
        if (cos_shuffle == nullptr) {
            GGML_LOG_ERROR("%s: failed to reshape cos for NORMAL mode\n", __func__);
            return nullptr;
        }
        cos_shuffle->setReshapeDimensions(trig_dims);
        cos_shuffle->setName((pfx + "_cos_reshape").c_str());
        cos_theta = cos_shuffle->getOutput(0);

        auto* sin_shuffle = network->addShuffle(*sin_theta);
        if (sin_shuffle == nullptr) {
            GGML_LOG_ERROR("%s: failed to reshape sin for NORMAL mode\n", __func__);
            return nullptr;
        }
        sin_shuffle->setReshapeDimensions(trig_dims);
        sin_shuffle->setName((pfx + "_sin_reshape").c_str());
        sin_theta = sin_shuffle->getOutput(0);
    }

    // ── Apply rotation ──
    // x0_rot = x0 * cos - x1 * sin
    // x1_rot = x0 * sin + x1 * cos

    auto* t0_layer = network->addElementWise(*x0, *cos_theta, nvinfer1::ElementWiseOperation::kPROD);
    auto* t1_layer = network->addElementWise(*x1, *sin_theta, nvinfer1::ElementWiseOperation::kPROD);
    if (t0_layer == nullptr || t1_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create rotation product layers\n", __func__);
        return nullptr;
    }
    t0_layer->setName((pfx + "_t0").c_str());
    t1_layer->setName((pfx + "_t1").c_str());

    auto* x0_rot_layer = network->addElementWise(
        *t0_layer->getOutput(0), *t1_layer->getOutput(0),
        nvinfer1::ElementWiseOperation::kSUB);
    if (x0_rot_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create x0_rot layer\n", __func__);
        return nullptr;
    }
    x0_rot_layer->setName((pfx + "_x0_rot").c_str());
    nvinfer1::ITensor* x0_rot = x0_rot_layer->getOutput(0);

    auto* t2_layer = network->addElementWise(*x0, *sin_theta, nvinfer1::ElementWiseOperation::kPROD);
    auto* t3_layer = network->addElementWise(*x1, *cos_theta, nvinfer1::ElementWiseOperation::kPROD);
    if (t2_layer == nullptr || t3_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create rotation product layers (x1)\n", __func__);
        return nullptr;
    }
    t2_layer->setName((pfx + "_t2").c_str());
    t3_layer->setName((pfx + "_t3").c_str());

    auto* x1_rot_layer = network->addElementWise(
        *t2_layer->getOutput(0), *t3_layer->getOutput(0),
        nvinfer1::ElementWiseOperation::kSUM);
    if (x1_rot_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create x1_rot layer\n", __func__);
        return nullptr;
    }
    x1_rot_layer->setName((pfx + "_x1_rot").c_str());
    nvinfer1::ITensor* x1_rot = x1_rot_layer->getOutput(0);

    // ── Reassemble ──
    nvinfer1::ITensor* rotated = nullptr;

    if (is_neox) {
        // NEOX: concatenate [x0_rot, x1_rot] along last axis
        nvinfer1::ITensor* concat_inputs[] = {x0_rot, x1_rot};
        auto* concat_layer = network->addConcatenation(concat_inputs, 2);
        if (concat_layer == nullptr) {
            GGML_LOG_ERROR("%s: failed to create concat layer (NEOX)\n", __func__);
            return nullptr;
        }
        concat_layer->setAxis(last_axis);
        concat_layer->setName((pfx + "_concat").c_str());
        rotated = concat_layer->getOutput(0);
    } else {
        // NORMAL: concatenate [x0_rot, x1_rot] along pair_axis, then reshape back
        int pair_axis = ndims;
        nvinfer1::ITensor* concat_inputs[] = {x0_rot, x1_rot};
        auto* concat_layer = network->addConcatenation(concat_inputs, 2);
        if (concat_layer == nullptr) {
            GGML_LOG_ERROR("%s: failed to create concat layer (NORMAL)\n", __func__);
            return nullptr;
        }
        concat_layer->setAxis(pair_axis);
        concat_layer->setName((pfx + "_concat").c_str());
        nvinfer1::ITensor* interleaved = concat_layer->getOutput(0);

        // Reshape back from (..., half, 2) to (..., n_dims)
        // Use 0 for dynamic dims (batch/tokens)
        nvinfer1::Dims flat_dims;
        flat_dims.nbDims = ndims;
        for (int i = 0; i < last_axis; i++) {
            flat_dims.d[i] = (input_dims.d[i] == -1) ? 0 : input_dims.d[i];
        }
        flat_dims.d[last_axis] = n_dims;

        auto* flat_shuffle = network->addShuffle(*interleaved);
        if (flat_shuffle == nullptr) {
            GGML_LOG_ERROR("%s: failed to reshape back from pairs (NORMAL)\n", __func__);
            return nullptr;
        }
        flat_shuffle->setReshapeDimensions(flat_dims);
        flat_shuffle->setName((pfx + "_flat").c_str());
        rotated = flat_shuffle->getOutput(0);
    }

    // ── Append passthrough if n_dims < head_dim ──
    nvinfer1::ITensor* output = rotated;
    if (has_passthrough && x_pass != nullptr) {
        nvinfer1::ITensor* final_inputs[] = {rotated, x_pass};
        auto* final_concat = network->addConcatenation(final_inputs, 2);
        if (final_concat == nullptr) {
            GGML_LOG_ERROR("%s: failed to create final concat layer\n", __func__);
            return nullptr;
        }
        final_concat->setAxis(last_axis);
        final_concat->setName((pfx + "_final").c_str());
        output = final_concat->getOutput(0);
    }

    // ── Cast back to original type ──
    output = builder->maybe_cast(output, input_type);

    GGML_LOG_DEBUG("%s: mode=%s, n_dims=%d, head_dim=%" PRId64 ", freq_base=%.1f, freq_scale=%.4f, output shape %s\n",
        __func__, is_neox ? "NEOX" : "NORMAL", n_dims, head_dim,
        freq_base, freq_scale,
        dims_to_string(output->getDimensions()).c_str());

    return output;
}

// Handle GGML_OP_ROPE — Native path using TRT IRotaryEmbeddingLayer
//
// Replaces the decomposed rotation (slice/multiply/subtract/add/concat) with
// a single addRotaryEmbedding call.  Still computes cos/sin dynamically from
// freq_base/freq_scale, then reshapes for the native API.
//
// TRT addRotaryEmbedding expects:
//   input:    [B, H, S, D]
//   cosCache: [B, S, half]  (without positionIds)
//   sinCache: [B, S, half]  (without positionIds)
//
// Our TRT tensors after ggml_tensor_to_dims reversal:
//   input: (n_tokens, n_heads, head_dim) = [S, H, D]
//   pos:   (n_tokens,) = [S]
//
nvinfer1::ITensor* handle_rope_native(NetworkBuilder* builder, const ggml_tensor* node) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(node->op == GGML_OP_ROPE);

    const ggml_tensor* src0 = node->src[0];  // Q/K data
    const ggml_tensor* src1 = node->src[1];  // position IDs (I32)
    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);

    nvinfer1::ITensor* input = builder->get_tensor(src0);
    nvinfer1::ITensor* pos   = builder->get_tensor(src1);
    if (input == nullptr || pos == nullptr) {
        GGML_LOG_ERROR("%s: input or position tensor not found in network\n", __func__);
        return nullptr;
    }

    auto* network = builder->get_network();

    // ── Extract op parameters ──
    const int n_dims = ((const int32_t *)node->op_params)[1];
    const int mode   = ((const int32_t *)node->op_params)[2];

    float freq_base  = 0.0f;
    float freq_scale = 0.0f;
    memcpy(&freq_base,  (const float *)node->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (const float *)node->op_params + 6, sizeof(float));

    const bool is_neox = (mode == GGML_ROPE_TYPE_NEOX);
    const int half = n_dims / 2;

    // Save original input type and upcast to F32 for computation
    nvinfer1::DataType input_type = input->getType();
    input = builder->maybe_cast(input, nvinfer1::DataType::kFLOAT);

    const int64_t head_dim = src0->ne[0];

    // Unique name prefix for layer naming
    std::string pfx = "rope_native_" + std::to_string(reinterpret_cast<uintptr_t>(node));

    // ── Build frequency constant table ──
    // freq[i] = freq_scale / pow(freq_base, 2.0*i/n_dims) for i = 0..half-1
    // Shape: (1, half) for broadcasting with pos (n_tokens, 1)
    std::vector<float> freq_data(half);
    for (int i = 0; i < half; i++) {
        freq_data[i] = freq_scale / powf(freq_base, 2.0f * (float)i / (float)n_dims);
    }

    nvinfer1::Dims freq_dims;
    freq_dims.nbDims = 2;
    freq_dims.d[0] = 1;
    freq_dims.d[1] = half;

    nvinfer1::ITensor* freq = builder->create_constant_tensor(
        freq_data.data(), freq_dims, nvinfer1::DataType::kFLOAT);
    if (freq == nullptr) {
        GGML_LOG_ERROR("%s: failed to create frequency constant\n", __func__);
        return nullptr;
    }

    // ── Compute cos/sin caches ──
    // pos: TRT shape (n_tokens,) [could be dynamic dim 0]
    // Reshape to (n_tokens, 1) for broadcasting with freq (1, half)
    nvinfer1::ITensor* pos_f32 = builder->maybe_cast(pos, nvinfer1::DataType::kFLOAT);

    nvinfer1::Dims pos_2d;
    pos_2d.nbDims = 2;
    nvinfer1::Dims pos_curr = pos_f32->getDimensions();
    pos_2d.d[0] = (pos_curr.d[0] == -1) ? 0 : pos_curr.d[0];
    pos_2d.d[1] = 1;

    auto* pos_reshape = network->addShuffle(*pos_f32);
    if (pos_reshape == nullptr) {
        GGML_LOG_ERROR("%s: failed to reshape position to 2D\n", __func__);
        return nullptr;
    }
    pos_reshape->setReshapeDimensions(pos_2d);
    pos_reshape->setName((pfx + "_pos_2d").c_str());
    nvinfer1::ITensor* pos_2d_t = pos_reshape->getOutput(0);

    // theta = pos_2d * freq  → (n_tokens, half)
    auto* theta_layer = network->addElementWise(
        *pos_2d_t, *freq, nvinfer1::ElementWiseOperation::kPROD);
    if (theta_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create theta layer\n", __func__);
        return nullptr;
    }
    theta_layer->setName((pfx + "_theta").c_str());
    nvinfer1::ITensor* theta = theta_layer->getOutput(0);

    // cos(theta) and sin(theta) → (n_tokens, half)
    auto* cos_layer = network->addUnary(*theta, nvinfer1::UnaryOperation::kCOS);
    auto* sin_layer = network->addUnary(*theta, nvinfer1::UnaryOperation::kSIN);
    if (cos_layer == nullptr || sin_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create cos/sin layers\n", __func__);
        return nullptr;
    }
    cos_layer->setName((pfx + "_cos").c_str());
    sin_layer->setName((pfx + "_sin").c_str());
    nvinfer1::ITensor* cos_theta = cos_layer->getOutput(0);  // (n_tokens, half)
    nvinfer1::ITensor* sin_theta = sin_layer->getOutput(0);  // (n_tokens, half)

    nvinfer1::ITensor* cos_cache = nullptr;
    nvinfer1::ITensor* sin_cache = nullptr;

    // ── Build shape tensor from pos for consistent dynamic S ──
    // The pos tensor always has the correct n_tokens (dynamic).  Using its
    // runtime shape to drive both cosCache and input reshapes ensures TRT sees
    // the same dynamic S dimension in both, avoiding static-vs-dynamic mismatch
    // when ggml_n_dims strips n_tokens=1 during decode.
    auto* pos_shape_layer = network->addShape(*pos_f32);
    if (pos_shape_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create pos shape layer\n", __func__);
        return nullptr;
    }
    pos_shape_layer->setName((pfx + "_pos_shape").c_str());
    nvinfer1::ITensor* pos_shape = pos_shape_layer->getOutput(0);  // 1D: [n_tokens]
    nvinfer1::DataType shape_type = pos_shape->getType();  // INT64 on TRT-RTX

    // Helper: create a 1-element shape constant matching pos_shape's type
    auto make_shape_scalar = [&](int64_t value) -> nvinfer1::ITensor* {
        nvinfer1::Dims d1{1, {1}};
        if (shape_type == nvinfer1::DataType::kINT64) {
            return builder->create_constant_tensor(&value, d1, nvinfer1::DataType::kINT64);
        } else {
            int32_t v32 = (int32_t)value;
            return builder->create_constant_tensor(&v32, d1, nvinfer1::DataType::kINT32);
        }
    };

    nvinfer1::ITensor* shape_1     = make_shape_scalar(1);
    nvinfer1::ITensor* shape_half  = make_shape_scalar(half);
    nvinfer1::ITensor* shape_heads = make_shape_scalar(src0->ne[1]);  // n_heads
    nvinfer1::ITensor* shape_dim   = make_shape_scalar(head_dim);

    // ── Reshape cos/sin to [B=1, S=n_tokens, half] via shape tensor ──
    // cosCache shape: (batchSize, sequenceLength, rotaryEmbeddingDim/2)
    {
        nvinfer1::ITensor* cache_parts[] = {shape_1, pos_shape, shape_half};
        auto* cache_shape_concat = network->addConcatenation(cache_parts, 3);
        if (cache_shape_concat == nullptr) {
            GGML_LOG_ERROR("%s: failed to build cache shape tensor\n", __func__);
            return nullptr;
        }
        cache_shape_concat->setAxis(0);
        cache_shape_concat->setName((pfx + "_cache_shape").c_str());
        nvinfer1::ITensor* cache_shape = cache_shape_concat->getOutput(0);

        auto* cos_reshape = network->addShuffle(*cos_theta);
        if (cos_reshape == nullptr) {
            GGML_LOG_ERROR("%s: failed to reshape cos cache\n", __func__);
            return nullptr;
        }
        cos_reshape->setInput(1, *cache_shape);
        cos_reshape->setName((pfx + "_cos_cache").c_str());
        cos_cache = cos_reshape->getOutput(0);

        auto* sin_reshape = network->addShuffle(*sin_theta);
        if (sin_reshape == nullptr) {
            GGML_LOG_ERROR("%s: failed to reshape sin cache\n", __func__);
            return nullptr;
        }
        sin_reshape->setInput(1, *cache_shape);
        sin_reshape->setName((pfx + "_sin_cache").c_str());
        sin_cache = sin_reshape->getOutput(0);
    }

    // ── Reshape input to [B=1, S=n_tokens, H=n_heads, D=head_dim] via shape tensor ──
    // Then permute to [B, H, S, D].
    // Using pos_shape for S ensures the dynamic dimension matches cosCache.
    // Input may be 2D (decode, n_tokens=1 stripped) or 3D (prompt) or 4D.
    nvinfer1::ITensor* input_bhsd = nullptr;
    {
        nvinfer1::ITensor* bshd_parts[] = {shape_1, pos_shape, shape_heads, shape_dim};
        auto* bshd_concat = network->addConcatenation(bshd_parts, 4);
        if (bshd_concat == nullptr) {
            GGML_LOG_ERROR("%s: failed to build input shape tensor\n", __func__);
            return nullptr;
        }
        bshd_concat->setAxis(0);
        bshd_concat->setName((pfx + "_bshd_shape").c_str());
        nvinfer1::ITensor* bshd_shape = bshd_concat->getOutput(0);

        auto* input_shuffle = network->addShuffle(*input);
        if (input_shuffle == nullptr) {
            GGML_LOG_ERROR("%s: failed to create input reshape\n", __func__);
            return nullptr;
        }
        input_shuffle->setInput(1, *bshd_shape);
        input_shuffle->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
        input_shuffle->setName((pfx + "_to_bhsd").c_str());
        input_bhsd = input_shuffle->getOutput(0);  // [B, H, S, D]
    }

    // ── addRotaryEmbedding ──
    // interleaved = true for NORMAL mode (pairs), false for NEOX (split halves)
    auto* rope_layer = network->addRotaryEmbedding(
        *input_bhsd, *cos_cache, *sin_cache,
        /*interleaved=*/ !is_neox,
        /*rotaryEmbeddingDim=*/ n_dims);
    if (rope_layer == nullptr) {
        GGML_LOG_ERROR("%s: addRotaryEmbedding failed\n", __func__);
        return nullptr;
    }
    rope_layer->setName((pfx + "_rope").c_str());
    nvinfer1::ITensor* rope_out = rope_layer->getOutput(0);  // [B, H, S, D]

    // ── Reshape output back to [S, H, D] ──
    // Step 1: permute [B, H, S, D] → [B, S, H, D]
    auto* perm_out = network->addShuffle(*rope_out);
    if (perm_out == nullptr) {
        GGML_LOG_ERROR("%s: failed to create output permute\n", __func__);
        return nullptr;
    }
    perm_out->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    perm_out->setName((pfx + "_perm_out").c_str());
    nvinfer1::ITensor* output_bshd = perm_out->getOutput(0);  // [1, S, H, D]

    // Step 2: strip batch dim → restore original input shape.
    // Use the original input's runtime shape (via IShapeLayer) as the reshape
    // target.  This handles all ndims cases (2D decode, 3D prompt, 4D batch)
    // and preserves dynamic dimensions correctly.
    auto* orig_shape_layer = network->addShape(*input);
    if (orig_shape_layer == nullptr) {
        GGML_LOG_ERROR("%s: failed to create original shape layer\n", __func__);
        return nullptr;
    }
    orig_shape_layer->setName((pfx + "_orig_shape").c_str());

    auto* strip_shuffle = network->addShuffle(*output_bshd);
    if (strip_shuffle == nullptr) {
        GGML_LOG_ERROR("%s: failed to strip batch dim\n", __func__);
        return nullptr;
    }
    strip_shuffle->setInput(1, *orig_shape_layer->getOutput(0));
    strip_shuffle->setName((pfx + "_strip_batch").c_str());
    nvinfer1::ITensor* output = strip_shuffle->getOutput(0);

    // ── Cast back to original type ──
    output = builder->maybe_cast(output, input_type);

    GGML_LOG_DEBUG("%s: mode=%s, n_dims=%d, head_dim=%" PRId64 ", freq_base=%.1f, freq_scale=%.4f, output shape %s\n",
        __func__, is_neox ? "NEOX" : "NORMAL", n_dims, head_dim,
        freq_base, freq_scale,
        dims_to_string(output->getDimensions()).c_str());

    return output;
}

// Register the ROPE operation handler — native path when GGML_TENSORRT_NATIVE_ATTN
// is enabled (default), decomposed path when explicitly disabled.
static void __attribute__((constructor)) register_rope_handler() {
    const char* env = getenv("GGML_TENSORRT_NATIVE_ATTN");
    bool native = (env == nullptr || atoi(env) != 0);
    if (native) {
        NetworkBuilder::register_op_handler(GGML_OP_ROPE, handle_rope_native);
    } else {
        NetworkBuilder::register_op_handler(GGML_OP_ROPE, handle_rope);
    }
}

} // namespace ggml_tensorrt
