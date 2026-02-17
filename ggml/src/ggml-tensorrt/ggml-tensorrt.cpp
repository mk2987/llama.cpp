#include "common.hpp"
#include "network-builder.hpp"
#include "engine-manager.hpp"
#include "utils/tensor-utils.hpp"
#include "utils/type-utils.hpp"
#include "kernels/set-rows.cuh"
#include "ggml-tensorrt.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ggml_tensorrt;

// Forward declarations
static const char * ggml_backend_tensorrt_get_name(ggml_backend_t backend);
static void ggml_backend_tensorrt_free(ggml_backend_t backend);
static ggml_backend_buffer_type_t ggml_backend_tensorrt_get_default_buffer_type(ggml_backend_t backend);
static void ggml_backend_tensorrt_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size);
static void ggml_backend_tensorrt_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size);
static bool ggml_backend_tensorrt_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst);
static void ggml_backend_tensorrt_synchronize(ggml_backend_t backend);
static enum ggml_status ggml_backend_tensorrt_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph);

// Buffer interface forward declarations
static void ggml_backend_tensorrt_buffer_free_buffer(ggml_backend_buffer_t buffer);
static void * ggml_backend_tensorrt_buffer_get_base(ggml_backend_buffer_t buffer);
static void ggml_backend_tensorrt_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size);
static void ggml_backend_tensorrt_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size);
static bool ggml_backend_tensorrt_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst);
static void ggml_backend_tensorrt_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value);

// Buffer type interface forward declarations
static const char * ggml_backend_tensorrt_buffer_type_get_name(ggml_backend_buffer_type_t buft);
static ggml_backend_buffer_t ggml_backend_tensorrt_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
static size_t ggml_backend_tensorrt_buffer_type_get_alignment(ggml_backend_buffer_type_t buft);
static size_t ggml_backend_tensorrt_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor);

//
// Buffer implementation
//

struct ggml_backend_tensorrt_buffer_context {
    int device;
    void * dev_ptr;
    std::string name;

    ggml_backend_tensorrt_buffer_context(int device, void * dev_ptr)
        : device(device), dev_ptr(dev_ptr) {
        name = GGML_TENSORRT_NAME + std::to_string(device);
    }
};

static void ggml_backend_tensorrt_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_tensorrt_buffer_context * ctx = (ggml_backend_tensorrt_buffer_context *) buffer->context;
    CUDA_CHECK(cudaFree(ctx->dev_ptr));
    delete ctx;
}

static void * ggml_backend_tensorrt_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_tensorrt_buffer_context * ctx = (ggml_backend_tensorrt_buffer_context *) buffer->context;
    return ctx->dev_ptr;
}

static void ggml_backend_tensorrt_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_tensorrt_buffer_context * ctx = (ggml_backend_tensorrt_buffer_context *) buffer->context;

    CUDA_CHECK(cudaSetDevice(ctx->device));
    CUDA_CHECK(cudaMemcpy((char*)tensor->data + offset, data, size, cudaMemcpyHostToDevice));
}

static void ggml_backend_tensorrt_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_tensorrt_buffer_context * ctx = (ggml_backend_tensorrt_buffer_context *) buffer->context;

    CUDA_CHECK(cudaSetDevice(ctx->device));
    CUDA_CHECK(cudaMemcpy(data, (const char*)tensor->data + offset, size, cudaMemcpyDeviceToHost));
}

static bool ggml_backend_tensorrt_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensorrt_buffer_context * ctx = (ggml_backend_tensorrt_buffer_context *) buffer->context;
        CUDA_CHECK(cudaSetDevice(ctx->device));
        CUDA_CHECK(cudaMemcpy(dst->data, src->data, ggml_nbytes(src), cudaMemcpyHostToDevice));
        return true;
    }
    return false;
}

static void ggml_backend_tensorrt_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_tensorrt_buffer_context * ctx = (ggml_backend_tensorrt_buffer_context *) buffer->context;

    CUDA_CHECK(cudaSetDevice(ctx->device));
    CUDA_CHECK(cudaMemset(ctx->dev_ptr, value, buffer->size));
}

static const ggml_backend_buffer_i ggml_backend_tensorrt_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_tensorrt_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_tensorrt_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_tensorrt_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_tensorrt_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_tensorrt_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_tensorrt_buffer_clear,
    /* .reset           = */ NULL,
};

//
// Buffer type implementation
//

struct ggml_backend_tensorrt_buffer_type_context {
    int device;
    std::string name;

    ggml_backend_tensorrt_buffer_type_context(int device)
        : device(device) {
        name = GGML_TENSORRT_NAME + std::to_string(device);
    }
};

static const char * ggml_backend_tensorrt_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_tensorrt_buffer_type_context * ctx = (ggml_backend_tensorrt_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_tensorrt_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_tensorrt_buffer_type_context * ctx = (ggml_backend_tensorrt_buffer_type_context *) buft->context;

    CUDA_CHECK(cudaSetDevice(ctx->device));

    void * dev_ptr;
    CUDA_CHECK(cudaMalloc(&dev_ptr, size));

    ggml_backend_tensorrt_buffer_context * buf_ctx = new ggml_backend_tensorrt_buffer_context(ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_tensorrt_buffer_interface, buf_ctx, size);
}

static size_t ggml_backend_tensorrt_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return 256;  // TensorRT alignment requirement
}

static size_t ggml_backend_tensorrt_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    (void) buft;
    return ggml_nbytes(tensor);
}

static const ggml_backend_buffer_type_i ggml_backend_tensorrt_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_tensorrt_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_tensorrt_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_tensorrt_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ ggml_backend_tensorrt_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_tensorrt_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_tensorrt_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_tensorrt_buffer_types[GGML_TENSORRT_MAX_DEVICES];
    static bool ggml_backend_tensorrt_buffer_type_initialized = false;

    if (!ggml_backend_tensorrt_buffer_type_initialized) {
        for (int i = 0; i < GGML_TENSORRT_MAX_DEVICES; i++) {
            ggml_backend_tensorrt_buffer_types[i] = {
                /* .iface   = */ ggml_backend_tensorrt_buffer_type_interface,
                /* .device  = */ nullptr,  // will be set when device is registered
                /* .context = */ new ggml_backend_tensorrt_buffer_type_context(i),
            };
        }
        ggml_backend_tensorrt_buffer_type_initialized = true;
    }

    return &ggml_backend_tensorrt_buffer_types[device];
}

//
// Host buffer type implementation (pinned memory)
//

struct ggml_backend_tensorrt_host_buffer_context {
    void * host_ptr;
};

static void ggml_backend_tensorrt_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_tensorrt_host_buffer_context * ctx = (ggml_backend_tensorrt_host_buffer_context *) buffer->context;
    CUDA_CHECK(cudaFreeHost(ctx->host_ptr));
    delete ctx;
}

static void * ggml_backend_tensorrt_host_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_tensorrt_host_buffer_context * ctx = (ggml_backend_tensorrt_host_buffer_context *) buffer->context;
    return ctx->host_ptr;
}

static const ggml_backend_buffer_i ggml_backend_tensorrt_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_tensorrt_host_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_tensorrt_host_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ NULL,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_tensorrt_host_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return GGML_TENSORRT_NAME " (Host)";
}

static ggml_backend_buffer_t ggml_backend_tensorrt_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    (void) buft;

    void * host_ptr;
    CUDA_CHECK(cudaMallocHost(&host_ptr, size));

    ggml_backend_tensorrt_host_buffer_context * ctx = new ggml_backend_tensorrt_host_buffer_context { host_ptr };

    return ggml_backend_buffer_init(buft, ggml_backend_tensorrt_host_buffer_interface, ctx, size);
}

static const ggml_backend_buffer_type_i ggml_backend_tensorrt_host_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_tensorrt_host_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_tensorrt_host_buffer_type_alloc_buffer,
    /* .get_alignment    = */ NULL,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ NULL,
    /* .is_host          = */ ggml_backend_buft_is_host,
};

ggml_backend_buffer_type_t ggml_backend_tensorrt_host_buffer_type() {
    static ggml_backend_buffer_type ggml_backend_tensorrt_host_buffer_type_instance = {
        /* .iface   = */ ggml_backend_tensorrt_host_buffer_type_interface,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };

    return &ggml_backend_tensorrt_host_buffer_type_instance;
}

//
// Backend implementation
//

static const char * ggml_backend_tensorrt_get_name(ggml_backend_t backend) {
    ggml_backend_tensorrt_context * ctx = (ggml_backend_tensorrt_context *) backend->context;
    static char name[128];
    snprintf(name, sizeof(name), "%s%d", GGML_TENSORRT_NAME, ctx->device);
    return name;
}

static void ggml_backend_tensorrt_free(ggml_backend_t backend) {
    ggml_backend_tensorrt_context * ctx = (ggml_backend_tensorrt_context *) backend->context;
    delete ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_tensorrt_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_tensorrt_context * ctx = (ggml_backend_tensorrt_context *) backend->context;
    return ggml_backend_tensorrt_buffer_type(ctx->device);
}

static void ggml_backend_tensorrt_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_tensorrt_context * ctx = (ggml_backend_tensorrt_context *) backend->context;

    CUDA_CHECK(cudaSetDevice(ctx->device));
    CUDA_CHECK(cudaMemcpyAsync((char*)tensor->data + offset, data, size, cudaMemcpyHostToDevice, ctx->stream));
}

static void ggml_backend_tensorrt_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_tensorrt_context * ctx = (ggml_backend_tensorrt_context *) backend->context;

    CUDA_CHECK(cudaSetDevice(ctx->device));
    CUDA_CHECK(cudaMemcpyAsync(data, (const char*)tensor->data + offset, size, cudaMemcpyDeviceToHost, ctx->stream));
}

static bool ggml_backend_tensorrt_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    // Only handle TRT→TRT copies (same stream, ordered).
    // Cross-backend copies (e.g. CUDA→TRT) need source stream sync —
    // returning false lets the scheduler's fallback path handle it.
    if (!ggml_backend_is_tensorrt(backend_src)) {
        return false;
    }

    GGML_ASSERT(src->buffer && dst->buffer);

    ggml_backend_tensorrt_context * ctx_dst = (ggml_backend_tensorrt_context *) backend_dst->context;

    CUDA_CHECK(cudaSetDevice(ctx_dst->device));
    CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, ctx_dst->stream));

    return true;
}

static void ggml_backend_tensorrt_synchronize(ggml_backend_t backend) {
    ggml_backend_tensorrt_context * ctx = (ggml_backend_tensorrt_context *) backend->context;

    CUDA_CHECK(cudaSetDevice(ctx->device));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

// Classify a leaf tensor as "static" (shape doesn't change between calls).
// Static leaves are model weights and normalization scales — loaded from
// GGUF once, shapes fixed by architecture.  Per-forward inputs (tokens,
// attention masks, positions) are marked by llama.cpp with
// GGML_TENSOR_FLAG_INPUT via ggml_set_input() — their shapes change with
// batch/sequence length and must be treated as dynamic.
static bool is_static_leaf(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_NONE &&
           !(tensor->flags & GGML_TENSOR_FLAG_INPUT);
}

// Check if an op is a shape/metadata operation (zero-copy in GGML).
// These share data pointers with their source and must NOT be marked as
// TRT engine outputs (that would create duplicate address bindings).
static bool is_shape_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// Check if an op is a "walk-through" op for leaf collection.
// These are TRT nodes whose source tensors should be recursively walked
// to find true leaf inputs, similar to shape ops.
// CPY/DUP: only src[0] is data; src[1] is dest template (shares output ptr).
// CONT: only src[0] is data.
static bool is_walkthrough_op(ggml_op op) {
    return is_shape_op(op) || op == GGML_OP_CPY || op == GGML_OP_DUP || op == GGML_OP_CONT;
}

// Check if an op is a trivial CUDA op executed outside the TRT engine.
// These are handled directly by custom CUDA kernels in graph_compute.
static bool is_trivial_op(ggml_op op) {
    return op == GGML_OP_SET_ROWS;
}

// Check if a tensor's op is TRT-compatible (built into the TRT engine).
// Uses NetworkBuilder::is_operation_supported for compute/shape/copy ops,
// plus the partial VIEW guard (partial views are subgraph boundaries).
static bool is_trt_compatible(const ggml_tensor * node) {
    if (node->op == GGML_OP_VIEW) {
        // Partial views (slices) extract a subset of elements and cannot be
        // expressed as a TRT reshape.  Only full views (same element count)
        // are reshapes → TRT node.
        return node->src[0] &&
               ggml_nelements(node) == ggml_nelements(node->src[0]);
    }
    return NetworkBuilder::is_operation_supported(node->op);
}

// Walk through shape ops (VIEW, RESHAPE, PERMUTE, TRANSPOSE) to find the
// originating compute node that owns the data.  Shape ops share data
// pointers with src[0], so consumers of shape ops are really consumers of
// the root compute node.
static const ggml_tensor * find_data_root(const ggml_tensor * t) {
    while (t && is_shape_op(t->op) && t->src[0]) {
        t = t->src[0];
    }
    return t;
}

// Consumer map: compute node → list of consumer node indices in the subgraph.
// Shape ops are transparent — their consumers register against the root compute
// node that owns the data (via find_data_root).
using consumer_map_t = std::unordered_map<const ggml_tensor *, std::vector<int>>;

// Recursively collect true leaf inputs, walking through shape op chains.
// Shape ops (VIEW, RESHAPE, PERMUTE, TRANSPOSE) pass through to their src[0].
// GGML_OP_NONE tensors are leaves.  Tensors produced by other TRT nodes in the
// same subgraph are skipped.  Walk-through ops (shape ops, CPY/DUP/CONT) that
// are part of our TRT subgraph are traversed to find their true data sources.
// Everything else (tensors from other backends, trivial ops like SET_ROWS)
// is treated as a leaf with valid GPU data.
static void collect_leaves_recursive(
    const ggml_tensor* tensor,
    std::vector<const ggml_tensor*>& leaf_tensors,
    std::unordered_set<const ggml_tensor*>& leaf_seen,
    const std::unordered_set<const ggml_tensor*>& trt_node_set
) {
    if (tensor == nullptr) {
        return;
    }
    if (tensor->op == GGML_OP_NONE) {
        // True leaf input (weight, embedding, etc.)
        if (leaf_seen.insert(tensor).second) {
            leaf_tensors.push_back(tensor);
        }
        return;
    }
    if (is_walkthrough_op(tensor->op)) {
        if (trt_node_set.count(tensor)) {
            // Walk-through op is part of our TRT subgraph — traverse src[0]
            // to find the true leaf.  Only src[0] — for CPY/DUP, src[1] is
            // the destination template sharing the output data pointer.
            collect_leaves_recursive(tensor->src[0], leaf_tensors, leaf_seen, trt_node_set);
        } else {
            // Walk-through op NOT in our subgraph (e.g. partial view) — its
            // data pointer is valid, so treat the tensor itself as a leaf.
            if (leaf_seen.insert(tensor).second) {
                leaf_tensors.push_back(tensor);
            }
        }
        return;
    }
    if (trt_node_set.count(tensor)) {
        // Produced by another TRT node in our subgraph — skip
        return;
    }
    // Tensor produced outside our TRT subgraph (another backend, trivial op
    // like SET_ROWS, etc.).  Its GPU data is valid, so treat as a leaf input.
    if (leaf_seen.insert(tensor).second) {
        leaf_tensors.push_back(tensor);
        GGML_LOG_DEBUG("%s: external tensor op=%s type=%s shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] added as leaf\n",
            __func__, ggml_op_name(tensor->op), ggml_type_name(tensor->type),
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    }
}

// Helper: read first N float values from GPU tensor for debug logging.
// Handles F32/F16/BF16 by converting to F32.  Returns up to n_vals floats.
static std::vector<float> debug_read_tensor_head(const ggml_tensor * tensor, cudaStream_t stream, int n_vals = 4) {
    std::vector<float> result;
    if (!tensor || !tensor->data) return result;

    int64_t nelements = ggml_nelements(tensor);
    int n = (int)std::min((int64_t)n_vals, nelements);
    if (n <= 0) return result;

    if (tensor->type == GGML_TYPE_F32) {
        result.resize(n);
        CUDA_CHECK(cudaMemcpyAsync(result.data(), tensor->data, n * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(n);
        CUDA_CHECK(cudaMemcpyAsync(buf.data(), tensor->data, n * sizeof(ggml_fp16_t),
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        result.resize(n);
        ggml_fp16_to_fp32_row(buf.data(), result.data(), n);
    } else if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> buf(n);
        CUDA_CHECK(cudaMemcpyAsync(buf.data(), tensor->data, n * sizeof(ggml_bf16_t),
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        result.resize(n);
        ggml_bf16_to_fp32_row(buf.data(), result.data(), n);
    }
    return result;
}

// Execute one TRT segment: collect leaves → hash → build/cache engine →
// I/O alias guard → bind inputs/outputs → enqueueV3 → debug logging.
// Each segment is a contiguous run of TRT-compatible nodes between trivial
// op boundaries.  The segment has its own leaf set and cached engine.
static enum ggml_status execute_trt_segment(
    ggml_backend_tensorrt_context * ctx,
    const ggml_cgraph * cgraph,
    const std::vector<int> & trt_node_indices,
    const std::unordered_set<const ggml_tensor *> & trt_node_set,
    const consumer_map_t & consumers,
    bool debug_enabled,
    int64_t segment_id
) {
    // Address tracking for debug mode (detect address changes between calls)
    static std::unordered_map<uint64_t, std::vector<void*>> prev_input_addrs;

    // ── Collect leaf inputs for this segment ──

    std::vector<const ggml_tensor*> leaf_tensors;
    std::unordered_set<const ggml_tensor*> leaf_seen;

    for (int idx : trt_node_indices) {
        const ggml_tensor * node = cgraph->nodes[idx];
        // CPY/DUP: only src[0] provides data; src[1] is the destination
        // template sharing the output data pointer.  Walking src[1] would
        // add it as a leaf input at the same address as the output → I/O aliasing.
        const int n_src = (node->op == GGML_OP_CPY || node->op == GGML_OP_DUP)
                        ? 1 : GGML_MAX_SRC;
        for (int j = 0; j < n_src; j++) {
            if (node->src[j]) {
                collect_leaves_recursive(node->src[j], leaf_tensors, leaf_seen, trt_node_set);
            }
        }
    }

    if (trt_node_indices.empty()) {
        return GGML_STATUS_SUCCESS;
    }

    // Check if segment has any compute (non-shape) ops.  A segment of only
    // shape ops (VIEW, RESHAPE, PERMUTE, TRANSPOSE) produces no TRT network
    // outputs and is a no-op — the data pointers already alias the correct
    // memory.  This happens after SET_ROWS when the next nodes are VIEWs of
    // the KV cache before the graph transitions to another backend (ROPE).
    {
        bool has_compute_op = false;
        for (int idx : trt_node_indices) {
            if (!is_shape_op(cgraph->nodes[idx]->op)) {
                has_compute_op = true;
                break;
            }
        }
        if (!has_compute_op) {
            GGML_LOG_DEBUG("%s: segment %" PRId64 " has only shape ops (%zu nodes), skipping\n",
                __func__, segment_id, trt_node_indices.size());
            return GGML_STATUS_SUCCESS;
        }
    }

    // ── Determine segment outputs ──
    //
    // A non-shape node is a segment output if:
    //   1. GGML_TENSOR_FLAG_SUBGRAPH_OUTPUT — consumed by another scheduler
    //      split (cross-subgraph output, set by the scheduler), OR
    //   2. Any consumer in the subgraph is NOT in this TRT segment — i.e.
    //      the node is consumed by a trivial op (SET_ROWS) or a node in
    //      another segment (inter-segment output).
    //
    // This is purely structural (address-independent), so all transformer
    // layers with identical graph structure produce the same output set →
    // one cached engine shared across layers.
    auto is_segment_output = [&](const ggml_tensor * node) -> bool {
        if (is_shape_op(node->op)) return false;
        // Cross-subgraph: scheduler flag
        if (node->flags & GGML_TENSOR_FLAG_SUBGRAPH_OUTPUT) return true;
        // Inter-segment: check if any consumer is outside this TRT segment
        auto it = consumers.find(node);
        if (it == consumers.end()) return true;  // no known consumers → must be an output
        for (int ci : it->second) {
            if (!trt_node_set.count(cgraph->nodes[ci])) return true;
        }
        return false;
    };

    // ── Classify leaves as static (weights) or dynamic (activations) ──
    //
    // Static leaves have constant shapes across all calls (model weights,
    // normalization scales).  Dynamic leaves have batch/token dims that change
    // between prompt and decode (hidden states, position IDs, KV views).

    std::unordered_set<const ggml_tensor *> static_leaf_set;
    bool has_dynamic_inputs = false;
    for (size_t k = 0; k < leaf_tensors.size(); k++) {
        if (is_static_leaf(leaf_tensors[k])) {
            static_leaf_set.insert(leaf_tensors[k]);
        } else {
            has_dynamic_inputs = true;
        }
    }

    // ── Engine cache lookup ──
    //
    // Shape-agnostic hash: static leaf shapes are fully hashed (weight dims
    // are constant), dynamic tensor shapes are excluded (only type + ndims).
    // This makes the hash batch-independent — one engine serves all batch sizes.

    uint64_t hash = compute_graph_hash(cgraph, trt_node_indices, static_leaf_set);

    nvinfer1::ICudaEngine* engine = ctx->engine_mgr->get_cached_engine(hash);

    bool was_cache_miss = (engine == nullptr);

    // ── Cache miss — build engine ──

    if (engine == nullptr) {
        // Diagnostic dump: log the subgraph structure before building
        static const bool dump_graph = (getenv("GGML_TENSORRT_DUMP_GRAPH") != nullptr &&
                                        atoi(getenv("GGML_TENSORRT_DUMP_GRAPH")) != 0);
        if (dump_graph) {
            fprintf(stderr, "[TensorRT-RTX] building TRT segment %" PRId64 " — hash 0x%016" PRIx64 ", %zu leaves, %zu nodes\n",
                segment_id, hash, leaf_tensors.size(), trt_node_indices.size());

            for (size_t k = 0; k < leaf_tensors.size(); k++) {
                const ggml_tensor * leaf = leaf_tensors[k];
                fprintf(stderr, "[TensorRT-RTX]   input_%zu: op=%-12s type=%-4s shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] name=%s\n",
                    k, ggml_op_name(leaf->op), ggml_type_name(leaf->type),
                    leaf->ne[0], leaf->ne[1], leaf->ne[2], leaf->ne[3],
                    leaf->name);
            }

            for (size_t ni = 0; ni < trt_node_indices.size(); ni++) {
                const ggml_tensor * node = cgraph->nodes[trt_node_indices[ni]];
                const char * op_str = ggml_op_name(node->op);

                char op_buf[64];
                if (node->op == GGML_OP_UNARY) {
                    snprintf(op_buf, sizeof(op_buf), "UNARY(%s)",
                        ggml_unary_op_name(ggml_get_unary_op(node)));
                    op_str = op_buf;
                } else if (node->op == GGML_OP_GLU) {
                    snprintf(op_buf, sizeof(op_buf), "GLU(%s)",
                        ggml_glu_op_name(ggml_get_glu_op(node)));
                    op_str = op_buf;
                }

                fprintf(stderr, "[TensorRT-RTX]   node %3zu [%3d]: %-20s -> type=%-4s shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]%s\n",
                    ni, trt_node_indices[ni], op_str,
                    ggml_type_name(node->type),
                    node->ne[0], node->ne[1], node->ne[2], node->ne[3],
                    is_shape_op(node->op) ? " (shape)" : "");

                for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                    const ggml_tensor * src = node->src[s];
                    fprintf(stderr, "[TensorRT-RTX]     src[%d]: op=%-12s type=%-4s shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] name=%s\n",
                        s, ggml_op_name(src->op), ggml_type_name(src->type),
                        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
                        src->name);
                }
            }
        }

        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(*ctx->logger));
        if (!builder) {
            GGML_LOG_ERROR("%s: failed to create TensorRT builder\n", __func__);
            return GGML_STATUS_FAILED;
        }

        const uint32_t explicit_batch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
        if (!network) {
            GGML_LOG_ERROR("%s: failed to create TensorRT network\n", __func__);
            return GGML_STATUS_FAILED;
        }

        NetworkBuilder net_builder(network.get(), ctx->logger);

        // Add leaf tensors as inputs with positional names.
        // Dynamic leaves get wildcard dims (-1); static leaves get concrete dims.
        for (size_t k = 0; k < leaf_tensors.size(); k++) {
            char name[64];
            snprintf(name, sizeof(name), "input_%zu", k);
            bool is_dynamic = !static_leaf_set.count(leaf_tensors[k]);
            if (!net_builder.add_input(leaf_tensors[k], name, is_dynamic)) {
                GGML_LOG_ERROR("%s: failed to add input tensor input_%zu\n", __func__, k);
                return GGML_STATUS_FAILED;
            }
        }

        // Process all TRT nodes.  Nodes identified as segment outputs (via
        // scheduler flag or consumer analysis) become TRT network outputs.
        // All other nodes are TRT-internal intermediates.
        int output_idx = 0;
        for (size_t ni = 0; ni < trt_node_indices.size(); ni++) {
            ggml_tensor * node = cgraph->nodes[trt_node_indices[ni]];

            nvinfer1::ITensor* output = net_builder.add_operation(node);
            if (!output) {
                GGML_LOG_ERROR("%s: operation %s not supported in TRT graph\n",
                               __func__, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
            }

            if (is_segment_output(node)) {
                nvinfer1::DataType expected_type = ggml_type_to_tensorrt(node->type);
                output = net_builder.maybe_cast(output, expected_type);

                char out_name[64];
                snprintf(out_name, sizeof(out_name), "output_%d", output_idx);
                net_builder.mark_output(output, out_name);
                output_idx++;
            }
        }

        // Build engine
        EngineConfig engine_config;

        // Minimum workspace floor — TRT's optimizer crashes with the assertion
        // "maxScratchSize > 0" if workspace is zero.  64 MB is enough for
        // most single-layer engines and prevents cascading failures.
        static constexpr size_t min_workspace = 64ULL << 20;  // 64 MB

        const char* workspace_env = getenv("GGML_TENSORRT_WORKSPACE_MB");
        if (workspace_env) {
            engine_config.max_workspace_size = (size_t)atoi(workspace_env) << 20;
        } else {
            // TRT needs VRAM both for workspace AND for internal build
            // allocations (serialized engine, compiler intermediates).
            // Cap workspace at 1/8 of free VRAM to leave headroom for
            // TRT's own allocations, which can be 256+ MB for large
            // dynamic-shape engines (logits matmul with big vocab).
            size_t free_bytes = 0, total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            size_t eighth_free = free_bytes / 8;
            if (eighth_free < engine_config.max_workspace_size) {
                engine_config.max_workspace_size = eighth_free;
            }
        }

        // Enforce minimum — never pass 0 workspace to TRT
        if (engine_config.max_workspace_size < min_workspace) {
            engine_config.max_workspace_size = min_workspace;
        }

        const char* aux_streams_env = getenv("GGML_TENSORRT_AUX_STREAMS");
        if (aux_streams_env) {
            engine_config.max_aux_streams = atoi(aux_streams_env);
        }

        // Build optimization profiles for dynamic inputs
        //
        // The key constraint: TRT's builder allocates intermediate buffers at
        // the MAX profile dimensions during tactic profiling.  For segments
        // containing a MUL_MAT with large weight matrices (e.g. n_vocab=262K),
        // the intermediate size is approximately:
        //   max_weight_dim × max_dynamic_dim0 × sizeof(float)
        //
        // We scale max_dynamic_dim0 inversely with the largest weight dimension
        // so this product stays within a memory budget.  This naturally gives
        // generous ranges for regular transformer layers (weight dims ~6K)
        // and tight ranges for the logits segment (weight dim = n_vocab).

        // Find the largest dimension across all static (weight) leaves.
        // This approximates the "cost factor" per unit of dynamic dim growth.
        int64_t max_weight_dim = 1;
        if (has_dynamic_inputs) {
            for (const auto* leaf : leaf_tensors) {
                if (!static_leaf_set.count(leaf)) continue;
                for (int d = 0; d < GGML_MAX_DIMS; d++) {
                    if (leaf->ne[d] > (int64_t)max_weight_dim) {
                        max_weight_dim = leaf->ne[d];
                    }
                }
            }
        }

        // Budget: cap the estimated largest intermediate tensor during build.
        // With ~10 live intermediates in a 29-node segment, 128 MB each keeps
        // total builder memory around 1-2 GB — well within typical free VRAM.
        const int64_t build_budget_bytes = 128LL * 1024 * 1024;
        int64_t dim0_cap = build_budget_bytes / (max_weight_dim * (int64_t)sizeof(float));
        if (dim0_cap < 4) dim0_cap = 4;
        if (dim0_cap > 4096) dim0_cap = 4096;

        std::vector<input_profile> profiles;
        if (has_dynamic_inputs) {
            for (size_t k = 0; k < leaf_tensors.size(); k++) {
                char name[64];
                snprintf(name, sizeof(name), "input_%zu", k);
                nvinfer1::Dims actual = ggml_tensor_to_dims(leaf_tensors[k]);

                input_profile p;
                p.name = name;

                if (static_leaf_set.count(leaf_tensors[k])) {
                    // Static: min = opt = max = actual
                    p.min_dims = actual;
                    p.opt_dims = actual;
                    p.max_dims = actual;
                } else {
                    // Dynamic: only dim 0 (token/batch) varies.
                    // Feature dims are architecturally fixed — varying them
                    // would violate matmul shape constraints (K must match).
                    p.min_dims = actual;
                    p.opt_dims = actual;
                    p.max_dims = actual;
                    p.min_dims.d[0] = 1;
                    int64_t expanded = actual.d[0] * 4;
                    if (expanded < 16) expanded = 16;
                    if (expanded > dim0_cap) expanded = dim0_cap;
                    // Never set max below actual (profile violation)
                    if (expanded < actual.d[0]) expanded = actual.d[0];
                    p.max_dims.d[0] = expanded;
                }
                profiles.push_back(p);
            }
        }

        // Diagnostic: log build details and VRAM state
        {
            size_t free_bytes = 0, total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            int64_t max_dyn_dim0 = 0;
            for (size_t k = 0; k < profiles.size(); k++) {
                if (k < leaf_tensors.size() && !static_leaf_set.count(leaf_tensors[k])) {
                    if (profiles[k].max_dims.d[0] > max_dyn_dim0) {
                        max_dyn_dim0 = profiles[k].max_dims.d[0];
                    }
                }
            }
            GGML_LOG_DEBUG("%s: building engine (segment %" PRId64 ", hash 0x%016" PRIx64
                ", %zu nodes, %zu leaves [%zu static, %zu dynamic], "
                "workspace %zu MB, GPU free %zu MB / %zu MB, "
                "dynamic=%s, dim0_cap=%" PRId64 ", max_weight_dim=%" PRId64
                ", max_dyn_dim0=%" PRId64 ")\n",
                __func__, segment_id, hash, trt_node_indices.size(), leaf_tensors.size(),
                static_leaf_set.size(), leaf_tensors.size() - static_leaf_set.size(),
                engine_config.max_workspace_size >> 20, free_bytes >> 20, total_bytes >> 20,
                has_dynamic_inputs ? "yes" : "no", dim0_cap, max_weight_dim, max_dyn_dim0);
        }

        if (has_dynamic_inputs) {
            engine = ctx->engine_mgr->build_engine(builder.get(), network.get(), engine_config, profiles);
        } else {
            engine = ctx->engine_mgr->build_engine(builder.get(), network.get(), engine_config);
        }
        if (!engine) {
            size_t free_after = 0, total_after = 0;
            cudaMemGetInfo(&free_after, &total_after);
            GGML_LOG_ERROR("%s: failed to build TensorRT engine "
                "(segment %" PRId64 ", GPU free %zu MB / %zu MB after failed build)\n",
                __func__, segment_id, free_after >> 20, total_after >> 20);
            return GGML_STATUS_FAILED;
        }

        ctx->engine_mgr->cache_engine(hash, engine);
    }

    // ── Execute ──

    // Disable CUDA graphs for dynamic-shape engines — shape changes break replay
    bool enable_cuda_graphs = !has_dynamic_inputs;
    nvinfer1::IExecutionContext* exec_ctx = ctx->engine_mgr->get_or_create_context(hash, enable_cuda_graphs);
    if (!exec_ctx) {
        GGML_LOG_ERROR("%s: failed to get execution context\n", __func__);
        return GGML_STATUS_FAILED;
    }

    // Set actual input shapes for all inputs (required before enqueueV3
    // when using optimization profiles — TRT needs concrete shapes for both
    // static and dynamic inputs in the profile).
    if (has_dynamic_inputs) {
        for (size_t k = 0; k < leaf_tensors.size(); k++) {
            char name[64];
            snprintf(name, sizeof(name), "input_%zu", k);
            nvinfer1::Dims actual = ggml_tensor_to_dims(leaf_tensors[k]);
            if (!exec_ctx->setInputShape(name, actual)) {
                GGML_LOG_ERROR("%s: failed to set input shape for %s\n", __func__, name);
                return GGML_STATUS_FAILED;
            }
        }
    }

    // ── I/O aliasing guard ──
    //
    // WHY: GGML's allocator reuses device addresses for tensors whose
    // lifetimes don't overlap when executed sequentially.  TRT's enqueueV3()
    // reads all inputs and writes all outputs in one fused launch — effectively
    // simultaneously.  If a leaf input and a segment output share an address,
    // TRT writes the output while still reading the input at that address →
    // read/write race → garbage.  We detect these collisions and copy the
    // affected leaf inputs to scratch memory so TRT sees distinct addresses.
    //
    // HOW: Collect all output addresses, then check each leaf.  Aliased
    // leaves are copied to scratch via cudaMemcpyAsync before enqueueV3.
    //
    // CRITICAL: Pre-compute the total scratch needed and do ONE alloc() call.
    // The scratch_buffer is a bump allocator with dynamic growth — when it
    // grows, it cudaFree()s the old buffer and cudaMalloc()s a new one.
    // If we call alloc() per leaf, a growth in the Nth call frees the buffer
    // that previous alloc() results point into → use-after-free.  A single
    // alloc() for the total ensures at most one realloc, and all sub-offsets
    // remain valid within the same buffer.

    std::unordered_set<void *> output_addrs;
    for (size_t ni = 0; ni < trt_node_indices.size(); ni++) {
        ggml_tensor * node = cgraph->nodes[trt_node_indices[ni]];
        if (is_segment_output(node)) {
            output_addrs.insert(node->data);
        }
    }

    // Pre-compute total scratch needed for all I/O aliases
    size_t total_scratch = 0;
    for (size_t k = 0; k < leaf_tensors.size(); k++) {
        if (output_addrs.count(leaf_tensors[k]->data)) {
            size_t nbytes = ggml_nbytes(leaf_tensors[k]);
            total_scratch += (nbytes + 255) & ~255;  // match alloc() alignment
        }
    }

    void * scratch_base = nullptr;
    if (total_scratch > 0) {
        scratch_base = ctx->scratch.alloc(total_scratch);
    }

    size_t scratch_off = 0;
    std::vector<void *> leaf_bind_addrs(leaf_tensors.size());
    for (size_t k = 0; k < leaf_tensors.size(); k++) {
        void * addr = leaf_tensors[k]->data;
        if (output_addrs.count(addr)) {
            size_t nbytes = ggml_nbytes(leaf_tensors[k]);
            void * scratch_addr = (char *)scratch_base + scratch_off;
            scratch_off += (nbytes + 255) & ~255;
            CUDA_CHECK(cudaMemcpyAsync(scratch_addr, addr, nbytes,
                                       cudaMemcpyDeviceToDevice, ctx->stream));
            leaf_bind_addrs[k] = scratch_addr;

            if (debug_enabled) {
                fprintf(stderr, "[TRT-DEBUG] I/O alias: seg %" PRId64 " leaf input_%zu addr %p (%zu bytes) "
                        "collides with output, copied to scratch %p\n",
                        segment_id, k, addr, nbytes, scratch_addr);
            }
        } else {
            leaf_bind_addrs[k] = addr;
        }
    }

    // Bind input addresses
    for (size_t k = 0; k < leaf_tensors.size(); k++) {
        char name[64];
        snprintf(name, sizeof(name), "input_%zu", k);
        if (!exec_ctx->setTensorAddress(name, leaf_bind_addrs[k])) {
            GGML_LOG_ERROR("%s: failed to set input tensor address for %s\n", __func__, name);
            return GGML_STATUS_FAILED;
        }
    }

    // Bind output addresses — only segment outputs (flag + consumer analysis)
    int n_outputs = 0;
    for (size_t ni = 0; ni < trt_node_indices.size(); ni++) {
        ggml_tensor * node = cgraph->nodes[trt_node_indices[ni]];
        if (!is_segment_output(node)) continue;
        char name[64];
        snprintf(name, sizeof(name), "output_%d", n_outputs);
        if (!exec_ctx->setTensorAddress(name, node->data)) {
            GGML_LOG_ERROR("%s: failed to set output tensor address for %s\n", __func__, name);
            return GGML_STATUS_FAILED;
        }
        n_outputs++;
    }

    // Debug: log input data before execution
    if (debug_enabled) {
        fprintf(stderr, "[TRT-DEBUG] seg %" PRId64 ", hash 0x%016" PRIx64 ", inputs: %zu, outputs: %d\n",
            segment_id, hash, leaf_tensors.size(), n_outputs);

        for (size_t k = 0; k < leaf_tensors.size(); k++) {
            const ggml_tensor * leaf = leaf_tensors[k];
            auto vals = debug_read_tensor_head(leaf, ctx->stream);
            fprintf(stderr, "[TRT-DEBUG]   input_%zu: addr=%p, type=%s, shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]",
                k, leaf->data, ggml_type_name(leaf->type),
                leaf->ne[0], leaf->ne[1], leaf->ne[2], leaf->ne[3]);
            if (!vals.empty()) {
                fprintf(stderr, ", first=[");
                for (size_t v = 0; v < vals.size(); v++) {
                    if (v > 0) fprintf(stderr, ", ");
                    fprintf(stderr, "%.4g", vals[v]);
                }
                fprintf(stderr, "]");
            }
            fprintf(stderr, "\n");
        }

        // Address change detection
        auto& prev_addrs = prev_input_addrs[hash];
        std::vector<void*> cur_addrs(leaf_tensors.size());
        for (size_t k = 0; k < leaf_tensors.size(); k++) {
            cur_addrs[k] = leaf_tensors[k]->data;
        }
        if (!prev_addrs.empty() && prev_addrs.size() == cur_addrs.size()) {
            for (size_t k = 0; k < cur_addrs.size(); k++) {
                if (cur_addrs[k] != prev_addrs[k]) {
                    fprintf(stderr, "[TRT-DEBUG]   input_%zu: address changed %p → %p\n",
                        k, prev_addrs[k], cur_addrs[k]);
                }
            }
        }
        prev_addrs = cur_addrs;
    }

    // Execute
    if (!exec_ctx->enqueueV3(ctx->stream)) {
        GGML_LOG_ERROR("%s: failed to execute TensorRT engine (segment %" PRId64 ")\n", __func__, segment_id);
        return GGML_STATUS_FAILED;
    }

    // Debug: log output data after execution
    if (debug_enabled) {
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        int out_idx = 0;
        for (size_t ni = 0; ni < trt_node_indices.size(); ni++) {
            ggml_tensor * node = cgraph->nodes[trt_node_indices[ni]];
            if (!is_segment_output(node)) continue;
            auto vals = debug_read_tensor_head(node, ctx->stream);
            fprintf(stderr, "[TRT-DEBUG]   output_%d: addr=%p, type=%s, shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]",
                out_idx, node->data, ggml_type_name(node->type),
                node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
            if (!vals.empty()) {
                fprintf(stderr, ", first=[");
                for (size_t v = 0; v < vals.size(); v++) {
                    if (v > 0) fprintf(stderr, ", ");
                    fprintf(stderr, "%.4g", vals[v]);
                }
                fprintf(stderr, "]");
            }
            fprintf(stderr, "\n");
            out_idx++;
        }
    }

    (void)was_cache_miss; // used only for profiling, which is done at graph_compute level

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_tensorrt_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_tensorrt_context * ctx = (ggml_backend_tensorrt_context *) backend->context;

    CUDA_CHECK(cudaSetDevice(ctx->device));

    // Instrumentation: env var checks (cached on first call)
    static const bool debug_enabled  = (getenv("GGML_TENSORRT_DEBUG") != nullptr &&
                                         atoi(getenv("GGML_TENSORRT_DEBUG")) != 0);
    static const bool profile_enabled = (getenv("GGML_TENSORRT_PROFILE") != nullptr &&
                                          atoi(getenv("GGML_TENSORRT_PROFILE")) != 0);
    static const bool dump_subgraph  = (getenv("GGML_TENSORRT_DUMP_SUBGRAPH") != nullptr &&
                                         atoi(getenv("GGML_TENSORRT_DUMP_SUBGRAPH")) != 0);

    static int64_t call_counter = 0;
    int64_t call_id = call_counter++;

    auto profile_now = []() { return std::chrono::high_resolution_clock::now(); };
    auto profile_start = profile_now();

    // Reset scratch buffer (shared across all segments this call)
    ctx->scratch.reset();

    // ── Build consumer map ──
    //
    // For each non-shape-op node, find its true consumers by walking through
    // transparent shape ops.  This map is shared across all segments and is
    // used by execute_trt_segment to determine which nodes are inter-segment
    // outputs (consumed by SET_ROWS or nodes in a different segment).
    consumer_map_t consumers;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (is_shape_op(node->op)) continue;
        for (int j = 0; j < GGML_MAX_SRC && node->src[j]; j++) {
            const ggml_tensor * root = find_data_root(node->src[j]);
            if (root) {
                consumers[root].push_back(i);
            }
        }
    }

    // ── Segmented execution ──
    //
    // Walk nodes in topological order.  Consecutive TRT-compatible nodes
    // accumulate into a segment.  Trivial ops (SET_ROWS) that depend on
    // the current segment are deferred.  When a TRT node appears after
    // deferred trivial ops, we flush the current segment (execute its TRT
    // engine + run deferred trivial ops) before starting a new segment.
    //
    // This correctly handles implicit memory dependencies through shared
    // buffers (KV cache) where SET_ROWS writes data that later TRT nodes
    // read via VIEW — there's no src[] edge, the dependency is positional.

    std::vector<int>                        seg_trt_indices;
    std::unordered_set<const ggml_tensor *> seg_trt_set;
    std::vector<ggml_tensor *>              pending_trivial;
    int64_t segment_id = 0;
    int64_t n_segments = 0;

    auto flush_segment = [&]() -> ggml_status {
        if (!seg_trt_indices.empty()) {
            ggml_status status = execute_trt_segment(
                ctx, cgraph, seg_trt_indices, seg_trt_set, consumers,
                debug_enabled, call_id * 100 + segment_id);
            if (status != GGML_STATUS_SUCCESS) return status;
            segment_id++;
            n_segments++;
        }
        // Execute deferred trivial ops (SET_ROWS) — their inputs are now
        // produced by the TRT engine we just ran.
        for (ggml_tensor * node : pending_trivial) {
            if (node->op == GGML_OP_SET_ROWS) {
                ggml_tensorrt_set_rows(node->src[0], node->src[1], node, ctx->stream);
            }
        }
        seg_trt_indices.clear();
        seg_trt_set.clear();
        pending_trivial.clear();
        ctx->scratch.reset();  // reclaim scratch for next segment
        return GGML_STATUS_SUCCESS;
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_NONE) continue;

        if (is_trivial_op(node->op)) {
            // Check if any source was produced by the current TRT segment.
            // If so, defer execution until after the segment's engine runs.
            // Otherwise, execute immediately (data from a previous subgraph).
            bool depends_on_segment = false;
            for (int j = 0; j < GGML_MAX_SRC && node->src[j]; j++) {
                if (seg_trt_set.count(node->src[j])) {
                    depends_on_segment = true;
                    break;
                }
            }
            if (depends_on_segment) {
                pending_trivial.push_back(node);
            } else {
                // No dependency on current segment — execute now.
                // (Data comes from previous subgraph or host.)
                ggml_tensorrt_set_rows(node->src[0], node->src[1], node, ctx->stream);
            }
            continue;
        }

        if (is_trt_compatible(node)) {
            // If there are pending trivial ops, we must flush the current
            // segment first.  The trivial ops write to shared buffers (KV
            // cache) that this new TRT node will read via implicit memory
            // dependencies (VIEW, no src[] edge).
            if (!pending_trivial.empty()) {
                ggml_status s = flush_segment();
                if (s != GGML_STATUS_SUCCESS) return s;
            }
            seg_trt_indices.push_back(i);
            seg_trt_set.insert(node);
            continue;
        }

        // Unrecognized op (e.g. partial VIEW treated as boundary) — skip.
        // It is not added to any segment.
    }

    // Flush the final segment
    ggml_status final_status = flush_segment();

    // ── Subgraph dump: log all node data after all segments execute ──
    if (dump_subgraph) {
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        fprintf(stderr, "[TRT-SUBGRAPH] call #%" PRId64 ", %d nodes, %" PRId64 " segments\n",
            call_id, cgraph->n_nodes, n_segments);
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * node = cgraph->nodes[i];
            auto vals = debug_read_tensor_head(node, ctx->stream, 8);
            fprintf(stderr, "[TRT-SUBGRAPH]   node[%3d] op=%-16s type=%-4s shape=[%4" PRId64 ",%4" PRId64 ",%4" PRId64 ",%4" PRId64 "] addr=%p",
                i, ggml_op_name(node->op), ggml_type_name(node->type),
                node->ne[0], node->ne[1], node->ne[2], node->ne[3],
                node->data);
            if (!vals.empty()) {
                fprintf(stderr, " first=[");
                for (size_t v = 0; v < vals.size(); v++) {
                    if (v > 0) fprintf(stderr, ", ");
                    fprintf(stderr, "%.6g", vals[v]);
                }
                fprintf(stderr, "]");
            }
            fprintf(stderr, "\n");
        }
    }

    auto compute_end = profile_now();

    // ── Profile: log timing and segment info ──
    if (profile_enabled) {
        auto to_ms = [](auto start, auto end) {
            return std::chrono::duration<double, std::milli>(end - start).count();
        };
        fprintf(stderr, "[TRT-PROF] call #%" PRId64 ": segments=%" PRId64 ", total=%.2fms\n",
            call_id, n_segments, to_ms(profile_start, compute_end));

        fprintf(stderr, "[TRT-PROF] cache: hits=%" PRId64 ", misses=%" PRId64 ", total_build_time=%.0fms\n",
            ctx->engine_mgr->cache_hits, ctx->engine_mgr->cache_misses,
            ctx->engine_mgr->total_build_time_ms);
    }

    return final_status;
}

static const ggml_backend_i ggml_backend_tensorrt_interface = {
    /* .get_name                = */ ggml_backend_tensorrt_get_name,
    /* .free                    = */ ggml_backend_tensorrt_free,
    /* .set_tensor_async        = */ ggml_backend_tensorrt_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_tensorrt_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_tensorrt_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_tensorrt_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_tensorrt_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_tensorrt_guid() {
    static ggml_guid guid = { 0x74, 0x72, 0x74, 0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };
    return &guid;
}

//
// Public API implementation
//

bool ggml_backend_is_tensorrt(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_tensorrt_guid());
}

int ggml_backend_tensorrt_get_device_count() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to get device count: %s\n", __func__, cudaGetErrorString(err));
        return 0;
    }
    return count;
}

void ggml_backend_tensorrt_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s (TensorRT-RTX)", prop.name);
}

void ggml_backend_tensorrt_get_device_memory(int device, size_t * free, size_t * total) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemGetInfo(free, total));
}

ggml_backend_t ggml_backend_tensorrt_init(int device) {
    if (device < 0 || device >= ggml_backend_tensorrt_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_tensorrt_context * ctx = new ggml_backend_tensorrt_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t tensorrt_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_tensorrt_guid(),
        /* .iface   = */ ggml_backend_tensorrt_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_tensorrt_reg(), device),
        /* .context = */ ctx,
    };

    return tensorrt_backend;
}

//
// Backend registry
//

static const char * ggml_backend_tensorrt_device_get_name(ggml_backend_dev_t dev) {
    int device_index = (int)(intptr_t)dev->context;
    static char name[512];
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_index);
    snprintf(name, sizeof(name), "%.480s (TensorRT-RTX)", prop.name);
    return name;
}

static const char * ggml_backend_tensorrt_device_get_description(ggml_backend_dev_t dev) {
    int device_index = (int)(intptr_t)dev->context;
    static char description[512];
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_index);
    snprintf(description, sizeof(description),
             "%.448s (TensorRT-RTX) - Compute Capability %d.%d",
             prop.name, prop.major, prop.minor);
    return description;
}

static void ggml_backend_tensorrt_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    int device_index = (int)(intptr_t)dev->context;
    ggml_backend_tensorrt_get_device_memory(device_index, free, total);
}

static enum ggml_backend_dev_type ggml_backend_tensorrt_device_get_type(ggml_backend_dev_t dev) {
    (void) dev;
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_tensorrt_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_tensorrt_device_get_name(dev);
    props->description = ggml_backend_tensorrt_device_get_description(dev);
    props->type        = ggml_backend_tensorrt_device_get_type(dev);
    ggml_backend_tensorrt_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_tensorrt_device_init(ggml_backend_dev_t dev, const char * params) {
    (void) params;
    int device_index = (int)(intptr_t)dev->context;
    return ggml_backend_tensorrt_init(device_index);
}

// Get the companion CUDA backend's default buffer type for the same CUDA device.
// Both TRT and CUDA use the same CUDA device index, so we call directly into the
// CUDA backend's public API.  Returns nullptr if CUDA backend is not available.
static ggml_backend_buffer_type_t find_cuda_buft_for_device(ggml_backend_dev_t dev) {
    int device_index = (int)(intptr_t)dev->context;
#ifdef GGML_USE_CUDA
    return ggml_backend_cuda_buffer_type(device_index);
#else
    (void) device_index;
    return nullptr;
#endif
}

static ggml_backend_buffer_type_t ggml_backend_tensorrt_device_get_buffer_type(ggml_backend_dev_t dev) {
    // Prefer companion CUDA buffer type (same physical GPU).
    // This makes sched->bufts[trt] == sched->bufts[cuda], enabling
    // the scheduler to freely route ops between TRT and CUDA.
    ggml_backend_buffer_type_t cuda_buft = find_cuda_buft_for_device(dev);
    if (cuda_buft) {
        return cuda_buft;
    }
    // Standalone mode (no CUDA backend)
    int device_index = (int)(intptr_t)dev->context;
    return ggml_backend_tensorrt_buffer_type(device_index);
}

static ggml_backend_buffer_t ggml_backend_tensorrt_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    (void) dev;
    (void) ptr;
    (void) size;
    (void) max_tensor_size;
    return nullptr;
}

static bool is_supported_compute_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F16;
}

static bool ggml_backend_tensorrt_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void) dev;

    // Metadata and data-movement ops — no type restriction, handled outside TRT engine.
    // CPY/DUP/CONT/SET_ROWS must be claimed so that the scheduler can assign them
    // to TRT when the tensors are on GPU memory (CUDA or TRT buffers).
    // These ops are NOT built into TRT engines — they run as trivial CUDA ops
    // (SET_ROWS) or are absorbed into TRT graphs (CPY/DUP/CONT) in graph_compute.
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_SET_ROWS:
            return true;
        default:
            break;
    }

    // Compute ops — F32, BF16, F16
    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SCALE:
        {
            // Output must be a supported compute type
            if (!is_supported_compute_type(op->type)) {
                return false;
            }
            // All sources must be supported compute types
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (op->src[i] && !is_supported_compute_type(op->src[i]->type)) {
                    return false;
                }
            }
            // SOFT_MAX: accept optional mask, reject ALiBi and sinks
            if (op->op == GGML_OP_SOFT_MAX) {
                // Reject sinks (src[2]) — rare, not worth complexity
                if (op->src[2] != nullptr) {
                    return false;
                }
                // Reject ALiBi (per-head slopes need complex constant generation)
                float max_bias = 0.0f;
                memcpy(&max_bias, &op->op_params[1], sizeof(float));
                if (max_bias != 0.0f) {
                    return false;
                }
                // Mask type (src[1]) must be a supported compute type
                if (op->src[1] && !is_supported_compute_type(op->src[1]->type)) {
                    return false;
                }
            }
            return true;
        }
        case GGML_OP_GET_ROWS:
        {
            // Output must be F32 (GET_ROWS always dequantizes)
            if (op->type != GGML_TYPE_F32) {
                return false;
            }
            // src[0] (data) must be F32/BF16/F16
            if (!op->src[0] || !is_supported_compute_type(op->src[0]->type)) {
                return false;
            }
            // src[1] (indices) must be I32
            if (!op->src[1] || op->src[1]->type != GGML_TYPE_I32) {
                return false;
            }
            return true;
        }
        case GGML_OP_GLU:
        {
            // Diagnostic: confirm the scheduler is probing GLU support
            {
                static bool logged = false;
                if (!logged) {
                    enum ggml_glu_op gop = ggml_get_glu_op(op);
                    GGML_LOG_WARN("%s: GLU probed — sub-op=%s out_type=%s src0_type=%s\n",
                        __func__, ggml_glu_op_name(gop),
                        ggml_type_name(op->type),
                        op->src[0] ? ggml_type_name(op->src[0]->type) : "null");
                    logged = true;
                }
            }
            // Output must be a supported compute type
            if (!is_supported_compute_type(op->type)) {
                return false;
            }
            // All sources must be supported compute types
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (op->src[i] && !is_supported_compute_type(op->src[i]->type)) {
                    return false;
                }
            }
            // Only supported GLU sub-ops
            enum ggml_glu_op gop = ggml_get_glu_op(op);
            switch (gop) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_OAI:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_ROPE:
        {
            // Output must be a supported compute type
            if (!is_supported_compute_type(op->type)) {
                return false;
            }
            // src[0] (Q/K data) must be a supported compute type
            if (!op->src[0] || !is_supported_compute_type(op->src[0]->type)) {
                return false;
            }
            // src[1] (positions) must be I32
            if (!op->src[1] || op->src[1]->type != GGML_TYPE_I32) {
                return false;
            }
            // Reject freq_factors (src[2]) — not needed for most models
            if (op->src[2] != nullptr) {
                return false;
            }
            // Only NORMAL and NEOX modes
            int mode = ((const int32_t *)op->op_params)[2];
            if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
                return false;
            }
            // Reject YaRN (complex frequency interpolation)
            float ext_factor = 0.0f;
            memcpy(&ext_factor, (const float *)op->op_params + 7, sizeof(float));
            if (ext_factor != 0.0f) {
                return false;
            }
            return true;
        }
        case GGML_OP_UNARY:
        {
            // Output must be a supported compute type
            if (!is_supported_compute_type(op->type)) {
                return false;
            }
            // All sources must be supported compute types
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (op->src[i] && !is_supported_compute_type(op->src[i]->type)) {
                    return false;
                }
            }
            // Only supported unary sub-ops
            enum ggml_unary_op uop = ggml_get_unary_op(op);
            switch (uop) {
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_EXP:
                    return true;
                default:
                    return false;
            }
        }
        default:
            break;
    }

    return false;
}

static bool ggml_backend_tensorrt_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Accept own device buffer types
    if (buft->iface.get_name == ggml_backend_tensorrt_buffer_type_get_name) {
        return true;
    }

    // Accept own host buffer types
    if (buft->iface.get_name == ggml_backend_tensorrt_host_buffer_type_get_name) {
        return true;
    }

    // Accept any host buffer type (enables scheduler copy path)
    if (ggml_backend_buft_is_host(buft)) {
        return true;
    }

    // Accept device buffers from companion backends on the same physical GPU.
    // Both TRT and CUDA allocate via cudaMalloc on the same device,
    // so device pointers are interchangeable.
    ggml_backend_buffer_type_t cuda_buft = find_cuda_buft_for_device(dev);
    if (cuda_buft && buft == cuda_buft) {
        return true;
    }

    return false;
}

static bool ggml_backend_tensorrt_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void) dev;

    // Offload compute-heavy ops to TensorRT
    return op->op == GGML_OP_MUL_MAT;
}

static const struct ggml_backend_device_i ggml_backend_tensorrt_device_interface = {
    /* .get_name                   = */ ggml_backend_tensorrt_device_get_name,
    /* .get_description            = */ ggml_backend_tensorrt_device_get_description,
    /* .get_memory                 = */ ggml_backend_tensorrt_device_get_memory,
    /* .get_type                   = */ ggml_backend_tensorrt_device_get_type,
    /* .get_props                  = */ ggml_backend_tensorrt_device_get_props,
    /* .init_backend               = */ ggml_backend_tensorrt_device_init,
    /* .get_buffer_type            = */ ggml_backend_tensorrt_device_get_buffer_type,
    /* .get_host_buffer_type       = */ NULL,
    /* .buffer_from_host_ptr       = */ ggml_backend_tensorrt_device_buffer_from_host_ptr,
    /* .supports_op                = */ ggml_backend_tensorrt_device_supports_op,
    /* .supports_buft              = */ ggml_backend_tensorrt_device_supports_buft,
    /* .offload_op                 = */ ggml_backend_tensorrt_device_offload_op,
    /* .event_new                  = */ NULL,
    /* .event_free                 = */ NULL,
    /* .event_synchronize          = */ NULL,
};

//
// Backend registry
//

static const char * ggml_backend_tensorrt_reg_get_name(ggml_backend_reg_t reg) {
    (void) reg;
    return GGML_TENSORRT_NAME;
}

static size_t ggml_backend_tensorrt_reg_get_device_count(ggml_backend_reg_t reg) {
    (void) reg;
    return ggml_backend_tensorrt_get_device_count();
}

static ggml_backend_dev_t ggml_backend_tensorrt_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void) reg;
    static std::vector<ggml_backend_device> devices;

    if (devices.empty()) {
        int device_count = ggml_backend_tensorrt_get_device_count();
        devices.resize(device_count);

        for (int i = 0; i < device_count; i++) {
            devices[i] = {
                /* .iface   = */ ggml_backend_tensorrt_device_interface,
                /* .reg     = */ reg,
                /* .context = */ (void *)(intptr_t)i,
            };
        }
    }

    if (index >= devices.size()) {
        return nullptr;
    }

    return &devices[index];
}

static void * ggml_backend_tensorrt_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    (void) reg;
    (void) name;
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_tensorrt_reg_interface = {
    /* .get_name         = */ ggml_backend_tensorrt_reg_get_name,
    /* .get_device_count = */ ggml_backend_tensorrt_reg_get_device_count,
    /* .get_device       = */ ggml_backend_tensorrt_reg_get_device,
    /* .get_proc_address = */ ggml_backend_tensorrt_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_tensorrt_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_tensorrt_reg_interface,
        /* .context     = */ nullptr,
    };

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_tensorrt_reg)
