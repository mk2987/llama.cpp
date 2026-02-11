#include "common.hpp"
#include "network-builder.hpp"
#include "engine-manager.hpp"
#include "utils/type-utils.hpp"
#include "ggml-tensorrt.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
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
    (void) backend_src;
    (void) backend_dst;

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

// Check if an op is a shape/metadata operation (zero-copy in GGML)
static bool is_shape_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// Recursively collect true leaf inputs, walking through shape op chains.
// Shape ops (VIEW, RESHAPE, PERMUTE, TRANSPOSE) pass through to their src[0].
// GGML_OP_NONE tensors are leaves.  Tensors produced by other TRT nodes in the
// same subgraph are skipped.  Everything else (tensors from other backends,
// trivial ops like CONT/CPY) is treated as a leaf with valid GPU data.
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
    if (is_shape_op(tensor->op)) {
        if (trt_node_set.count(tensor)) {
            // Shape op is part of our TRT subgraph — walk through it
            collect_leaves_recursive(tensor->src[0], leaf_tensors, leaf_seen, trt_node_set);
        } else {
            // Shape op NOT in our subgraph (e.g. partial view) — its
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
    // like CONT/CPY, etc.).  Its GPU data is valid, so treat as a leaf input.
    if (leaf_seen.insert(tensor).second) {
        leaf_tensors.push_back(tensor);
        GGML_LOG_DEBUG("%s: external tensor op=%s type=%s shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] added as leaf\n",
            __func__, ggml_op_name(tensor->op), ggml_type_name(tensor->type),
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    }
}

static enum ggml_status ggml_backend_tensorrt_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_tensorrt_context * ctx = (ggml_backend_tensorrt_context *) backend->context;

    CUDA_CHECK(cudaSetDevice(ctx->device));

    // ── Phase 1a: Categorize nodes into trivial vs TRT ──

    std::vector<int> trt_node_indices;
    std::unordered_set<const ggml_tensor*> trt_node_set;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_CPY:
            case GGML_OP_DUP:
            case GGML_OP_CONT:
            case GGML_OP_SET_ROWS:
                break; // trivial — handled separately
            case GGML_OP_VIEW:
                // Partial views (slices) extract a subset of elements and
                // cannot be expressed as a TRT reshape.  Treat as boundary.
                // Full views (same element count) are reshapes → TRT node.
                if (node->src[0] &&
                    ggml_nelements(node) != ggml_nelements(node->src[0])) {
                    break; // partial view — subgraph boundary
                }
                trt_node_indices.push_back(i);
                trt_node_set.insert(node);
                break;
            default:
                trt_node_indices.push_back(i);
                trt_node_set.insert(node);
                break;
        }
    }

    // ── Phase 1b: Execute trivial ops and collect leaf inputs ──

    std::vector<const ggml_tensor*> leaf_tensors;
    std::unordered_set<const ggml_tensor*> leaf_seen;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_NONE:
                continue;

            // Copy ops — handle via CUDA memcpy directly
            case GGML_OP_CPY:
            case GGML_OP_DUP:
            case GGML_OP_CONT:
            {
                const size_t nb = ggml_nbytes(node);
                if (node->src[0] && node->src[0]->data && node->data) {
                    CUDA_CHECK(cudaMemcpyAsync(node->data, node->src[0]->data, nb,
                                               cudaMemcpyDeviceToDevice, ctx->stream));
                }
                continue;
            }

            // SET_ROWS — scatter F32 source rows into destination (KV cache)
            case GGML_OP_SET_ROWS:
            {
                const ggml_tensor * src0 = node->src[0];  // source rows (F32)
                const ggml_tensor * src1 = node->src[1];  // indices

                const int64_t ne00 = src0->ne[0];
                const int64_t ne01 = src0->ne[1];
                const int64_t ne02 = src0->ne[2];
                const int64_t ne03 = src0->ne[3];

                const int64_t ne11 = src1->ne[1];
                const int64_t ne12 = src1->ne[2];

                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

                const size_t idx_bytes = ggml_nbytes(src1);
                std::vector<uint8_t> idx_host(idx_bytes);

                if (src1->buffer && ggml_backend_buffer_is_host(src1->buffer)) {
                    memcpy(idx_host.data(), src1->data, idx_bytes);
                } else {
                    CUDA_CHECK(cudaMemcpy(idx_host.data(), src1->data, idx_bytes, cudaMemcpyDeviceToHost));
                }

                const bool idx_i64 = (src1->type == GGML_TYPE_I64);
                auto read_idx = [&](size_t byte_offset) -> int64_t {
                    if (idx_i64) {
                        int64_t v;
                        memcpy(&v, idx_host.data() + byte_offset, sizeof(int64_t));
                        return v;
                    } else {
                        int32_t v;
                        memcpy(&v, idx_host.data() + byte_offset, sizeof(int32_t));
                        return (int64_t)v;
                    }
                };

                const size_t src_row_bytes = ne00 * sizeof(float);
                const size_t dst_row_bytes = ggml_row_size(node->type, ne00);
                const bool need_convert = (node->type != GGML_TYPE_F32);

                std::vector<float>   host_f32;
                std::vector<uint8_t> host_cvt;
                if (need_convert) {
                    host_f32.resize(ne00);
                    host_cvt.resize(dst_row_bytes);
                }

                for (int64_t i03 = 0; i03 < ne03; ++i03) {
                    for (int64_t i02 = 0; i02 < ne02; ++i02) {
                        for (int64_t i01 = 0; i01 < ne01; ++i01) {
                            const int64_t i12 = i03 % ne12;
                            const int64_t i11 = i02 % ne11;
                            const int64_t i10 = i01;

                            const size_t idx_off = i10 * src1->nb[0]
                                                 + i11 * src1->nb[1]
                                                 + i12 * src1->nb[2];
                            const int64_t dst_row = read_idx(idx_off);

                            const char * src_ptr = (const char *)src0->data
                                + i01 * src0->nb[1]
                                + i02 * src0->nb[2]
                                + i03 * src0->nb[3];

                            char * dst_ptr = (char *)node->data
                                + dst_row * node->nb[1]
                                + i02     * node->nb[2]
                                + i03     * node->nb[3];

                            if (!need_convert) {
                                CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, src_row_bytes,
                                                           cudaMemcpyDeviceToDevice, ctx->stream));
                            } else {
                                CUDA_CHECK(cudaMemcpy(host_f32.data(), src_ptr, src_row_bytes,
                                                      cudaMemcpyDeviceToHost));

                                if (node->type == GGML_TYPE_F16) {
                                    ggml_fp32_to_fp16_row(host_f32.data(),
                                                          (ggml_fp16_t *)host_cvt.data(), ne00);
                                } else if (node->type == GGML_TYPE_BF16) {
                                    ggml_fp32_to_bf16_row(host_f32.data(),
                                                          (ggml_bf16_t *)host_cvt.data(), ne00);
                                } else {
                                    GGML_LOG_ERROR("%s: SET_ROWS unsupported dst type %s\n",
                                                   __func__, ggml_type_name(node->type));
                                    return GGML_STATUS_FAILED;
                                }

                                CUDA_CHECK(cudaMemcpy(dst_ptr, host_cvt.data(), dst_row_bytes,
                                                      cudaMemcpyHostToDevice));
                            }
                        }
                    }
                }
                continue;
            }

            default:
            {
                // Only collect leaf inputs for nodes in the TRT subgraph.
                // Other nodes (e.g. partial VIEWs excluded from trt_node_set)
                // would add their parents as unnecessary inputs.
                if (trt_node_set.count(node)) {
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        if (node->src[j]) {
                            collect_leaves_recursive(node->src[j], leaf_tensors, leaf_seen, trt_node_set);
                        }
                    }
                }
                break;
            }
        }
    }

    // No compute nodes — all handled above
    if (trt_node_indices.empty()) {
        return GGML_STATUS_SUCCESS;
    }

    // ── Phase 2: Engine cache lookup ──

    uint64_t hash = compute_graph_hash(cgraph, trt_node_indices);

    nvinfer1::ICudaEngine* engine = ctx->engine_mgr->get_cached_engine(hash);

    // ── Phase 3: Cache miss — build engine ──

    if (engine == nullptr) {
        // ── Diagnostic dump: log the subgraph structure before building ──
        // Controlled by GGML_TENSORRT_DUMP_GRAPH=1.  Uses fprintf(stderr)
        // directly to bypass GGML's log callback which llama-cli filters.
        static const bool dump_graph = (getenv("GGML_TENSORRT_DUMP_GRAPH") != nullptr &&
                                        atoi(getenv("GGML_TENSORRT_DUMP_GRAPH")) != 0);
        if (dump_graph) {
            fprintf(stderr, "[TensorRT-RTX] building TRT subgraph — hash 0x%016" PRIx64 ", %zu leaves, %zu nodes\n",
                hash, leaf_tensors.size(), trt_node_indices.size());

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

                // For UNARY ops, append the sub-op name
                char op_buf[64];
                if (node->op == GGML_OP_UNARY) {
                    snprintf(op_buf, sizeof(op_buf), "UNARY(%s)",
                        ggml_unary_op_name(ggml_get_unary_op(node)));
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

        // Add leaf tensors as inputs with positional names
        for (size_t k = 0; k < leaf_tensors.size(); k++) {
            char name[64];
            snprintf(name, sizeof(name), "input_%zu", k);
            if (!net_builder.add_input(leaf_tensors[k], name)) {
                GGML_LOG_ERROR("%s: failed to add input tensor input_%zu\n", __func__, k);
                return GGML_STATUS_FAILED;
            }
        }

        // Process all TRT nodes (shape ops + compute ops)
        // Only mark compute ops (non-shape-ops) as network outputs.
        // Shape ops share data pointers with their source in GGML,
        // so binding them as outputs would create duplicate address bindings.
        int output_idx = 0;
        for (int idx : trt_node_indices) {
            ggml_tensor * node = cgraph->nodes[idx];

            nvinfer1::ITensor* output = net_builder.add_operation(node);
            if (!output) {
                GGML_LOG_ERROR("%s: operation %s not supported in TRT graph\n",
                               __func__, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
            }

            // Only mark compute ops as outputs
            if (!is_shape_op(node->op)) {
                // Cast TRT output to match GGML's expected output type.
                // E.g. ggml_mul_mat always produces F32 but TRT with BF16
                // inputs produces BF16 output in strongly-typed mode.
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

        // Workspace: use env var override, else cap at 256 MB or 25% of free VRAM
        const char* workspace_env = getenv("GGML_TENSORRT_WORKSPACE_MB");
        if (workspace_env) {
            engine_config.max_workspace_size = (size_t)atoi(workspace_env) << 20;
        } else {
            size_t free_bytes = 0, total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            size_t quarter_free = free_bytes / 4;
            if (quarter_free < engine_config.max_workspace_size) {
                engine_config.max_workspace_size = quarter_free;
            }
        }

        const char* aux_streams_env = getenv("GGML_TENSORRT_AUX_STREAMS");
        if (aux_streams_env) {
            engine_config.max_aux_streams = atoi(aux_streams_env);
        }

        {
            size_t free_bytes = 0, total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            GGML_LOG_WARN("%s: building engine (hash 0x%016" PRIx64 ", %zu nodes, workspace %zu MB, GPU free %zu MB / %zu MB)\n",
                __func__, hash, trt_node_indices.size(),
                engine_config.max_workspace_size >> 20, free_bytes >> 20, total_bytes >> 20);
        }

        engine = ctx->engine_mgr->build_engine(builder.get(), network.get(), engine_config);
        if (!engine) {
            GGML_LOG_ERROR("%s: failed to build TensorRT engine\n", __func__);
            return GGML_STATUS_FAILED;
        }

        // Cache the engine (transfers ownership)
        ctx->engine_mgr->cache_engine(hash, engine);
    }

    // ── Phase 4: Execute ──

    nvinfer1::IExecutionContext* exec_ctx = ctx->engine_mgr->get_or_create_context(hash);
    if (!exec_ctx) {
        GGML_LOG_ERROR("%s: failed to get execution context\n", __func__);
        return GGML_STATUS_FAILED;
    }

    // Bind input addresses (positional names)
    for (size_t k = 0; k < leaf_tensors.size(); k++) {
        char name[64];
        snprintf(name, sizeof(name), "input_%zu", k);
        if (!exec_ctx->setTensorAddress(name, leaf_tensors[k]->data)) {
            GGML_LOG_ERROR("%s: failed to set input tensor address for %s\n", __func__, name);
            return GGML_STATUS_FAILED;
        }
    }

    // Bind output addresses — only for compute ops (skip shape ops)
    int output_idx = 0;
    for (int idx : trt_node_indices) {
        ggml_tensor * node = cgraph->nodes[idx];
        if (is_shape_op(node->op)) {
            continue;
        }
        char name[64];
        snprintf(name, sizeof(name), "output_%d", output_idx);
        if (!exec_ctx->setTensorAddress(name, node->data)) {
            GGML_LOG_ERROR("%s: failed to set output tensor address for %s\n", __func__, name);
            return GGML_STATUS_FAILED;
        }
        output_idx++;
    }

    // Execute
    if (!exec_ctx->enqueueV3(ctx->stream)) {
        GGML_LOG_ERROR("%s: failed to execute TensorRT engine\n", __func__);
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
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

static ggml_backend_buffer_type_t ggml_backend_tensorrt_device_get_buffer_type(ggml_backend_dev_t dev) {
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

    // Metadata ops — no type restriction, no actual computation
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
            // SOFT_MAX: reject mask and ALiBi
            if (op->op == GGML_OP_SOFT_MAX) {
                if (op->src[1] != nullptr) {
                    return false;
                }
                float max_bias = 0.0f;
                memcpy(&max_bias, &op->op_params[1], sizeof(float));
                if (max_bias != 0.0f) {
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
    (void) dev;

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
