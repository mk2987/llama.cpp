#include "common.hpp"
#include "network-builder.hpp"
#include "engine-manager.hpp"
#include "ggml-tensorrt.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <cassert>
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
// GGML_OP_NONE tensors are leaves. Other ops are intermediate compute results
// already in the subgraph, so we skip them.
static void collect_leaves_recursive(
    const ggml_tensor* tensor,
    std::vector<const ggml_tensor*>& leaf_tensors,
    std::unordered_set<const ggml_tensor*>& leaf_seen
) {
    if (tensor == nullptr) {
        return;
    }
    if (tensor->op == GGML_OP_NONE) {
        // True leaf input
        if (leaf_seen.insert(tensor).second) {
            leaf_tensors.push_back(tensor);
        }
        return;
    }
    if (is_shape_op(tensor->op)) {
        // Walk through shape op to find real leaf
        collect_leaves_recursive(tensor->src[0], leaf_tensors, leaf_seen);
        return;
    }
    // Intermediate compute op already in the TRT subgraph — skip
}

static enum ggml_status ggml_backend_tensorrt_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_tensorrt_context * ctx = (ggml_backend_tensorrt_context *) backend->context;

    CUDA_CHECK(cudaSetDevice(ctx->device));

    // ── Phase 1: Handle trivial ops, collect TRT nodes and leaf inputs ──

    // TRT node indices within cgraph->nodes (includes shape ops and compute ops)
    std::vector<int> trt_node_indices;

    // Leaf inputs in stable discovery order (deduped)
    std::vector<const ggml_tensor*> leaf_tensors;
    std::unordered_set<const ggml_tensor*> leaf_seen;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            // NONE is a no-op, skip
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

            default:
            {
                // Shape ops and compute ops both go into TRT node list
                trt_node_indices.push_back(i);

                // Collect leaf inputs recursively (walks through shape ops)
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j]) {
                        collect_leaves_recursive(node->src[j], leaf_tensors, leaf_seen);
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

    uint64_t hash = compute_graph_hash(cgraph);
    nvinfer1::ICudaEngine* engine = ctx->engine_mgr->get_cached_engine(hash);

    // ── Phase 3: Cache miss — build engine ──

    if (engine == nullptr) {
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
                char out_name[64];
                snprintf(out_name, sizeof(out_name), "output_%d", output_idx);
                net_builder.mark_output(output, out_name);
                output_idx++;
            }
        }

        // Build engine
        EngineConfig engine_config;
        engine_config.max_workspace_size = 1024ULL * 1024 * 1024;  // 1 GB
        engine_config.use_fp16 = false;

        engine = ctx->engine_mgr->build_engine(network.get(), engine_config);
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
            return true;
        default:
            break;
    }

    // Compute ops — F32 only for correctness
    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_SOFT_MAX:
        {
            // Output must be F32
            if (op->type != GGML_TYPE_F32) {
                return false;
            }
            // All sources must be F32
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
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
        case GGML_OP_UNARY:
        {
            // Output must be F32
            if (op->type != GGML_TYPE_F32) {
                return false;
            }
            // All sources must be F32
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
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
