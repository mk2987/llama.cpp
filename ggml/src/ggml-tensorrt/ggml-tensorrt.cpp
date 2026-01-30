#include "common.hpp"
#include "ggml-tensorrt.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cuda_runtime.h>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
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

static enum ggml_status ggml_backend_tensorrt_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_tensorrt_context * ctx = (ggml_backend_tensorrt_context *) backend->context;

    CUDA_CHECK(cudaSetDevice(ctx->device));

    // For now, we don't support any operations - they will all fall back to CUDA
    // This will be implemented in Milestone 4
    (void) cgraph;

    GGML_LOG_DEBUG("%s: graph computation not yet implemented, falling back to CUDA\n", __func__);

    return GGML_STATUS_FAILED;
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

    // Support GGML_OP_NONE (storage tensors like KV cache)
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    // For now, we don't support any compute operations
    // This will be implemented in Milestone 2-4
    return false;
}

static bool ggml_backend_tensorrt_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_tensorrt_buffer_type_get_name &&
        buft->iface.get_name != ggml_backend_tensorrt_host_buffer_type_get_name) {
        return false;
    }

    (void) dev;
    return true;
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
    /* .offload_op                 = */ NULL,
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
