#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>

// Forward declarations
namespace nvinfer1 {
    class IRuntime;
    class ILogger;
}

namespace ggml_tensorrt {
    class EngineManager;
}

namespace ggml_tensorrt {

// TensorRT logger implementation
class Logger : public nvinfer1::ILogger {
public:
    Logger(Severity min_severity = Severity::kWARNING);
    void log(Severity severity, const char* msg) noexcept override;

    void set_min_severity(Severity severity);

private:
    Severity min_severity_;
};

// Return the global singleton Logger instance.
// TensorRT internally registers a single global logger; using one shared
// instance avoids the "logger differs" warning when multiple backends are
// created in the same process.
Logger & get_global_logger();

// Scratch buffer for I/O aliasing workaround.
// When a TRT output binding and a leaf input binding share the same device
// address (GGML allocator reuse), we copy the leaf to scratch before
// enqueueV3 so TRT sees distinct addresses.  Bump allocator, reset per
// graph_compute call, persists across calls (grows once, stays).
struct scratch_buffer {
    void * ptr      = nullptr;
    size_t capacity = 0;
    size_t offset   = 0;

    void * alloc(size_t nbytes) {
        // Align to 256 bytes
        nbytes = (nbytes + 255) & ~255;
        if (offset + nbytes > capacity) {
            size_t new_cap = capacity == 0 ? (1 << 20)   // 1 MB initial
                           : capacity * 2;
            while (new_cap < offset + nbytes) new_cap *= 2;
            void * new_ptr = nullptr;
            cudaMalloc(&new_ptr, new_cap);
            if (ptr) {
                // Preserve existing allocations from this graph_compute call
                if (offset > 0) {
                    cudaMemcpy(new_ptr, ptr, offset, cudaMemcpyDeviceToDevice);
                }
                cudaFree(ptr);
            }
            ptr      = new_ptr;
            capacity = new_cap;
        }
        void * result = (char *)ptr + offset;
        offset += nbytes;
        return result;
    }

    void reset() { offset = 0; }

    void free_mem() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
        capacity = 0;
        offset   = 0;
    }
};

// Backend context structure
struct ggml_backend_tensorrt_context {
    int device;
    cudaStream_t stream;

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    Logger * logger;  // non-owning, points to global singleton
    std::unique_ptr<EngineManager> engine_mgr;

    scratch_buffer scratch;  // I/O aliasing workaround

    ggml_backend_tensorrt_context(int device);
    ~ggml_backend_tensorrt_context();

    // Disable copy and move
    ggml_backend_tensorrt_context(const ggml_backend_tensorrt_context&) = delete;
    ggml_backend_tensorrt_context& operator=(const ggml_backend_tensorrt_context&) = delete;
};

// Helper functions
const char* cuda_error_to_str(cudaError_t err);
void check_cuda(cudaError_t err, const char* file, int line);

#define CUDA_CHECK(err) check_cuda(err, __FILE__, __LINE__)

} // namespace ggml_tensorrt
