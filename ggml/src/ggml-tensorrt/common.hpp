#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <string>

// Forward declarations
namespace nvinfer1 {
    class IRuntime;
    class ILogger;
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

// Backend context structure
struct ggml_backend_tensorrt_context {
    int device;
    cudaStream_t stream;

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<Logger> logger;

    // Engine cache: maps graph hash to compiled engines
    // Will be implemented in later milestones
    // std::map<uint64_t, std::unique_ptr<nvinfer1::ICudaEngine>> engine_cache;

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
