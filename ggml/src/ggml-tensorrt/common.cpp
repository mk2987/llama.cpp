#include "common.hpp"
#include "engine-manager.hpp"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ggml_tensorrt {

// Logger implementation
Logger::Logger(Severity min_severity)
    : min_severity_(min_severity) {
}

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity > min_severity_) {
        return;
    }

    const char* level = "";
    switch (severity) {
        case Severity::kINTERNAL_ERROR: level = "INTERNAL_ERROR"; break;
        case Severity::kERROR:          level = "ERROR"; break;
        case Severity::kWARNING:        level = "WARN"; break;
        case Severity::kINFO:           level = "INFO"; break;
        case Severity::kVERBOSE:        level = "DEBUG"; break;
    }

    fprintf(stderr, "[TensorRT-RTX] [%s] %s\n", level, msg);
}

void Logger::set_min_severity(Severity severity) {
    min_severity_ = severity;
}

Logger & get_global_logger() {
    // Determine log level from environment variable (read once)
    static Logger instance = []() {
        nvinfer1::ILogger::Severity log_level = nvinfer1::ILogger::Severity::kWARNING;
        const char* env = getenv("GGML_TENSORRT_LOG_LEVEL");
        if (env) {
            if (strcmp(env, "VERBOSE") == 0 || strcmp(env, "DEBUG") == 0) {
                log_level = nvinfer1::ILogger::Severity::kVERBOSE;
            } else if (strcmp(env, "INFO") == 0) {
                log_level = nvinfer1::ILogger::Severity::kINFO;
            } else if (strcmp(env, "ERROR") == 0) {
                log_level = nvinfer1::ILogger::Severity::kERROR;
            }
        }
        return Logger(log_level);
    }();
    return instance;
}

// Backend context implementation
ggml_backend_tensorrt_context::ggml_backend_tensorrt_context(int device_id)
    : device(device_id), stream(nullptr), logger(nullptr) {

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(device));

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Use the process-wide singleton logger (avoids TRT "logger differs" warning)
    logger = &get_global_logger();

    // Create TensorRT runtime
    runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(*logger));

    if (!runtime) {
        GGML_LOG_ERROR("%s: failed to create TensorRT runtime\n", __func__);
        GGML_ABORT("Failed to create TensorRT runtime");
    }

    // Create engine manager
    engine_mgr = std::make_unique<EngineManager>(runtime.get(), logger);

    GGML_LOG_INFO("%s: initialized TensorRT-RTX backend on device %d\n", __func__, device);
}

ggml_backend_tensorrt_context::~ggml_backend_tensorrt_context() {
    // Reset in dependency order: engine_mgr uses runtime, runtime uses logger
    engine_mgr.reset();
    runtime.reset();
    // logger is a non-owning pointer to the global singleton — do not delete

    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

// Helper functions
const char* cuda_error_to_str(cudaError_t err) {
    return cudaGetErrorString(err);
}

void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[TensorRT-RTX] CUDA error at %s:%d: %s\n",
                file, line, cuda_error_to_str(err));
        GGML_ABORT("CUDA error");
    }
}

} // namespace ggml_tensorrt
