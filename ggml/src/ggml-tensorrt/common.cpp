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
        // Default to ERROR — TRT's WARN level is noisy (strongly-typed
        // reminders, unused input warnings for dead branches, etc.).
        // Use GGML_TENSORRT_LOG_LEVEL=WARN to see warnings.
        nvinfer1::ILogger::Severity log_level = nvinfer1::ILogger::Severity::kERROR;
        const char* env = getenv("GGML_TENSORRT_LOG_LEVEL");
        if (env) {
            if (strcmp(env, "VERBOSE") == 0 || strcmp(env, "DEBUG") == 0) {
                log_level = nvinfer1::ILogger::Severity::kVERBOSE;
            } else if (strcmp(env, "INFO") == 0) {
                log_level = nvinfer1::ILogger::Severity::kINFO;
            } else if (strcmp(env, "WARN") == 0 || strcmp(env, "WARNING") == 0) {
                log_level = nvinfer1::ILogger::Severity::kWARNING;
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

    // CUDA graphs enabled by default, disable with GGML_TENSORRT_CUDA_GRAPHS=0
    bool cuda_graphs = true;
    const char* cuda_graphs_env = getenv("GGML_TENSORRT_CUDA_GRAPHS");
    if (cuda_graphs_env && (strcmp(cuda_graphs_env, "0") == 0 || strcmp(cuda_graphs_env, "OFF") == 0)) {
        cuda_graphs = false;
        GGML_LOG_INFO("%s: CUDA graph capture disabled by env var\n", __func__);
    }
    engine_mgr->set_cuda_graphs(cuda_graphs);
    if (cuda_graphs) {
        GGML_LOG_INFO("%s: CUDA graph capture enabled\n", __func__);
    }

    GGML_LOG_INFO("%s: initialized TensorRT-RTX backend on device %d\n", __func__, device);
}

ggml_backend_tensorrt_context::~ggml_backend_tensorrt_context() {
    // Reset in dependency order: engine_mgr uses runtime, runtime uses logger
    engine_mgr.reset();
    runtime.reset();
    // logger is a non-owning pointer to the global singleton — do not delete

    scratch.free_mem();

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
