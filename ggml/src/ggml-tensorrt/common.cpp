#include "common.hpp"
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

// Backend context implementation
ggml_backend_tensorrt_context::ggml_backend_tensorrt_context(int device_id)
    : device(device_id), stream(nullptr) {

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(device));

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Determine log level from environment variable
    nvinfer1::ILogger::Severity log_level = nvinfer1::ILogger::Severity::kWARNING;
    const char* env_log_level = getenv("GGML_TENSORRT_LOG_LEVEL");
    if (env_log_level) {
        if (strcmp(env_log_level, "VERBOSE") == 0 || strcmp(env_log_level, "DEBUG") == 0) {
            log_level = nvinfer1::ILogger::Severity::kVERBOSE;
            GGML_LOG_INFO("%s: TensorRT log level set to VERBOSE\n", __func__);
        } else if (strcmp(env_log_level, "INFO") == 0) {
            log_level = nvinfer1::ILogger::Severity::kINFO;
            GGML_LOG_INFO("%s: TensorRT log level set to INFO\n", __func__);
        } else if (strcmp(env_log_level, "WARNING") == 0 || strcmp(env_log_level, "WARN") == 0) {
            log_level = nvinfer1::ILogger::Severity::kWARNING;
        } else if (strcmp(env_log_level, "ERROR") == 0) {
            log_level = nvinfer1::ILogger::Severity::kERROR;
            GGML_LOG_INFO("%s: TensorRT log level set to ERROR\n", __func__);
        } else {
            GGML_LOG_WARN("%s: unknown GGML_TENSORRT_LOG_LEVEL value '%s', using WARNING\n",
                         __func__, env_log_level);
        }
    }

    // Create logger
    logger = std::make_unique<Logger>(log_level);

    // Create TensorRT runtime
    runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(*logger));

    if (!runtime) {
        GGML_LOG_ERROR("%s: failed to create TensorRT runtime\n", __func__);
        GGML_ABORT("Failed to create TensorRT runtime");
    }

    GGML_LOG_INFO("%s: initialized TensorRT-RTX backend on device %d\n", __func__, device);
}

ggml_backend_tensorrt_context::~ggml_backend_tensorrt_context() {
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
