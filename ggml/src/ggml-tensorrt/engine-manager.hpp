#pragma once

#include "common.hpp"
#include <NvInfer.h>
#include <memory>
#include <map>

namespace ggml_tensorrt {

// Configuration for engine building
struct EngineConfig {
    size_t max_workspace_size = 1ULL << 30; // 1 GB default
    nvinfer1::BuilderFlag flags = static_cast<nvinfer1::BuilderFlag>(0);
    bool use_fp16 = true;
    bool use_bf16 = false;
    bool use_int8 = false;
    int32_t dla_core = -1; // -1 means no DLA

    // Optimization profile for dynamic shapes
    bool use_dynamic_shapes = false;

    EngineConfig() = default;
};

// EngineManager handles building, caching, and executing TensorRT engines
class EngineManager {
public:
    EngineManager(nvinfer1::IRuntime* runtime, nvinfer1::ILogger* logger);
    ~EngineManager();

    // Build an engine from a network definition
    nvinfer1::ICudaEngine* build_engine(
        nvinfer1::INetworkDefinition* network,
        const EngineConfig& config
    );

    // Create an execution context from an engine
    nvinfer1::IExecutionContext* create_context(nvinfer1::ICudaEngine* engine);

    // Execute inference with the given context
    bool execute(
        nvinfer1::IExecutionContext* context,
        void** bindings,
        cudaStream_t stream
    );

    // Get cached engine by hash (for future use)
    // Returns nullptr if not found
    nvinfer1::ICudaEngine* get_cached_engine(uint64_t hash);

    // Cache an engine with a hash key
    void cache_engine(uint64_t hash, nvinfer1::ICudaEngine* engine);

private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ILogger* logger_;

    // Engine cache: hash -> engine
    // For Milestone 2, this is simple; will be enhanced later
    std::map<uint64_t, std::unique_ptr<nvinfer1::ICudaEngine>> engine_cache_;

    // Helper: Configure builder with the given config
    void configure_builder(
        nvinfer1::IBuilder* builder,
        nvinfer1::IBuilderConfig* builder_config,
        const EngineConfig& config
    );
};

} // namespace ggml_tensorrt
