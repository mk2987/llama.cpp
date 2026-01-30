#include "engine-manager.hpp"
#include "ggml-impl.h"

#include <NvInferRuntime.h>

namespace ggml_tensorrt {

EngineManager::EngineManager(nvinfer1::IRuntime* runtime, nvinfer1::ILogger* logger)
    : runtime_(runtime), logger_(logger) {
    GGML_ASSERT(runtime_ != nullptr);
    GGML_ASSERT(logger_ != nullptr);
}

EngineManager::~EngineManager() {
    // Engines are managed by unique_ptr and will be automatically deleted
    engine_cache_.clear();
}

nvinfer1::ICudaEngine* EngineManager::build_engine(
    nvinfer1::INetworkDefinition* network,
    const EngineConfig& config
) {
    GGML_ASSERT(network != nullptr);

    // Create builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(*logger_);
    if (builder == nullptr) {
        GGML_LOG_ERROR("%s: failed to create TensorRT builder\n", __func__);
        return nullptr;
    }

    // Create builder config
    nvinfer1::IBuilderConfig* builder_config = builder->createBuilderConfig();
    if (builder_config == nullptr) {
        GGML_LOG_ERROR("%s: failed to create builder config\n", __func__);
        delete builder;
        return nullptr;
    }

    // Configure builder
    configure_builder(builder, builder_config, config);

    // Build serialized network
    GGML_LOG_INFO("%s: building TensorRT engine (this may take a while)...\n", __func__);

    nvinfer1::IHostMemory* serialized_engine = builder->buildSerializedNetwork(*network, *builder_config);
    if (serialized_engine == nullptr) {
        GGML_LOG_ERROR("%s: failed to build serialized network\n", __func__);
        delete builder_config;
        delete builder;
        return nullptr;
    }

    // Deserialize engine
    nvinfer1::ICudaEngine* engine = runtime_->deserializeCudaEngine(
        serialized_engine->data(),
        serialized_engine->size()
    );

    // Clean up
    delete serialized_engine;
    delete builder_config;
    delete builder;

    if (engine == nullptr) {
        GGML_LOG_ERROR("%s: failed to deserialize engine\n", __func__);
        return nullptr;
    }

    GGML_LOG_INFO("%s: successfully built TensorRT engine\n", __func__);

    return engine;
}

nvinfer1::IExecutionContext* EngineManager::create_context(nvinfer1::ICudaEngine* engine) {
    GGML_ASSERT(engine != nullptr);

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (context == nullptr) {
        GGML_LOG_ERROR("%s: failed to create execution context\n", __func__);
        return nullptr;
    }

    return context;
}

bool EngineManager::execute(
    nvinfer1::IExecutionContext* context,
    void** bindings,
    cudaStream_t stream
) {
    GGML_ASSERT(context != nullptr);
    GGML_ASSERT(bindings != nullptr);

    // Execute asynchronously
    bool success = context->enqueueV3(stream);

    if (!success) {
        GGML_LOG_ERROR("%s: execution failed\n", __func__);
        return false;
    }

    return true;
}

nvinfer1::ICudaEngine* EngineManager::get_cached_engine(uint64_t hash) {
    auto it = engine_cache_.find(hash);
    if (it != engine_cache_.end()) {
        return it->second.get();
    }
    return nullptr;
}

void EngineManager::cache_engine(uint64_t hash, nvinfer1::ICudaEngine* engine) {
    GGML_ASSERT(engine != nullptr);

    // Store engine in cache
    // Note: We transfer ownership to the cache
    engine_cache_[hash] = std::unique_ptr<nvinfer1::ICudaEngine>(engine);

    GGML_LOG_DEBUG("%s: cached engine with hash %llu\n", __func__,
                  static_cast<unsigned long long>(hash));
}

void EngineManager::configure_builder(
    nvinfer1::IBuilder* builder,
    nvinfer1::IBuilderConfig* builder_config,
    const EngineConfig& config
) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(builder_config != nullptr);

    // Set maximum workspace size
    builder_config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE,
        config.max_workspace_size
    );

    // Set precision flags
    // Note: kFP16 and kBF16 flags are deprecated in TensorRT 10+
    // but still functional. Will be updated to use strongly-typed flags in future.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (config.use_fp16) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        GGML_LOG_DEBUG("%s: enabled FP16 mode\n", __func__);
    }

    if (config.use_bf16) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kBF16);
        GGML_LOG_DEBUG("%s: enabled BF16 mode\n", __func__);
    }
#pragma GCC diagnostic pop

    if (config.use_int8) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        GGML_LOG_DEBUG("%s: enabled INT8 mode\n", __func__);
    }

    // Set DLA core if specified
    if (config.dla_core >= 0) {
        builder_config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        builder_config->setDLACore(config.dla_core);
        GGML_LOG_DEBUG("%s: using DLA core %d\n", __func__, config.dla_core);
    }

    // Enable GPU fallback for DLA
    if (config.dla_core >= 0) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }

    GGML_LOG_DEBUG("%s: max workspace size: %zu bytes\n",
                  __func__, config.max_workspace_size);
}

} // namespace ggml_tensorrt
