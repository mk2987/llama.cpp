#include "engine-manager.hpp"
#include "ggml-impl.h"

#include <NvInferRuntime.h>

namespace ggml_tensorrt {

// FNV-1a hash helpers
static const uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
static const uint64_t FNV_PRIME        = 1099511628211ULL;

static inline uint64_t fnv1a_hash_bytes(uint64_t hash, const void* data, size_t len) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

template<typename T>
static inline uint64_t fnv1a_hash_value(uint64_t hash, T value) {
    return fnv1a_hash_bytes(hash, &value, sizeof(value));
}

uint64_t compute_graph_hash(const ggml_cgraph * cgraph) {
    uint64_t hash = FNV_OFFSET_BASIS;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];

        // Hash op type
        hash = fnv1a_hash_value(hash, static_cast<int32_t>(node->op));

        // Hash output data type
        hash = fnv1a_hash_value(hash, static_cast<int32_t>(node->type));

        // Hash output shape
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            hash = fnv1a_hash_value(hash, node->ne[d]);
        }

        // Hash op_params (all 16 int32 values)
        hash = fnv1a_hash_bytes(hash, node->op_params, sizeof(node->op_params));

        // Hash source tensor types and shapes (NOT pointers)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                hash = fnv1a_hash_value(hash, static_cast<int32_t>(node->src[j]->type));
                for (int d = 0; d < GGML_MAX_DIMS; d++) {
                    hash = fnv1a_hash_value(hash, node->src[j]->ne[d]);
                }
            } else {
                // Sentinel for null source
                hash = fnv1a_hash_value(hash, static_cast<int32_t>(-1));
            }
        }
    }

    return hash;
}

EngineManager::EngineManager(nvinfer1::IRuntime* runtime, nvinfer1::ILogger* logger)
    : runtime_(runtime), logger_(logger) {
    GGML_ASSERT(runtime_ != nullptr);
    GGML_ASSERT(logger_ != nullptr);
}

EngineManager::~EngineManager() {
    // Destroy execution contexts before engines (contexts reference engines)
    context_cache_.clear();
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

nvinfer1::IExecutionContext* EngineManager::get_or_create_context(uint64_t hash) {
    // Check context cache first
    auto it = context_cache_.find(hash);
    if (it != context_cache_.end()) {
        return it->second.get();
    }

    // Need a cached engine to create the context from
    auto engine_it = engine_cache_.find(hash);
    if (engine_it == engine_cache_.end()) {
        GGML_LOG_ERROR("%s: no cached engine for hash %llu\n", __func__,
                      static_cast<unsigned long long>(hash));
        return nullptr;
    }

    nvinfer1::IExecutionContext* ctx = engine_it->second->createExecutionContext();
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to create execution context\n", __func__);
        return nullptr;
    }

    context_cache_[hash] = std::unique_ptr<nvinfer1::IExecutionContext>(ctx);

    GGML_LOG_DEBUG("%s: created and cached execution context for hash %llu\n", __func__,
                  static_cast<unsigned long long>(hash));

    return ctx;
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

    // TRT-RTX uses strongly-typed mode by default — precision is determined
    // by input tensor types, not global builder flags. kFP16/kBF16/kINT8
    // builder flags are deprecated no-ops.

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
