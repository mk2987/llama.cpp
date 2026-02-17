#include "engine-manager.hpp"
#include "ggml-impl.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <chrono>
#include <cinttypes>

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

uint64_t compute_graph_hash(const ggml_cgraph * cgraph, const std::vector<int> & node_indices) {
    uint64_t hash = FNV_OFFSET_BASIS;

    for (int i : node_indices) {
        GGML_ASSERT(i >= 0 && i < cgraph->n_nodes);
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
                hash = fnv1a_hash_value(hash, static_cast<int32_t>(-1));
            }
        }
    }

    return hash;
}

uint64_t compute_graph_hash(
    const ggml_cgraph * cgraph,
    const std::vector<int> & node_indices,
    const std::unordered_set<const ggml_tensor *> & static_leaves
) {
    uint64_t hash = FNV_OFFSET_BASIS;

    for (int i : node_indices) {
        GGML_ASSERT(i >= 0 && i < cgraph->n_nodes);
        const ggml_tensor * node = cgraph->nodes[i];

        // Hash op type (always)
        hash = fnv1a_hash_value(hash, static_cast<int32_t>(node->op));

        // Hash output data type (always)
        hash = fnv1a_hash_value(hash, static_cast<int32_t>(node->type));

        // Hash output ndims (always) — but NOT output shape values
        // (output shapes depend on dynamic input shapes)
        int n_dims = 0;
        for (int d = GGML_MAX_DIMS - 1; d >= 0; d--) {
            if (node->ne[d] > 1) { n_dims = d + 1; break; }
        }
        if (n_dims == 0) n_dims = 1;
        hash = fnv1a_hash_value(hash, n_dims);

        // Hash op_params — but skip for VIEW ops whose params contain
        // byte offsets and strides that change per layer and per token
        // (each layer's KV cache VIEW has a different offset).  The
        // structural properties of a VIEW (output shape, source shape)
        // are already captured by the ndims/type hashing above.
        if (node->op != GGML_OP_VIEW) {
            hash = fnv1a_hash_bytes(hash, node->op_params, sizeof(node->op_params));
        }

        // Hash source tensor types and shapes
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                const ggml_tensor * src = node->src[j];
                hash = fnv1a_hash_value(hash, static_cast<int32_t>(src->type));

                if (static_leaves.count(src)) {
                    // Static leaf: hash full shape (weight dimensions are constant)
                    for (int d = 0; d < GGML_MAX_DIMS; d++) {
                        hash = fnv1a_hash_value(hash, src->ne[d]);
                    }
                } else {
                    // Dynamic source (activation, position, or intermediate node):
                    // hash only ndims, not shape values
                    int src_ndims = 0;
                    for (int d = GGML_MAX_DIMS - 1; d >= 0; d--) {
                        if (src->ne[d] > 1) { src_ndims = d + 1; break; }
                    }
                    if (src_ndims == 0) src_ndims = 1;
                    hash = fnv1a_hash_value(hash, src_ndims);
                }
            } else {
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
    nvinfer1::IBuilder* builder,
    nvinfer1::INetworkDefinition* network,
    const EngineConfig& config
) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(network != nullptr);

    // Create builder config
    nvinfer1::IBuilderConfig* builder_config = builder->createBuilderConfig();
    if (builder_config == nullptr) {
        GGML_LOG_ERROR("%s: failed to create builder config\n", __func__);
        return nullptr;
    }

    // Configure builder
    configure_builder(builder, builder_config, config);

    // Build serialized network
    GGML_LOG_INFO("%s: building TensorRT engine (this may take a while)...\n", __func__);

    auto build_start = std::chrono::high_resolution_clock::now();

    nvinfer1::IHostMemory* serialized_engine = builder->buildSerializedNetwork(*network, *builder_config);
    if (serialized_engine == nullptr) {
        GGML_LOG_ERROR("%s: failed to build serialized network\n", __func__);
        delete builder_config;
        return nullptr;
    }

    // Deserialize engine
    nvinfer1::ICudaEngine* engine = runtime_->deserializeCudaEngine(
        serialized_engine->data(),
        serialized_engine->size()
    );

    auto build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    total_build_time_ms += build_ms;

    // Clean up serialization artifacts (engine is self-contained after deserialization)
    delete serialized_engine;
    delete builder_config;

    if (engine == nullptr) {
        GGML_LOG_ERROR("%s: failed to deserialize engine\n", __func__);
        return nullptr;
    }

    GGML_LOG_INFO("%s: successfully built TensorRT engine (%.1f ms)\n", __func__, build_ms);

    return engine;
}

nvinfer1::ICudaEngine* EngineManager::build_engine(
    nvinfer1::IBuilder* builder,
    nvinfer1::INetworkDefinition* network,
    const EngineConfig& config,
    const std::vector<input_profile>& profiles
) {
    GGML_ASSERT(builder != nullptr);
    GGML_ASSERT(network != nullptr);

    nvinfer1::IBuilderConfig* builder_config = builder->createBuilderConfig();
    if (builder_config == nullptr) {
        GGML_LOG_ERROR("%s: failed to create builder config\n", __func__);
        return nullptr;
    }

    configure_builder(builder, builder_config, config);

    // Create optimization profile for dynamic input shapes
    if (!profiles.empty()) {
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        if (profile == nullptr) {
            GGML_LOG_ERROR("%s: failed to create optimization profile\n", __func__);
            delete builder_config;
            return nullptr;
        }

        for (const auto& p : profiles) {
            if (!profile->setDimensions(p.name.c_str(),
                    nvinfer1::OptProfileSelector::kMIN, p.min_dims)) {
                GGML_LOG_ERROR("%s: failed to set min dims for %s\n", __func__, p.name.c_str());
                delete builder_config;
                return nullptr;
            }
            if (!profile->setDimensions(p.name.c_str(),
                    nvinfer1::OptProfileSelector::kOPT, p.opt_dims)) {
                GGML_LOG_ERROR("%s: failed to set opt dims for %s\n", __func__, p.name.c_str());
                delete builder_config;
                return nullptr;
            }
            if (!profile->setDimensions(p.name.c_str(),
                    nvinfer1::OptProfileSelector::kMAX, p.max_dims)) {
                GGML_LOG_ERROR("%s: failed to set max dims for %s\n", __func__, p.name.c_str());
                delete builder_config;
                return nullptr;
            }
        }

        builder_config->addOptimizationProfile(profile);
    }

    GGML_LOG_INFO("%s: building TensorRT engine with optimization profiles (%zu inputs)...\n",
        __func__, profiles.size());

    auto build_start = std::chrono::high_resolution_clock::now();

    nvinfer1::IHostMemory* serialized_engine = builder->buildSerializedNetwork(*network, *builder_config);
    if (serialized_engine == nullptr) {
        GGML_LOG_ERROR("%s: failed to build serialized network\n", __func__);
        delete builder_config;
        return nullptr;
    }

    nvinfer1::ICudaEngine* engine = runtime_->deserializeCudaEngine(
        serialized_engine->data(),
        serialized_engine->size()
    );

    auto build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    total_build_time_ms += build_ms;

    delete serialized_engine;
    delete builder_config;

    if (engine == nullptr) {
        GGML_LOG_ERROR("%s: failed to deserialize engine\n", __func__);
        return nullptr;
    }

    GGML_LOG_INFO("%s: successfully built TensorRT engine with dynamic shapes (%.1f ms)\n", __func__, build_ms);

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
        cache_hits++;
        return it->second.get();
    }
    cache_misses++;
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

void EngineManager::evict_engine(uint64_t hash) {
    // Context must be destroyed before engine (it references the engine)
    context_cache_.erase(hash);
    engine_cache_.erase(hash);
    GGML_LOG_DEBUG("%s: evicted engine and context for hash 0x%016" PRIx64 "\n",
        __func__, hash);
}

nvinfer1::IExecutionContext* EngineManager::get_or_create_context(uint64_t hash, bool use_cuda_graphs) {
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

    nvinfer1::IExecutionContext* ctx = nullptr;

    // Try CUDA graph capture path if enabled (both instance-level and call-level)
    if (use_cuda_graphs_ && use_cuda_graphs) {
        nvinfer1::IRuntimeConfig* rt_config = engine_it->second->createRuntimeConfig();
        if (rt_config) {
            rt_config->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);
            ctx = engine_it->second->createExecutionContext(rt_config);
            delete rt_config;
        }
        if (!ctx) {
            GGML_LOG_WARN("%s: CUDA graph context creation failed, falling back to regular context\n", __func__);
        }
    }

    // Fallback: create context without CUDA graph capture
    if (!ctx) {
        ctx = engine_it->second->createExecutionContext();
    }

    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to create execution context\n", __func__);
        return nullptr;
    }

    context_cache_[hash] = std::unique_ptr<nvinfer1::IExecutionContext>(ctx);

    GGML_LOG_DEBUG("%s: created and cached execution context for hash %llu%s\n", __func__,
                  static_cast<unsigned long long>(hash),
                  (use_cuda_graphs_ && use_cuda_graphs) ? "" : " (CUDA graphs disabled)");

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

    // Set auxiliary streams for parallel layer execution
    if (config.max_aux_streams > 0) {
        builder_config->setMaxAuxStreams(config.max_aux_streams);
        GGML_LOG_DEBUG("%s: max auxiliary streams: %d\n", __func__, config.max_aux_streams);
    }

    GGML_LOG_DEBUG("%s: max workspace size: %zu bytes\n",
                  __func__, config.max_workspace_size);
}

void EngineManager::set_cuda_graphs(bool enabled) {
    use_cuda_graphs_ = enabled;
}

} // namespace ggml_tensorrt
