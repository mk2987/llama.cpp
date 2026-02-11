#pragma once

#include "common.hpp"
#include "ggml.h"
#include <NvInfer.h>
#include <cstdint>
#include <memory>
#include <map>
#include <vector>

namespace ggml_tensorrt {

// Configuration for engine building
struct EngineConfig {
    size_t max_workspace_size = 256ULL << 20; // 256 MB default
    int32_t dla_core = -1; // -1 means no DLA
    bool use_cuda_graphs = true;    // CUDA graph capture (on by default)
    int32_t max_aux_streams = 0;    // 0 = TRT default, >0 = allow parallel streams

    EngineConfig() = default;
};

// Compute a structural hash of the compute nodes in a graph.
// Hashes op type, output type/shape, op_params, and source type/shapes.
// Independent of pointer addresses — same structure yields same hash.
uint64_t compute_graph_hash(const ggml_cgraph * cgraph);

// Compute a structural hash over a subset of nodes identified by indices.
// Only these nodes contribute to the hash — trivial ops (SET_ROWS, CPY, etc.)
// whose parameters change every token are excluded.
uint64_t compute_graph_hash(const ggml_cgraph * cgraph, const std::vector<int> & node_indices);

// EngineManager handles building, caching, and executing TensorRT engines
class EngineManager {
public:
    EngineManager(nvinfer1::IRuntime* runtime, nvinfer1::ILogger* logger);
    ~EngineManager();

    // Build an engine from a network definition.
    // The builder must be the same one that created the network.
    nvinfer1::ICudaEngine* build_engine(
        nvinfer1::IBuilder* builder,
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

    // Get cached engine by hash
    // Returns nullptr if not found
    nvinfer1::ICudaEngine* get_cached_engine(uint64_t hash);

    // Cache an engine with a hash key (transfers ownership)
    void cache_engine(uint64_t hash, nvinfer1::ICudaEngine* engine);

    // Get or create an execution context for a cached engine.
    // Returns a reusable context — caller must rebind tensor addresses before use.
    nvinfer1::IExecutionContext* get_or_create_context(uint64_t hash);

    // Enable or disable CUDA graph capture for new execution contexts
    void set_cuda_graphs(bool enabled);

    // Cache statistics (for profiling instrumentation)
    int64_t cache_hits   = 0;
    int64_t cache_misses = 0;
    double  total_build_time_ms = 0.0;

private:
    bool use_cuda_graphs_ = true;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ILogger* logger_;

    // Engine cache: hash -> engine
    std::map<uint64_t, std::unique_ptr<nvinfer1::ICudaEngine>> engine_cache_;

    // Execution context cache: hash -> context (reusable after rebinding addresses)
    std::map<uint64_t, std::unique_ptr<nvinfer1::IExecutionContext>> context_cache_;

    // Helper: Configure builder with the given config
    void configure_builder(
        nvinfer1::IBuilder* builder,
        nvinfer1::IBuilderConfig* builder_config,
        const EngineConfig& config
    );
};

} // namespace ggml_tensorrt
