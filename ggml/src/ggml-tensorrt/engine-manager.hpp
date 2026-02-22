#pragma once

#include "common.hpp"
#include "ggml.h"
#include <NvInfer.h>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ggml_tensorrt {

// Configuration for engine building
struct EngineConfig {
    size_t max_workspace_size = 128ULL << 20; // 128 MB default
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

// Compute a shape-agnostic hash for dynamic input support.
// Like the node_indices overload, but shapes of dynamic tensors (those NOT
// in static_leaves) are excluded from the hash.  Only type + a constant rank
// placeholder (GGML_MAX_DIMS) are hashed for dynamic tensors.  Static tensor
// shapes (weights) are fully hashed.  This produces batch-independent hashes
// — different batch sizes yield the same hash, enabling one cached engine to
// serve all batch sizes.
uint64_t compute_graph_hash(
    const ggml_cgraph * cgraph,
    const std::vector<int> & node_indices,
    const std::unordered_set<const ggml_tensor *> & static_leaves
);

// Per-node hash diagnostic entry — captures each node's contribution to the
// graph hash so that on a cache miss we can diff against the previous call
// and pinpoint exactly which node changed.
struct node_hash_entry {
    uint64_t cumulative_hash;  // running hash AFTER this node
    int32_t  op;               // GGML_OP_*
    int32_t  type;             // output type
    int64_t  ne[GGML_MAX_DIMS];// output shape
    int32_t  n_src;            // number of non-null sources
};

// Diagnostic hash: returns the same hash as compute_graph_hash (3-arg) but
// also fills `entries` with per-node snapshots for post-hoc diffing.
uint64_t compute_graph_hash_diagnostic(
    const ggml_cgraph * cgraph,
    const std::vector<int> & node_indices,
    const std::unordered_set<const ggml_tensor *> & static_leaves,
    std::vector<node_hash_entry> & entries
);

// Per-input optimization profile info for dynamic shape engines
struct input_profile {
    std::string name;
    nvinfer1::Dims min_dims;
    nvinfer1::Dims opt_dims;
    nvinfer1::Dims max_dims;
};

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

    // Build an engine with optimization profiles for dynamic input shapes.
    // Each input_profile specifies min/opt/max dims for one network input.
    // Static inputs have min=opt=max.  Dynamic inputs allow range of shapes.
    nvinfer1::ICudaEngine* build_engine(
        nvinfer1::IBuilder* builder,
        nvinfer1::INetworkDefinition* network,
        const EngineConfig& config,
        const std::vector<input_profile>& profiles
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

    // Evict a cached engine and its execution context.
    // Used when the cached engine's optimization profile can no longer
    // accommodate the current input shapes (profile range exceeded).
    void evict_engine(uint64_t hash);

    // Get or create an execution context for a cached engine.
    // Returns a reusable context — caller must rebind tensor addresses before use.
    // use_cuda_graphs overrides the instance-level setting for this specific context.
    nvinfer1::IExecutionContext* get_or_create_context(uint64_t hash, bool use_cuda_graphs);

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
    std::unordered_map<uint64_t, std::unique_ptr<nvinfer1::ICudaEngine>> engine_cache_;

    // Execution context cache: hash -> context (reusable after rebinding addresses)
    std::unordered_map<uint64_t, std::unique_ptr<nvinfer1::IExecutionContext>> context_cache_;

public:
    // Helper: Configure builder with the given config
    void configure_builder(
        nvinfer1::IBuilder* builder,
        nvinfer1::IBuilderConfig* builder_config,
        const EngineConfig& config
    );
};

} // namespace ggml_tensorrt
