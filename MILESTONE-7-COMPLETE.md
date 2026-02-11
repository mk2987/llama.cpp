# Milestone E2E: End-to-End Model Testing Plan

## Status: IN PROGRESS

## Goal

Validate the TensorRT backend with a real GGUF model end-to-end: model loading,
multi-backend scheduling (TRT + CUDA fallback), correct text generation, and
stable performance. This is the first time the backend runs against a real
transformer model rather than synthetic unit test graphs.

---

## Model Selection

### Primary: Gemma 3 1B Instruct (F16)

```bash
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -f gemma-3-1b-it-F16.gguf
```

Why this model:
- Small (2 GB F16), fits easily in GPU memory
- Available pre-built in F16 GGUF from https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF
- Standard transformer architecture with good op coverage

### Alternative: Llama 3.2 1B Instruct (F16)

Also available from https://huggingface.co/Mungert/Llama-3.2-1B-Instruct-GGUF.
Uses SILU instead of GELU for FFN activation — both are supported. Very similar
op profile.

### Important: Use F16, not quantized

The TRT backend only supports F32/BF16/F16. Quantized variants (Q4_K_M, Q8_0)
use integer types that will be rejected by `supports_op`, so everything would
fall back to CUDA. Make sure to grab the F16 GGUF file.

---

## Op Coverage Analysis

Checked the Gemma 3 graph builder (`src/models/gemma3.cpp`). Here's how ops
partition between TRT and CUDA:

### TRT-claimed ops (supported in `supports_op`)

| Op | Used For | Notes |
|----|----------|-------|
| MUL_MAT | All linear layers (Q/K/V, FFN up/gate/down, output) | Offloaded, compute-heavy |
| RMS_NORM | All layer norms | FP32 upcast internally |
| ADD | Residual connections | Native type |
| MUL | Norm scaling | Native type |
| GELU | FFN activation | Decomposed via kERF |
| SOFT_MAX | Attention weights (non-flash path) | Only if no mask |
| TANH | Logit soft-capping | Via UNARY handler |
| RESHAPE, PERMUTE, VIEW, CONT, TRANSPOSE | Shape ops | IShuffleLayer |

### CUDA/CPU fallback ops (not in `supports_op`)

| Op | Used For | Why fallback |
|----|----------|--------------|
| ROPE_EXT | Rotary position embeddings | Not implemented |
| FLASH_ATTN_EXT | Optimized attention | Not implemented |
| SCALE | Embedding/attention scaling | Implemented in M6, but not offloaded (elementwise) |
| CAST | Type conversion for KV cache | Not implemented |
| GET_ROWS | Embedding lookup | Implemented in M6, but not offloaded (scatter op) |

The compute-dominant ops (MUL_MAT for all linear layers, RMS_NORM, GELU, ADD)
go to TRT. The attention mechanism and positional encoding fall back — this is
expected and fine since the GGML scheduler handles the partitioning
automatically.

---

## Prerequisites

### Docker container with GPU

```bash
docker run -it --gpus "device=GPU-de95b637-d204-aff0-6de6-8f95992d2807" \
  -v /home/scratch.mkaushik_sw_1/llama-rtx/llama.cpp:/app/llama.cpp \
  -v /home/scratch.mkaushik_sw_1/llama-rtx/data/:/data/ \
  rtx-llama-dev
```

### Download the model (before entering container, or from inside if network available)

```bash
# Option A: Use huggingface-cli
huggingface-cli download ggml-org/gemma-3-1b-it-GGUF \
  gemma-3-1b-it-F16.gguf --local-dir /data/models/

# Option B: Direct download
wget -P /data/models/ \
  https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-F16.gguf

# Option C: Let llama-cli download via -hf flag (needs network in container)
# The -hf flag auto-downloads, but -f specifies which file from the repo
```

### Build inside the container

```bash
cd /app/llama.cpp
cmake -B build \
  -DGGML_TENSORRT=ON \
  -DTENSORRT_ROOT=/opt/TensorRT-RTX \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8
```

---

## Test Plan

### Phase 0: Regression — All existing unit tests pass

**Purpose**: Confirm nothing is broken before attempting e2e.

```bash
cd /app/llama.cpp/build
ctest -R tensorrt -V
```

**Expected**: All 23 tests pass (M1: 4, M2: 4, M3: 8, M5a: 6, M5b: 5).

**Pass criteria**: 100% pass rate. If any fail, fix before proceeding.

---

### Phase 1: Smoke test — Model loads and generates text

**Purpose**: Verify the model loads, TRT backend initializes, scheduler
partitions the graph, and text is produced without crashes.

```bash
GGML_TENSORRT_LOG_LEVEL=INFO \
  ./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "What is 2+2?" \
  -n 32 \
  2>trt_stderr.log
```

**Expected**:
1. No crash or segfault
2. TRT backend initialization log in stderr:
   ```
   [TensorRT-RTX] [INFO] initialized TensorRT-RTX backend on device 0
   ```
3. Engine building log messages (first run only):
   ```
   [TensorRT-RTX] [INFO] building TensorRT engine for hash ...
   ```
4. Coherent text output (not garbage/NaN)
5. Process exits cleanly (exit code 0)

**Pass criteria**: Text output is readable and process exits without error.
Content doesn't need to be perfect — just not garbage.

**Failure modes to watch for**:
- Segfault during graph_compute → likely a tensor address binding bug
- NaN/Inf in output → likely a precision mismatch or missing cast layer
- All ops falling back to CUDA → check that F16 model is loaded (not quantized)
- TRT engine build failure → check TRT-RTX logs for unsupported layer config

---

### Phase 2: Op partitioning verification

**Purpose**: Confirm the GGML scheduler is actually routing ops to TRT (not
falling back entirely to CUDA).

```bash
GGML_TENSORRT_LOG_LEVEL=VERBOSE \
  ./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "Hello" \
  -n 8 \
  2>trt_verbose.log
```

Then inspect the log:

```bash
# Check that TRT engines were built
grep "building TensorRT engine" trt_verbose.log | wc -l

# Check that engines were cached and reused on subsequent tokens
grep "cached engine" trt_verbose.log | wc -l

# Look for any TRT errors
grep -i "error\|failed\|abort" trt_verbose.log
```

**Expected**:
- At least 1 engine built (for the compute-heavy subgraph)
- Engine reuse on token 2+ (cache hits > 0)
- No TRT errors

**Pass criteria**: TRT engines are built and reused. Some ops run on TRT, some
on CUDA — this is correct multi-backend behavior.

---

### Phase 3: Correctness — Compare TRT+CUDA vs CUDA-only output

**Purpose**: Verify that TRT backend produces the same (or acceptably similar)
output as pure CUDA.

#### Step 3a: Generate with TRT enabled (default)

```bash
./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "The capital of France is" \
  -n 64 \
  --seed 42 \
  --temp 0 \
  > trt_output.txt 2>/dev/null
```

#### Step 3b: Generate with TRT disabled (CUDA-only)

Build without TRT or use a CUDA-only binary:

```bash
# Option A: Rebuild without TRT
cmake -B build-cuda \
  -DGGML_CUDA=ON \
  -DGGML_TENSORRT=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release -j8

./build-cuda/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "The capital of France is" \
  -n 64 \
  --seed 42 \
  --temp 0 \
  > cuda_output.txt 2>/dev/null
```

#### Step 3c: Compare outputs

```bash
diff trt_output.txt cuda_output.txt
```

**Expected**: Outputs should be identical or nearly identical. With temp=0 and
same seed, greedy decoding should produce the same token sequence. Small
floating-point differences from TRT's different kernel implementations may
occasionally cause a token divergence after many tokens.

**Pass criteria**:
- First 16 tokens are identical
- If divergence occurs, it happens late (after 32+ tokens) and both outputs
  remain coherent
- No NaN/garbage in either output

**Note**: If SOFT_MAX falls back to CUDA (because the model uses masked
softmax), TRT handles fewer ops, reducing the chance of divergence.

---

### Phase 4: Performance — Timing comparison

**Purpose**: Measure whether TRT backend provides any speedup or at least
doesn't significantly regress performance.

#### Step 4a: Benchmark with TRT

```bash
./build/bin/llama-bench \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p 128 -n 64 \
  -r 3 \
  2>trt_bench_stderr.log
```

#### Step 4b: Benchmark without TRT

```bash
./build-cuda/bin/llama-bench \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p 128 -n 64 \
  -r 3 \
  2>/dev/null
```

**Metrics to record**:

| Metric | TRT+CUDA | CUDA-only | Ratio |
|--------|----------|-----------|-------|
| Prompt eval (tok/s) | | | |
| Token generation (tok/s) | | | |
| Memory usage (MB) | | | |
| First engine build time (s) | | | N/A |

**Expected**:
- First run is slower (TRT engine compilation overhead)
- Subsequent runs: TRT+CUDA should be within 80-120% of CUDA-only
- If TRT is slower, this is acceptable for now — the backend is still
  immature and only handles a subset of ops
- Engine build time should be < 30s for a 1B model

**Pass criteria**: No crash, performance within 2x of CUDA-only. Absolute
speedup is a nice-to-have, not a requirement for this milestone.

---

### Phase 5: Stability — Repeated execution and memory

**Purpose**: Verify no memory leaks or crashes over sustained use.

#### Step 5a: Repeated short generations

```bash
for i in $(seq 1 20); do
  echo "=== Run $i ==="
  ./build/bin/llama-cli \
    -m /data/models/gemma-3-1b-it-F16.gguf \
    -p "Count to five:" \
    -n 16 \
    2>/dev/null
  echo ""
done
```

**Expected**: All 20 runs produce coherent output. No crash on any run.

#### Step 5b: GPU memory stability

Terminal 1:
```bash
watch -n 1 nvidia-smi
```

Terminal 2:
```bash
for i in $(seq 1 10); do
  echo "=== Run $i ==="
  ./build/bin/llama-cli \
    -m /data/models/gemma-3-1b-it-F16.gguf \
    -p "Hello" \
    -n 16 \
    2>/dev/null
done
```

**Expected**: GPU memory usage stabilizes after run 1. Should NOT grow linearly
across runs. Small fluctuations (< 50 MB) are acceptable.

**Pass criteria**: No OOM, no growing memory leak.

#### Step 5c: Longer generation

```bash
./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "Write a detailed essay about the history of computing:" \
  -n 512 \
  2>trt_long.log
```

**Expected**: Completes without crash. Output is coherent throughout (no
degradation or garbage mid-generation).

**Pass criteria**: Generation completes, output is coherent, no crash.

---

### Phase 6: CUDA graph interaction

**Purpose**: Verify CUDA graph capture works with real model graphs (not just
synthetic unit tests).

```bash
# With CUDA graphs (default)
GGML_TENSORRT_LOG_LEVEL=INFO \
  ./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "Hello" -n 16 \
  2>cuda_graph_on.log

# Without CUDA graphs
GGML_TENSORRT_CUDA_GRAPHS=0 GGML_TENSORRT_LOG_LEVEL=INFO \
  ./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "Hello" -n 16 \
  2>cuda_graph_off.log
```

**Expected**: Both produce coherent output. Check logs:

```bash
grep -i "cuda graph" cuda_graph_on.log
# Should see: "CUDA graph capture enabled"

grep -i "cuda graph" cuda_graph_off.log
# Should see: "CUDA graph capture disabled by env var"
```

**Pass criteria**: Both modes work. If CUDA graph capture fails for a particular
graph structure, the graceful fallback should kick in (warning in log, not crash).

---

### Phase 7 (Optional): Alternative model — Llama 3.2 1B

**Purpose**: Validate with a second model architecture to increase confidence.

```bash
# Download
wget -P /data/models/ \
  https://huggingface.co/Mungert/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-F16.gguf

# Run
./build/bin/llama-cli \
  -m /data/models/Llama-3.2-1B-Instruct-F16.gguf \
  -p "What is the meaning of life?" \
  -n 64 \
  2>llama_trt.log
```

**Expected**: Coherent output. Llama uses SILU instead of GELU — both are
handled by the TRT unary handler.

**Pass criteria**: Same as Phase 1 (no crash, coherent output).

---

## Success Criteria Summary

| Phase | Requirement | Blocking? |
|-------|-------------|-----------|
| 0. Regression | All 23 unit tests pass | Yes |
| 1. Smoke test | Model loads, text generated, no crash | Yes |
| 2. Op partitioning | TRT engines built and reused | Yes |
| 3. Correctness | Output matches CUDA-only (first 16 tokens) | Yes |
| 4. Performance | No crash, within 2x of CUDA-only | No |
| 5. Stability | 20 runs without crash, no memory leak | Yes |
| 6. CUDA graphs | Both modes work without crash | No |
| 7. Alt model | Llama 3.2 works too | No |

Phases 0-3 and 5 are blocking. Phases 4, 6, 7 are informational.

---

## Known Risks and Expected Issues

### 1. SOFT_MAX with mask
Gemma 3 may use masked softmax in attention (src[1] != nullptr). Our
`supports_op` rejects this, so it falls back to CUDA. This is correct behavior.
The FFN softmax (if any) without mask will run on TRT.

### 2. F16 model with F32 MUL_MAT output
`ggml_mul_mat` always creates F32 output tensors regardless of input type. The
M5a output cast layer handles this, but it's the first time it runs on a real
model's graph topology. Watch for type mismatch errors.

### 3. Engine build time
TRT JIT compilation for real model subgraphs may take 10-30s on first run.
This is one-time cost per unique graph structure. Subsequent tokens reuse cached
engines.

### 4. Graph structure changes across tokens
The prefill (prompt evaluation) graph and decode (token generation) graph have
different shapes. Expect 2+ engine builds: one for prefill, one for decode.
Both should be cached after first occurrence.

### 5. Memory pressure
F16 model (2 GB) + TRT engine memory + KV cache may approach GPU memory limits
on smaller GPUs (< 8 GB). Monitor with nvidia-smi.

### 6. Multi-backend data transfers
The GGML scheduler inserts copy ops between TRT and CUDA subgraphs. Our
`supports_buft` accepts host buffers to enable this. Watch for correctness
issues at backend boundaries.

---

## Figuring Out Which Ops Run on Which Backend

There are two complementary views: the **GGML scheduler** decides which backend
gets each op (top-down), and the **TRT backend** logs which ops it actually
executes (bottom-up). Use both to get the full picture.

### Method 1: GGML Scheduler Debug Output (`GGML_SCHED_DEBUG`)

The GGML multi-backend scheduler has built-in debug logging controlled by the
`GGML_SCHED_DEBUG` environment variable. This is the authoritative source for
backend assignment — it shows the scheduler's decisions *before* any backend
code runs.

#### Level 1 — Split-level assignments

```bash
GGML_SCHED_DEBUG=1 \
  ./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "Hello" -n 4 \
  2>sched_debug.log
```

This prints one line per **split** (a contiguous run of ops on the same
backend). Example output:

```
## SPLIT #0: TensorRT-RTX # 2 inputs: [token_embd.weight (...)] [inp_pos (...)]
## SPLIT #1: CUDA0 # 1 inputs: [blk.0.attn_norm (...)]
## SPLIT #2: TensorRT-RTX # 3 inputs: [...]
...
```

Each split line tells you:
- **Split number**: Sequential index
- **Backend name**: `TensorRT-RTX` or `CUDA0` (or `CPU`)
- **Number of inputs**: Tensors transferred in from a different backend
- **Input names**: The tensors that cross the backend boundary

**What to look for**:
- Splits should alternate between TensorRT-RTX and CUDA0
- TRT splits should contain the compute-heavy ops (MUL_MAT, ADD, RMS_NORM, etc.)
- CUDA splits should contain fallback ops (ROPE_EXT, FLASH_ATTN_EXT, etc.)
- If there is only 1 split on CUDA0 and 0 on TensorRT-RTX, all ops fell back

**Summarize the splits**:
```bash
grep "^## SPLIT" sched_debug.log | sort | uniq -c | sort -rn
# Shows how many splits each backend got
```

#### Level 2 — Per-node assignments

```bash
GGML_SCHED_DEBUG=2 \
  ./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "Hello" -n 1 \
  2>sched_debug2.log
```

This prints **every node** in the graph with its assigned backend. Example:

```
node #  0 (   MUL_MAT):     blk.0.attn_q.weight (...) [TensorRT-RTX  sup_op]
node #  1 (   MUL_MAT):     blk.0.attn_k.weight (...) [TensorRT-RTX  sup_op]
node #  2 (  ROPE_EXT):     rope_ext-0           (...) [CUDA0    sup_op]
node #  3 (  ROPE_EXT):     rope_ext-1           (...) [CUDA0    sup_op]
node #  4 (FLASH_ATTN):     flash_attn_ext-0     (...) [CUDA0    sup_op]
node #  5 (       ADD):     add-0                (...) [TensorRT-RTX  sup_op]
...
```

Each line shows:
- **Node index**: Position in the computation graph
- **Op name**: The GGML operation (MUL_MAT, ADD, ROPE_EXT, etc.)
- **Tensor name**: The output tensor name
- **Backend**: Which backend was assigned (`TensorRT-RTX`, `CUDA0`, `CPU`)
- **Cause**: Why (e.g. `sup_op` = `supports_op` returned true)

**Generate a per-op backend summary**:
```bash
# Count ops per backend
grep "^node #" sched_debug2.log \
  | sed 's/.*\[\(.*\)\s\s*.*/\1/' \
  | sort | uniq -c | sort -rn

# Example output:
#   847 TensorRT-RTX
#   203 CUDA0
#    12 CPU
```

**Count specific op types per backend**:
```bash
# Which ops go to TRT?
grep "TensorRT-RTX" sched_debug2.log \
  | grep -oP '\(\s*\K[A-Z_]+' \
  | sort | uniq -c | sort -rn

# Which ops fall back to CUDA?
grep "CUDA0" sched_debug2.log \
  | grep -oP '\(\s*\K[A-Z_]+' \
  | sort | uniq -c | sort -rn
```

Expected for Gemma 3 1B F16:
```
TRT ops:                    CUDA fallback ops:
  ~400 MUL_MAT                ~52 ROPE_EXT
  ~100 ADD                     ~26 FLASH_ATTN_EXT
   ~52 RMS_NORM                ~26 SCALE
   ~52 MUL                     ~26 CAST
   ~26 GELU (unary)            ~26 GET_ROWS
   ~200 RESHAPE/PERMUTE/VIEW   ...
```

(Exact counts depend on model architecture — 26 layers for Gemma 3 1B.)

### Method 2: TRT Backend Verbose Logging (`GGML_TENSORRT_LOG_LEVEL`)

This shows what the TRT backend *actually does* when `graph_compute` is called
— which ops it builds into TRT engines, and engine cache behavior.

```bash
GGML_TENSORRT_LOG_LEVEL=VERBOSE \
  ./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "Hello" -n 4 \
  2>trt_verbose.log
```

**What it logs**:
- `added input <name> with shape <dims>` — leaf tensors added as TRT inputs
- `added operation <OP>` — each op processed by the TRT network builder
- `marked output <name> with shape <dims>` — compute ops marked as TRT outputs
- `building TensorRT engine` — engine build started (cache miss)
- `successfully built TensorRT engine` — engine build completed
- `created and cached execution context` — new context for this graph hash
- Per-op details: matmul shapes, softmax axes, normalization epsilon, etc.

**Extract the ops TRT actually processes**:
```bash
grep "added operation" trt_verbose.log \
  | sed 's/.*added operation //' \
  | sort | uniq -c | sort -rn
```

**Count engine builds vs cache hits**:
```bash
echo "Engine builds:"
grep -c "building TensorRT engine" trt_verbose.log

echo "Cache hits (reused engines):"
grep -c "cached engine\|cached execution context" trt_verbose.log
```

### Method 3: Combined — Full Picture in One Run

Use both env vars together for the most complete view:

```bash
GGML_SCHED_DEBUG=2 GGML_TENSORRT_LOG_LEVEL=VERBOSE \
  ./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "Hello" -n 1 \
  2>full_debug.log
```

Then analyze:

```bash
# 1. Scheduler split summary (how many backend transitions)
echo "=== Scheduler Splits ==="
grep "^## SPLIT" full_debug.log

# 2. Per-backend op counts
echo -e "\n=== TRT-assigned ops ==="
grep "TensorRT-RTX" full_debug.log | grep "^node #" \
  | grep -oP '\(\s*\K[A-Z_]+' | sort | uniq -c | sort -rn

echo -e "\n=== CUDA-assigned ops ==="
grep "CUDA0" full_debug.log | grep "^node #" \
  | grep -oP '\(\s*\K[A-Z_]+' | sort | uniq -c | sort -rn

# 3. TRT engine builds
echo -e "\n=== TRT engines built ==="
grep "building TensorRT engine" full_debug.log

# 4. Any errors
echo -e "\n=== Errors ==="
grep -i "error\|failed\|abort" full_debug.log
```

### Method 4: Quick Sanity Check (One-Liner)

If you just want to know "is TRT doing anything at all?":

```bash
GGML_SCHED_DEBUG=1 ./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf \
  -p "Hi" -n 1 2>&1 \
  | grep "## SPLIT" | grep -c "TensorRT-RTX"
```

If the count is 0, TRT is not being used. Check `supports_op` and model types.
If > 0, TRT is handling some subgraphs.

### Interpreting the Results

**Healthy multi-backend split pattern** (expected):
```
SPLIT #0: CUDA0        ← GET_ROWS (embedding lookup)
SPLIT #1: TensorRT-RTX ← MUL_MAT, ADD, RMS_NORM, MUL, GELU (FFN + norms)
SPLIT #2: CUDA0        ← ROPE_EXT, FLASH_ATTN_EXT (attention)
SPLIT #3: TensorRT-RTX ← MUL_MAT, ADD, RMS_NORM (post-attention)
SPLIT #4: CUDA0        ← ROPE_EXT, FLASH_ATTN_EXT
... (repeats per layer)
```

**All-CUDA (TRT not contributing)**:
```
SPLIT #0: CUDA0        ← everything
```
This means `supports_op` rejected all ops. Check: is the model quantized?

**All-TRT (unrealistic but possible for tiny synthetic graphs)**:
```
SPLIT #0: TensorRT-RTX ← everything
```
This shouldn't happen with a real model — ROPE/attention should fall back.

---

## Debugging Tips

### If model crashes immediately
```bash
# Run with maximum verbosity
GGML_TENSORRT_LOG_LEVEL=VERBOSE ./build/bin/llama-cli \
  -m /data/models/gemma-3-1b-it-F16.gguf -p "test" -n 1 2>&1 | head -100
```

### If output is garbage/NaN
```bash
# Check if TRT engines are actually being used
GGML_TENSORRT_LOG_LEVEL=VERBOSE ./build/bin/llama-cli ... 2>&1 | grep -c "enqueueV3"
```
If 0 engines are executing, all ops fell back to CUDA — check supports_op.

### If all ops fall back to CUDA
```bash
# Verify model is F16 not quantized
./build/bin/llama-cli -m /data/models/gemma-3-1b-it-F16.gguf --verbose 2>&1 | grep "type"
```
Look for `f16` in tensor types. If you see `q4_0` or `q8_0`, wrong model file.

### If engine build fails
```bash
# TRT-RTX detailed builder logs
GGML_TENSORRT_LOG_LEVEL=VERBOSE ./build/bin/llama-cli ... 2>&1 | grep -i "builder\|network\|layer"
```

---

## Sources

- https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF
- https://huggingface.co/Mungert/Llama-3.2-1B-Instruct-GGUF
- https://huggingface.co/docs/hub/en/gguf-llamacpp
- https://github.com/ggml-org/llama.cpp

---

## Bugs Found and Fixed During E2E Testing

### Bug 1: F16→F32 matmul promotion OOM (commit `43005dc98`)

**Symptom**: OOM crash during first engine build on Gemma 3 1B F16.

**Root cause**: When matmul inputs had mixed types (F16 weights + F32 activation),
the network builder promoted BOTH inputs to F32. For the logits matmul with the
262K-vocab embedding tensor `[1152, 262144]` in F16, the cast intermediate was
~1.15 GB — exceeding available VRAM on any GPU already loaded with model weights
and KV cache.

**Fix**: Cast to the *narrower* type (F16/BF16) instead of promoting to F32.
The small F32 activation tensor is cast down to match the weight type. The matmul
runs on tensor cores which natively support F16/BF16 — no precision loss for
this operation.

### Bug 2: Matmul output type mismatch cascade (commit `1929a04e8`)

**Symptom**: After fixing Bug 1, TRT engine build failed with type mismatch
errors on downstream elementwise ops (e.g., MUL with F32 norm weights following
an F16 matmul output).

**Root cause**: After Bug 1's fix, matmul produced F16 output (matching input
type). But `ggml_mul_mat` always sets `node->type = GGML_TYPE_F32`. The output
cast in `graph_compute` only applied to *network outputs* (final results copied
back to ggml tensors), not to intermediate tensors within the same TRT subgraph.
So downstream ops in the same subgraph received F16 where they expected F32.

**Fix**: Added an explicit cast of the matmul output to `node->type` (F32)
inside the MUL_MAT handler itself, so downstream ops always see the correct type
regardless of what type the matmul computed in internally.

### Bug 3: Graph hash included non-TRT nodes (commit `43005dc98`)

**Symptom**: Engine cache never hit — every token triggered a full TRT engine
rebuild, causing a memory leak (each engine allocated ~50-100 MB) and extremely
slow generation.

**Root cause**: `compute_graph_hash(cgraph)` hashed ALL nodes in the subgraph,
including SET_ROWS and CPY ops. These ops have `op_params` that change every
token (e.g., row indices for KV cache scatter). Since the hash changed every
token, the engine cache key never matched, forcing a rebuild.

**Fix**: New `compute_graph_hash(cgraph, trt_node_indices)` overload that only
hashes the TRT compute nodes (skipping SET_ROWS/CPY which are handled separately
in `graph_compute`). The hash now stays stable across tokens for the same graph
structure.

### Bug 4: Workspace too large for inference (commit `43005dc98`)

**Symptom**: Engine build OOM even after fixing Bug 1, because the 1 GB default
workspace reservation consumed too much of the remaining VRAM.

**Root cause**: Default TRT builder workspace was 1 GB, appropriate for training
but excessive for inference where GPU memory is already consumed by model weights
and KV cache.

**Fix**:
- Reduced default workspace to 256 MB
- Added dynamic cap at 25% of free VRAM at engine build time
- Added `GGML_TENSORRT_WORKSPACE_MB` env var for user override
- Added free memory diagnostic logging before each engine build

### Bug 5: Elementwise ops missing type cast — GARBAGE OUTPUT (this milestone)

**Symptom**: Model produces garbage text despite engines building successfully.

**Root cause**: `handle_elementwise_binary()` in `ops/elementwise.cpp` did NOT cast
inputs to a common type. In real models, `MUL(rms_norm_output_F32, norm_weight_F16)`
passes mismatched types to TRT's `addElementWise`. TRT strongly-typed mode silently
produces garbage output when types don't match.

**Fix**: Added type reconciliation before `addElementWise`, using the same narrowing
strategy as matmul — cast the F32 operand down to F16/BF16. The subgraph output
cast at the boundary converts back to the node's expected ggml type.

### Bug 6: CPY/CONT raw memcpy ignores type conversion — DATA CORRUPTION (this milestone)

**Symptom**: Potential silent data corruption when CPY source and dest have different types.

**Root cause**: CPY/DUP/CONT were handled with `cudaMemcpyAsync(dst, src, ggml_nbytes(node))`.
When src is F32 and dst is F16, `ggml_nbytes(node)` returns the F16 byte count — only half
the source data is copied, and F32 bits are reinterpreted as F16.

**Fix**: Removed CPY/DUP/CONT from `supports_op()`. CUDA backend handles these correctly.

### Bug 7: SET_ROWS CPU-side scatter kills performance — 4 t/s (this milestone)

**Symptom**: Extremely slow generation (4 t/s vs 114 t/s CUDA-only).

**Root cause**: Each SET_ROWS did `cudaStreamSynchronize()` + D2H index copy + per-row
D2H/CPU-convert/H2D. For ~52 SET_ROWS per token, each stalling the GPU pipeline.

**Fix**: Removed SET_ROWS from `supports_op()`. CUDA backend handles it with native
GPU kernels that run entirely on-device.

---

## Milestone 7: Debug Instrumentation Added

### Correctness Instrumentation (`GGML_TENSORRT_DEBUG=1`)

Zero overhead when disabled (env var cached at first call). When enabled, logs around
each TRT execution:

- **Call counter**: Each `graph_compute` call numbered (`call #N`)
- **Input checksums**: Before `enqueueV3`, first 4 float values of each input (D2H, converts F16/BF16)
- **Output checksums**: After `enqueueV3`, first 4 float values of each output
- **Address tracking**: Detects address changes between calls to same engine hash

```
[TRT-DEBUG] call #0, hash 0x..., inputs: 5, outputs: 3
[TRT-DEBUG]   input_0: addr=0x..., type=f16, shape=[4096,1536,1,1], first=[0.0012, -0.0034, ...]
[TRT-DEBUG]   output_0: addr=0x..., type=f32, shape=[1,4096,1,1], first=[0.123, -0.456, ...]
```

### Performance Instrumentation (`GGML_TENSORRT_PROFILE=1`)

Zero overhead when disabled. When enabled:

- **Phase timing**: Phase 1 (categorize+collect), Phase 2 (hash), Phase 3 (build/cache), Phase 4 (execute)
- **Cache hit/miss**: Whether Phase 3 was a hit or miss
- **Running stats**: Cumulative hits, misses, total build time

```
[TRT-PROF] call #0: hash=0x..., nodes=12, leaves=5, phase1=0.02ms, phase2=0.01ms, phase3=2340.00ms (miss), phase4=0.15ms
[TRT-PROF] cache: hits=0, misses=1, total_build_time=2340ms
```

### Engine Manager Stats

`EngineManager` now exposes `cache_hits`, `cache_misses`, `total_build_time_ms`.

---

## Milestone 7: Test Plan

### 1. Build
```bash
cd /app/llama.cpp
cmake -B build -DGGML_TENSORRT=ON -DTENSORRT_ROOT=/opt/TensorRT-RTX -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release -j 8
```

### 2. Unit Tests
```bash
cd build && ctest -R tensorrt -V
```
- All tests pass (m6a is removed — should NOT appear)
- Expected tests: init, devices, buffers, transfer, m2, m3, m5, m5b, m6

### 3. Scheduler Routing
```bash
GGML_SCHED_DEBUG=2 ./build/bin/llama-cli -m /path/to/gemma-3-1b-f16.gguf -p "Hello" -n 10 2>&1 | grep -E "SET_ROWS|CPY|CONT"
```
- SET_ROWS, CPY, CONT assigned to `CUDA0` (not `TensorRT-RTX0`)

### 4. Correctness with Debug
```bash
GGML_TENSORRT_DEBUG=1 GGML_TENSORRT_LOG_LEVEL=INFO \
  ./build/bin/llama-cli -m /path/to/gemma-3-1b-f16.gguf -p "The meaning of life is" -n 20 2>&1 | head -200
```
- call #0 input/output values: not NaN, not all zero
- If garbage on call #0 → fundamental op bug
- If garbage on call #2+ → state/caching issue
- Generated text should be coherent

### 5. Performance with Profiling
```bash
GGML_TENSORRT_PROFILE=1 \
  ./build/bin/llama-cli -m /path/to/gemma-3-1b-f16.gguf -p "Hello world" -n 50 2>&1 | grep TRT-PROF
```
- Engine builds (`miss`) only on first few unique shapes
- Token 2+ should be all `hit`
- No SET_ROWS stalls (those ops are on CUDA now)
- `misses` count stabilizes (no per-token rebuilds)

### 6. Performance Comparison
```bash
# TRT+CUDA hybrid
./build/bin/llama-cli -m /path/to/gemma-3-1b-f16.gguf -p "Hello" -n 100

# CUDA-only
./build/bin/llama-cli -m /path/to/gemma-3-1b-f16.gguf -p "Hello" -n 100 --no-tensorrt
```
- TRT should be significantly faster than the previous 4 t/s
- Compare against CUDA-only baseline (114 t/s)

### 7. Memory Leak Check
```bash
GGML_TENSORRT_PROFILE=1 \
  ./build/bin/llama-cli -m /path/to/gemma-3-1b-f16.gguf -p "Hello" -n 200 2>&1 | grep "cache:"
```
- `misses` count stabilizes (not growing every token)

---

## All Environment Variables (TRT Backend)

| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_TENSORRT_DEBUG` | 0 | Correctness instrumentation (input/output dumps) |
| `GGML_TENSORRT_PROFILE` | 0 | Performance instrumentation (phase timing, cache stats) |
| `GGML_TENSORRT_CUDA_GRAPHS` | 1 | CUDA graph capture |
| `GGML_TENSORRT_AUX_STREAMS` | 0 | Auxiliary streams |
| `GGML_TENSORRT_WORKSPACE_MB` | 256 | Workspace size in MB |
| `GGML_TENSORRT_DUMP_GRAPH` | 0 | Dump subgraph structure before engine build |
| `GGML_TENSORRT_LOG_LEVEL` | ERROR | TRT logger level (VERBOSE/INFO/WARN/ERROR) |

---

## Files Modified (This Milestone)

| File | Change |
|------|--------|
| `ggml/src/ggml-tensorrt/ops/elementwise.cpp` | Type casting before `addElementWise` (Bug 5) |
| `ggml/src/ggml-tensorrt/ggml-tensorrt.cpp` | Remove CPY/DUP/CONT/SET_ROWS (Bug 6+7); add debug+profile instrumentation |
| `ggml/src/ggml-tensorrt/engine-manager.hpp` | Add cache stats members |
| `ggml/src/ggml-tensorrt/engine-manager.cpp` | Track cache stats; add build timing |
| `tests/test-tensorrt-ops-m5b.cpp` | Updated comments |
| `tests/test-tensorrt-ops-m6a.cpp` | Deleted |
| `tests/CMakeLists.txt` | Removed m6a registration |
