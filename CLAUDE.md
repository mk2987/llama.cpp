# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository integrates **llama.cpp** (LLM inference in C/C++) with **NVIDIA TensorRT-RTX 1.3.0.35** for optimized inference on NVIDIA GeForce RTX GPUs. The project uses Docker to create a containerized environment with CUDA 12.9 support.

## Docker-Based Development

### Build and Run Container

```bash
# Build the Docker image
docker build -t llama-rtx .

# Run container with GPU support
docker run --gpus all -it -v $(pwd)/llama.cpp:/app/llama.cpp llama-rtx

# Run with specific user permissions (matching host UID/GID)
docker build --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g) -t llama-rtx .
docker run --gpus all -it -v $(pwd)/llama.cpp:/app/llama.cpp llama-rtx
```

The Dockerfile:
- Uses CUDA 12.9 base image on Ubuntu 24.04
- Installs TensorRT-RTX from the tarball in `/opt/TensorRT-RTX`
- Sets up environment variables: `PATH` and `LD_LIBRARY_PATH` for TensorRT-RTX
- Creates a non-root user with configurable UID/GID
- Mounts the llama.cpp directory at `/app/llama.cpp`

## TensorRT-RTX Integration

**Version**: 1.3.0.35 (Linux x86_64, CUDA 13.1)

**Installation Location** (in container): `/opt/TensorRT-RTX/`
- Binaries: `/opt/TensorRT-RTX/bin/tensorrt_rtx`
- Libraries: `/opt/TensorRT-RTX/lib/`
- Headers: `/opt/TensorRT-RTX/include/`
- Python bindings: `/opt/TensorRT-RTX/python/`

**Resources**:
- Documentation: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html
- C++ API: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/index.html
- Python API: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/python-api/index.html

## llama.cpp Architecture

### Directory Structure

- **`llama.cpp/src/`** - Core inference engine (llama library)
  - Public API: `include/llama.h`
  - Model loading, context management, KV cache, sampling

- **`llama.cpp/ggml/`** - Tensor library with multi-backend support
  - CPU, CUDA, Metal, Vulkan, SYCL, HIP, and more
  - Can be used as system library or built from source

- **`llama.cpp/common/`** - Utility library
  - Argument parsing, chat handling, JSON schema, sampling

- **`llama.cpp/tools/`** - Standalone tools
  - `server/` - HTTP inference server
  - `quantize/`, `imatrix/`, `perplexity/` - Model utilities

- **`llama.cpp/examples/`** - Example applications
- **`llama.cpp/tests/`** - Test suite

### Build System

llama.cpp uses **CMake** as the primary build system (a Makefile also exists for simple CPU builds).

**Basic CPU Build**:
```bash
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

**CUDA Build** (for NVIDIA GPU acceleration):
```bash
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

**Parallel Build**:
```bash
cmake --build build --config Release -j 8
```

**CMake Presets** (see `CMakePresets.json`):
```bash
# Examples
cmake --preset x64-linux-gcc-release
cmake --build build-x64-linux-gcc-release
```

**Build Options** (key flags in `CMakeLists.txt`):
- `GGML_CUDA=ON` - Enable CUDA backend
- `GGML_METAL=ON` - Enable Metal (macOS)
- `GGML_VULKAN=ON` - Enable Vulkan
- `BUILD_SHARED_LIBS=OFF` - Static build
- `LLAMA_BUILD_TESTS=ON` - Build test suite
- `LLAMA_BUILD_TOOLS=ON` - Build tools
- `LLAMA_BUILD_SERVER=ON` - Build HTTP server

### Key Architectural Components

**Model & Context Hierarchy**:
- `llama_model` - Holds trained weights, architecture, hyperparameters
- `llama_context` - Manages inference state, KV cache, batch processing

**Backend Integration**:
- GGML provides hardware abstraction layer
- Backend scheduler handles multi-GPU and device distribution
- Computation graphs (`ggml_cgraph`) execute tensor operations

**Memory Management**:
- Memory-mapped model loading (GGUF format)
- Flexible KV cache implementations (standard, ISWA, hybrid, recurrent)
- Backend-specific buffer allocation

**Model Loading Pipeline**:
- GGUF format (versions V1/V2/V3)
- Lazy tensor loading with memory mapping
- Quantized weight support (1.5-bit to 8-bit)

**Computation Graph Pattern**:
- Per-architecture graph builders (120+ models supported)
- Template-based polymorphism (`llm_build_llama`, `llm_build_qwen2`, etc.)
- Graphs created on-demand per inference step

### Testing

**Run All Tests**:
```bash
cd llama.cpp/build
ctest
```

**Key Test Executables** (in `llama.cpp/tests/`):
- `test-backend-ops` - Verify backend operator consistency
- `test-backend-sampler` - Sampling algorithms
- `test-chat-template` - Chat template parsing
- `test-tokenizer-*` - Tokenizer validation

**Backend Testing**:
If you modify GGML operators, run `test-backend-ops` with multiple backends to ensure consistent results.

### Important Files

**Coordination Files**:
- `src/llama.cpp` - Main API implementation
- `src/llama-model.cpp` - Model loading and graph building dispatch
- `src/llama-context.cpp` - Context lifecycle and scheduler
- `src/llama-arch.h` - Architecture registry (120+ models)

**Build Configuration**:
- `CMakeLists.txt` - Main build orchestration
- `CMakePresets.json` - Preset configurations
- `cmake/` - Build system modules

**Documentation**:
- `docs/build.md` - Build instructions
- `docs/docker.md` - Docker usage
- `CONTRIBUTING.md` - Contribution guidelines
- `AGENTS.md` - **CRITICAL**: AI agent usage policy

## Critical: AI Usage Policy

**This project does NOT accept AI-generated pull requests.** Before contributing to llama.cpp:

1. Read `llama.cpp/AGENTS.md` thoroughly
2. Read `llama.cpp/CONTRIBUTING.md`
3. AI can ONLY be used in an assistive capacity:
   - Asking about codebase structure
   - Code review and suggestions
   - Learning about techniques used
4. Contributors must understand and explain every line of code
5. Explicit disclosure required for any AI usage

**Forbidden**:
- Writing full implementations or large code blocks
- Generating entire PRs
- Bypassing human understanding

## Common Commands

### Building

```bash
# Inside Docker container
cd /app/llama.cpp

# CPU build
cmake -B build
cmake --build build --config Release -j 8

# CUDA build (with TensorRT-RTX available)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 8

# Build with tests
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release -j 8
```

### Running

```bash
# Run inference with a local model
./build/bin/llama-cli -m /path/to/model.gguf

# Download and run a model from Hugging Face
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Start HTTP server
./build/bin/llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

### Testing

```bash
cd build

# Run all tests
ctest

# Run specific test
ctest -R test-tokenizer

# Run with verbose output
ctest -V

# Run backend operations test (multi-backend validation)
./bin/test-backend-ops
```

### Model Conversion

```bash
# Convert HuggingFace model to GGUF
python3 convert_hf_to_gguf.py /path/to/hf/model

# Quantize model
./build/bin/llama-quantize input.gguf output.gguf q4_0
```

## Code Style (llama.cpp)

From `CONTRIBUTING.md`:
- Use `snake_case` for functions, variables, types
- Avoid STL templates, use basic `for` loops
- 4 spaces for indentation
- Vertical alignment preferred
- Use sized integer types (`int32_t`) in public APIs
- Follow longest common prefix naming (e.g., `number_small`, `number_big`)
- Enum values uppercase with prefix: `LLAMA_VOCAB_TYPE_SPM`
- Naming pattern: `<class>_<method>` where method is `<action>_<noun>`
- Constructor/destructor actions: `init`/`free`

## Development Workflow

1. **Environment Setup**: Build Docker image and run container with GPU support
2. **Build llama.cpp**: Use CMake inside container with appropriate backend flags
3. **Make Changes**: Focus on CPU support initially for new features
4. **Test Changes**:
   - Run relevant tests with `ctest`
   - Check backend compatibility with `test-backend-ops`
   - Verify perplexity/performance with `llama-perplexity` and `llama-bench`
5. **Backend Support**: Add GPU backend support (CUDA, etc.) in follow-up work

## Model Support

llama.cpp supports 120+ model architectures. Key architectures:
- LLaMA (1, 2, 3)
- Mistral, Mixtral, Jamba
- Qwen, Gemma, Phi
- BERT, Starcoder
- Multimodal: LLaVA, MiniCPM, Qwen2-VL

Check `src/llama-arch.h` for the full architecture registry.

## GGUF Format

**GGUF** is the native model format:
- Supports quantization (1.5-bit to 8-bit)
- Memory-mappable for efficient loading
- Version support: V1, V2, V3
- Use `convert_hf_to_gguf.py` to convert from HuggingFace

## Resources

**llama.cpp**:
- Main repository: https://github.com/ggml-org/llama.cpp
- GGML library: https://github.com/ggml-org/ggml

**TensorRT-RTX**:
- Developer page: https://developer.nvidia.com/tensorrt-rtx
- Documentation: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html
