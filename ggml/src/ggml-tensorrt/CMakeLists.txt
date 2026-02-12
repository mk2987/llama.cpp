cmake_minimum_required(VERSION 3.18)

# Include the TensorRT configuration
include(../../cmake/ggml-tensorrt.cmake)

if (NOT TENSORRT_FOUND)
    message(FATAL_ERROR "TensorRT-RTX not found. Please set TENSORRT_ROOT to your TensorRT-RTX installation directory.")
    return()
endif()

# Find CUDA (required for TensorRT)
find_package(CUDAToolkit REQUIRED)

if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. TensorRT backend requires CUDA.")
    return()
endif()

# Enable CUDA language for .cu kernel files
enable_language(CUDA)

message(STATUS "TensorRT Backend: Building with TensorRT-RTX support")
message(STATUS "  CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "  TensorRT include: ${TENSORRT_INCLUDE_DIRS}")

# Collect headers
file(GLOB GGML_HEADERS_TENSORRT "*.hpp" "*.h")
list(APPEND GGML_HEADERS_TENSORRT "../../include/ggml-tensorrt.h")

# Collect sources
set(GGML_SOURCES_TENSORRT
    ggml-tensorrt.cpp
    common.cpp
    network-builder.cpp
    engine-manager.cpp
    ops/matmul.cpp
    ops/elementwise.cpp
    ops/normalization.cpp
    ops/rope.cpp
    ops/softmax.cpp
    ops/attention.cpp
    ops/unary.cpp
    ops/shape.cpp
    ops/scale.cpp
    ops/gather.cpp
    utils/tensor-utils.cpp
    utils/type-utils.cpp
    utils/logger.cpp
    kernels/set-rows.cu
)

# Create the backend library
ggml_add_backend_library(ggml-tensorrt
    ${GGML_HEADERS_TENSORRT}
    ${GGML_SOURCES_TENSORRT}
)

# Set C++ and CUDA standard (TensorRT requires C++17 or later)
set_target_properties(ggml-tensorrt PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Include directories
# Mark TensorRT headers as SYSTEM to suppress their warnings
target_include_directories(ggml-tensorrt SYSTEM PRIVATE
    ${TENSORRT_INCLUDE_DIRS}
)
target_include_directories(ggml-tensorrt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Link libraries
target_link_libraries(ggml-tensorrt PRIVATE
    ${TENSORRT_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
)

# Add compile definitions
target_compile_definitions(ggml-tensorrt PRIVATE
    GGML_USE_TENSORRT
)

# Compiler warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ggml-tensorrt PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-unused-function
        -Wno-deprecated-declarations  # Suppress TensorRT-RTX header deprecation warnings
    )
endif()

# CUDA compile options
target_compile_options(ggml-tensorrt PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>"
)
