# Find TensorRT-RTX installation

if (NOT DEFINED TENSORRT_ROOT)
    if (DEFINED ENV{TENSORRT_ROOT})
        set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
    else()
        set(TENSORRT_ROOT "/opt/TensorRT-RTX")
    endif()
endif()

message(STATUS "TensorRT-RTX: Using TENSORRT_ROOT=${TENSORRT_ROOT}")

# Find TensorRT-RTX headers
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TENSORRT_ROOT}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)

# Find TensorRT-RTX libraries
find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    PATHS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)

find_library(TENSORRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    PATHS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)

if (TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    message(STATUS "TensorRT-RTX: Found headers at ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT-RTX: Found library at ${TENSORRT_LIBRARY}")

    set(TENSORRT_FOUND TRUE)

    # Set up include directories and libraries
    set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
    set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY})

    if (TENSORRT_PLUGIN_LIBRARY)
        message(STATUS "TensorRT-RTX: Found plugin library at ${TENSORRT_PLUGIN_LIBRARY}")
        list(APPEND TENSORRT_LIBRARIES ${TENSORRT_PLUGIN_LIBRARY})
    endif()

else()
    message(WARNING "TensorRT-RTX: Could not find TensorRT installation")
    message(WARNING "  Looked in: ${TENSORRT_ROOT}")
    message(WARNING "  Set TENSORRT_ROOT to your TensorRT-RTX installation directory")
    set(TENSORRT_FOUND FALSE)
endif()
