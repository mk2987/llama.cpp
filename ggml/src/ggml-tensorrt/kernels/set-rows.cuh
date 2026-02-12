#pragma once

#include "ggml.h"
#include <cuda_runtime.h>

// CUDA scatter kernel for SET_ROWS — replaces CPU round-trip path.
// Dispatches on src1->type (I32/I64) x dst->type (F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL).
void ggml_tensorrt_set_rows(const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst, cudaStream_t stream);
