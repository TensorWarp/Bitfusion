

#pragma once

#include "../common/cudaUtils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bitfusion
{
namespace kernels
{

template <typename T>
void invokeGeneralRmsNorm(T* out, const T* input, const T* gamma, const T* beta, const float eps, const int tokens,
    const int hidden_dim, cudaStream_t stream = 0, const float* scale = nullptr, float* dynamic_scale = nullptr,
    int8_t* out_quant = nullptr);

} // namespace kernels
} // namespace bitfusion
