
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#if defined(ENABLE_BF16)
#include <cuda_bf16.h>
#endif

#include <type_traits>
#include <vector>

namespace bitfusion
{
namespace kernels
{

template <typename T>
void apply_per_channel_scale_kernel_launcher(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream = 0);

} // namespace kernels
} // namespace bitfusion
