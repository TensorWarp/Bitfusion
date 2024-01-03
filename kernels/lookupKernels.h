

#pragma once

#include "../common/cudaUtils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bitfusion
{
namespace kernels
{
template <typename T, typename Idx>
void invokeLookUp(T* out, const Idx* input, const T* weight, const Idx batch_size, const Idx offset, const Idx size,
    const int n_embed, cudaStream_t stream = 0);

} // namespace kernels
} // namespace bitfusion
