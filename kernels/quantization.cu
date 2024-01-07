

#include "../common/assert.h"
#include "../common/cudaTypeUtils.cuh"
#include "../common/cudaUtils.h"
#include "../common/reduceKernelUtils.cuh"
#include "quantization.h"

using namespace bitfusion::common;

namespace bitfusion
{
namespace kernels
{

__global__ void quantizedKernel(char4* dst, const float4* src, const int64_t sizeDiv4, const float* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        const float scale = __ldg(scalePtr);
        char4 tmp;
        const float4 floatTmp = __ldg(src + idx);
        tmp.x = cuda_cast<int8_t>(floatTmp.x * scale);
        tmp.y = cuda_cast<int8_t>(floatTmp.y * scale);
        tmp.z = cuda_cast<int8_t>(floatTmp.z * scale);
        tmp.w = cuda_cast<int8_t>(floatTmp.w * scale);
        dst[idx] = tmp;
    }
}

__global__ void quantizedKernel(char4* dst, const half2* src, const int64_t sizeDiv4, const float* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        const float scale = __ldg(scalePtr);
        char4 tmp;
        int srcId = idx << 1;

        const uint2 h2 = __ldg(reinterpret_cast<const uint2*>(src + srcId));

        const half2 half2Tmp = reinterpret_cast<const half2&>(h2.x);
        const half2 half2Tmp2 = reinterpret_cast<const half2&>(h2.y);

        tmp.x = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.x) * scale);
        tmp.y = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.y) * scale);
        tmp.z = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.x) * scale);
        tmp.w = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.y) * scale);
        dst[idx] = tmp;
    }
}

template <typename T>
void invokeQuantization(
    int8_t* dst, const T* src, const int64_t size, const float* scalePtr, cudaStream_t stream, int maxGridSize)
{
    CHECK_WITH_INFO(size % 4 == 0, "[ERROR][invokeQuantization] size should be a multiple of 4.\n");

    int numBlocks{static_cast<int>((size + 255) / 256)};
    dim3 grid(std::min(numBlocks, maxGridSize));
    CHECK_WITH_INFO(grid.x <= maxGridSize, "[ERROR][invokeQuantization] grid max size is exceeded\n");
    dim3 block(64);
    if (std::is_same_v<T, float>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (const float4*) src, size / 4, scalePtr);
    }
    else if (std::is_same_v<T, half>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (const half2*) src, size / 4, scalePtr);
    }
}

template void invokeQuantization<float>(
    int8_t* dst, const float* src, const int64_t size, const float* scalePtr, cudaStream_t stream, int maxGridSize);

template void invokeQuantization<half>(
    int8_t* dst, const half* src, const int64_t size, const float* scalePtr, cudaStream_t stream, int maxGridSize);

template <typename T>
__global__ void perTokenQuantization(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr)
{
    const T* srcRow = src + blockIdx.x * numCols;
    int8_t* dstRow = dst + blockIdx.x * numCols;

    T localMax = 1e-6f;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x)
    {
        localMax = cuda_max(localMax, cuda_abs(srcRow[i]));
    }
    const float rowMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0)
    {
        scalePtr[blockIdx.x] = rowMax / 127.f;
    }

    const float scaleOrigQuant = 127.f / rowMax;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x)
    {
        dstRow[i] = cuda_cast<int8_t>(cuda_cast<float>(srcRow[i]) * scaleOrigQuant);
    }
}

template <typename T>
void invokePerTokenQuantization(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, cudaStream_t stream)
{
    // each block is responsible for a single row
    const dim3 block(512);
    const dim3 grid(numRows);

    perTokenQuantization<<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr);
}

#define INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(T)                                                                   \
    template void invokePerTokenQuantization(                                                                          \
        int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, cudaStream_t stream)

INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(float);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(__nv_bfloat16);
#endif

} // namespace kernels
} // namespace bitfusion
