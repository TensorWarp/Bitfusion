#include "gpuTypes.h"
#include "types.h"
#include <limits>

static __constant__ GpuData cData;

__device__ inline float atomicMax(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

void SetKActivationGpuData()
{
    cudaError_t status;
    status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));
}

void GetKActivationGpuData()
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));
}

__global__ void
__launch_bounds__(256, 4)
invokeSigmoidActivation_kernel(float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float a = 1.0f / (1.0f + exp(-pData[pos]));
        pData[pos] = a;
    }
}


void invokeSigmoidActivation(float* pData, uint64_t size)
{
    uint32_t blocks = CalculateBlocks(size);

    void* kernelArgs[] = { &pData, &size };

    cudaLaunchKernel(reinterpret_cast<void*>(invokeSigmoidActivation_kernel), dim3(blocks), dim3(getGpu()._threadsPerBlock), kernelArgs, 0, cudaStreamDefault);
}

__global__ void
__launch_bounds__(256, 4)
invokeTanhActivation_kernel(float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        pData[pos] = tanh(pData[pos]);
}

void invokeTanhActivation(float* pData, uint64_t size)
{
    uint32_t blocks = CalculateBlocks(size);

    void* kernelArgs[] = { &pData, &size };

    cudaLaunchKernel(reinterpret_cast<void*>(invokeTanhActivation_kernel), dim3(blocks), dim3(getGpu()._threadsPerBlock), kernelArgs, 0, cudaStreamDefault);
}

__global__ void
__launch_bounds__(256, 4)
invokeRELUActivation_kernel(float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        pData[pos] = max(0.0f, pData[pos]);
}

void invokeRELUActivation(float* pData, uint64_t size)
{
    uint32_t blocks = CalculateBlocks(size);

    void* kernelArgs[] = { &pData, &size };

    cudaLaunchKernel(reinterpret_cast<void*>(invokeRELUActivation_kernel), dim3(blocks), dim3(getGpu()._threadsPerBlock), kernelArgs, 0, cudaStreamDefault);
}

__global__ void
__launch_bounds__(256, 4)
invokeLRELUActivation_kernel(float* pData, uint64_t size, float slope)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float val = pData[pos];
        pData[pos] = max(val, val * slope);
    }
}

void invokeLRELUActivation(float* pData, uint64_t size, float slope)
{
    uint32_t blocks = CalculateBlocks(size);

    void* kernelArgs[] = { &pData, &size, &slope };

    cudaLaunchKernel(reinterpret_cast<void*>(invokeLRELUActivation_kernel), dim3(blocks), dim3(getGpu()._threadsPerBlock), kernelArgs, 0, cudaStreamDefault);
}

__global__ void
__launch_bounds__(256, 4)
invokeELUActivation_kernel(float* pData, uint64_t size, float alpha)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x = pData[pos];
        pData[pos] = (x > (float)0.0) ? x : alpha * (exp(x) - (float)1.0);
    }
}

void invokeELUActivation(float* pData, uint64_t size, float alpha)
{
    uint32_t blocks = CalculateBlocks(size);

    void* kernelArgs[] = { &pData, &size, &alpha };

    cudaLaunchKernel(reinterpret_cast<void*>(invokeELUActivation_kernel), dim3(blocks), dim3(getGpu()._threadsPerBlock), kernelArgs, 0, cudaStreamDefault);
}

__global__ void
__launch_bounds__(256, 4)
invokeSELUActivation_kernel(float* pData, uint64_t size, float alpha, float lambda)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x = pData[pos];
        pData[pos] = (x > (float)0.0) ? lambda * x : lambda * alpha * (exp(x) - (float)1.0);
    }
}

void invokeSELUActivation(float* pData, uint64_t size, float alpha, float lambda)
{
    uint32_t blocks = CalculateBlocks(size);

    void* kernelArgs[] = { &pData, &size, &alpha, &lambda };

    cudaLaunchKernel(reinterpret_cast<void*>(invokeSELUActivation_kernel), dim3(blocks), dim3(getGpu()._threadsPerBlock), kernelArgs, 0, cudaStreamDefault);
}

__global__ void
__launch_bounds__(256, 4)
invokeSoftMaxActivation_kernel(float* pData, uint32_t stride)
{
    __shared__ unsigned long long int sAccumulator;
    __shared__ float sMaxValue;

    if (threadIdx.x == 0)
    {
        sAccumulator = 0;
        sMaxValue = (float)-99999999.0f;
    }
    __syncthreads();


    pData += blockIdx.x * stride;
    uint32_t pos = threadIdx.x;
    float maxValue = (float)-9999999999.0;

    while (pos < stride)
    {
        float z = pData[pos];
        maxValue = max(z, maxValue);
        pos += blockDim.x;
    }

    uint32_t tgx = threadIdx.x & cData._warpMask;
    maxValue = max(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 1));
    maxValue = max(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 2));
    maxValue = max(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 4));
    maxValue = max(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 8));
    maxValue = max(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 16));

    if (tgx == 0)
        atomicMax(&sMaxValue, maxValue);
    __syncthreads();
    maxValue = sMaxValue;

    pos = threadIdx.x;
    float sum = (float)0.0;
    while (pos < stride)
    {
        float z = pData[pos];
        sum += exp(z - maxValue);
        pos += blockDim.x;
    }

    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 1);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 2);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 4);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 8);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 16);
    unsigned long long int lsum = llitoulli(llrintf(ERRORSCALEF * sum));
    if (tgx == 0)
        atomicAdd(&sAccumulator, lsum);
    __syncthreads();
    float norm = (float)1.0 / (float)((double)sAccumulator * ONEOVERERRORSCALE);


    pos = threadIdx.x;
    while (pos < stride)
    {
        float z = pData[pos];
        float a = exp(z - maxValue);
        pData[pos] = min((float)1.0, a * norm);
        pos += blockDim.x;
    }

}

void invokeSoftMaxActivation(float* pData, uint32_t batch, uint32_t stride)
{
    uint32_t warps = getGpu()._threadsPerBlock / getGpu()._warpSize;

    void* kernelArgs[] = { &pData, &stride };

    cudaLaunchKernel(reinterpret_cast<void*>(invokeSoftMaxActivation_kernel), dim3(batch), dim3(getGpu()._threadsPerBlock), kernelArgs, 0, cudaStreamDefault);
}