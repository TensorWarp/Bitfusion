#ifdef __CUDACC__

#include "GpuTypes.h"
#include "Types.h"
#include <limits>
#include "Kernels.cuh"

/// <summary>
/// CUDA constant data structure containing GPU data.
/// </summary>
static __constant__ GpuData cData;

/// <summary>
/// Atomically updates the maximum value at the specified memory address with the given value.
/// </summary>
/// <param name="address">Pointer to the memory address to update.</param>
/// <param name="val">The value to compare and update with the current maximum.</param>
/// <returns>The previous maximum value.</returns>
__device__ inline float atomicMax(float* address, float val)
{
    int* address_as_int = reinterpret_cast<int*>(address);
    int old_as_int = __float_as_int(*address);
    int assumed_as_int = old_as_int;
    int max_as_int = __float_as_int(val);

    do
    {
        old_as_int = assumed_as_int;
        assumed_as_int = atomicCAS(address_as_int, old_as_int, max_as_int);
    } while (assumed_as_int != old_as_int);

    return __int_as_float(old_as_int);
}

/// <summary>
/// Copies GPU data to a CUDA constant symbol `cData`.
/// </summary>
void SetKActivationGpuData()
{
    cudaError_t status;
    status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));
    RTERROR(status, "cudaMemcpyToSymbol: SetKActivationGpuData copy to cData failed");
}

/// <summary>
/// Retrieves GPU data from a CUDA constant symbol `cData`.
/// </summary>
void GetKActivationGpuData()
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));
    RTERROR(status, "cudaMemcpyFromSymbol: GetKActivationGpuData copy From cData failed");
}

/// <summary>
/// CUDA kernel function for calculating the Sigmoid activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidActivation_kernel(float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float a = 1.0f / (1.0f + exp(-pData[pos]));
        pData[pos] = a;
    }
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the Sigmoid activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
void kCalculateSigmoidActivation(float* pData, uint64_t size)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size
    };

    cudaLaunchKernel((void*)kCalculateSigmoidActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    LAUNCHERROR("kCalculateSigmoidActivation_kernel");
}

/// <summary>
/// CUDA kernel function for calculating the Hyperbolic Tangent (tanh) activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
__global__ void LAUNCH_BOUNDS() kCalculateTanhActivation_kernel(float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        pData[pos] = tanh(pData[pos]);
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the tanh activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
void kCalculateTanhActivation(float* pData, uint64_t size)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size
    };

    cudaLaunchKernel((void*)kCalculateTanhActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    LAUNCHERROR("kCalculateTanhActivation_kernel");
}

/// <summary>
/// CUDA kernel function for calculating the Rectified Linear Unit (ReLU) activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
__global__ void LAUNCH_BOUNDS() kCalculateRELUActivation_kernel(float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        pData[pos] = max(0.0f, pData[pos]);
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the ReLU activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
void kCalculateRELUActivation(float* pData, uint64_t size)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size
    };

    cudaLaunchKernel((void*)kCalculateRELUActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    LAUNCHERROR("kCalculateRELUActivation_kernel");
}

/// <summary>
/// CUDA kernel function for calculating the Leaky Rectified Linear Unit (LReLU) activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="slope">Slope of the activation for negative inputs.</param>
__global__ void LAUNCH_BOUNDS() kCalculateLRELUActivation_kernel(float* pData, uint64_t size, float slope)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float val = pData[pos];
        pData[pos] = max(val, val * slope);
    }
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the LReLU activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="slope">Slope of the activation for negative inputs.</param>
void kCalculateLRELUActivation(float* pData, uint64_t size, float slope)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size,
        &slope
    };

    cudaLaunchKernel((void*)kCalculateLRELUActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    LAUNCHERROR("kCalculateLRELUActivation_kernel");
}

/// <summary>
/// CUDA kernel function for calculating the Exponential Linear Unit (ELU) activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="alpha">Scaling factor for negative inputs.</param>
__global__ void LAUNCH_BOUNDS() kCalculateELUActivation_kernel(float* pData, uint64_t size, float alpha)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x = pData[pos];
        pData[pos] = (x > (float)0.0) ? x : alpha * (exp(x) - (float)1.0);
    }
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the ELU activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="alpha">Scaling factor for negative inputs.</param>
void kCalculateELUActivation(float* pData, uint64_t size, float alpha)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size,
        &alpha
    };

    cudaLaunchKernel((void*)kCalculateELUActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    LAUNCHERROR("kCalculateELUActivation_kernel");
}

/// <summary>
/// CUDA kernel function for calculating the Scaled Exponential Linear Unit (SELU) activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="alpha">Scaling factor for negative inputs.</param>
/// <param name="lambda">Scaling factor for positive inputs.</param>
__global__ void LAUNCH_BOUNDS() kCalculateSELUActivation_kernel(float* pData, uint64_t size, float alpha, float lambda)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x = pData[pos];
        pData[pos] = (x > (float)0.0) ? lambda * x : lambda * alpha * (exp(x) - (float)1.0);
    }
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the SELU activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="alpha">Scaling factor for negative inputs.</param>
/// <param name="lambda">Scaling factor for positive inputs.</param>
void kCalculateSELUActivation(float* pData, uint64_t size, float alpha, float lambda)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size,
        &alpha,
        &lambda
    };

    cudaLaunchKernel((void*)kCalculateSELUActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    LAUNCHERROR("kCalculateSELUActivation_kernel");
}

/// <summary>
/// This CUDA kernel function calculates the SoftMax activation for a given array.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="stride">Stride of the input data.</param>
__global__ void kCalculateSoftMaxActivation_kernel(float* pData, uint32_t stride)
{
    // Shared memory for maximum value and sum of exponentials
    __shared__ float sMaxValue;
    __shared__ float sSumExp;

    // Initialize shared memory variables for the first thread in the block
    if (threadIdx.x == 0)
    {
        sMaxValue = -99999999.0f;
        sSumExp = 0.0f;
    }
    __syncthreads();

    // Calculate the starting position for this block
    pData += blockIdx.x * stride;
    uint32_t pos = threadIdx.x;
    float maxValue = -9999999999.0;

    // Compute the maximum value within the block
    while (pos < stride)
    {
        float z = pData[pos];
        maxValue = fmaxf(z, maxValue);
        pos += blockDim.x;
    }

    // Perform warp-level parallel reduction to find the block maximum
    uint32_t tgx = threadIdx.x & cData._warpMask;
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 1));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 2));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 4));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 8));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 16));

    // Store the block maximum in shared memory
    if (tgx == 0)
        sMaxValue = maxValue;
    __syncthreads();

    // Compute the sum of exponentials for the SoftMax function
    pos = threadIdx.x;
    float sum = 0.0f;
    while (pos < stride)
    {
        float z = pData[pos];
        sum += expf(z - sMaxValue);
        pos += blockDim.x;
    }

    // Perform warp-level parallel reduction to find the block sum
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 1);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 2);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 4);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 8);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 16);

    // Accumulate the sum of exponentials atomically
    atomicAdd(&sSumExp, sum);
    __syncthreads();

    // Normalize and apply SoftMax to the input data
    pos = threadIdx.x;
    while (pos < stride)
    {
        float z = pData[pos];
        float a = expf(z - sMaxValue);
        pData[pos] = fminf(1.0f, a / sSumExp);
        pos += blockDim.x;
    }
}

/// <summary>
/// This function calculates the SoftMax activation for a given data batch.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="batch">Number of data points in the batch.</param>
/// <param name="stride">Stride of the input data.</param>
void kCalculateSoftMaxActivation(float* pData, uint32_t batch, uint32_t stride)
{
    // Calculate the number of warps per block
    uint32_t warps = getGpu()._threadsPerBlock / getGpu()._warpSize;

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &batch,
        &stride
    };

    cudaLaunchKernel((void*)kCalculateSoftMaxActivation_kernel, batch, blockDim, args, 0, nullptr);

    // Check for launch errors
    LAUNCHERROR("kCalculateSoftMaxActivation_kernel");
}

#endif