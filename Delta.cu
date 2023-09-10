#ifdef __CUDACC__

#include "GpuTypes.h"
#include "Types.h"
#include <limits>
#include <iostream>
#include "Kernels.cuh"

/// <summary>
/// Constant GPU data structure.
/// </summary>
static __constant__ GpuData cData;

/// <summary>
/// Minimum activation value.
/// </summary>
#define MIN_ACTIVATION 0.1f

/// <summary>
/// Maximum activation value.
/// </summary>
#define MAX_ACTIVATION 0.9f

/// <summary>
/// Copy data from CPU to GPU constant memory.
/// </summary>
void SetKDeltaGpuData()
{
    /// <summary>
    /// Copy the GPU data structure to the constant memory symbol cData.
    /// </summary>
    cudaError_t status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));

    if (status != cudaSuccess)
    {
        /// <summary>
        /// Print an error message if the cudaMemcpyToSymbol operation fails.
        /// </summary>
        std::cerr << "cudaMemcpyToSymbol failed: " << cudaGetErrorString(status) << std::endl;
    }
}

/// <summary>
/// Copy data from GPU to CPU.
/// </summary>
void GetKDeltaGpuData()
{
    /// <summary>
    /// Copy data from the constant memory symbol cData to the CPU's GPU data structure.
    /// </summary>
    cudaError_t status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));

    if (status != cudaSuccess)
    {
        /// <summary>
        /// Print an error message if the cudaMemcpyFromSymbol operation fails.
        /// </summary>
        std::cerr << "cudaMemcpyFromSymbol failed: " << cudaGetErrorString(status) << std::endl;
    }
}

/// <summary>
/// Template kernel function for calculating the sigmoid output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    const int tileSize = 32;
    const int numTiles = (stride + tileSize - 1) / tileSize;

    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;

        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        __shared__ float shared_pUnit[tileSize];
        __shared__ float shared_pDelta[tileSize];

        float t;

        for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx)
        {
            int tileStart = tileIdx * tileSize;
            int tileEnd = min(tileStart + tileSize, stride);

#pragma unroll
            for (int i = tileStart + threadIdx.x; i < tileEnd; i += blockDim.x)
            {
                float a = pUnit[uOffset + i];
                t = static_cast<float>(pData[dOffset + i]) * scalingFactor;
                shared_pUnit[i - tileStart] = a;
            }

            __syncthreads();

#pragma unroll
            for (int i = tileStart + threadIdx.x; i < tileEnd; i += blockDim.x)
            {
                float delta = w * (shared_pUnit[i - tileStart] - t) * shared_pUnit[i - tileStart] * (1.0f - shared_pUnit[i - tileStart]);
                shared_pDelta[i - tileStart] = delta;
            }

            __syncthreads();

#pragma unroll
            for (int i = tileStart + threadIdx.x; i < tileEnd; i += blockDim.x)
            {
                pDelta[uOffset + i] = shared_pDelta[i - tileStart];
            }
        }
    }
}

/// <summary>
/// Template kernel function for calculating the hyperbolic tangent (tanh) output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0 / 256.0f : 1.0 / 128.0f;
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;

        // Calculate and update the delta value using the hyperbolic tangent (tanh) derivative.
        pDelta[uOffset + pos] = w * (a - t) * (1.0 - a * a);
    }
}

/// <summary>
/// Template kernel function for calculating the linear output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0 / 256.0f : 1.0 / 128.0f;
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;

        // Calculate and update the delta value using the linear activation function.
        pDelta[uOffset + pos] = w * (a - t);
    }
}

/// <summary>
/// Template kernel function for calculating the Rectified Linear Unit (ReLU) output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0 / 256.0f : 1.0 / 128.0f;
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;

        // Calculate and update the delta value using the ReLU derivative.
        pDelta[uOffset + pos] = w * (a - t) * (a > 0.0f);
    }
}

/// <summary>
/// Template kernel function for calculating the Leaky Rectified Linear Unit (LReLU) output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
/// <param name="slope">The slope for negative activations.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;

        // Calculate and update the delta value using the Leaky ReLU derivative with a specified slope.
        pDelta[uOffset + pos] = w * (a - t) * ((a > 0.0f) + (a <= 0.0f) * slope);
    }
}

/// <summary>
/// Template kernel function for calculating the Exponential Linear Unit (ELU) output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
/// <param name="alpha">The alpha value for negative activations.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;

        // Calculate and update the delta value using the Exponential Linear Unit (ELU) derivative with a specified alpha.
        pDelta[uOffset + pos] = w * (a - t) * ((a >= 0.0f) + (a < 0.0f) * (a + alpha));
    }
}

/// <summary>
/// Template kernel function for calculating the Scaled Exponential Linear Unit (SELU) output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
/// <param name="alpha">The alpha value for the SELU activation.</param>
/// <param name="lambda">The lambda value for the SELU activation.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;

        // Calculate and update the delta value using the Scaled Exponential Linear Unit (SELU) derivative.
        pDelta[uOffset + pos] = w * (a - t) * ((a >= 0.0f) * lambda + (a < 0.0f) * (lambda * alpha * expf(a)));
    }
}

/// <summary>
/// Template kernel function for calculating the Softmax output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;

        // Calculate and update the delta value for Softmax activation.
        pDelta[uOffset + pos] = w * (a - t);
    }
}

/// <summary>
/// Calculates the output delta using the specified activation function and updates pDelta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="activation">The activation function to be applied.</param>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data (output).</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
/// <param name="slope">The slope for Leaky Rectified Linear Unit (LReLU) activation.</param>
/// <param name="alpha">The alpha value for Exponential Linear Unit (ELU) activation.</param>
/// <param name="lambda">The lambda value for Scaled Exponential Linear Unit (SELU) activation.</param>
template<typename T> void kCalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);

    try {
        switch (activation)
        {
        case Sigmoid:
            kCalculateSigmoidOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateTanhOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Linear:
            kCalculateLinearOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateRELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateLRELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateSELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha, lambda);
            break;

        case SoftMax:
            kCalculateSoftMaxOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

/// <summary>
/// Template kernel function for calculating the Sigmoid output delta with indexed data.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pIndex">Pointer to the index data for data lookup.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;

        // Calculate and update the delta value for Sigmoid activation.
        pDelta[uOffset + pos] = w * (a - t) * a * (1.0f - a);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;
        pDelta[uOffset + pos] = w * (a - t) * (1.0f - a * a);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;
        pDelta[uOffset + pos] = w * (a - t);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;
        pDelta[uOffset + pos] = w * (a - t) * (a > 0.0f);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;
        pDelta[uOffset + pos] = w * (a - t) * ((a > 0.0f) + (a <= 0.0f) * slope);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;
        pDelta[uOffset + pos] = w * (a - t) * ((a >= 0.0f) + (a < 0.0f) * (a + alpha));
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;

        float selu_derivative = (a >= 0.0f) ? lambda : (lambda * alpha * exp(a));
        pDelta[uOffset + pos] = w * (a - t) * selu_derivative;
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;

        pDelta[uOffset + pos] = w * (a - t);
    }
}

template<typename T> void kCalculateIndexedOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);

    try {
        switch (activation)
        {
        case Sigmoid:
            kCalculateIndexedSigmoidOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateIndexedTanhOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Linear:
            kCalculateIndexedLinearOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateIndexedRELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateIndexedLRELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateIndexedELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateIndexedSELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha, lambda);
            break;

        case SoftMax:
            kCalculateIndexedSoftMaxOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;
        float diff = a - fabsf(t);

        // Modify the diff based on the sign of t
        if (t > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        pDelta[uOffset + pos] = w * diff * a * (1.0f - a);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;
        float diff = a - fabsf(t);

        // Modify the diff based on the sign of t
        if (t > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        pDelta[uOffset + pos] = w * diff * (1.0f - a * a);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]) * scalingFactor;
        float diff = a - fabsf(t);

        // Modify the diff based on the sign of t
        if (t > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        pDelta[uOffset + pos] = w * diff;
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]);

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(t * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        pDelta[uOffset + pos] = w * diff * (a > 0.0f);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]);

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(t * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        pDelta[uOffset + pos] = w * diff * ((a > 0.0f) + (a <= 0.0f) * slope);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]);

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(t * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) + (a < 0.0f) * (a + alpha));
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]);

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(t * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Compute the SELU-specific update
        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) * lambda + (a < 0.0f) * (lambda * alpha * expf(a)));
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]);

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(t * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Calculate the final delta
        pDelta[uOffset + pos] = w * diff;
    }
}

template<typename T> void kCalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);

    try {
        switch (activation)
        {
        case Sigmoid:
            kCalculateSigmoidL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateTanhL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Linear:
            kCalculateLinearL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateRELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateLRELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateSELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha, lambda);
            break;

        case SoftMax:
            kCalculateSoftMaxL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]);

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(t * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Calculate the final delta
        pDelta[uOffset + pos] = w * diff * a * (1.0f - a);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]);

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(t * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Calculate the final delta
        pDelta[uOffset + pos] = w * diff * (1.0f - a * a);
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]);

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(t * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Calculate the final delta
        pDelta[uOffset + pos] = w * diff;
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        float t = static_cast<float>(pData[dOffset + pos]);

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(t * scalingFactor);

        // Apply ReLU activation and calculate the final delta
        if (a > 0.0f)
            pDelta[uOffset + pos] = w * fminf(0.0f, diff);
        else
            pDelta[uOffset + pos] = 0.0f;
    }
} // TODO: Consolidate kernels

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float diff            = a - fabsf(t);
        diff                    = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float diff            = a - fabsf(t);
        diff                    = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float diff            = a - fabsf(t);
        diff                    = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float diff            = a - fabsf(t);
        diff                    = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff * ((a >= (float)0.0) + (a < (float)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float diff            = a - fabsf(t);
        diff                    = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff * ((a >= (float)0.0) + (a < (float)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float diff            = a - fabsf(t);
        diff                    = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff * ((a >= (float)0.0) + (a < (float)0.0) * (a + alpha));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float diff            = a - fabsf(t);
        diff                    = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff * ((a >= (float)0.0) * lambda + (a < (float)0.0) * (lambda * alpha * exp(a)));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float diff            = a - fabsf(t);
        diff                    = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff * ((a >= (float)0.0) * lambda + (a < (float)0.0) * (lambda * alpha * exp(a)));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float diff            = a - fabsf(t);
        diff                    = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff * ((a >= (float)0.0) * lambda + (a < (float)0.0) * (lambda * alpha * exp(a)));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float diff            = a - fabsf(t);
        diff                    = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff;      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float diff            = a - fabsf(t);
        diff                    = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff;      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float diff            = a - fabsf(t);
        diff                    = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);          
        pDelta[uOffset + pos]   = w * diff;      
    }
}

template<typename T> void kCalculateIndexedL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);

    try {
        switch (activation)
        {
        case Sigmoid:
            kCalculateIndexedSigmoidL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateIndexedTanhL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Linear:
            kCalculateIndexedLinearL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateIndexedRELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateIndexedLRELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateIndexedELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateIndexedSELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha, lambda);
            break;

        case SoftMax:
            kCalculateIndexedSoftMaxL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dpos               = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset            = dpos * stride;
    float w                   = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        float a               = pUnit[pos];
        float t               = pData[pos];
        pDelta[pos]             = w * ((a < (float)0.0) ? -t : (float)0.0);
        pos                    += blockDim.x;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dpos               = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset            = dpos * stride;
    float w                   = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        float a               = pUnit[pos];
        float t               = (float)pData[pos] * (float)(1.0 / 256.0);
        pDelta[pos]             = w * ((a < (float)0.0) ? -t : (float)0.0);
        pos                    += blockDim.x;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dpos               = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset            = dpos * stride;
    float w                   = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        float a               = pUnit[pos];
        float t               = (float)pData[pos] * (float)(1.0 / 128.0);
        pDelta[pos]             = w * ((a < (float)0.0) ? -t : (float)0.0);
        pos                    += blockDim.x;
    }
}


template<typename T> void kCalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    unsigned long threads = std::max(32UL, std::min(static_cast<unsigned long>(stride), static_cast<unsigned long>(getGpu()._threadsPerBlock)));

    try {
        kCalculateHingeOutputDelta_kernel << <batch, threads >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dpos               = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset            = dpos * stride;
    float w                   = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;    
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        float a               = pUnit[pos];
        float t               = pData[pos];
        pDelta[pos]             = w * ((a < (float)0.0) ? -t : (float)0.0);
        pos                    += blockDim.x;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dpos               = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset            = dpos * stride;
    float w                   = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0; 
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        float a               = pUnit[pos];
        float t               = (float)pData[pos] * (float)(1.0 / 256.0);
        pDelta[pos]             = w * ((a < (float)0.0) ? -t : (float)0.0);
        pos                    += blockDim.x;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dpos               = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset            = dpos * stride;
    float w                   = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0; 
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        float a               = pUnit[pos];
        float t               = (float)pData[pos] * (float)(1.0 / 128.0);
        pDelta[pos]             = w * ((a < (float)0.0) ? -t : (float)0.0);
        pos                    += blockDim.x;
    }
}


template<typename T> void kCalculateIndexedHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    unsigned long threads = std::max(32UL, std::min(static_cast<unsigned long>(stride), static_cast<unsigned long>(getGpu()._threadsPerBlock)));

    try {
        kCalculateIndexedHingeOutputDelta_kernel << <batch, threads >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * a * a * ((float)1.0 - a);      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawTanhOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];    
        pDelta[pos]             = w * a * ((float)1.0 - a * a);       
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLinearOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];    
        pDelta[pos]             = w * a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawRELUOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];         
        pDelta[pos]             = w * a * (a > (float)0.0);   
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLRELUOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float slope)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * a * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawELUOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float alpha)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * a * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSELUOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * a * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * (a - (float)1.0) * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSoftMaxOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];    
        pDelta[pos]             = w * a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = a - w;   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = {
            &position,
            &batch,
            &stride,
            &size,
            &pUnit,
            &pDelta,
            &pSparseStart,
            &pSparseEnd,
            &pSparseIndex,
            &pDataWeight,
            &slope,
            &alpha,
            &lambda
        };

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSigmoidOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroTanhOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroLinearOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroRELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroLRELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSoftMaxOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * (a - (float)1.0) * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = a - w;   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = {
            &position,
            &batch,
            &stride,
            &size,
            &pUnit,
            &pDelta,
            &pIndex,
            &pSparseStart,
            &pSparseEnd,
            &pSparseIndex,
            &pDataWeight,
            &slope,
            &alpha,
            &lambda
        };

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSigmoidOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroTanhOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroLinearOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroRELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroLRELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSoftMaxOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, slope);
            }
            kCalculateSparseAnalogNonZeroLRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, slope);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha);
            }
            kCalculateSparseAnalogNonZeroELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha, lambda);
            }
            kCalculateSparseAnalogNonZeroSELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha, lambda);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t *pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t);   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float *pDataWeight, unsigned char* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float *pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t); 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, slope);
            }
            kCalculateIndexedSparseAnalogNonZeroLRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, slope);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha);
            }
            kCalculateIndexedSparseAnalogNonZeroELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha, lambda);
            }
            kCalculateIndexedSparseAnalogNonZeroSELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha, lambda);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);
        pDelta[pos]             = w * a * a * ((float)1.0 - a);      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawTanhL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);    
        pDelta[pos]             = w * a * ((float)1.0 - a * a);       
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);                
            pDelta[pos2]        = w * diff * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLinearL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);    
        pDelta[pos]             = w * a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff;   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawRELUL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);         
        pDelta[pos]             = w * a * (a > (float)0.0);   
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLRELUL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float slope)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);
        pDelta[pos]             = w * a * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawELUL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float alpha)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        float diff            = min((float)0.0, a - (float)1.0);         
        pDelta[pos]             = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSELUL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);
        pDelta[pos]             = w * a * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));            
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSoftMaxL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);    
        pDelta[pos]             = w * a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - w);             
            pDelta[pos2]        = diff;   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = {
            &position,
            &batch,
            &stride,
            &size,
            &pUnit,
            &pDelta,
            &pSparseStart,
            &pSparseEnd,
            &pSparseIndex,
            &pDataWeight,
            &slope,
            &alpha,
            &lambda
        };

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSigmoidL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroTanhL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroLinearL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroRELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroLRELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSoftMaxL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff;   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - w);             
            pDelta[pos2]        = diff;   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);

    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, &slope, &alpha, &lambda };

        dim3 blockDim(getGpu()._threadsPerBlock);


        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSigmoidL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroTanhL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroLinearL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroRELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroLRELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSoftMaxL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);        
            pDelta[pos2]        = w * diff * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff;   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff;   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff;   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);  
            pDelta[pos2]        = w * diff * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff; 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff; 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff; 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);

    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, slope);
            }
            kCalculateSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, slope);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha);
            }
            kCalculateSparseAnalogNonZeroELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha, lambda);
            }
            kCalculateSparseAnalogNonZeroSELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha, lambda);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t *pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff;   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff;   
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff;   
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float *pDataWeight, unsigned char* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff; 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff; 
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float *pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);              
            pDelta[pos2]        = w * diff; 
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, slope);
            }
            kCalculateIndexedSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, slope);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha);
            }
            kCalculateIndexedSparseAnalogNonZeroELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha, lambda);
            }
            kCalculateIndexedSparseAnalogNonZeroSELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha, lambda);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * (a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * (a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * (a - t);      
    }
}

template<typename T> void kCalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
        case SoftMax:
            kCalculateSigmoidCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * (a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * (a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * (a - t);      
    }
}

template<typename T> 
void kCalculateIndexedCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
        case SoftMax:
            kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * a;       
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };

        dim3 blockDim(getGpu()._threadsPerBlock);

        switch (activation)
        {
        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
                break;
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSoftMaxOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
                break;
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };

        dim3 blockDim(getGpu()._threadsPerBlock);

        switch (activation)
        {
        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
                break;
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSoftMaxOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
                break;
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateSparseAnalogCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case SoftMax:
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case SoftMax:
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float output          = (float)0.0;
        if ((t == (float)1.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t); 
        pDelta[uOffset + pos]   = w * output;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float output          = (float)0.0;
        if ((t == (float)1.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t); 
        pDelta[uOffset + pos]   = w * output;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;
    }
}

template<typename T> void kCalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
            kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case SoftMax:
            kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float output          = (float)0.0;
        if ((t == (float)1.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t); 
        pDelta[uOffset + pos]   = w * output;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float output          = (float)0.0;
        if ((t == (float)1.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t); 
        pDelta[uOffset + pos]   = w * output;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;
    }
}

template<typename T> void kCalculateIndexedScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
            kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case SoftMax:
            kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._SMCE_zeroScale;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        float output          = (float)0.0;
        if (a > cData._SMCE_zeroTarget)
            output              = w * a;
        pDelta[pos]             = output;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float output      = (float)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = w * (a - (float)1.0);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float a               = pUnit[pos];
        float output          = (float)0.0;
        if (a > cData._SMCE_zeroTarget)
            output              = cData._SMCE_zeroScale * a;
        pDelta[pos]             = output;   
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1));
        uint64_t offset         = pos * stride;
        pos1                   += threadIdx.x & cData._warpMask;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float output      = (float)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = (a - w);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };

        dim3 blockDim(getGpu()._threadsPerBlock);

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float output      = (float)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = w * (a - (float)1.0);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1);
        uint64_t offset         = pos * stride;
        pos1                   += threadIdx.x & cData._warpMask;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float output      = (float)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = cData._SMCE_oneScale * (a - w);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };

        dim3 blockDim(getGpu()._threadsPerBlock);

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel(uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
          float a               = pUnit[pos];
          float output          = (float)0.0;
          if (a > cData._SMCE_zeroTarget)
          {
              output              = cData._SMCE_zeroScale * a;
          }
          pDelta[pos]             = output;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
              uint64_t pos2       = offset + pSparseIndex[pos1];
              float a           = pUnit[pos2];
              T t                 = pSparseData[pos1];
              float output      = (float)0.0;
              if (a < cData._SMCE_oneTarget)
              {
                  output          = cData._SMCE_oneScale * t * (a - (float)1.0);
              }
              pDelta[pos2]        = output;
              pos1               += cData._warpSize;
        }
    }
}

template<typename T>
void kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            }
            kCalculateSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            break;

        case SoftMax:
            std::cout << "unsupported activation for this cost function" << std::endl;
            getGpu().Shutdown();
            exit(-1);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
              uint64_t pos2       = offset + pSparseIndex[pos1];
              float a           = pUnit[pos2];
              T t                 = pSparseData[pos1];
              float output      = (float)0.0;
              if (a < cData._SMCE_oneTarget)
              {
                  output          = cData._SMCE_oneScale * t * (a - (float)1.0);
              }
              pDelta[pos2]        = output;
              pos1               += cData._warpSize;
        }
    }
}

template<typename T>
void kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            }
            kCalculateIndexedSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            break;

        case SoftMax:
            std::cout << "unsupported activation for this cost function" << std::endl;
            getGpu().Shutdown();
            exit(-1);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((float)1.0 - a * a);      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t);  
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a- t) * ((float)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t); 
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);   
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((float)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);  
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<typename T> void kCalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
            kCalculateSigmoidL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateTanhL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Linear:
            kCalculateLinearL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateLRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateSELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha, lambda);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((float)1.0 - a * a);      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t);  
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a- t) * ((float)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t); 
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);   
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((float)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);  
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<typename T> void kCalculateIndexedL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
            kCalculateIndexedSigmoidL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateIndexedTanhL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Linear:
            kCalculateIndexedLinearL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateIndexedRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateIndexedLRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateIndexedELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateIndexedSELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha, lambda);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w *  sgn(a) * a * ((float)1.0 - a);      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0) * a * ((float)1.0 - a);      
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawTanhL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];    
        pDelta[pos]             = w * sgn(a) * ((float)1.0 - a * a);          
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((float)1.0 - a * a);     
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLinearL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];  
        pDelta[pos]             = w * sgn(a);
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0);    
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawRELUL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * (a > (float)0.0);          
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (float)1.0) * (a > (float)0.0);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawELUL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta, float alpha)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * sgn(a) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));    
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a > (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSELUL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * sgn(a) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));  
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLRELUL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta, float slope)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * sgn(a) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRawLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];  
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, &slope, &alpha, &lambda };

        dim3 blockDim(getGpu()._threadsPerBlock);

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSigmoidL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroTanhL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroLinearL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroRELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroRawLRELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0) * a * ((float)1.0 - a);      
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((float)1.0 - a * a);     
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0);    
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (float)1.0) * (a > (float)0.0);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a > (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));  
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroRawLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];  
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        dim3 blockDim(getGpu()._threadsPerBlock);

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, &slope, &alpha, &lambda };

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSigmoidL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroTanhL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroLinearL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroRELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroRawLRELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparsenessPenalty_kernel(uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float p, float beta)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos < stride)
    {
        float pi              = (float)0.0;
        for (int i = 0; i < batch; i++)
        {
            pi                 += pUnit[pos];
            pos                += stride;
        }

        pi                     /= (float)batch;
        pi                      = max(MIN_ACTIVATION, min(MAX_ACTIVATION, pi));
        float penalty         = beta * (-p / pi + ((float)1.0 - p) / ((float)1.0 - pi));
        
        pos                     = blockIdx.x * blockDim.x + threadIdx.x;
        for (int i = 0; i < batch; i++)
        {
            pDelta[pos]        += penalty;
            pos                += stride;
        }
    }
}


void kCalculateSparsenessPenalty(uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float p, float beta)
{
    try {
        dim3 grid1(CalculateBlocks(stride));

        void* args[] = {
            &batch,
            &stride,
            &pUnit,
            &pDelta,
            &p,
            &beta
        };

        cudaLaunchKernel((void*)kCalculateSparsenessPenalty_kernel, grid1, blockDim, args, 0, nullptr);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];
        float d               = pDelta[pos];
        pDelta[pos]             = x * ((float)1.0 - x) * d;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateTanhHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta, float scale, float oneOverScale)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];
        float d               = pDelta[pos];
        x                      *= oneOverScale;
        pDelta[pos]             = scale * ((float)1.0 - x * x) * d;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateRELUHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];
        if (x <= (float)0.0)
            pDelta[pos]         = (float)0.0;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta, float slope)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];
        if (x <= (float)0.0)
        {
            pDelta[pos]         *= slope;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateELUHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta, float alpha)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];
        if (x <= (float)0.0)
            pDelta[pos]        *= (x + alpha);            
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSELUHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta, float alpha, float lambda)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];        
        float delta           = pDelta[pos];
        if (x > (float)0.0)
        {
            delta              *= lambda;
        }
        else
        {
            delta              *= (x + lambda * alpha);
        }
        pDelta[pos]             = delta;
    }
}

void kCalculateHadamardProduct(Activation activation, uint64_t size, float scale, float* pUnit, float* pDelta, float slope, float alpha, float lambda)
{
    try {
        uint32_t blocks = CalculateBlocks(size);
        float oneOverScale = 1.0f / scale;

        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &size, pUnit, pDelta, &oneOverScale, &slope, &alpha, &lambda };

        switch (activation)
        {
        case Sigmoid:
            cudaLaunchKernel((void*)kCalculateSigmoidHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            cudaLaunchKernel((void*)kCalculateTanhHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;

        case Linear:
            break;

        case RectifiedLinear:
            cudaLaunchKernel((void*)kCalculateRELUHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            cudaLaunchKernel((void*)kCalculateLRELUHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            cudaLaunchKernel((void*)kCalculateELUHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            cudaLaunchKernel((void*)kCalculateSELUHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kNormalizeDeltas_kernel(float norm, uint32_t batch, uint32_t stride, float* pDelta)
{
    uint32_t dpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;
    pDelta += dpos * stride;

    if (dpos < batch)
    {
        uint32_t pos = tgx;
        float r2 = 0.0f;

        while (pos < stride)
        {
            float x = pDelta[pos];
            r2 += x * x;
            pos += cData._warpSize;
        }

        for (int offset = cData._warpSize / 2; offset > 0; offset /= 2)
        {
            r2 += __shfl_down_sync(0xffffffff, r2, offset);
        }

        if (tgx == 0)
        {
            r2 = sqrtf(r2);
            if (r2 > norm)
            {
                norm = 1.0f / r2;
                pos = tgx;

                while (pos < stride)
                {
                    pDelta[pos] *= norm;
                    pos += cData._warpSize;
                }
            }
        }
    }
}

void kNormalizeDeltas(float norm, uint32_t batch, uint32_t stride, float* pDelta)
{
    try {
        uint32_t blocks = (batch + 3) / 4;
        unsigned long threadsPerBlock = 128;
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &norm, &batch, &stride, &pDelta };
        cudaLaunchKernel((void*)kNormalizeDeltas_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateDeltaMagnitudes_kernel(uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude)
{
    uint32_t dpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;
    pDelta += dpos * stride;

    if (dpos < batch)
    {
        uint32_t pos = tgx;
        float r2 = 0.0f;

        while (pos < stride)
        {
            float x = pDelta[pos];
            r2 += x * x;
            pos += cData._warpSize;
        }

        for (int offset = cData._warpSize / 2; offset > 0; offset /= 2)
        {
            r2 += __shfl_down_sync(0xffffffff, r2, offset);
        }

        if (tgx == 0)
            pMagnitude[dpos] = sqrtf(r2);
    }
}

void kCalculateDeltaMagnitudes(uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude)
{
    try {
        uint32_t blocks = (batch + 3) / 4;
        unsigned long threadsPerBlock = 128;
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &batch, &stride, &pDelta, &pMagnitude };
        cudaLaunchKernel((void*)kCalculateDeltaMagnitudes_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kNormalizeDeltaMagnitudes_kernel(float norm, uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude)
{
    uint32_t dpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    pDelta                                 += dpos * stride;
    if (dpos < batch)
    {    
        float r2                          = pMagnitude[dpos];
        if (r2 > norm * norm)
        {
            norm                           *= rsqrt(r2);
            uint32_t pos                    = tgx;
            while (pos < stride)
            {
                pDelta[pos]                *= norm;
                pos                        += cData._warpSize;
            }
        }    
        
   }  
}

void kNormalizeDeltaMagnitudes(float norm, uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude)
{
    try {
        uint32_t blocks = (batch + 3) / 4;
        unsigned long threadsPerBlock = 128;
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &norm, &batch, &stride, &pDelta, &pMagnitude };
        cudaLaunchKernel((void*)kNormalizeDeltaMagnitudes_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();  // Wait for the kernel to finish
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateMaxoutDelta_kernel(float* pSrc, float* pSrcDelta, size_t size, float beta, float* pDst, float* pDstDelta)
{
    uint64_t pos                        = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float s = pSrc[pos];
        float sdelta = pSrcDelta[pos];
        float d = pDst[pos];
        float delta                   = (s == d) ? sdelta : (float)0;
        
        if (beta == (float)0)
            pDstDelta[pos]              = delta;
        else if (delta != (float)0.0)
            pDstDelta[pos]              = beta * pDstDelta[pos] + delta;
    }
}


void kCalculateMaxoutDelta(float* pSrc, float* pSrcDelta, size_t size, float beta, float* pDst, float* pDstDelta)
{
    try {
        unsigned long blocks = CalculateBlocks(size);
        unsigned long threadsPerBlock = getGpu()._threadsPerBlock;
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &pSrc, &pSrcDelta, &size, &beta, &pDst, &pDstDelta };
        cudaLaunchKernel((void*)kCalculateMaxoutDelta_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateCosineDelta_kernel(float* pDPDelta, float* pDP, float* pA, float* pB, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride)
{
    p0Vector               += blockIdx.x * inputStride + threadIdx.x;
    pVector                += blockIdx.x * inputStride + threadIdx.x;
    pDPDelta               += blockIdx.x * stride;
    pDP                    += blockIdx.x * stride;
    pA                     += blockIdx.x * stride;
    pB                     += blockIdx.x * stride;    
    pDelta0                += blockIdx.x * inputStride + threadIdx.x;
    pDelta                 += blockIdx.x * inputStride + threadIdx.x;    
    uint32_t pos            = threadIdx.x;
    float dp              = *pDP;
    float dpDelta         = *pDPDelta;
    float a               = *pA;
    float b               = *pB;
    float ab              = a * b;
    float a2              = a * a;
    float b2              = b * b;
    
    while (pos < inputStride)
    {
        float ai          = *p0Vector;
        float bi          = *pVector;

        float delta0      = dpDelta * ((bi / ab) - (ai * dp / a2));
        float delta       = dpDelta * ((ai / ab) - (bi * dp / b2));
        if (beta0 == (float)0)
            *pDelta0        = delta0;
        else
            *pDelta0        = *pDelta0 + beta0 * delta0;
        if (beta == (float)0)
            *pDelta         = delta;
        else
            *pDelta         = *pDelta + beta * delta;        
    
        pDelta0            += blockDim.x;
        pDelta             += blockDim.x;     
        p0Vector           += blockDim.x;
        pVector            += blockDim.x;
        pos                += blockDim.x;
    }
}

void kCalculateCosineDelta(float* pDPDeltaIn, float* pDPIn, float* pA, float* pB, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride)
{
    try {
        unsigned long blocks = batch;
        unsigned long threadsPerBlock = std::min(stride, getGpu()._threadsPerBlock);
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &pDPDeltaIn, &pDPIn, &pA, &pB, &p0Vector, &pVector, &batch, &stride, &pDelta0, &beta0, &pDelta, &beta, &inputStride };
        cudaLaunchKernel((void*)kCalculateCosineDelta_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();  // Wait for the kernel to finish
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateDotProductDelta_kernel(float* pDPDelta, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride)
{
    p0Vector               += blockIdx.x * inputStride + threadIdx.x;
    pVector                += blockIdx.x * inputStride + threadIdx.x;
    pDPDelta               += blockIdx.x * stride; 
    pDelta0                += blockIdx.x * inputStride + threadIdx.x;
    pDelta                 += blockIdx.x * inputStride + threadIdx.x;    
    uint32_t pos            = threadIdx.x;
    float dpDelta         = *pDPDelta;
    
    while (pos < inputStride)
    {
        float ai          = *p0Vector;
        float bi          = *pVector;
        float delta0      = dpDelta * bi;
        float delta       = dpDelta * ai;
        if (beta0 == (float)0)
            *pDelta0        = delta0;
        else
            *pDelta0        = *pDelta0 + beta0 * delta0;
        if (beta == (float)0)
            *pDelta         = delta;
        else
            *pDelta         = *pDelta + beta * delta;        
    
        pDelta0            += blockDim.x;
        pDelta             += blockDim.x;     
        p0Vector           += blockDim.x;
        pVector            += blockDim.x;
        pos                += blockDim.x;
    }
}

void kCalculateDotProductDelta(float* pDPDelta, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride)
{
    try {
        unsigned long blocks = batch;
        unsigned long threadsPerBlock = std::min(stride, getGpu()._threadsPerBlock);
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &pDPDelta, &p0Vector, &pVector, &batch, &stride, &pDelta0, &beta0, &pDelta, &beta, &inputStride };
        cudaLaunchKernel((void*)kCalculateDotProductDelta_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}


#define EXPLICITLY_INSTANTIATE_KERNELS(T)\
template void kCalculateL1OutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*, float, float, float);\
template void kCalculateIndexedL1OutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*, float, float, float);\
template void kCalculateL2HingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*, float, float, float);\
template void kCalculateIndexedL2HingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*, float, float, float);\
template void kCalculateCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*);\
template void kCalculateIndexedCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*);\
template void kCalculateScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*);\
template void kCalculateIndexedScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*);\
template void kCalculateOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*, float, float, float);\
template void kCalculateIndexedOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*, float, float, float);\
template void kCalculateHingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*);\
template void kCalculateIndexedHingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*);\
template void kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint64_t*, uint64_t*, uint32_t*, T*, bool);\
template void kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, T*, bool);\
template void kCalculateSparseAnalogOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*,  float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, bool, float, float, float);\
template void kCalculateIndexedSparseAnalogOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*,  float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, bool, float, float, float);\
template void kCalculateSparseAnalogL2HingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*,  float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, bool, float, float, float);\
template void kCalculateIndexedSparseAnalogL2HingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*,  float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, bool, float, float, float);\

EXPLICITLY_INSTANTIATE_KERNELS(float)
EXPLICITLY_INSTANTIATE_KERNELS(double)
EXPLICITLY_INSTANTIATE_KERNELS(unsigned char)
EXPLICITLY_INSTANTIATE_KERNELS(char)
EXPLICITLY_INSTANTIATE_KERNELS(uint32_t)
EXPLICITLY_INSTANTIATE_KERNELS(uint64_t)
EXPLICITLY_INSTANTIATE_KERNELS(int32_t)
EXPLICITLY_INSTANTIATE_KERNELS(int64_t)

#endif