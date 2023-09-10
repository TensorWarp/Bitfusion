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
/// Sets the GPU data in the constant memory.
/// </summary>
void SetKLossGpuData()
{
    cudaError_t status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));
    if (status != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyToSymbol: SetKLossGpuData copy to cData failed - " + std::string(cudaGetErrorString(status)));
    }
}

/// <summary>
/// Retrieves the GPU data from the constant memory.
/// </summary>
void GetKLossGpuData()
{
    cudaError_t status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));
    if (status != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyFromSymbol: GetKLossGpuData copy from cData failed - " + std::string(cudaGetErrorString(status)));
    }
}

/// <summary>
/// CUDA kernel to calculate sparse raw L1 error.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="size">The size parameter.</param>
__global__ void LAUNCH_BOUNDS() kCalculateSparseRawL1Error_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint64_t stride, uint64_t size)
{
    uint64_t pos = blockDim.x * blockIdx.x + threadIdx.x;
    float error = (float)0.0;
    if (pos < size)
    {
        float w = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }

        float a = pUnit[pos];
        error = w * fabsf(a);
    }
}

/// <summary>
/// CUDA kernel to calculate sparse L1 error for non-zero elements.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pSparseStart">Pointer to the sparse data start indices.</param>
/// <param name="pSparseEnd">Pointer to the sparse data end indices.</param>
/// <param name="pSparseIndex">Pointer to the sparse data indices.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
__global__ void LAUNCH_BOUNDS() kCalculateSparseNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            error += w * (fabsf(a - (float)1.0) - fabsf(a));
            pos1 += cData._warpSize;
        }
    }
}

/// <summary>
/// CUDA kernel to calculate sparse L1 error for non-zero elements.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pSparseStart">Pointer to the sparse data start indices.</param>
/// <param name="pSparseEnd">Pointer to the sparse data end indices.</param>
/// <param name="pSparseIndex">Pointer to the sparse data indices.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
__global__ void LAUNCH_BOUNDS() kCalculateSparseOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            error += w * fabsf(a - (float)1.0);
            pos1 += cData._warpSize;
        }
    }
}

/// <summary>
/// Function to calculate sparse L1 error with optional zero element exclusion.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pSparseStart">Pointer to the sparse data start indices.</param>
/// <param name="pSparseEnd">Pointer to the sparse data end indices.</param>
/// <param name="pSparseIndex">Pointer to the sparse data indices.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="bSparseIgnoreZero">Flag to ignore zero elements.</param>
/// <returns>The calculated sparse L1 error.</returns>
float kCalculateSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseOnlyNonZeroL1Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL1Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseNonZeroL1Error_kernel, gridDim, blockDim, args2, 0, nullptr);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            error              += w * (fabsf(a - (float)1.0) - fabsf(a));   
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            error              += w * fabsf(a - (float)1.0);   
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateIndexedSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseOnlyNonZeroL1Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL1Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroL1Error_kernel, gridDim, blockDim, args2, 0, nullptr);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            T t                 = pSparseData[pos1];
            error              += w * fabsf(a - t);   
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            T t                 = pSparseData[pos1];
            error              += w * (fabsf(a - t) - fabsf(a));   
            pos1               += cData._warpSize;
        }
    }  

}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * fabsf(a - t);   
            pos1               += cData._warpSize;
        }
    }  

}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * (fabsf(a - t) - fabsf(a));   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * fabsf(a - t);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * (fabsf(a - t) - fabsf(a));   
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
float kCalculateSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogOnlyNonZeroL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogNonZeroL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseAnalogL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            T t                 = pSparseData[pos1];
            error              += w * fabsf(a - t);   
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            T t                 = pSparseData[pos1];
            error              += w * (fabsf(a - t) - fabsf(a));   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * fabsf(a - t);   
            pos1               += cData._warpSize;
        }
    }  

}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * (fabsf(a - t) - fabsf(a));   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * fabsf(a - t);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * (fabsf(a - t) - fabsf(a));   
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
float kCalculateIndexedSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogNonZeroL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseAnalogL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawL2Error_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    float error               = (float)0.0;
    if (pos < size)
    {
        float w               = (float)0.5;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        error                   = w * a * a;     
    }
    
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += w * ((a - (float)1.0) * (a - (float)1.0));   
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += w * ((a - (float)1.0) * (a - (float)1.0) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}


float kCalculateSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseOnlyNonZeroL2Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL2Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseNonZeroL2Error_kernel, gridDim, blockDim, args2, 0, nullptr);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseL2Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            error              += w * ((a - t) * (a - t));   
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            error              += w * ((a - t) * (a - t) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * ((a - t) * (a - t));   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * ((a - t) * (a - t) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * ((a - t) * (a - t));   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * ((a - t) * (a - t) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}


template<typename T>
float kCalculateSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogOnlyNonZeroL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = batch * stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogNonZeroL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseAnalogL2Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += w * ((a - (float)1.0) * (a - (float)1.0));   
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += w * ((a - (float)1.0) * (a - (float)1.0) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}


#include <stdexcept>

float kCalculateIndexedSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseOnlyNonZeroL2Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL2Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroL2Error_kernel, gridDim, blockDim, args2, 0, nullptr);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseL2Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            error              += w * ((a - t) * (a - t));   
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            error              += w * ((a - t) * (a - t) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * ((a - t) * (a - t));   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * ((a - t) * (a - t) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * ((a - t) * (a - t));   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * ((a - t) * (a - t) - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
float kCalculateIndexedSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogOnlyNonZeroL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = batch * stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogNonZeroL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseAnalogL2Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawL2HingeError_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    float error               = (float)0.0;
    if (pos < size)
    {
        float w               = (float)0.5;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);
        error                   = w * a * a;
    }
    
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);
            a                   = max((float)0.0, a);
            error              += w * (diff * diff - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseOnlyNonZeroL2HingeError_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL2HingeError_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseNonZeroL2HingeError_kernel, gridDim, blockDim, args2, 0, nullptr);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseL2HingeError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            a                   = max((float)0.0, a);
            diff                = (t > (T)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);          
            error              += w * (diff * diff - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - t;
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);             
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - t;
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);   
            a                   = max((float)0.0, a);  
            error              += w * (diff * diff - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf((float)t);
            diff                = (t > (float)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);           
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            a                   = max((float)0.0, a);
            diff                = (t > (float)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);       
            error              += w * (diff * diff - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}


#include <stdexcept>

template<typename T>
float kCalculateSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogOnlyNonZeroL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = batch * stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogNonZeroL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseAnalogL2HingeError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float diff        = min((float)0.0, pUnit[pos2] - (float)1.0);
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);
            a                   = max((float)0.0, a);
            error              += w * (diff * diff - a * a); 
            pos1               += cData._warpSize;
        }
    }  

}


float kCalculateIndexedSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseOnlyNonZeroL2HingeError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL2HingeError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroL2HingeError_kernel, gridDim, blockDim, args2, 0, nullptr);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseL2HingeError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);         
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            a                   = max((float)0.0, a);
            diff                = (t > (T)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);          
            error              += w * (diff * diff - a * a);               
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - t;
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff); 
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - t;
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);    
            a                   = max((float)0.0, a);  
            error              += w * (diff * diff - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);      
            error              += w * diff * diff;  
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            a                   = max((float)0.0, a);
            diff                = (t > (float)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);          
            error              += w * (diff * diff - a * a);  
            pos1               += cData._warpSize;
        }
    }  

}


template<typename T>
float kCalculateIndexedSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = batch * stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseAnalogL2HingeError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawCrossEntropyError_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    float error               = (float)0.0;
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
        error                   = -w * log(max(MIN_ERROR, (float)1.0 - a));     
    }

}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            error              += -w * log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  


}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            error              += w * (-log(max(MIN_ERROR, a)) + log(max(MIN_ERROR, (float)1.0 - a)));   
            pos1               += cData._warpSize;
        }
    }  

}

#include <stdexcept>

float kCalculateSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseOnlyNonZeroCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseNonZeroCrossEntropyError_kernel, gridDim, blockDim, args2, 0, nullptr);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseOnlyNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            error              += -w * log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            error              += w * (-log(max(MIN_ERROR, a)) + log(max(MIN_ERROR, (float)1.0 - a)));   
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateIndexedSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseOnlyNonZeroCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroCrossEntropyError_kernel, gridDim, blockDim, args2, 0, nullptr);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight == NULL) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos];
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += -w * log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
        cudaLaunchKernel((void*)kCalculateSparseMultinomialCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseMultinomialCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight == NULL) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos];
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += -w * log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateIndexedSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
        cudaLaunchKernel((void*)kCalculateIndexedSparseMultinomialCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseMultinomialCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            T t                 = pSparseData[pos1];
            error              += w * (-t * log(max(MIN_ERROR, a)));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * (-t * log(max(MIN_ERROR, a)));   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * (-t * log(max(MIN_ERROR, a)));   
            pos1               += cData._warpSize;
        }
    }  

}


template<typename T>
float kCalculateSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseAnalogMultinomialCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseAnalogMultinomialCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            T t                 = pSparseData[pos1];
            error              += w * (-t * log(max(MIN_ERROR, a)));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * (-t * log(max(MIN_ERROR, a)));   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * (-t * log(max(MIN_ERROR, a)));   
            pos1               += cData._warpSize;
        }
    }  

}


template<typename T>
float kCalculateIndexedSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseAnalogMultinomialCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseAnalogMultinomialCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawScaledMarginalCrossEntropyError_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    float error               = (float)0.0;
    if (pos < size)
    {
        float w               = cData._SMCE_zeroScale;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = pos / stride;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        if (a > cData._SMCE_zeroTarget)
            error               = -w * log(max(MIN_ERROR, (float)1.0 - a));     
    }
    
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            if (a < cData._SMCE_oneTarget)
                error          += -w * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            if (a > cData._SMCE_zeroTarget)
            {
                error          += w * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
            }
            if (a < cData._SMCE_oneTarget)
            {
                error          += -w * cData._SMCE_oneScale * log(max(MIN_ERROR, a));
            }
            pos1               += cData._warpSize;
        }


    }  

}

float kCalculateSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args2, 0, nullptr);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            if (a < cData._SMCE_oneTarget)
                error          += -w * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            if (a > cData._SMCE_zeroTarget)
            {
                error          += w * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
            }
            if (a < cData._SMCE_oneTarget)
            {
                error          += -w * cData._SMCE_oneScale * log(max(MIN_ERROR, a));
            }
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateIndexedSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args2, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus2 = cudaGetLastError();
            if (cudaStatus2 != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus2)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawDataScaledMarginalCrossEntropyError_kernel(float* pUnit, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    float error               = (float)0.0;
    if (pos < size)
    {
          float a               = pUnit[pos];
          if (a > cData._SMCE_zeroTarget)
          {
              error               = -cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
          }
    }

}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroDataScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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

              if (a > cData._SMCE_zeroTarget)
              {
                  error          += cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
              }

              if (a < cData._SMCE_oneTarget)
              {
                  error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
              }
              pos1               += cData._warpSize;
        }
    }

}

template<typename T>
float kCalculateSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (!bSparseIgnoreZero)
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawDataScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (pUnit, size);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseNonZeroDataScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus2 = cudaGetLastError();
        if (cudaStatus2 != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus2)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseDataScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}



template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroDataScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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

              if (a > cData._SMCE_zeroTarget)
              {
                  error          += cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
              }

              if (a < cData._SMCE_oneTarget)
              {
                  error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
              }
              pos1               += cData._warpSize;
        }
    }

}

template<typename T>
float kCalculateIndexedSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (!bSparseIgnoreZero)
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawDataScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (pUnit, size);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseNonZeroDataScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus2 = cudaGetLastError();
        if (cudaStatus2 != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus2)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseDataScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight == NULL) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos]);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            if (a < cData._SMCE_oneTarget)
                error          += -w * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
        cudaLaunchKernel((void*)kCalculateSparseNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseMultinomialScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight == NULL) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos]);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            if (a < cData._SMCE_oneTarget)
                error          += -w * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
        cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}




template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            T t                 = pSparseData[pos1];  
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = pSparseData[pos1] * (float)(1.0 / 256.0);
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = pSparseData[pos1] * (float)(1.0 / 128.0);
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
float kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            T t                 = pSparseData[pos1];  
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = pSparseData[pos1] * (float)(1.0 / 256.0);
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
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
            float t           = pSparseData[pos1] * (float)(1.0 / 128.0);
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
float kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error               = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;    
        float a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        error                   = w * fabsf(a - t);        
    }

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error               = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;        
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error                   = w * fabsf(a - t);        
    }

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error               = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;            
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error                   = w * fabsf(a - t);        
    }

}

template<typename T>
float kCalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateL1Error_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error               = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        error                   = w * fabsf(a - t);        
    }

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error               = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error                   = w * fabsf(a - t);        
    }

}

/// <summary>
/// CUDA kernel to calculate the L1 error for indexed data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * fabsf(a - t);
    }
}

/// <summary>
/// Calculate the L1 error for indexed data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated L1 error.</returns>
template<typename T>
float kCalculateIndexedL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedL1Error_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the L2 error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 error for data with unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array with unsigned char type.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 error for data with unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array with unsigned char type.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 error for data with unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array with unsigned char type.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated indexed L2 error.</returns>
template<typename T>
float kCalculateIndexedL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedL2Error_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedL2Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the L2 hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        float diff = a - fabsf(t);
        diff = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error += w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 hinge error for data with an unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (unsigned char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float diff = a - t;
        diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error = w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 hinge error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float diff = a - fabsf(t);
        diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error += w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateL2HingeError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateL2HingeError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        float diff = a - fabsf(t);
        diff = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error += w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 hinge error for data with an unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array (unsigned char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float diff = a - t;
        diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error = w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 hinge error for data with an unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array (unsigned char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float diff = a - fabsf(t);
        diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error += w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateIndexedL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedL2HingeError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedL2HingeError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = pData[pos];
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }
}

/// <summary>
/// CUDA kernel to calculate the hinge error for data with an unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (unsigned char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = pData[pos] * (float)(1.0 / 128.0);
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }
}

/// <summary>
/// CUDA kernel to calculate the hinge error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{

    pUnit += blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = (float)pData[pos] * (float)(1.0 / 256.0);
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }

}

/// <summary>
/// Calculate hinge error using CUDA for a given batch of data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated hinge error.</returns>
template<typename T>
float kCalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        unsigned long threads = max(32, min(stride, 128));
        kCalculateHingeError_kernel << <batch, threads >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateHingeError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the indexed hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = pData[pos];
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed hinge error for data with an unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data array.</param>
/// <param name="pData">Pointer to the data array (unsigned char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = pData[pos] * (float)(1.0 / 256.0);
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed hinge error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data array.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = (float)pData[pos] * (float)(1.0 / 128.0);
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }
}

/// <summary>
/// CUDA function to calculate the indexed hinge error for a batch of data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated indexed hinge error.</returns>
template<typename T>
float kCalculateIndexedHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        unsigned long threads = max(32, min(stride, 128));
        kCalculateIndexedHingeError_kernel << <batch, threads >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedHingeError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the cross-entropy error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA function to calculate the cross-entropy error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated cross-entropy error.</returns>
template<typename T>
float kCalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the indexed cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed cross-entropy error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// Calculate the indexed cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated indexed cross-entropy error.</returns>
template<typename T>
float kCalculateIndexedCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the multinomial cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// Calculate the multinomial cross-entropy error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array with char data type.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the multinomial cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// Calculate the multinomial cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error.</returns>
template<typename T>
float kCalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateMultinomialCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateMultinomialCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the multinomial cross-entropy error for indexed data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the multinomial cross-entropy error for indexed data with a specific data type (char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the multinomial cross-entropy error for indexed data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// Function to calculate the multinomial cross-entropy error for indexed data.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateIndexedMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedMultinomialCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedMultinomialCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        if (((t == (T)1.0) && (a < cData._SMCE_oneTarget)) ||
            ((t == (T)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for data with a specific data type (char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        if (((t == (float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (float)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for data with a specific data type (unsigned char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        if (((t == (float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (float)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// Function to calculate the scaled marginal cross-entropy error for data.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateScaledMarginalCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateScaledMarginalCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        if (((t == (T)1.0) && (a < cData._SMCE_oneTarget)) ||
            ((t == (T)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed data with a specific data type (char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        if (((t == (float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (float)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed data with a specific data type (unsigned char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        if (((t == (float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (float)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// Function to calculate the scaled marginal cross-entropy error for indexed data.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateIndexedScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedScaledMarginalCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedScaledMarginalCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for multinomial data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        if ((t != (T)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for multinomial data with a specific data type (char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        if ((t != (float)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for multinomial data with a specific data type (unsigned char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        if ((t != (float)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// Function to calculate the scaled marginal cross-entropy error for multinomial data.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateMultinomialScaledMarginalCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateMultinomialScaledMarginalCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed multinomial data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        if ((t != (T)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed multinomial data with a specific data type (char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        if ((t != (float)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed multinomial data.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        if ((t != (float)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// Function to calculate the scaled marginal cross-entropy error for indexed multinomial data.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateIndexedMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

#define EXPLICITLY_INSTANTIATE_KERNELS(T)\
template float kCalculateL1Error<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedL1Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateL2Error<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedL2Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateHingeError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedHingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateSparseAnalogL1Error<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateIndexedSparseAnalogL1Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateSparseAnalogL2Error<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateIndexedSparseAnalogL2Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateSparseAnalogL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateIndexedSparseAnalogL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateSparseAnalogMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*);\
template float kCalculateIndexedSparseAnalogMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*);\
template float kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*);\
template float kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*);\
template float kCalculateSparseDataScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, T*, bool);\
template float kCalculateIndexedSparseDataScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, T*, bool);\

EXPLICITLY_INSTANTIATE_KERNELS(float)
EXPLICITLY_INSTANTIATE_KERNELS(double)
EXPLICITLY_INSTANTIATE_KERNELS(unsigned char)
EXPLICITLY_INSTANTIATE_KERNELS(char)
EXPLICITLY_INSTANTIATE_KERNELS(uint32_t)
EXPLICITLY_INSTANTIATE_KERNELS(uint64_t)
EXPLICITLY_INSTANTIATE_KERNELS(int32_t)
EXPLICITLY_INSTANTIATE_KERNELS(int64_t)

#endif