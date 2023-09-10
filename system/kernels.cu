#ifdef __CUDACC__

#include "GpuTypes.h"
#include "Types.h"
#include <limits>
#include <stdexcept>

static __constant__ GpuData cData;

void SetKernelsGpuData()
{
    cudaError_t status;
    status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyToSymbol: SetKernelsGpuData copy to cData failed");
    }
}

void GetKernelsGpuData()
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyFromSymbol: GetKernelsGpuData copy From cData failed");
    }
}


uint32_t CalculateBlocks(uint64_t size)
{
    return (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
}

__global__ void
LAUNCH_BOUNDS()
kScaleAndBias_kernel(float* pData, uint64_t size, float scale, float bias)
{
    const uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < size)
    {
        float value = pData[offset];
        pData[offset] = scale * value - bias;
    }
}

void kScaleAndBias(float* pData, uint64_t size, float scale, float bias)
{
    const uint64_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    kScaleAndBias_kernel<<<blocks, threadsPerBlock>>>(pData, size, scale, bias);
    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}


__global__ void LAUNCH_BOUNDS() kClearUnit_kernel(float* pUnit, float* pBias, uint32_t stride, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias[bpos];
    }
}

void kClearUnit(float* pUnit, float* pBias, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    kClearUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias, stride, size);
    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void LAUNCH_BOUNDS() kClearDualSourceUnit_kernel(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t size)
{
    const uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        const uint32_t bpos = pos % stride;
        pUnit[pos] = pBias1[bpos] + pBias2[bpos];
    }
}

void kClearDualSourceUnit(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch)
{
    const uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    kClearDualSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, stride, size);
    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void LAUNCH_BOUNDS() kClearTripleSourceUnit_kernel(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    if (pos < size)
    {
        pUnit[pos] = pBias1[bpos] + pBias2[bpos] + pBias3[pos];
    }
}

void kClearTripleSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    kClearTripleSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, stride, size);
    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void LAUNCH_BOUNDS() kClearQuadSourceUnit_kernel(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    if (pos < size)
    {
        pUnit[pos] = pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

void kClearQuadSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    kClearQuadSourceUnit_kernel<<<blocks, threadsPerBlock>>>(pUnit, pBias1, pBias2, pBias3, pBias4, stride, size);
    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void LAUNCH_BOUNDS() kLoadSparseInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    const uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1                   = pos + position;                            
        pos1                            = cData._bShuffleIndices ?  cData._pShuffleIndex[pos1] : pos1;
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = (pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0;
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            const uint64_t pos2 = offset + pSparseIndex[start];
            pUnit[pos2] = w;
            start += cData._warpSize;
        }
    }
}

void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    const uint32_t last = position + batch;
    const uint32_t count = last - position;
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    kLoadSparseInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void LAUNCH_BOUNDS() kLoadIndexedSparseInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    const uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1                   = pos + position;                            
        pos1                            = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[pos1] : pos1];
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = (pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0;
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            const uint64_t pos2 = offset + pSparseIndex[start];
            pUnit[pos2] = w;
            start += cData._warpSize;
        }
    }
}

void kLoadIndexedSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    const uint32_t last = position + batch;
    const uint32_t count = last - position;
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    kLoadIndexedSparseInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight);

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kLoadSparseAnalogInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    const uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1 = pos + position;
        pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1;
        const uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        const uint64_t end = pSparseEnd[pos1];
        const float w = (pDataWeight != nullptr) ? pDataWeight[pos1] : 1.0f;
        const uint64_t offset = pos * stride;

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            const uint64_t pos2 = offset + pSparseIndex[i];
            T data = pSparseData[i];
            pUnit[pos2] = w * data;
        }
    }
}

template<typename T>
void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    try {
        uint32_t last = position + batch;
        uint32_t count = last - position;
        uint32_t blocks = (count * getGpu()._warpSize + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
        cudaError_t status = cudaMemset(pUnit, 0, (uint64_t)batch * (uint64_t)stride * sizeof(float));
        kLoadSparseAnalogInputUnit_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kLoadSparseAnalogInputUnit: " + std::string(e.what()));
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadIndexedSparseAnalogInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t pos1                   = pos + position;                            
        pos1                            = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[pos1] : pos1];
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = (pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0;
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            uint64_t pos2               = offset + pSparseIndex[start];
            T data                      = pSparseData[start];
            pUnit[pos2]                 = w * data;
            start                      += cData._warpSize;
        }
    }
}

template<typename T>
void kLoadIndexedSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    kLoadIndexedSparseAnalogInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void
LAUNCH_BOUNDS()
kLoadSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0);
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            float value               = pRandom[start];
            uint64_t pos2               = offset + pSparseIndex[start];
            if (value >= cData._denoising_p)
                pUnit[pos2]             = w;
            start                      += cData._warpSize;
        }
    }
}


void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    kLoadSparseDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void
LAUNCH_BOUNDS()
kLoadIndexedSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1                   = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0);
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            float value               = pRandom[start];
            uint64_t pos2               = offset + pSparseIndex[start];
            if (value >= cData._denoising_p)
                pUnit[pos2]             = w;
            start                      += cData._warpSize;
        }
    }
}

void kLoadIndexedSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    kLoadIndexedSparseDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0);
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            float value               = pRandom[start];
            uint64_t pos2               = offset + pSparseIndex[start];
            T data                      = pSparseData[start];
            if (value >= cData._denoising_p)
                pUnit[pos2]             = w * data;
            start                      += cData._warpSize;
        }
    }
}

template<typename T>
void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    kLoadSparseAnalogDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadIndexedSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1                   = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0);
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            float value               = pRandom[start];
            uint64_t pos2               = offset + pSparseIndex[start];
            T data                      = pSparseData[start];
            if (value >= cData._denoising_p)
                pUnit[pos2]             = w * data;
            start                      += cData._warpSize;
        }
    }
}

template<typename T>
void kLoadIndexedSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    kLoadIndexedSparseAnalogDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData)
{
    uint64_t pos                        = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = pData[soffset];
    }
}

__global__ void
kLoadNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData)
{
    uint64_t pos                        = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = (float)pData[soffset] * (float)(1.0 / 256.0) - (float)0.5;
    }
}

__global__ void
kLoadNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData)
{
    uint64_t pos          = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = (float)pData[soffset] * (float)(1.0 / 128.0);
    }
}

template<typename T>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template<>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, unsigned char* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template<>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, char* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadIndexedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData)
{
    uint64_t pos                        = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position];
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = pData[soffset];
    }
}

__global__ void
kLoadIndexedNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData)
{
    uint64_t pos                        = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position];
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = (float)pData[soffset] * (float)(1.0 / 256.0) - (float)0.5;
    }
}

__global__ void
kLoadIndexedNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData)
{
    uint64_t pos          = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position];
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = (float)pData[soffset] * (float)(1.0 / 128.0);
    }
}

template<typename T>
void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadIndexedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pIndex, pData);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template<>
void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadIndexedNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pIndex, pData);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template<>
void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadIndexedNormalizedInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pIndex, pData);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void
LAUNCH_BOUNDS()
kAddBias_kernel(float* pUnit, float* pBias, uint32_t stride, uint32_t size)
{
    uint32_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias[bpos];
    }
}


void kAddBias(float* pUnit, float* pBias, uint32_t stride, uint32_t batch)
{
    uint32_t size = stride * batch;
    uint32_t blocks = CalculateBlocks(size);
    kAddBias_kernel<<<blocks, getGpu()._threadsPerBlock>>>(pUnit, pBias, stride, size);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}


__global__ void
LAUNCH_BOUNDS()
kAddDualBias_kernel(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias1[bpos] + pBias2[bpos];
    }
}

void kAddDualBias(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    kAddDualBias_kernel << <blocks, threadsPerBlock >> > (pUnit, pBias1, pBias2, stride, size);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void
LAUNCH_BOUNDS()
kAddTripleBias_kernel(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias1[bpos] + pBias2[bpos] + pBias3[pos];
    }
}

void kAddTripleBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    kAddTripleBias_kernel << <blocks, threadsPerBlock >> > (pUnit, pBias1, pBias2, pBias3, stride, size);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void
LAUNCH_BOUNDS()
kAddQuadBias_kernel(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

void kAddQuadBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    kAddQuadBias_kernel << <blocks, threadsPerBlock >> > (pUnit, pBias1, pBias2, pBias3, pBias4, stride, size);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

#if (__CUDA_ARCH__ >= 600)
static const uint32_t MAXSPARSE = SM_6X_MAXSPARSE;
static const uint32_t MAXSPARSEANALOG = SM_6X_MAXSPARSEANALOG;
#elif (__CUDA_ARCH__ >= 500)
static const uint32_t MAXSPARSE = SM_5X_MAXSPARSE;
static const uint32_t MAXSPARSEANALOG = SM_5X_MAXSPARSEANALOG;
#else
static const uint32_t MAXSPARSE = SM_3X_MAXSPARSE;
static const uint32_t MAXSPARSEANALOG = SM_3X_MAXSPARSEANALOG;
#endif


__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSE];

    position                        = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start                  = pSparseStart[position];
    uint64_t end                    = pSparseEnd[position];
    float w                       = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit                          += blockIdx.x * stride;
    while (start < end)
    {
        sOpos                       = blockDim.x;
        uint32_t inputs             = ullmin(end - start, (uint64_t)MAXSPARSE);
        uint64_t tend               = start + inputs;
        uint64_t tstart             = start + threadIdx.x;
        uint32_t pos                = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos]            = pSparseIndex[tstart] * stride;
            pos                    += blockDim.x;
            tstart                 += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit           += w * pWeight[offset + opos];
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}


void kCalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseZ_kernel << <batch, threads >> > (position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void
LAUNCH_BOUNDS256()
kCalculateIndexedSparseZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSE];

    sOpos                       = blockDim.x;
    position                    = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        sOpos                       = blockDim.x;
        uint32_t inputs             = ullmin(end - start, (uint64_t)MAXSPARSE);
        uint64_t tend               = start + inputs;
        uint64_t tstart             = start + threadIdx.x;
        uint32_t pos                = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos]            = pSparseIndex[tstart] * stride;
            pos                    += blockDim.x;
            tstart                 += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit           += w * pWeight[offset + opos];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}


void kCalculateIndexedSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateIndexedSparseZ_kernel << <batch, threads >> > (position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ T sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        sOpos                       = blockDim.x;
        uint32_t inputs             = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend               = start + inputs;
        uint64_t tstart             = start + threadIdx.x;
        uint32_t pos                = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos]        = pSparseIndex[tstart] * stride;
            sValue[pos]         = w * pSparseData[tstart];
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit           += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ float sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        sOpos                       = blockDim.x;
        uint32_t inputs             = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend               = start + inputs;
        uint64_t tstart             = start + threadIdx.x;
        uint32_t pos                = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos]        = pSparseIndex[tstart] * stride;
            sValue[pos]         = w * ((float)pSparseData[tstart] * (float)(1.0 / 256.0));
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit           += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ float sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        sOpos                       = blockDim.x;
        uint32_t inputs             = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend               = start + inputs;
        uint64_t tstart             = start + threadIdx.x;
        uint32_t pos                = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos]        = pSparseIndex[tstart] * stride;
            sValue[pos]         = w * ((float)pSparseData[tstart] * (float)(1.0 / 256.0));
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit           += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<typename T>
void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogZ_kernel << <batch, threads >> > (position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS256()
kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ T sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        sOpos                       = blockDim.x;
        uint32_t inputs             = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend               = start + inputs;
        uint64_t tstart             = start + threadIdx.x;
        uint32_t pos                = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos]        = pSparseIndex[tstart] * stride;
            sValue[pos]         = w * pSparseData[tstart];
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit           += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ float sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        sOpos                       = blockDim.x;
        uint32_t inputs             = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend               = start + inputs;
        uint64_t tstart             = start + threadIdx.x;
        uint32_t pos                = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos]        = pSparseIndex[tstart] * stride;
            sValue[pos]         = w * ((float)pSparseData[tstart] * (float)(1.0 / 256.0));
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit           += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ float sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        sOpos                       = blockDim.x;
        uint32_t inputs             = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend               = start + inputs;
        uint64_t tstart             = start + threadIdx.x;
        uint32_t pos                = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos]        = pSparseIndex[tstart] * stride;
            sValue[pos]         = w * ((float)pSparseData[tstart] * (float)(1.0 / 128.0));
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit           += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateIndexedSparseAnalogZ_kernel << <batch, threads >> > (position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}


__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSE];

    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        sOpos                       = blockDim.x;
        uint32_t inputs             = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend               = start + inputs;
        uint64_t tstart             = start + threadIdx.x;
        uint32_t pos                = threadIdx.x;

        while (tstart < tend)
        {
            float value           = pRandom[tstart];
            sOffset[pos]            = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            pos                    += blockDim.x;
            tstart                 += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit       += pWeight[offset + opos];
                }

                pUnit[opos]         = w * unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

void kCalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseDenoisedZ_kernel << <batch, threads >> > (position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void
LAUNCH_BOUNDS256()
kCalculateIndexedSparseDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSE];

    sOpos                       = blockDim.x;
    position                    = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit                      += blockIdx.x * stride;

    while (start < end)
    {
        sOpos                       = blockDim.x;
        uint32_t inputs             = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend               = start + inputs;
        uint64_t tstart             = start + threadIdx.x;
        uint32_t pos                = threadIdx.x;

        while (tstart < tend)
        {
            float value           = pRandom[tstart];
            sOffset[pos]            = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            pos                    += blockDim.x;
            tstart                 += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit       += pWeight[offset + opos];
                }

                pUnit[opos]         = w * unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

void kCalculateIndexedSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateIndexedSparseDenoisedZ_kernel << <batch, threads >> > (position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ T sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs         = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend           = start + inputs;
        uint64_t tstart         = start + threadIdx.x;
        uint32_t pos            = threadIdx.x;

        while (tstart < tend)
        {
            float value       = pRandom[tstart];
            sOffset[pos]        = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos]         = pSparseData[tstart] * w;
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : pUnit[opos];
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit       += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pRandom, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ int32_t sOffset[MAXSPARSEANALOG];
__shared__ float sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs         = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend           = start + inputs;
        uint64_t tstart         = start + threadIdx.x;
        uint32_t pos            = threadIdx.x;
        while (tstart < tend)
        {
            float value       = pRandom[tstart];
            sOffset[pos]        = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos]         = (float)pSparseData[tstart] * (float)(1.0 / 256.0) * w;
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit       += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pRandom, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ float sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs         = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend           = start + inputs;
        uint64_t tstart         = start + threadIdx.x;
        uint32_t pos            = threadIdx.x;
        while (tstart < tend)
        {
            float value       = pRandom[tstart];
            sOffset[pos]        = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos]         = (float)pSparseData[tstart] * (float)(1.0 / 128.0) * w;
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit       += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<typename T>
void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogDenoisedZ_kernel << <batch, threads >> > (position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS256()
kCalculateIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ T sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs         = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend           = start + inputs;
        uint64_t tstart         = start + threadIdx.x;
        uint32_t pos            = threadIdx.x;

        while (tstart < tend)
        {
            float value       = pRandom[tstart];
            sOffset[pos]        = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos]         = pSparseData[tstart] * w;
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit       += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pRandom, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ int32_t sOffset[MAXSPARSEANALOG];
__shared__ float sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs         = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend           = start + inputs;
        uint64_t tstart         = start + threadIdx.x;
        uint32_t pos            = threadIdx.x;

        while (tstart < tend)
        {
            float value       = pRandom[tstart];
            sOffset[pos]        = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos]         = (float)pSparseData[tstart] * (float)(1.0 / 256.0) * w;
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit       += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
kCalculateIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pRandom, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ float sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs         = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend           = start + inputs;
        uint64_t tstart         = start + threadIdx.x;
        uint32_t pos            = threadIdx.x;

        while (tstart < tend)
        {
            float value       = pRandom[tstart];
            sOffset[pos]        = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos]         = (float)pSparseData[tstart] * (float)(1.0 / 128.0) * w;
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit       += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateIndexedSparseAnalogDenoisedZ_kernel << <batch, threads >> > (position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            start                          += cData._warpSize;                   
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateWeightedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = pDataWeight[position];
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            pSparseTransposedData[opos]     = w;
            start                          += cData._warpSize;                   
        }
    }
}

void kCalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    try {
        if (pDataWeight == NULL) {
            kCalculateSparseTransposedMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pSparseStart, pSparseEnd, pSparseIndex, pSparseTransposedEnd, pSparseTransposedIndex);
        }
        else {
            kCalculateWeightedSparseTransposedMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
        }

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedMatrix: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            start                          += cData._warpSize;                   
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedWeightedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = pDataWeight[position];
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            pSparseTransposedData[opos]     = w;
            start                          += cData._warpSize;                   
        }
    }
}

void kCalculateIndexedSparseTransposedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch);

    try {
        if (pDataWeight == NULL) {
            kCalculateIndexedSparseTransposedMatrix_kernel << <batch, getGpu()._warpSize >> > (position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseTransposedEnd, pSparseTransposedIndex);
        }
        else {
            kCalculateIndexedWeightedSparseTransposedMatrix_kernel << <batch, getGpu()._warpSize >> > (position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
        }

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateIndexedSparseTransposedMatrix: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
            }
            start                          += cData._warpSize;                   
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateWeightedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = cData._denoising_q * pDataWeight[position];
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w;
            }
            start                          += cData._warpSize;                   
        }
    }
}

void kCalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    try {
        if (pDataWeight == NULL) {
            kCalculateSparseTransposedDenoisedMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pSparseStart, pSparseEnd, pSparseIndex, pRandom, pSparseTransposedEnd, pSparseTransposedIndex);
        }
        else {
            kCalculateWeightedSparseTransposedDenoisedMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
        }

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedDenoisedMatrix: " + std::string(e.what()));
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
            }
            start                          += cData._warpSize;                   
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedWeightedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = cData._denoising_q * pDataWeight[position];
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w;
            }
            start                          += cData._warpSize;                   
        }
    }
}

void kCalculateIndexedSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    try {
        if (pDataWeight == NULL) {
            kCalculateIndexedSparseTransposedDenoisedMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pRandom, pSparseTransposedEnd, pSparseTransposedIndex);
        }
        else {
            kCalculateIndexedWeightedSparseTransposedDenoisedMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
        }

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateIndexedSparseTransposedDenoisedMatrix: " + std::string(e.what()));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            T value                         = pSparseData[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            pSparseTransposedData[opos]     = w * value;
            start                          += cData._warpSize;                   
        }
    }
}

template<typename T>
void kCalculateSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    try {
        kCalculateSparseTransposedAnalogMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedAnalogMatrix: " + std::string(e.what()));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            T value                         = pSparseData[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            pSparseTransposedData[opos]     = w * value;
            start                          += cData._warpSize;                   
        }
    }
}

template<typename T>
void kCalculateIndexedSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    try {
        kCalculateIndexedSparseTransposedAnalogMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateIndexedSparseTransposedAnalogMatrix: " + std::string(e.what()));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                T value                     = pSparseData[start];
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value               = (float)pSparseData[start] * (float)(1.0 / 256.0);
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value               = (float)pSparseData[start] * (float)(1.0 / 128.0);
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}

template<typename T>
void kCalculateSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    try {
        kCalculateSparseTransposedAnalogDenoisedMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedAnalogDenoisedMatrix: " + std::string(e.what()));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                T value                     = pSparseData[start];
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value               = (float)pSparseData[start] * (float)(1.0 / 256.0);
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value               = (float)pSparseData[start] * (float)(1.0 / 128.0);
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}

template<typename T>
void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    try {
        kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateIndexedSparseTransposedAnalogDenoisedMatrix: " + std::string(e.what()));
    }
}


__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseTransposedWeightGradient_kernel(float alpha, float beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pDelta, float* pWeightGradient)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSE];

    uint64_t start              = pSparseTransposedStart[blockIdx.x];
    uint64_t end                = pSparseTransposedEnd[blockIdx.x];
    alpha                      *= cData._denoising_q;
    pWeightGradient            += blockIdx.x * n;
    do
    {
        sOpos                   = blockDim.x;         
        uint32_t inputs         = ullmin(end - start, (uint64_t)MAXSPARSE);
        uint64_t tend           = start + inputs;
        uint64_t tstart         = start + threadIdx.x;
        uint32_t pos            = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos]        = pSparseTransposedIndex[tstart] * n;
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t opos           = threadIdx.x;
        uint32_t tgx            = threadIdx.x & cData._warpMask;    
        while (opos < n)
        {        
            float oldgradient = (beta == (float)0.0) ? (float)0.0 : beta * pWeightGradient[opos];
            int64_t sum         = 0;
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                sum            += llrintf(ERRORSCALEF * pDelta[offset + opos]);  
            }

            float fsum        = alpha * (float)((double)sum * ONEOVERERRORSCALE);
            pWeightGradient[opos] = oldgradient + fsum;            

            if (tgx == 0)
            {
                opos            = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                = SHFL(opos, 0);
            opos               += tgx;
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
            beta                = (float)1.0;
        }
    }
    while (start < end);
}


void kCalculateSparseTransposedWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pDelta, float* pWeightGradient)
{
    uint32_t threads = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    try {
        kCalculateSparseTransposedWeightGradient_kernel << <m, threads >> > (alpha, beta, n, pSparseTransposedStart, pSparseTransposedEnd, pSparseTransposedIndex, pDelta, pWeightGradient);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedWeightGradient: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseTransposedAnalogWeightGradient_kernel(float alpha, float beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData, float* pDelta, float* pWeightGradient)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ float sValue[MAXSPARSEANALOG];

    uint64_t start              = pSparseTransposedStart[blockIdx.x];
    uint64_t end                = pSparseTransposedEnd[blockIdx.x];
    alpha                      *= cData._denoising_q;
    pWeightGradient            += blockIdx.x * n;
    do
    {
        sOpos                   = blockDim.x;        
        uint32_t inputs         = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend           = start + inputs;
        uint64_t tstart         = start + threadIdx.x;
        uint32_t pos            = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos]        = pSparseTransposedIndex[tstart] * n;
            sValue[pos]         = pSparseTransposedData[start];
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();


        uint32_t opos           = threadIdx.x;
        uint32_t tgx            = threadIdx.x & cData._warpMask;    
        while (opos < n)
        {        
            float oldgradient = (beta == (float)0.0) ? (float)0.0 : beta * pWeightGradient[opos];
            int64_t sum         = 0;
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                float value   = sValue[i]; 
                sum            += llrintf(ERRORSCALEF * value * pDelta[offset + opos]);  
            }

            float fsum        = alpha * (float)((double)sum * ONEOVERERRORSCALE);
            pWeightGradient[opos] = oldgradient + fsum;            

            if (tgx == 0)
            {
                opos            = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                = SHFL(opos, 0);
            opos               += tgx;
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
            beta                = (float)1.0;
        }
    }
    while (start < end);
}

void kCalculateSparseTransposedAnalogWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData, float* pDelta, float* pWeightGradient)
{
    uint32_t threads = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    try {
        kCalculateSparseTransposedAnalogWeightGradient_kernel << <m, threads >> > (alpha, beta, n, pSparseTransposedStart, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData, pDelta, pWeightGradient);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedAnalogWeightGradient: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kUpdateBiases_kernel(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum             = (float)0.0;
        pDelta                 += pos;
        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }
        pBias[pos]             -= alpha * sum;
    }
}

void kUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);

    try {
        kUpdateBiases_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, batch, width, pDelta, pBias);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kUpdateBiases: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateRegularizationError_kernel(float* pWeight, uint64_t size, float lambda, float lambda1)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    float error               = (float)0.0;
    if (pos < size)
    {
        float w               = pWeight[pos];
        error                   = lambda * w * w + lambda1 * abs(w);   
    }

    REDUCEERROR(error)
}

float kCalculateRegularizationError(float lambda, float lambda1, float* pWeight, uint64_t size)
{
    uint32_t blocks = CalculateBlocks(size);

    try {
        cudaError_t status = cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
        if (status != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed");
        }

        kCalculateRegularizationError_kernel << <blocks, getGpu()._threadsPerBlock >> > (pWeight, size, (float)0.5 * lambda, lambda1);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }

        getGpu()._pbAccumulator->Download();
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateRegularizationError: " + std::string(e.what()));
    }

    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kSGDUpdateWeights_kernel(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightGradient, float* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g               = pWeightGradient[pos];
        float w               = pWeight[pos];
        pWeight[pos]            = w + alpha * (g - lambda * w - lambda1 * sgn(w));
    }
}

void kSGDUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightGradient, float* pWeight)
{
    uint32_t blocks = CalculateBlocks(size);

    try {
        kSGDUpdateWeights_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, lambda, lambda1, size, pWeightGradient, pWeight);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kSGDUpdateWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kSGDUpdateBiases_kernel(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint32_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum             = 0.0f;
        pDelta                 += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }
        sum                    /= (float)batch;

        float bias            = pBias[pos];
        pBias[pos]              = bias - alpha * sum;
    }
}

void kSGDUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);

    try {
        kSGDUpdateBiases_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, batch, width, pDelta, pBias);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kSGDUpdateBiases: " + std::string(e.what()));
    }
}


__global__ void
LAUNCH_BOUNDS()
kMomentumUpdateWeights_kernel(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g               = pWeightGradient[pos];
        float w               = pWeight[pos];
        float v               = pWeightVelocity[pos];
        v                       = mu * v + alpha * (g - lambda * w - lambda1 * sgn(w));
        pWeightVelocity[pos]    = v;
        pWeight[pos]            = w + v;
    }
}

void kMomentumUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint32_t blocks = CalculateBlocks(size);

    try {
        kMomentumUpdateWeights_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, lambda, lambda1, mu, size, pWeightVelocity, pWeightGradient, pWeight);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kMomentumUpdateWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kMomentumUpdateBiases_kernel(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint32_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum             = 0.0f;
        pDelta                 += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }
        sum                    /= (float)batch;

        float v               = pBiasVelocity[pos];
        v                       = mu * v - alpha * sum;
        pBiasVelocity[pos]      = v;
        pBias[pos]             += v;
    }
}

void kMomentumUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);

    try {
        kMomentumUpdateBiases_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, mu, batch, width, pDelta, pBiasVelocity, pBias);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kMomentumUpdateBiases: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kAdaGradUpdateWeights_kernel(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g               = pWeightGradient[pos];
        float w               = pWeight[pos];
        float v               = pWeightVelocity[pos];
        g                      -= lambda * w + lambda1 * sgn(w);
        v                      += g * g;
        pWeightVelocity[pos]    = v;
        pWeight[pos]            = w + alpha * g * rsqrt(max(0.000000001f, v));
    }
}

void kAdaGradUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    unsigned long blocks = CalculateBlocks(size);

    try {
        kAdaGradUpdateWeights_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, lambda, lambda1, size, pWeightVelocity, pWeightGradient, pWeight);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAdaGradUpdateWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kAdaGradUpdateBiases_kernel(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum             = 0.0f;
        pDelta                 += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }
        sum                    /= (float)batch;

        float v               = pBiasVelocity[pos];
        v                      += sum * sum;
        pBiasVelocity[pos]      = v;
        pBias[pos]             -= alpha * sum * rsqrt(max(0.000000001f, v));
    }
}

void kAdaGradUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);

    try {
        kAdaGradUpdateBiases_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, batch, width, pDelta, pBiasVelocity, pBias);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAdaGradUpdateBiases: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kAdaDeltaUpdateWeights_kernel(float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g                       = pWeightGradient[pos];
        float w                       = pWeight[pos];
        float v                       = pWeightVelocity[pos];
        float vg                      = pWeightGradientVelocity[pos];
        g                              -= lambda * w + lambda1 * sgn(w);
        vg                              = mu * vg + ((float)1.0 - mu) * g * g;
        float dw                      = sqrt(max((float)0.000000001, v) / max((float)0.000000001, vg)) * g;
        v                               = mu * v + ((float)1.0 - mu) * dw * dw;
        pWeightVelocity[pos]            = v;
        pWeightGradientVelocity[pos]    = vg;
        pWeight[pos]                    = w + dw;
    }
}

void kAdaDeltaUpdateWeights(float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight)
{
    unsigned long blocks = CalculateBlocks(size);

    try {
        kAdaDeltaUpdateWeights_kernel << <blocks, getGpu()._threadsPerBlock >> > (lambda, lambda1, mu, size, pWeightVelocity, pWeightGradient, pWeightGradientVelocity, pWeight);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAdaDeltaUpdateWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kAdaDeltaUpdateBiases_kernel(float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias)
{
    uint64_t pos                    = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum                 = (float)0.0;
        pDelta                     += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum                    += *pDelta;
            pDelta                 += width;
        }
        sum                        /= (float)batch;

        float v                   = pBiasVelocity[pos];
        float vg                  = pBiasGradientVelocity[pos];        
        vg                          = mu * vg + ((float)1.0 - mu) * sum * sum;
        float dw                  = sqrt(max((float)0.000000001, v) / max((float)0.000000001, vg)) * sum;        
        v                           = mu * v + ((float)1.0 - mu) * dw * dw;
        pBiasVelocity[pos]          = v;
        pBiasGradientVelocity[pos]  = vg;        
        pBias[pos]                 -= dw;
    }
}

void kAdaDeltaUpdateBiases(float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);

    try {
        kAdaDeltaUpdateBiases_kernel << <blocks, getGpu()._threadsPerBlock >> > (mu, batch, width, pDelta, pBiasVelocity, pBiasGradientVelocity, pBias);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAdaDeltaUpdateBiases: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kAdamUpdateWeights_kernel(float alpha, float lambda, float lambda1, float beta1, float beta2, float t, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float dw                      = pWeightGradient[pos];
        float w                       = pWeight[pos];
        float vdw                     = pWeightVelocity[pos];
        float sdw                     = pWeightGradientVelocity[pos];
        dw                             -= lambda * w + lambda1 * sgn(w);
        vdw                             = beta1 * vdw + ((float)1.0 - beta1) * dw;
        sdw                             = beta2 * sdw + ((float)1.0 - beta2) * dw * dw;
        t                              += (float)1.0;
        pWeightVelocity[pos]            = vdw;
        pWeightGradientVelocity[pos]    = sdw;
        vdw                            /= (float)1.0 - pow(beta1, t);
        sdw                            /= (float)1.0 - pow(beta2, t);        
        dw                              = alpha * vdw / (sqrt(sdw) + (float)1.0e-8);
        pWeight[pos]                    = w + dw;
    }
}

void kAdamUpdateWeights(float alpha, float lambda, float lambda1, float beta1, float beta2, float t, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight)
{
    unsigned long blocks = CalculateBlocks(size);

    try {
        kAdamUpdateWeights_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, lambda, lambda1, beta1, beta2, t, size, pWeightVelocity, pWeightGradient, pWeightGradientVelocity, pWeight);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAdamUpdateWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kAdamUpdateBiases_kernel(float alpha, float beta1, float beta2, float t, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias)
{
    uint64_t pos                    = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum                 = (float)0.0;
        pDelta                     += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum                    += *pDelta;
            pDelta                 += width;
        }
        sum                        /= (float)batch;

        float vdw                 = pBiasVelocity[pos];
        float sdw                 = pBiasGradientVelocity[pos];
        vdw                         = beta1 * vdw + ((float)1.0 - beta1) * sum;
        sdw                         = beta2 * sdw + ((float)1.0 - beta2) * sum * sum;
        t                          += (float)1.0;
        pBiasVelocity[pos]          = vdw;
        pBiasGradientVelocity[pos]  = sdw;
        vdw                        /= (float)1.0 - pow(beta1, t);
        sdw                        /= (float)1.0 - pow(beta2, t);        
        float dw                  = alpha * vdw / (sqrt(sdw) + (float)1.0e-8);
        pBias[pos]                 -= dw;
    }
}

void kAdamUpdateBiases(float alpha, float mu, float mu1, float t, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);

    try {
        kAdamUpdateBiases_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, mu, mu1, t, batch, width, pDelta, pBiasVelocity, pBiasGradientVelocity, pBias);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAdamUpdateBiases: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kNesterovUpdateWeights_kernel(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g               = pWeightGradient[pos];
        float w               = pWeight[pos];
        float vOld            = pWeightVelocity[pos];
        float vNew            = mu * vOld + alpha * (g - lambda * w - lambda1 * sgn(w));
        pWeightVelocity[pos]    = vNew;
        w                       = w + vNew + mu * (vNew - vOld);
        pWeight[pos]            = w;      
    }
}

void kNesterovUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint32_t blocks = CalculateBlocks(size);

    try {
        kNesterovUpdateWeights_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, lambda, lambda1, mu, size, pWeightVelocity, pWeightGradient, pWeight);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kNesterovUpdateWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kNesterovUpdateBiases_kernel(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum             = 0.0f;
        pDelta                 += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }
        sum                    /= (float)batch;

        float vOld            = pBiasVelocity[pos];
        float vNew            = mu * vOld - alpha * sum;
        pBiasVelocity[pos]      = vNew;
        pBias[pos]             += vNew + mu * (vNew - vOld);
    }
}

void kNesterovUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    try {
        uint32_t blocks = CalculateBlocks(width);
        kNesterovUpdateBiases_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, mu, batch, width, pDelta, pBiasVelocity, pBias);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kNesterovUpdateBiases: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kNesterovShiftWeights_kernel(float mu, uint64_t size, float* pWeightVelocity, float* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float w               = pWeight[pos];
        float v               = pWeightVelocity[pos];
        pWeight[pos]            = w + mu * v;
    }
}

void kNesterovShiftWeights(float mu, uint64_t size, float* pWeightVelocity, float* pWeight)
{
    try {
        uint32_t blocks = CalculateBlocks(size);
        kNesterovShiftWeights_kernel << <blocks, getGpu()._threadsPerBlock >> > (mu, size, pWeightVelocity, pWeight);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kNesterovShiftWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kNesterovShiftBiases_kernel(float mu, uint32_t width, float* pBiasVelocity, float* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float b               = pBias[pos];
        float v               = pBiasVelocity[pos];
        pBias[pos]              = b + mu * v;
    }
}

void kNesterovShiftBiases(float mu, uint32_t width, float* pBiasVelocity, float* pBias)
{
    try {
        uint32_t blocks = CalculateBlocks(width);
        kNesterovShiftBiases_kernel << <blocks, getGpu()._threadsPerBlock >> > (mu, width, pBiasVelocity, pBias);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kNesterovShiftBiases: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kRMSPropUpdateWeights_kernel(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint64_t pos  = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g               = pWeightGradient[pos];
        float w               = pWeight[pos];
        float v               = pWeightVelocity[pos];
        g                      -= lambda * w + lambda1 * sgn(w);
        v                       = mu * v + (1.0f - mu) * g * g;
        pWeightVelocity[pos]    = v;
        pWeight[pos]            = w + alpha * g * rsqrt(max(0.000000001f, v));
    }
}

void kRMSPropUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    try {
        uint32_t blocks = CalculateBlocks(size);
        kRMSPropUpdateWeights_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, lambda, lambda1, mu, size, pWeightVelocity, pWeightGradient, pWeight);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kRMSPropUpdateWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kRMSPropUpdateBiases_kernel(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum             = 0.0f;
        pDelta                 += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }
        sum                    /= (float)batch;

        float v               = pBiasVelocity[pos];
        v                       = mu * v + (1.0f - mu) * sum * sum;
        pBiasVelocity[pos]      = v;
        pBias[pos]             -= alpha * sum * rsqrt(max(0.000000001f, v));
    }
}

void kRMSPropUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    try {
        uint32_t blocks = CalculateBlocks(width);
        kRMSPropUpdateBiases_kernel << <blocks, getGpu()._threadsPerBlock >> > (alpha, mu, batch, width, pDelta, pBiasVelocity, pBias);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kRMSPropUpdateBiases: " + std::string(e.what()));
    }
}

#include "Bitonic.cuh"
__global__ void
LAUNCH_BOUNDS()
CalculateOutput_32_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
__shared__ volatile float sKey[64 * 4];
__shared__ volatile uint32_t sValue[64 * 4];


    uint32_t pos                    = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                    = threadIdx.x & cData._warpMask;
            
    if (pos < batch)
    {
        float *pOutput            = pOutputBuffer + pos * width;
        uint32_t offset             = threadIdx.x >> cData._warpBits;
        volatile float* psKey     = &sKey[64 * offset];
        volatile uint32_t* psValue  = &sValue[64 * offset];

        float k0                  = -FLT_MAX;
        float k1                  = -FLT_MAX;
        uint32_t v0                 = 0;
        uint32_t v1                 = 0;

        uint32_t wpos               = tgx;
        if (wpos < width)
        {
            k0                      = pOutput[wpos];
            v0                      = wpos;
        }
        wpos                       += cData._warpSize;

        float minValue            = -FLT_MAX;
        uint32_t rpos               = 32;
        uint32_t bufferSize         = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos           = rpos + tgx;
            float key             = -FLT_MAX;
            uint32_t value          = wpos;
            if (wpos < width)
            {
                key                 = pOutput[wpos];                
            }
            
            uint32_t count          = BALLOT(key > minValue);
            if (key > minValue)
            {
                uint32_t mask       = 0xffffffff >> (32 - tgx);
                uint32_t offset     = __popc(count & mask);
                offset             += bufferSize;
                psKey[offset]       = key;
                psValue[offset]     = value;
            }
            bufferSize             += __popc(count);

            if (bufferSize >= 32)
            {
                k1                  = psKey[tgx];
                v1                  = psValue[tgx];
                bool flag;
                BITONICSORT64_64();

                minValue = SHFL(k0, cData._warpSize - 1);

                bufferSize         -= 32;
                if (tgx < bufferSize)
                {
                    psKey[tgx]      = psKey[tgx + 32];
                    psValue[tgx]    = psValue[tgx + 32];
                }
            }

            rpos                    += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 32))
        {
            k1                       = -FLT_MAX;
            v1                       = 0;

            if (tgx < bufferSize)
            {
                k1                   = psKey[tgx];
                v1                   = psValue[tgx];
            }
            BITONICSORT64_64();
        }

        float* pKey                = pKeyBuffer + pos * k;
        uint32_t* pValue             = pValueBuffer + pos * k;                
        wpos                         = tgx;
        if (wpos < k)
        {
            pKey[wpos]               = k0;
            pValue[wpos]             = v0;
        }
        wpos                        += cData._warpSize;
    }
}


__global__ void
LAUNCH_BOUNDS()
CalculateOutput_64_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
__shared__ volatile float sKey[96 * 4];
__shared__ volatile uint32_t sValue[96 * 4];


    uint32_t pos                    = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                    = threadIdx.x & cData._warpMask;
            
    if (pos < batch)
    {
        float *pOutput            = pOutputBuffer + pos * width;
        uint32_t offset             = threadIdx.x >> cData._warpBits;
        volatile float* psKey     = &sKey[96 * offset];
        volatile uint32_t* psValue  = &sValue[96 * offset];

        float k0                  = -FLT_MAX;
        float k1                  = -FLT_MAX;
        float k2                  = -FLT_MAX;
        float k3                  = -FLT_MAX;
        uint32_t v0                 = 0;
        uint32_t v1                 = 0;
        uint32_t v2                 = 0;
        uint32_t v3                 = 0;

        uint32_t wpos               = tgx;
        if (wpos < width)
        {
            k0                      = pOutput[wpos];
            v0                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k1                      = pOutput[wpos];
            v1                      = wpos;
        }
        wpos                       += cData._warpSize;

     
        float minValue            = -FLT_MAX;
        uint32_t rpos               = 64;
        uint32_t bufferSize         = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos           = rpos + tgx;
            float key             = -FLT_MAX;
            uint32_t value          = wpos;
            if (wpos < width)
            {
                key                 = pOutput[wpos];                
            }
            
            uint32_t count          = BALLOT(key > minValue);
            if (key > minValue)
            {
                uint32_t mask       = 0xffffffff >> (32 - tgx);
                uint32_t offset     = __popc(count & mask);
                offset             += bufferSize;
                psKey[offset]       = key;
                psValue[offset]     = value;
            }
            bufferSize             += __popc(count);

            if (bufferSize >= 64)
            {
                k2                  = psKey[tgx];
                v2                  = psValue[tgx];
                k3                  = psKey[tgx + cData._warpSize];
                v3                  = psValue[tgx + cData._warpSize];
                bool flag;
                BITONICSORT128_128();

                minValue = SHFL(k1, cData._warpSize - 1);

                bufferSize         -= 64;
                if (tgx < bufferSize)
                {
                    psKey[tgx]      = psKey[tgx + 64];
                    psValue[tgx]    = psValue[tgx + 64];
                }
            }

            rpos                    += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 64))
        {
            k2                       = -FLT_MAX;
            k3                       = -FLT_MAX;
            v2                       = 0;
            v3                       = 0;

            if (tgx < bufferSize)
            {
                k2                   = psKey[tgx];
                v2                   = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k3                   = psKey[tgx + cData._warpSize];
                v3                   = psValue[tgx + cData._warpSize];
            }

            BITONICSORT128_128();
        }

        float* pKey                = pKeyBuffer + pos * k;
        uint32_t* pValue             = pValueBuffer + pos * k;                
        wpos                         = tgx;
        if (wpos < k)
        {
            pKey[wpos]               = k0;
            pValue[wpos]             = v0;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k1;
            pValue[wpos]             = v1;
        }
        wpos                        += cData._warpSize;
    }
}

__global__ void
LAUNCH_BOUNDS()
CalculateOutput_128_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
__shared__ volatile float sKey[160 * 4];
__shared__ volatile uint32_t sValue[160 * 4];


    uint32_t pos                    = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                    = threadIdx.x & cData._warpMask;
        
    
    if (pos < batch)
    {
        float *pOutput            = pOutputBuffer + pos * width;
        uint32_t offset             = threadIdx.x >> cData._warpBits;
        volatile float* psKey     = &sKey[160 * offset];
        volatile uint32_t* psValue  = &sValue[160 * offset];

        float k0                  = -FLT_MAX;
        float k1                  = -FLT_MAX;
        float k2                  = -FLT_MAX;
        float k3                  = -FLT_MAX;
        float k4                  = -FLT_MAX;
        float k5                  = -FLT_MAX;
        float k6                  = -FLT_MAX;
        float k7                  = -FLT_MAX;
        uint32_t v0                 = 0;
        uint32_t v1                 = 0;
        uint32_t v2                 = 0;
        uint32_t v3                 = 0;
        uint32_t v4                 = 0;
        uint32_t v5                 = 0;
        uint32_t v6                 = 0;
        uint32_t v7                 = 0;

        uint32_t wpos               = tgx;
        if (wpos < width)
        {
            k0                      = pOutput[wpos];
            v0                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k1                      = pOutput[wpos];
            v1                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k2                      = pOutput[wpos];
            v2                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k3                      = pOutput[wpos];
            v3                      = wpos;
        }
     
        float minValue            = -FLT_MAX;
        uint32_t rpos               = 128;
        uint32_t bufferSize         = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos           = rpos + tgx;
            float key             = -FLT_MAX;
            uint32_t value          = wpos;
            if (wpos < width)
            {
                key                 = pOutput[wpos];                
            }
            
            uint32_t count          = BALLOT(key > minValue);
            if (key > minValue)
            {
                uint32_t mask       = 0xffffffff >> (32 - tgx);
                uint32_t offset     = __popc(count & mask);
                offset             += bufferSize;
                psKey[offset]       = key;
                psValue[offset]     = value;
            }
            bufferSize             += __popc(count);

            if (bufferSize >= 128)
            {
                k4                  = psKey[tgx];
                v4                  = psValue[tgx];
                k5                  = psKey[tgx + cData._warpSize];
                v5                  = psValue[tgx + cData._warpSize];
                k6                  = psKey[tgx + 2 * cData._warpSize];
                v6                  = psValue[tgx + 2 * cData._warpSize];
                k7                  = psKey[tgx + 3 * cData._warpSize];
                v7                  = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                minValue = SHFL(k3, cData._warpSize - 1);

                bufferSize         -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx]      = psKey[tgx + 128];
                    psValue[tgx]    = psValue[tgx + 128];
                }
            }

            rpos                    += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 128))
        {
            k4                       = -FLT_MAX;
            k5                       = -FLT_MAX;
            k6                       = -FLT_MAX;
            k7                       = -FLT_MAX;
            v4                       = 0;
            v5                       = 0;
            v6                       = 0;
            v7                       = 0;

            if (tgx < bufferSize)
            {
                k4                   = psKey[tgx];
                v4                   = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k5                   = psKey[tgx + cData._warpSize];
                v5                   = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k6                   = psKey[tgx + 2 * cData._warpSize];
                v6                   = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k7                   = psKey[tgx + 3 * cData._warpSize];
                v7                   = psValue[tgx + 3 * cData._warpSize];
            }

            BITONICSORT256_256();
        }

        float* pKey                = pKeyBuffer + pos * k;
        uint32_t* pValue             = pValueBuffer + pos * k;                
        wpos                         = tgx;
        if (wpos < k)
        {
            pKey[wpos]               = k0;
            pValue[wpos]             = v0;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k1;
            pValue[wpos]             = v1;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k2;
            pValue[wpos]             = v2;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k3;
            pValue[wpos]             = v3;
        }
    }
}


__global__ void
LAUNCH_BOUNDS()
CalculateOutput_256_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
__shared__ volatile float sKey[288 * 4];
__shared__ volatile uint32_t sValue[288 * 4];


    uint32_t pos                    = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                    = threadIdx.x & cData._warpMask;
        
    
    if (pos < batch)
    {
        float *pOutput            = pOutputBuffer + pos * width;
        uint32_t offset             = threadIdx.x >> cData._warpBits;
        volatile float* psKey     = &sKey[288 * offset];
        volatile uint32_t* psValue  = &sValue[288 * offset];

        float k0                  = -FLT_MAX;
        float k1                  = -FLT_MAX;
        float k2                  = -FLT_MAX;
        float k3                  = -FLT_MAX;
        float k4                  = -FLT_MAX;
        float k5                  = -FLT_MAX;
        float k6                  = -FLT_MAX;
        float k7                  = -FLT_MAX;
        float k8                  = -FLT_MAX;
        float k9                  = -FLT_MAX;
        float k10                 = -FLT_MAX;
        float k11                 = -FLT_MAX;
        float k12                 = -FLT_MAX;
        float k13                 = -FLT_MAX;
        float k14                 = -FLT_MAX;
        float k15                 = -FLT_MAX;
        uint32_t v0                 = 0;
        uint32_t v1                 = 0;
        uint32_t v2                 = 0;
        uint32_t v3                 = 0;
        uint32_t v4                 = 0;
        uint32_t v5                 = 0;
        uint32_t v6                 = 0;
        uint32_t v7                 = 0;
        uint32_t v8                 = 0;
        uint32_t v9                 = 0;
        uint32_t v10                = 0;
        uint32_t v11                = 0;
        uint32_t v12                = 0;
        uint32_t v13                = 0;
        uint32_t v14                = 0;
        uint32_t v15                = 0;
        
        uint32_t wpos               = tgx;
        if (wpos < width)
        {
            k0                      = pOutput[wpos];
            v0                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k1                      = pOutput[wpos];
            v1                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k2                      = pOutput[wpos];
            v2                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k3                      = pOutput[wpos];
            v3                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k4                      = pOutput[wpos];
            v4                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k5                      = pOutput[wpos];
            v5                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k6                      = pOutput[wpos];
            v6                      = wpos;
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k7                      = pOutput[wpos];
            v7                      = wpos;
        }
             
        float minValue            = -FLT_MAX;
        uint32_t rpos               = 256;
        uint32_t bufferSize         = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos           = rpos + tgx;
            float key             = -FLT_MAX;
            uint32_t value          = wpos;
            if (wpos < width)
            {
                key                 = pOutput[wpos];                
            }
            
            uint32_t count          = BALLOT(key > minValue);
            if (key > minValue)
            {
                uint32_t mask       = 0xffffffff >> (32 - tgx);
                uint32_t offset     = __popc(count & mask);
                offset             += bufferSize;
                psKey[offset]       = key;
                psValue[offset]     = value;
            }
            bufferSize             += __popc(count);

            if (bufferSize >= 256)
            {
                k8                  = psKey[tgx];
                v8                  = psValue[tgx];
                k9                  = psKey[tgx + cData._warpSize];
                v9                  = psValue[tgx + cData._warpSize];
                k10                 = psKey[tgx + 2 * cData._warpSize];
                v10                 = psValue[tgx + 2 * cData._warpSize];
                k11                 = psKey[tgx + 3 * cData._warpSize];
                v11                 = psValue[tgx + 3 * cData._warpSize];
                k12                 = psKey[tgx + 4 * cData._warpSize];
                v12                 = psValue[tgx + 4 * cData._warpSize];
                k13                 = psKey[tgx + 5 * cData._warpSize];
                v13                 = psValue[tgx + 5 * cData._warpSize];
                k14                 = psKey[tgx + 6 * cData._warpSize];
                v14                 = psValue[tgx + 6 * cData._warpSize];                
                k15                 = psKey[tgx + 7 * cData._warpSize];
                v15                 = psValue[tgx + 7 * cData._warpSize];
                bool flag;
                BITONICSORT512_512();

                minValue = SHFL(k7, cData._warpSize - 1);

                bufferSize         -= 256;
                if (tgx < bufferSize)
                {
                    psKey[tgx]      = psKey[tgx + 256];
                    psValue[tgx]    = psValue[tgx + 256];
                }
            }

            rpos                    += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 256))
        {
            k8                       = -FLT_MAX;
            k9                       = -FLT_MAX;
            k10                      = -FLT_MAX;
            k11                      = -FLT_MAX;
            k12                      = -FLT_MAX;
            k13                      = -FLT_MAX;
            k14                      = -FLT_MAX;
            k15                      = -FLT_MAX;
            v8                       = 0;
            v9                       = 0;
            v10                      = 0;
            v11                      = 0;
            v12                      = 0;
            v13                      = 0;
            v14                      = 0;
            v15                      = 0;

            if (tgx < bufferSize)
            {
                k8                   = psKey[tgx];
                v8                   = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k9                   = psKey[tgx + cData._warpSize];
                v9                   = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k10                  = psKey[tgx + 2 * cData._warpSize];
                v10                  = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k11                  = psKey[tgx + 3 * cData._warpSize];
                v11                  = psValue[tgx + 3 * cData._warpSize];
            }
            if (tgx + 4 * cData._warpSize < bufferSize)
            {
                k12                  = psKey[tgx + 4 * cData._warpSize];
                v12                  = psValue[tgx + 4 * cData._warpSize];
            }
            if (tgx + 5 * cData._warpSize < bufferSize)
            {
                k13                  = psKey[tgx + 5 * cData._warpSize];
                v13                  = psValue[tgx + 5 * cData._warpSize];
            }  
            if (tgx + 6 * cData._warpSize < bufferSize)
            {
                k14                  = psKey[tgx + 6 * cData._warpSize];
                v14                  = psValue[tgx + 6 * cData._warpSize];
            }
            if (tgx + 7 * cData._warpSize < bufferSize)
            {
                k15                  = psKey[tgx + 7 * cData._warpSize];
                v15                  = psValue[tgx + 7 * cData._warpSize];
            } 

            BITONICSORT512_512();
        }

        float* pKey                = pKeyBuffer + pos * k;
        uint32_t* pValue             = pValueBuffer + pos * k;                
        wpos                         = tgx;
        if (wpos < k)
        {
            pKey[wpos]               = k8;
            pValue[wpos]             = v8;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k9;
            pValue[wpos]             = v9;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k10;
            pValue[wpos]             = v10;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k11;
            pValue[wpos]             = v11;
        }
        if (wpos < k)
        {
            pKey[wpos]               = k12;
            pValue[wpos]             = v12;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k13;
            pValue[wpos]             = v13;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k14;
            pValue[wpos]             = v14;
        }
        wpos                        += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]               = k15;
            pValue[wpos]             = v15;
        }
    }
}



void CalculateOutput(float* pOutput, float* pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    try {
        uint32_t blocks = (batch + 3) / 4;
        if (k <= 32)
        {
            CalculateOutput_32_kernel << <blocks, 128 >> > (pOutput, pKey, pValue, batch, width, k);
        }
        else if (k <= 64)
        {
            CalculateOutput_64_kernel << <blocks, 128 >> > (pOutput, pKey, pValue, batch, width, k);
        }
        else if (k <= 128)
        {
            CalculateOutput_128_kernel << <blocks, 128 >> > (pOutput, pKey, pValue, batch, width, k);
        }
        else
        {
            CalculateOutput_256_kernel << <blocks, 128 >> > (pOutput, pKey, pValue, batch, width, k);
        }

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in CalculateOutput: " + std::string(e.what()));
    }
}


__global__ void
LAUNCH_BOUNDS()
CalculateOutput_kernel(float* pOutputKey, float* pOutputValue, float* pKeyBuffer, float* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
__shared__ volatile float sKey[160 * 4];
__shared__ volatile float sValue[160 * 4];


    uint32_t pos                    = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                    = threadIdx.x & cData._warpMask;
        
    
    if (pos < batch)
    {
        pOutputKey                 += pos * width;
        pOutputValue               += pos * width;
        uint32_t offset             = threadIdx.x >> cData._warpBits;
        volatile float* psKey     = &sKey[160 * offset];
        volatile float* psValue   = &sValue[160 * offset];

        float k0                  = -FLT_MAX;
        float k1                  = -FLT_MAX;
        float k2                  = -FLT_MAX;
        float k3                  = -FLT_MAX;
        float k4                  = -FLT_MAX;
        float k5                  = -FLT_MAX;
        float k6                  = -FLT_MAX;
        float k7                  = -FLT_MAX;
        float v0                  = 0.0f;
        float v1                  = 0.0f;
        float v2                  = 0.0f;
        float v3                  = 0.0f;
        float v4                  = 0.0f;
        float v5                  = 0.0f;
        float v6                  = 0.0f;
        float v7                  = 0.0f;

        uint32_t wpos               = tgx;
        if (wpos < width)
        {
            k0                      = pOutputKey[wpos];
            v0                      = pOutputValue[wpos];
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k1                      = pOutputKey[wpos];
            v1                      = pOutputValue[wpos];
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k2                      = pOutputKey[wpos];
            v2                      = pOutputValue[wpos];
        }
        wpos                       += cData._warpSize;
        if (wpos < width)
        {
            k3                      = pOutputKey[wpos];
            v3                      = pOutputValue[wpos];
        }
  
        float minValue            = -FLT_MAX;
        uint32_t rpos               = 128;
        uint32_t bufferSize         = 0;
        float key1, key2;
        float value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos           = rpos + tgx;
            float key             = -FLT_MAX;
            float value           = 0.0f;
            if (wpos < width)
            {
                key                 = pOutputKey[wpos];
                value               = pOutputValue[wpos];              
            }
            
            uint32_t count          = BALLOT(key > minValue);
            if (key > minValue)
            {
                uint32_t mask       = 0xffffffff >> (32 - tgx);
                uint32_t offset     = __popc(count & mask);
                offset             += bufferSize;
                psKey[offset]       = key;
                psValue[offset]     = value;
            }
            bufferSize             += __popc(count);

            if (bufferSize >= 128)
            {
                k4                  = psKey[tgx];
                v4                  = psValue[tgx];
                k5                  = psKey[tgx + cData._warpSize];
                v5                  = psValue[tgx + cData._warpSize];
                k6                  = psKey[tgx + 2 * cData._warpSize];
                v6                  = psValue[tgx + 2 * cData._warpSize];
                k7                  = psKey[tgx + 3 * cData._warpSize];
                v7                  = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                bufferSize         -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx]      = psKey[tgx + 128];
                    psValue[tgx]    = psValue[tgx + 128];
                }
            }

            rpos                   += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 128))
        {
            k4                      = -FLT_MAX;
            k5                      = -FLT_MAX;
            k6                      = -FLT_MAX;
            k7                      = -FLT_MAX;
            v4                      = 0;
            v5                      = 0;
            v6                      = 0;
            v7                      = 0;
            
            if (tgx < bufferSize)
            {
                k4                  = psKey[tgx];
                v4                  = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k5                  = psKey[tgx + cData._warpSize];
                v5                  = psValue[tgx + cData._warpSize];
            }
            if (tgx  + 2 * cData._warpSize < bufferSize)
            {
                k6                  = psKey[tgx + 2 * cData._warpSize];
                v6                  = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {          
                k7                  = psKey[tgx + 3 * cData._warpSize];
                v7                  = psValue[tgx + 3 * cData._warpSize];
            }
            BITONICSORT256_256();
        }

        float* pKey               = pKeyBuffer + pos * k;
        float* pValue             = pValueBuffer + pos * k;                
        wpos                        = tgx;
        if (wpos < k)
        {
            pKey[wpos]              = k0;
            pValue[wpos]            = v0;
        }
        wpos                       += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]              = k1;
            pValue[wpos]            = v1;
        }
        wpos                       += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]              = k2;
            pValue[wpos]            = v2;
        }
        wpos                       += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]              = k3;
            pValue[wpos]            = v3;
        }
    }
}

void CalculateOutput(float* pOutputKey, float* pOutputValue, float* pKey, float* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    try {
        uint32_t blocks = (batch + 3) / 4;
        CalculateOutput_kernel << <blocks, 128 >> > (pOutputKey, pOutputValue, pKey, pValue, batch, width, k);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in CalculateOutput: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
CalculateOutput_kernel(float* pOutputKey, uint32_t* pOutputValue, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
__shared__ volatile float sKey[160 * 4];
__shared__ volatile uint32_t sValue[160 * 4];
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                        = threadIdx.x & cData._warpMask;        
    
    if (pos < batch)
    {
        pOutputKey                     += pos * width;
        pOutputValue                   += pos * width;
        uint32_t offset                 = threadIdx.x >> cData._warpBits;
        volatile float* psKey         = &sKey[160 * offset];
        volatile uint32_t* psValue      = &sValue[160 * offset];

        float k0                      = -FLT_MAX;
        float k1                      = -FLT_MAX;
        float k2                      = -FLT_MAX;
        float k3                      = -FLT_MAX;
        float k4                      = -FLT_MAX;
        float k5                      = -FLT_MAX;
        float k6                      = -FLT_MAX;
        float k7                      = -FLT_MAX;
        uint32_t v0                     = 0;
        uint32_t v1                     = 0;
        uint32_t v2                     = 0;
        uint32_t v3                     = 0;
        uint32_t v4                     = 0;
        uint32_t v5                     = 0;
        uint32_t v6                     = 0;
        uint32_t v7                     = 0;

        uint32_t wpos                   = tgx;
        if (wpos < width)
        {
            k0                              = pOutputKey[wpos];
            v0                              = pOutputValue[wpos];
        }
        wpos                               += cData._warpSize;
        if (wpos < width)
        {
            k1                              = pOutputKey[wpos];
            v1                              = pOutputValue[wpos];
        }
        wpos                               += cData._warpSize;
        if (wpos < width)
        {
            k2                              = pOutputKey[wpos];
            v2                              = pOutputValue[wpos];
        }
        wpos                               += cData._warpSize;
        if (wpos < width)
        {
            k3                              = pOutputKey[wpos];
            v3                              = pOutputValue[wpos];
        }
     
        float minValue                    = -FLT_MAX;
        uint32_t rpos                       = 128;
        uint32_t bufferSize                 = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos                   = rpos + tgx;
            float key                     = -FLT_MAX;
            float value                   = 0.0f;
            if (wpos < width)
            {
                key                         = pOutputKey[wpos];
                value                       = pOutputValue[wpos];              
            }
            
            uint32_t count                  = BALLOT(key > minValue);
            if (key > minValue)
            {
                uint32_t mask               = 0xffffffff >> (32 - tgx);
                uint32_t offset             = __popc(count & mask);
                offset                     += bufferSize;
                psKey[offset]               = key;
                psValue[offset]             = value;
            }
            bufferSize                     += __popc(count);

            if (bufferSize >= 128)
            {
                k4                          = psKey[tgx];
                v4                          = psValue[tgx];
                k5                          = psKey[tgx + cData._warpSize];
                v5                          = psValue[tgx + cData._warpSize];
                k6                          = psKey[tgx + 2 * cData._warpSize];
                v6                          = psValue[tgx + 2 * cData._warpSize];
                k7                          = psKey[tgx + 3 * cData._warpSize];
                v7                          = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                bufferSize                 -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx]              = psKey[tgx + 128];
                    psValue[tgx]            = psValue[tgx + 128];
                }
            }

            rpos                           += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 128))
        {
            k4                              = -FLT_MAX;
            k5                              = -FLT_MAX;
            k6                              = -FLT_MAX;
            k7                              = -FLT_MAX;
            v4                              = 0;
            v5                              = 0;
            v6                              = 0;
            v7                              = 0;

            if (tgx < bufferSize)
            {
                k4                          = psKey[tgx];
                v4                          = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k5                          = psKey[tgx + cData._warpSize];
                v5                          = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k6                          = psKey[tgx + 2 * cData._warpSize];
                v6                          = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k7                          = psKey[tgx + 3 * cData._warpSize];
                v7                          = psValue[tgx + 3 * cData._warpSize];
            }

            BITONICSORT256_256();
        }

        float* pKey                       = pKeyBuffer + pos * k;
        uint32_t* pValue                    = pValueBuffer + pos * k;                
        wpos                                = tgx;
        if (wpos < k)
        {
            pKey[wpos]                      = k0;
            pValue[wpos]                    = v0;
        }
        wpos                               += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]                      = k1;
            pValue[wpos]                    = v1;
        }
        wpos                               += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]                      = k2;
            pValue[wpos]                    = v2;
        }
        wpos                               += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos]                      = k3;
            pValue[wpos]                    = v3;
        }
    }
}


void CalculateOutput(float* pOutputKey, uint32_t* pOutputValue, float* pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    try {
        uint32_t blocks = (batch + 3) / 4;
        CalculateOutput_kernel << <blocks, 128 >> > (pOutputKey, pOutputValue, pKey, pValue, batch, width, k);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in CalculateOutput: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kNormalizeWeights_kernel(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight)
{
    uint32_t pos                            = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        float r2                          = 0.0f;
        float* pEnd                       = pWeight + outputStride * inputStride; 
        pWeight                            += pos;
        float* p                          = pWeight;
        
        while (p < pEnd)
        {
            float x                       = *p;
            r2                             += x * x;
            p                              += outputStride;
        } 
        
        if (r2 > norm * norm)
        {
            norm                           *= rsqrt(r2);
            p                               = pWeight;
            while (p < pEnd)
            {
                *p                         *= norm;
                p                          += outputStride;
            }             
        }
    }

}

void kNormalizeWeights(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight)
{
    try {
        uint32_t blocks = (outputStride + 127) / 128;
        kNormalizeWeights_kernel << <blocks, 128 >> > (norm, outputStride, inputStride, pWeight);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kNormalizeWeights: " + std::string(e.what()));
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateWeightMagnitudes_kernel(uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude)
{
    uint32_t pos                            = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        float r2                          = 0.0f;
        float* pEnd                       = pWeight + outputStride * inputStride; 
        pWeight                            += pos;
        float* p                          = pWeight;
        
        while (p < pEnd)
        {
            float x                       = *p;
            r2                             += x * x;
            p                              += outputStride;
        } 
        
        pMagnitude[pos]                     = r2;
    }

}

void kCalculateWeightMagnitudes(uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude)
{
    try {
        uint32_t blocks = (outputStride + 127) / 128;
        kCalculateWeightMagnitudes_kernel << <blocks, 128 >> > (outputStride, inputStride, pWeight, pMagnitude);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateWeightMagnitudes: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kNormalizeWeightMagnitudes_kernel(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude)
{
    uint32_t pos                            = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        float r2                          = pMagnitude[pos];
        float* pEnd                       = pWeight + outputStride * inputStride; 
        pWeight                            += pos;
        float* p                          = pWeight;
        
        if (r2 > norm * norm)
        {
            norm                           *= rsqrt(r2);
            p                               = pWeight;
            while (p < pEnd)
            {
                *p                         *= norm;
                p                          += outputStride;
            }             
        }
    }

}

void kNormalizeWeightMagnitudes(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude)
{
    try {
        uint32_t blocks = (outputStride + 127) / 128;
        kNormalizeWeightMagnitudes_kernel << <blocks, 128 >> > (norm, outputStride, inputStride, pWeight, pMagnitude);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kNormalizeWeightMagnitudes: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateScaledBiasedDropout_kernel(float* pUnit, float* pRandom, float p, float target, float a, float b, size_t size)
{
    uint64_t pos                            = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float r                           = pRandom[pos];
        pUnit[pos]                          = (r < p) ? target : a * pUnit[pos] + b;
    }
}

void kCalculateScaledBiasedDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target, float a, float b)
{
    try {
        curandGenerateUniform(getGpu()._RNG, pRandom, batch * stride);
        unsigned long blocks = CalculateBlocks(batch * stride);
        kCalculateScaledBiasedDropout_kernel << <blocks, getGpu()._threadsPerBlock >> > (pUnit, pRandom, p, a * target + b, a, b, batch * stride);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateScaledBiasedDropout: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateDropout_kernel(float* pUnit, float* pRandom, float p, float scale, float target, size_t size)
{
    uint64_t pos                            = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float r                           = pRandom[pos];
        pUnit[pos]                          = (r < p) ? target : scale * pUnit[pos];
    }
}

void kCalculateDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target)
{
    try {
        if (batch == 0 || stride == 0) {
            return;
        }

        curandGenerateUniform(getGpu()._RNG, pRandom, batch * stride);
        unsigned long blocks = CalculateBlocks(batch * stride);
        float scale = (target == 0.0f) ? 1.0f / (1.0f - p) : 1.0f;

        kCalculateDropout_kernel << <blocks, getGpu()._threadsPerBlock >> > (pUnit, pRandom, p, scale, target, batch * stride);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateDropout: " + std::string(e.what()));
    }
}

__global__ void 
LAUNCH_BOUNDS()
kCalculateMaxout_kernel(float* pSrc, size_t size, float* pDst)
{
    uint64_t pos                        = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float s = pSrc[pos];
        float d = pDst[pos];
        if (s > d)
            pDst[pos]                   = s;
    }
}

void kCalculateMaxout(float* pSrc, size_t size, float* pDst)
{
    try {
        if (size == 0)
            return;

        unsigned long blocks = CalculateBlocks(size);
        kCalculateMaxout_kernel << <blocks, getGpu()._threadsPerBlock >> > (pSrc, size, pDst);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateMaxout: " + std::string(e.what()));
    }
}

__global__ void 
LAUNCH_BOUNDS()
kCalculateCosine_kernel(float* pVector1, float* pVector2, uint32_t stride, float* pDPOut, float* pAOut, float* pBOut, uint32_t outStride)
{
__shared__ float sDP[64];
__shared__ float sA[64];
__shared__ float sB[64];


    pVector1               += blockIdx.x * stride + threadIdx.x;
    pVector2               += blockIdx.x * stride + threadIdx.x;
    pDPOut                 += blockIdx.x * outStride;
    pAOut                  += blockIdx.x * outStride;
    pBOut                  += blockIdx.x * outStride;    
    uint32_t pos            = threadIdx.x;
    float dp              = (float)0;
    float al              = (float)0;
    float bl              = (float)0;
    
    while (pos < stride)
    {
        float a           = *pVector1;
        float b           = *pVector2;
        dp                 += a * b;
        al                 += a * a;
        bl                 += b * b;
        pVector1           += blockDim.x;
        pVector2           += blockDim.x;
        pos                += blockDim.x;
    }
    
    
    uint32_t tgx            = threadIdx.x & cData._warpMask;
    dp                     += SHFL(dp, tgx ^ 1);
    al                     += SHFL(al, tgx ^ 1);
    bl                     += SHFL(bl, tgx ^ 1);
    dp                     += SHFL(dp, tgx ^ 2);
    al                     += SHFL(al, tgx ^ 2);
    bl                     += SHFL(bl, tgx ^ 2);
    dp                     += SHFL(dp, tgx ^ 4);
    al                     += SHFL(al, tgx ^ 4);
    bl                     += SHFL(bl, tgx ^ 4);
    dp                     += SHFL(dp, tgx ^ 8);
    al                     += SHFL(al, tgx ^ 8);
    bl                     += SHFL(bl, tgx ^ 8);    
    dp                     += SHFL(dp, tgx ^ 16); 
    al                     += SHFL(al, tgx ^ 16);
    bl                     += SHFL(bl, tgx ^ 16);
    if (tgx == 0)           
    {
        uint32_t index      = threadIdx.x >> cData._warpBits;
        sDP[index]          = dp;
        sA[index]           = al;
        sB[index]           = bl;
    }
    __syncthreads();
    
    if (threadIdx.x < cData._warpSize)
    {
        uint32_t limit      = (blockDim.x + cData._warpSize -1) >> cData._warpBits;
        al                  = (threadIdx.x < limit) ? sA[threadIdx.x]     : (float)0;      
        bl                  = (threadIdx.x < limit) ? sB[threadIdx.x]     : (float)0; 
        dp                  = (threadIdx.x < limit) ? sDP[threadIdx.x]    : (float)0;
        dp                 += SHFL(dp, tgx ^ 1);
        al                 += SHFL(al, tgx ^ 1);
        bl                 += SHFL(bl, tgx ^ 1);
        dp                 += SHFL(dp, tgx ^ 2);
        al                 += SHFL(al, tgx ^ 2);
        bl                 += SHFL(bl, tgx ^ 2);
        dp                 += SHFL(dp, tgx ^ 4);
        al                 += SHFL(al, tgx ^ 4);
        bl                 += SHFL(bl, tgx ^ 4);
        dp                 += SHFL(dp, tgx ^ 8);
        al                 += SHFL(al, tgx ^ 8);
        bl                 += SHFL(bl, tgx ^ 8);    
        dp                 += SHFL(dp, tgx ^ 16); 
        al                 += SHFL(al, tgx ^ 16);
        bl                 += SHFL(bl, tgx ^ 16);        
                         
        
        if (threadIdx.x == 0)
        {
            al              = sqrt(al) + (float)1.0e-08;
            bl              = sqrt(bl) + (float)1.0e-08;
            dp             /= al * bl;
            *pAOut          = al;
            *pBOut          = bl;
            *pDPOut         = dp;
        }
    }
} 

void kCalculateCosine(float* pVector1In, float* pVector2In, uint32_t batch, uint32_t stride, float* pDPOut, float* pAOut, float* pBOut, uint32_t outStride)
{
    try {
        if (batch == 0 || stride == 0)
            return;

        unsigned long threads = max(32, min(stride, getGpu()._threadsPerBlock));
        kCalculateCosine_kernel << <batch, threads >> > (pVector1In, pVector2In, stride, pDPOut, pAOut, pBOut, outStride);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateCosine: " + std::string(e.what()));
    }
}

__global__ void 
LAUNCH_BOUNDS()
kCalculateDotProduct_kernel(float* pVector1In, float* pVector2In, uint32_t strideIn, float* pDPOut, uint32_t strideOut)
{
__shared__ float sDP[32];

    pVector1In             += blockIdx.x * strideIn + threadIdx.x;
    pVector2In             += blockIdx.x * strideIn + threadIdx.x;
    pDPOut                 += blockIdx.x * strideOut;
    uint32_t pos            = threadIdx.x;
    float dp              = (float)0;

    
    while (pos < strideIn)
    {
        float a           = *pVector1In;
        float b           = *pVector2In;
        dp                 += a * b;
        pVector1In         += blockDim.x;
        pVector2In         += blockDim.x;
        pos                += blockDim.x;
    }
    
    
    REDUCE(dp)
    uint32_t tgx            = threadIdx.x & cData._warpMask;    
    if (tgx == 0)           
    {
        uint32_t index      = threadIdx.x >> cData._warpBits;
        sDP[index]          = dp;
    }
    __syncthreads();
    
    if (threadIdx.x < cData._warpSize)
    {
        uint32_t limit      = (blockDim.x + cData._warpSize -1) >> cData._warpBits;
        dp                  = (threadIdx.x < limit) ? sDP[threadIdx.x]    : (float)0;
        REDUCE(dp)                 
        
        if (threadIdx.x == 0)
        {
            *pDPOut         = dp;      
        }
    }
} 

void kCalculateDotProduct(float* pVector1In, float* pVector2In, uint32_t batch, uint32_t strideIn, float* pDPOut, uint32_t strideOut)
{
    try {
        if (batch == 0 || strideIn == 0)
            return;

        unsigned long threads = max(32, min(strideIn, getGpu()._threadsPerBlock));
        kCalculateDotProduct_kernel << <batch, threads >> > (pVector1In, pVector2In, strideIn, pDPOut, strideOut);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateDotProduct: " + std::string(e.what()));
    }
}

#include "cub/util_allocator.cuh"
#include "cub/device/device_radix_sort.cuh"

template<typename KeyType, typename ValueType> size_t kInitSort(uint32_t items, GpuBuffer<KeyType>* pbKey, GpuBuffer<ValueType>* pbValue)
{
    uint32_t itemStride                     = ((items + 511) >> 9) << 9;
    size_t tempBytes;
    cub::DoubleBuffer<KeyType> d_keys(pbKey->_pDevData, pbKey->_pDevData + itemStride);
    cub::DoubleBuffer<ValueType> d_values(pbValue->_pDevData, pbValue->_pDevData + itemStride);
    cub::DeviceRadixSort::SortPairs(NULL, tempBytes, d_keys, d_values, items);
    return tempBytes;
}

template<typename KeyType, typename ValueType> bool kSort(uint32_t items, KeyType* pKey0, KeyType* pKey1, ValueType* pValue0, ValueType* pValue1, char* pTemp, size_t tempBytes)
{
    cub::DoubleBuffer<KeyType>  d_keys(pKey0, pKey1);
    cub::DoubleBuffer<ValueType> d_values(pValue0, pValue1);
    cub::DeviceRadixSort::SortPairs(pTemp, tempBytes, d_keys, d_values, items);
    return true;   
}

__global__ void
LAUNCH_BOUNDS()
kAddScaleBuffers_kernel(float* pDst, float* pSrc, float scale, uint64_t size)
{
    uint64_t pos                            = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        *(pDst + pos)                      += *(pSrc + pos) * scale;
}

void kAddScaleBuffers(float* pDst, float* pSrc, float scale, uint64_t size)
{
    try {
        if (size == 0)
            return;

        uint32_t blocks = CalculateBlocks(size);
        kAddScaleBuffers_kernel << <blocks, getGpu()._threadsPerBlock >> > (pDst, pSrc, scale, size);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAddScaleBuffers: " + std::string(e.what()));
    }
}

__global__ void LAUNCH_BOUNDS() kAddBuffers_kernel(float* pDst, float* pSrc, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        *(pDst + pos) += *(pSrc + pos);
}

void kAddBuffers(float* pDst, float* pSrc, uint64_t size, cudaStream_t stream)
{
    if (size == 0)
        return;

    try {
        uint32_t blocks = CalculateBlocks(size);
        kAddBuffers_kernel << <blocks, getGpu()._threadsPerBlock, 0, stream >> > (pDst, pSrc, size);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAddBuffers: " + std::string(e.what()));
    }
}

__global__ void LAUNCH_BOUNDS() kAddBuffers2D_kernel(float* pDst, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width)
{
    uint64_t yOffset = blockIdx.y * blockDim.x + threadIdx.x;
    if (yOffset < width)
    {
        uint64_t dpos = blockIdx.x * dpitch + yOffset;
        uint64_t spos = blockIdx.x * spitch + yOffset;
        pDst[dpos] += pSrc[spos];
    }
}

void kAddBuffers2D(float* pDst, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream)
{
    if ((height == 0) || (width == 0))
        return;

    try {
        dim3 grid(height, (width + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kAddBuffers2D_kernel << <grid, getGpu()._threadsPerBlock, 0, stream >> > (pDst, dpitch, pSrc, spitch, width);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAddBuffers2D: " + std::string(e.what()));
    }
}

__global__ void LAUNCH_BOUNDS() kCopy2D_kernel(float* pDst, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width)
{
    uint64_t yOffset = blockIdx.y * blockDim.x + threadIdx.x;
    if (yOffset < width)
    {
        uint64_t dpos = blockIdx.x * dpitch + yOffset;
        uint64_t spos = blockIdx.x * spitch + yOffset;
        pDst[dpos] = pSrc[spos];
    }
}

void kCopy2D(float* pDst, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream)
{
    if ((height == 0) || (width == 0))
        return;

    try {
        dim3 grid(height, (width + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCopy2D_kernel << <grid, getGpu()._threadsPerBlock, 0, stream >> > (pDst, dpitch, pSrc, spitch, width);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCopy2D: " + std::string(e.what()));
    }
}

template size_t kInitSort<float, float>(uint32_t, GpuBuffer<float>*, GpuBuffer<float>*);
template size_t kInitSort<uint32_t, float>(uint32_t, GpuBuffer<uint32_t>*, GpuBuffer<float>*);
template size_t kInitSort<float, uint32_t>(uint32_t, GpuBuffer<float>*, GpuBuffer<uint32_t>*);
template size_t kInitSort<uint32_t, uint32_t>(uint32_t, GpuBuffer<uint32_t>*, GpuBuffer<uint32_t>*);

template bool kSort<float, float>(uint32_t, float*, float*, float*, float*, char*, size_t);
template bool kSort<float, uint32_t>(uint32_t, float*, float*, uint32_t*, uint32_t*, char*, size_t);
template bool kSort<uint32_t, float>(uint32_t, uint32_t*, uint32_t*, float*, float*, char*, size_t);
template bool kSort<uint32_t, uint32_t>(uint32_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, char*, size_t);

#define EXPLICITLY_INSTANTIATE_KERNELS(T) \
template void kLoadSparseAnalogDenoisedInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*); \
template void kLoadIndexedSparseAnalogDenoisedInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*); \
template void kLoadSparseAnalogInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*); \
template void kLoadIndexedSparseAnalogInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*); \
template void kCalculateSparseAnalogZ<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float); \
template void kCalculateIndexedSparseAnalogZ<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float); \
template void kCalculateSparseAnalogDenoisedZ<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float*, float); \
template void kCalculateIndexedSparseAnalogDenoisedZ<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float*, float); \
template void kCalculateSparseTransposedAnalogMatrix<T>(uint32_t, uint32_t, uint64_t*, uint64_t*, uint32_t*, float*, T*, uint32_t*, uint32_t*, float*); \
template void kCalculateIndexedSparseTransposedAnalogMatrix<T>(uint32_t, uint32_t, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, uint32_t*, uint32_t*, float*); \
template void kCalculateSparseTransposedAnalogDenoisedMatrix<T>(uint32_t, uint32_t, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, uint32_t*, uint32_t*, float*); \
template void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix<T>(uint32_t, uint32_t, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, uint32_t*, uint32_t*, float*); \
template void kLoadInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, T*); \
template void kLoadIndexedInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*);

EXPLICITLY_INSTANTIATE_KERNELS(float)
EXPLICITLY_INSTANTIATE_KERNELS(double)
EXPLICITLY_INSTANTIATE_KERNELS(unsigned char)
EXPLICITLY_INSTANTIATE_KERNELS(char)
EXPLICITLY_INSTANTIATE_KERNELS(uint32_t)
EXPLICITLY_INSTANTIATE_KERNELS(uint64_t)
EXPLICITLY_INSTANTIATE_KERNELS(int32_t)
EXPLICITLY_INSTANTIATE_KERNELS(int64_t)

#endif