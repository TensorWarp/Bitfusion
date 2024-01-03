#include "gpuTypes.h"
#include "types.h"
#include "kernels.cuh"
#include <limits>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <nccl.h>
#include <mpi.h>


namespace cg = cooperative_groups;

static __constant__ GpuData cData;

void SetsparseGpuData(const GpuData& gpuData)
{
    cudaError_t status = cudaMemcpyToSymbol(cData, &gpuData, sizeof(GpuData));
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol: SetKLossGpuData copy to cData failed: %s\n", cudaGetErrorString(status));
        throw std::runtime_error("cudaMemcpyToSymbol failed");
    }
}

void GetsparseGpuData(GpuData& gpuData)
{
    cudaError_t status = cudaMemcpyFromSymbol(&gpuData, cData, sizeof(GpuData));
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyFromSymbol: GetKLossGpuData copy from cData failed: %s\n", cudaGetErrorString(status));
        throw std::runtime_error("cudaMemcpyFromSymbol failed");
    }
}

__global__ void __launch_bounds__(256, 4) invokeSparseRawL1Error_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint64_t stride, uint64_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    __shared__ float sharedError[14 * 32];

    cg::thread_block block = cg::this_thread_block();

    while (pos < size) {
        float w = 1.0f;
        uint64_t dpos;

        if (pos < size) {
            if (pDataWeight != nullptr) {
                dpos = (pos / stride) + position;
                dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
                w *= __ldg(&pDataWeight[dpos]);
            }

            float a = __ldg(&pUnit[pos]);
            sharedError[threadIdx.x] = w * fabsf(a);
        }
        else {
            sharedError[threadIdx.x] = 0.0f;
        }

        block.sync();

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sharedError[threadIdx.x] += __shfl_down_sync(0xFFFFFFFF, sharedError[threadIdx.x], offset);
        }

        if (threadIdx.x % warpSize == 0) {
            REDUCEERROR(sharedError[threadIdx.x]);
        }

        pos += static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    }
}

__global__ void __launch_bounds__(256, 4) invokeSparseNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = static_cast<uint64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    __shared__ float sharedError[14 * 32];

    if (pos < batch) {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x % warpSize);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;
        float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;

        int sharedIndex = threadIdx.x % warpSize;

        cg::thread_block block = cg::this_thread_block();

#pragma unroll
        for (int i = 0; i < 14; ++i) {
            if (pos1 < end) {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = __ldg(&pUnit[pos2]);
                sharedError[sharedIndex * 14 + i] = w * (fabsf(a - 1.0f) - fabsf(a));
                pos1 += warpSize;
            }
            else {
                sharedError[sharedIndex * 14 + i] = 0.0f;
            }
        }

        block.sync();

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            for (int i = 0; i < 14; ++i) {
                sharedError[sharedIndex * 14 + i] += __shfl_down_sync(0xFFFFFFFF, sharedError[sharedIndex * 14 + i], offset);
            }
        }

        if ((threadIdx.x % warpSize) == 0) {
            for (int i = 0; i < 14; ++i) {
                REDUCEERROR(sharedError[i]);
            }
        }
    }
}

__global__ void __launch_bounds__(256, 4) invokeSparseOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            error += w * fabsf(a - (float)1.0);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));
    uint32_t blocks;
    dim3 gridSize;
    dim3 blockSize;

    if (bSparseIgnoreZero)
    {
        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseOnlyNonZeroL1Error_kernel<T>), gridSize, blockSize, args);
    }
    else
    {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        blocks = CalculateBlocks(size);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pUnit, &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL1Error_kernel<T>), gridSize, blockSize, args);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseNonZeroL1Error_kernel<T>), gridSize, blockSize, args2);
    }

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void __launch_bounds__(256, 4) invokeIndexedSparseNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        float w[14] = { 1.0f };
        uint64_t dpos[14];
        uint64_t offset[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                uint32_t shuffled_dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
                dpos[i] = __ldg(&pIndex[shuffled_dpos]);
                offset[i] = pos * stride;
                float weight = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos[i]]) : 1.0f;

                uint64_t pos1 = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                uint64_t end = pSparseEnd[dpos[i]];

                while (pos1 < end)
                {
                    uint64_t pos2 = offset[i] + pSparseIndex[pos1];
                    float a = __ldg(&pUnit[pos2]);
                    error[i] += w[i] * (fabsf(a - 1.0f) - fabsf(a));
                    pos1 += cData._warpSize;
                }

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

__global__ void __launch_bounds__(256, 4) invokeIndexedSparseOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        float w[14] = { 1.0f };
        uint64_t dpos[14];
        uint64_t offset[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                uint32_t shuffled_dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
                dpos[i] = __ldg(&pIndex[shuffled_dpos]);
                offset[i] = pos * stride;
                float weight = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos[i]]) : 1.0f;

                uint64_t pos1 = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                uint64_t end = pSparseEnd[dpos[i]];

                while (pos1 < end)
                {
                    uint64_t pos2 = offset[i] + pSparseIndex[pos1];
                    float a = __ldg(&pUnit[pos2]);
                    error[i] += w[i] * fabsf(a - 1.0f);
                    pos1 += cData._warpSize;
                }

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<typename T>
float invokeIndexedSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks = CalculateBlocks(local_size);
    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        if (bSparseIgnoreZero) {
            void* args[] = { &position, &batch, &stride, pLocalUnit[i], pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseOnlyNonZeroL1Error_kernel<T>), gridSize, blockSize, args, 0, streams[i]);
        }
        else {
            void* args1[] = { &position, pDataWeight, pLocalUnit[i], &stride, &local_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL1Error_kernel<T>), gridSize, blockSize, args1, 0, streams[i]);

            void* args2[] = { &position, &batch, &stride, pLocalUnit[i], pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseNonZeroL1Error_kernel<T>), gridSize, blockSize, args2, 0, streams[i]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            error += w * fabsf(a - t);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            error += w * (fabsf(a - t) - fabsf(a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}


template<>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error += w * fabsf(a - t);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}


template<>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error += w * (fabsf(a - t) - fabsf(a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error += w * fabsf(a - t);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error += w * (fabsf(a - t) - fabsf(a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks;
    dim3 gridSize;
    dim3 blockSize;

    if (bSparseIgnoreZero)
    {
        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pLocalUnit[local_rank], pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseAnalogOnlyNonZeroL1Error_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);
    }
    else
    {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        blocks = CalculateBlocks(size);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pLocalUnit[local_rank], &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL1Error_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pLocalUnit[local_rank], pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseAnalogNonZeroL1Error_kernel<T>), gridSize, blockSize, args2, 0, streams[local_rank]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            error += w * fabsf(a - t);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            error += w * (fabsf(a - t) - fabsf(a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<>
__global__ void __launch_bounds__(256, 4) invokeIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error += w * fabsf(a - t);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}


template<>
__global__ void __launch_bounds__(256, 4) invokeIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error += w * (fabsf(a - t) - fabsf(a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<>
__global__ void __launch_bounds__(256, 4) invokeIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error += w * fabsf(a - t);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<>
__global__ void __launch_bounds__(256, 4) invokeIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        float w[14] = { 1.0f };
        uint32_t dpos[14];
        uint64_t offset[14];
        char sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                uint32_t shuffled_dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
                dpos[i] = __ldg(&pIndex[shuffled_dpos]);
                offset[i] = pos * stride;
                float weight = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos[i]]) : 1.0f;

                uint64_t pos1 = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                uint64_t end = pSparseEnd[dpos[i]];

                while (pos1 < end)
                {
                    uint64_t pos2 = offset[i] + pSparseIndex[pos1];
                    float a = __ldg(&pUnit[pos2]);
                    sparseData[i] = static_cast<char>(__ldg(&pSparseData[pos1])) * (1.0f / 128.0f);
                    error[i] += w[i] * (fabsf(a - sparseData[i]) - fabsf(a));
                    pos1 += cData._warpSize;
                }

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<typename T>
float invokeIndexedSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks;
    dim3 gridSize;
    dim3 blockSize;

    if (bSparseIgnoreZero)
    {
        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pLocalUnit[local_rank], pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseAnalogOnlyNonZeroL1Error_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);
    }
    else
    {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        blocks = CalculateBlocks(size);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pLocalUnit[local_rank], &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL1Error_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pLocalUnit[local_rank], pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseAnalogNonZeroL1Error_kernel<T>), gridSize, blockSize, args2, 0, streams[local_rank]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void __launch_bounds__(256, 4) invokeSparseRawL2Error_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    float error = (float)0.0;
    if (pos < size)
    {
        float w = (float)0.5;
        if (pDataWeight != nullptr)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }
        float a = pUnit[pos];
        error = w * a * a;
    }

    REDUCEERROR(error)
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                error[i] += w[i] * ((a - 1.0f) * (a - 1.0f));
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                error[i] += w[i] * ((a - 1.0f) * (a - 1.0f) - a * a);
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}


template<typename T>
float invokeSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks;
    dim3 gridSize;
    dim3 blockSize;

    if (bSparseIgnoreZero)
    {
        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pLocalUnit[local_rank], pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseOnlyNonZeroL2Error_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);
    }
    else
    {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        blocks = CalculateBlocks(size);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pLocalUnit[local_rank], &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL2Error_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pLocalUnit[local_rank], pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseNonZeroL2Error_kernel<T>), gridSize, blockSize, args2, 0, streams[local_rank]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        T sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = pSparseData[pos1[i]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                T t = sparseData[i];
                error[i] += w[i] * ((a - t) * (a - t));
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        T sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = pSparseData[pos1[i]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                T t = sparseData[i];
                error[i] += w[i] * ((a - t) * (a - t) - a * a);
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        float sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = static_cast<float>(pSparseData[pos1[i]]) * (1.0f / 256.0f);

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                float t = sparseData[i];
                error[i] += w[i] * ((a - t) * (a - t));
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        float sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = static_cast<float>(pSparseData[pos1[i]]) * (1.0f / 256.0f);

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                float t = sparseData[i];
                error[i] += w[i] * ((a - t) * (a - t));
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        float sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = static_cast<float>(pSparseData[pos1[i]]) * (1.0f / 128.0f);

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                float t = sparseData[i];
                error[i] += w[i] * ((a - t) * (a - t));
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        float sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = static_cast<float>(pSparseData[pos1[i]]) * (1.0f / 128.0f);

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                float t = sparseData[i];
                error[i] += w[i] * ((a - t) * (a - t) - a * a);
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}


template<typename T>
float invokeSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint32_t blocks;
    dim3 gridSize;
    dim3 blockSize;

    if (bSparseIgnoreZero)
    {
        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseAnalogOnlyNonZeroL2Error_kernel<T>), gridSize, blockSize, args);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }
    else
    {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        blocks = CalculateBlocks(size);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pUnit, &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL2Error_kernel<T>), gridSize, blockSize, args);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseAnalogNonZeroL2Error_kernel<T>), gridSize, blockSize, args2);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[batch * stride * i], pUnit, static_cast<unsigned long long>(batch) * stride * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                error[i] += w[i] * ((a - 1.0f) * (a - 1.0f));
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                error[i] += w[i] * ((a - 1.0f) * (a - 1.0f) - a * a);
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}


template<typename T>
float invokeIndexedSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    if (batch == 0 || stride == 0)
        return 0.0f;

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint32_t blocks;
    dim3 gridSize;
    dim3 blockSize;

    if (bSparseIgnoreZero)
    {
        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseOnlyNonZeroL2Error_kernel<T>), gridSize, blockSize, args);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }
    else
    {
        uint64_t size = static_cast<uint64_t>(batch) * stride;
        blocks = CalculateBlocks(size);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pUnit, &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL2Error_kernel<T>), gridSize, blockSize, args);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseNonZeroL2Error_kernel<T>), gridSize, blockSize, args2);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[batch * stride * i], pUnit, static_cast<unsigned long long>(batch) * stride * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        T sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = pSparseData[pos1[i]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                T t = sparseData[i];
                error[i] += w[i] * ((a - t) * (a - t));
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        T sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = pSparseData[pos1[i]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                T t = sparseData[i];
                error[i] += w[i] * ((a - t) * (a - t) - a * a);
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        unsigned char sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = pSparseData[pos1[i]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                float t = (float)sparseData[i] * (1.0f / 256.0f);
                error[i] += w[i] * ((a - t) * (a - t));
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        unsigned char sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = pSparseData[pos1[i]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                float t = (float)sparseData[i] * (1.0f / 256.0f);
                error[i] += w[i] * ((a - t) * (a - t) - a * a);
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        char sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = pSparseData[pos1[i]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                float t = (float)sparseData[i] * (1.0f / 128.0f);
                error[i] += w[i] * ((a - t) * (a - t));
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        uint64_t pos1[14];
        uint64_t end[14];
        float w[14];
        uint64_t offset[14];
        float unit[14];
        char sparseData[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                pos1[i] = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                end[i] = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                offset[i] = pos * stride;
                unit[i] = pUnit[offset[i] + pSparseIndex[pos1[i]]];
                sparseData[i] = pSparseData[pos1[i]];

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            while (pos1[i] < end[i])
            {
                float a = unit[i];
                float t = (float)sparseData[i] * (1.0f / 128.0f);
                error[i] += w[i] * ((a - t) * (a - t) - a * a);
                pos1[i] += cData._warpSize;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}


template<typename T>
float invokeIndexedSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    if (batch == 0 || stride == 0)
        return 0.0f;

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint32_t blocks;
    dim3 gridSize;
    dim3 blockSize;

    if (bSparseIgnoreZero)
    {
        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseAnalogOnlyNonZeroL2Error_kernel<T>), gridSize, blockSize, args);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }
    else
    {
        uint64_t size = static_cast<uint64_t>(batch) * stride;
        blocks = CalculateBlocks(size);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pUnit, &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL2Error_kernel<T>), gridSize, blockSize, args);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseAnalogNonZeroL2Error_kernel<T>), gridSize, blockSize, args2);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[batch * stride * i], pUnit, static_cast<unsigned long long>(batch) * stride * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


__global__ void __launch_bounds__(256, 4) invokeSparseRawL2HingeError_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    float error = (float)0.0;
    if (pos < size)
    {
        float w = (float)0.5;
        if (pDataWeight != nullptr)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }
        float a = max((float)0.0, pUnit[pos]);
        error = w * a * a;
    }

    REDUCEERROR(error)
}

__global__ void __launch_bounds__(256, 4) invokeSparseOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (float)0.5 * ((pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float diff = min((float)0.0, a - (float)1.0);
            error += w * diff * diff;
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

__global__ void __launch_bounds__(256, 4) invokeSparseNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (float)0.5 * ((pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float diff = min((float)0.0, a - (float)1.0);
            a = max((float)0.0, a);
            error += w * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    if (batch == 0 || stride == 0)
        return 0.0f;

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint32_t blocks;
    dim3 gridSize;
    dim3 blockSize;

    if (bSparseIgnoreZero)
    {
        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseOnlyNonZeroL2HingeError_kernel<T>), gridSize, blockSize, args);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }
    else
    {
        uint64_t size = static_cast<uint64_t>(batch) * stride;
        blocks = CalculateBlocks(size);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pUnit, &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL2HingeError_kernel<T>), gridSize, blockSize, args);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseNonZeroL2HingeError_kernel<T>), gridSize, blockSize, args2);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[batch * stride * i], pUnit, batch * stride * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (float)0.5 * ((pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            float diff = a - fabsf(t);
            diff = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
            error += w * diff * diff;
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (float)0.5 * (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            float diff = a - fabsf(t);
            a = max((float)0.0, a);
            diff = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
            error += w * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (float)0.5 * (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff = a - t;
            diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
            error += w * diff * diff;
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (float)0.5 * (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff = a - t;
            diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
            a = max((float)0.0, a);
            error += w * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (float)0.5 * (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff = a - fabsf((float)t);
            diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
            error += w * diff * diff;
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (float)0.5 * (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff = a - fabsf(t);
            a = max((float)0.0, a);
            diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
            error += w * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}


template<typename T>
float invokeSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint32_t blocks;
    dim3 gridSize;
    dim3 blockSize;

    if (bSparseIgnoreZero)
    {
        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseAnalogOnlyNonZeroL2HingeError_kernel<T>), gridSize, blockSize, args);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }
    else
    {
        uint64_t size = static_cast<uint64_t>(batch) * stride;
        blocks = CalculateBlocks(size);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pUnit, &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL2HingeError_kernel<T>), gridSize, blockSize, args);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseAnalogNonZeroL2HingeError_kernel<T>), gridSize, blockSize, args2);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, batch * stride, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[batch * stride * i], pUnit, batch * stride * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void __launch_bounds__(256, 4) invokeIndexedSparseOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (float)0.5 * ((pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float diff = min((float)0.0, pUnit[pos2] - (float)1.0);
            error += w * diff * diff;
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        float w[14];
        float unit[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                uint64_t pos1 = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                uint64_t end = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                uint64_t offset = pos * stride;

                float diff = 0.0f;

                while (pos1 < end)
                {
                    uint64_t pos2 = offset + pSparseIndex[pos1];
                    diff = min(0.0f, pUnit[pos2] - 1.0f);
                    float a = max(0.0f, pUnit[pos2]);
                    unit[i] = w[i] * (diff * diff - a * a);
                    pos1 += cData._warpSize;
                }

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}


template<typename T>
float invokeIndexedSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS];
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS];
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint32_t blocks;
    dim3 gridSize;
    dim3 blockSize;

    if (bSparseIgnoreZero)
    {
        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseOnlyNonZeroL2HingeError_kernel<T>), gridSize, blockSize, args);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }
    else
    {
        uint64_t size = static_cast<uint64_t>(batch) * stride;
        blocks = CalculateBlocks(size);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pUnit, &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL2HingeError_kernel<T>), gridSize, blockSize, args);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        gridSize = dim3(blocks, getGpu()._threadsPerBlock);
        blockSize = dim3(getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseNonZeroL2HingeError_kernel<T>), gridSize, blockSize, args2);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[batch * stride * i], pUnit, static_cast<unsigned long long>(batch) * stride * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        float w[14];
        float unit[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                uint64_t pos1 = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                uint64_t end = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                uint64_t offset = pos * stride;

                float diff = 0.0f;

                while (pos1 < end)
                {
                    uint64_t pos2 = offset + pSparseIndex[pos1];
                    float a = pUnit[pos2];
                    T t = pSparseData[pos1];
                    diff = a - fabsf(t);
                    diff = (t > (T)0.0) ? min(0.0f, diff) : max(0.0f, diff);
                    unit[i] = w[i] * (diff * diff);
                    pos1 += cData._warpSize;
                }

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        float w[14];
        float unit[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                uint64_t pos1 = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                uint64_t end = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                uint64_t offset = pos * stride;

                float diff = 0.0f;

                while (pos1 < end)
                {
                    uint64_t pos2 = offset + pSparseIndex[pos1];
                    float a = pUnit[pos2];
                    T t = pSparseData[pos1];
                    diff = a - fabsf(t);
                    a = max(0.0f, a);
                    diff = (t > (T)0.0) ? min(0.0f, diff) : max(0.0f, diff);
                    unit[i] = w[i] * (diff * diff - a * a);
                    pos1 += cData._warpSize;
                }

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        float w[14];
        float unit[14];

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            if (pos < batch)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                uint64_t pos1 = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                uint64_t end = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                uint64_t offset = pos * stride;

                float diff = 0.0f;

                while (pos1 < end)
                {
                    uint64_t pos2 = offset + pSparseIndex[pos1];
                    float a = pUnit[pos2];
                    float t = (float)pSparseData[pos1] * (1.0f / 256.0f);
                    diff = a - t;
                    diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
                    unit[i] = w[i] * diff * diff;
                    pos1 += cData._warpSize;
                }

                pos++;
            }
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] = unit[i];
        }

#pragma unroll
        for (int i = 0; i < 14; ++i)
        {
            error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
        }

        if ((threadIdx.x & cData._warpMask) == 0)
        {
            REDUCEERROR(error[0]);
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error[14] = { 0.0f };

    while (pos < batch)
    {
        uint32_t dpos[14];
        float w[14];
        float unit[14];

        if (pos < batch)
        {
#pragma unroll
            for (int i = 0; i < 14; ++i)
            {
                dpos[i] = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
                uint64_t pos1 = pSparseStart[dpos[i]] + (threadIdx.x & cData._warpMask);
                uint64_t end = pSparseEnd[dpos[i]];
                w[i] = 0.5f * ((pDataWeight != nullptr) ? pDataWeight[dpos[i]] : 1.0f);
                uint64_t offset = pos * stride;

                float diff = 0.0f;

                while (pos1 < end)
                {
                    uint64_t pos2 = offset + pSparseIndex[pos1];
                    float a = pUnit[pos2];
                    float t = (float)pSparseData[pos1] * (1.0f / 256.0f);
                    diff = a - t;
                    diff = (t > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
                    a = max(0.0f, a);
                    unit[i] = w[i] * (diff * diff - a * a);
                    pos1 += cData._warpSize;
                }

                pos++;
            }

#pragma unroll
            for (int i = 0; i < 14; ++i)
            {
                error[i] = unit[i];
            }

#pragma unroll
            for (int i = 0; i < 14; ++i)
            {
                error[i] += __shfl_down_sync(0xFFFFFFFF, error[i], 1);
            }

            if ((threadIdx.x & cData._warpMask) == 0)
            {
                REDUCEERROR(error[0]);
            }
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (float)0.5 * ((pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff = a - fabsf(t);
            diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
            error += w * diff * diff;
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (float)0.5 * ((pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff = a - fabsf(t);
            a = max((float)0.0, a);
            diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
            error += w * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}


template<typename T>
float invokeIndexedSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    if (bSparseIgnoreZero)
    {
        uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        dim3 gridSize(blocks, getGpu()._threadsPerBlock);
        dim3 blockSize(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel<T>), gridSize, blockSize, args);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }
    else
    {
        uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
        uint32_t blocks = CalculateBlocks(size);
        dim3 gridSize(blocks, getGpu()._threadsPerBlock);
        dim3 blockSize(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pUnit, &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawL2HingeError_kernel<T>), gridSize, blockSize, args);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        dim3 gridSize2(blocks, getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseAnalogNonZeroL2HingeError_kernel<T>), gridSize2, blockSize, args2);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[batch * stride * i], pUnit, static_cast<unsigned long long>(batch) * stride * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseRawCrossEntropyError_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos = blockDim.x * blockIdx.x + threadIdx.x;
    float error = (float)0.0;
    if (pos < size)
    {
        float w = (float)1.0;
        if (pDataWeight != nullptr)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }
        float a = pUnit[pos];
        error = -w * log(max(MIN_ERROR, (float)1.0 - a));
    }

    REDUCEERROR(error)
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseOnlyNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            error += -w * log(max(MIN_ERROR, a));
            pos1 += cData._warpSize;
        }
    }


    REDUCEERROR(error)
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            error += w * (-log(max(MIN_ERROR, a)) + log(max(MIN_ERROR, (float)1.0 - a)));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS];
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS];
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    if (bSparseIgnoreZero)
    {
        uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        dim3 gridSize(blocks, getGpu()._threadsPerBlock);
        dim3 blockSize(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseOnlyNonZeroCrossEntropyError_kernel<T>), gridSize, blockSize, args);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }
    else
    {
        uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
        uint32_t blocks = CalculateBlocks(size);
        dim3 gridSize(blocks, getGpu()._threadsPerBlock);
        dim3 blockSize(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pUnit, &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawCrossEntropyError_kernel<T>), gridSize, blockSize, args);

        blocks = CalculateBlocks(batch * getGpu()._warpSize);
        dim3 gridSize2(blocks, getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseNonZeroCrossEntropyError_kernel<T>), gridSize2, blockSize, args2);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[batch * stride * i], pUnit, static_cast<unsigned long long>(batch) * stride * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseOnlyNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            error += -w * log(max(MIN_ERROR, a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            error += w * (-log(max(MIN_ERROR, a)) + log(max(MIN_ERROR, (float)1.0 - a)));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeIndexedSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    if (bSparseIgnoreZero)
    {
        uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        dim3 gridSize(blocks, getGpu()._threadsPerBlock);
        dim3 blockSize(getGpu()._threadsPerBlock);

        void* args[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseOnlyNonZeroCrossEntropyError_kernel<T>), gridSize, blockSize, args);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }
    else
    {
        uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
        uint32_t blocks = CalculateBlocks(size);
        dim3 gridSize(blocks, getGpu()._threadsPerBlock);
        dim3 blockSize(getGpu()._threadsPerBlock);

        void* args[] = { &position, pDataWeight, pUnit, &stride, &size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawCrossEntropyError_kernel<T>), gridSize, blockSize, args);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        dim3 gridSize2(blocks, getGpu()._threadsPerBlock);

        void* args2[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseNonZeroCrossEntropyError_kernel<T>), gridSize2, blockSize, args2);

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            ncclAllReduce(pUnit, pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
        }
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, batch * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, static_cast<size_t>(batch) * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[batch * stride * i], pUnit, static_cast<unsigned long long>(batch) * stride * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos];
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight == nullptr) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos];
        pos1 += threadIdx.x & cData._warpMask;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            error += -w * log(max(MIN_ERROR, a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks = CalculateBlocks(local_size);
    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    void* args[] = { &position, &batch, &stride, pLocalUnit[local_rank], pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
    cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseMultinomialCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos];
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight == nullptr) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos];
        pos1 += threadIdx.x & cData._warpMask;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            error += -w * log(max(MIN_ERROR, a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeIndexedSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks = CalculateBlocks(local_size);
    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    void* args[] = { &position, &batch, &stride, pLocalUnit[local_rank], pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
    cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseMultinomialCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;

        T t;
        if constexpr (std::is_same<T, char>::value)
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
                error += w * (-t * log(max(MIN_ERROR, a)));
                pos1 += cData._warpSize;
            }
        }
        else if constexpr (std::is_same<T, unsigned char>::value)
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
                error += w * (-t * log(max(MIN_ERROR, a)));
                pos1 += cData._warpSize;
            }
        }
        else
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = pSparseData[pos1];
                error += w * (-t * log(max(MIN_ERROR, a)));
                pos1 += cData._warpSize;
            }
        }
    }

    REDUCEERROR(error)
}


template<typename T>
float invokeSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks = CalculateBlocks(local_size);
    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    void* args[] = { &position, &batch, &stride, pLocalUnit[local_rank], pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
    cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseAnalogMultinomialCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeIndexedSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;

        T t;
        if constexpr (std::is_same<T, char>::value)
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
                error += w * (-t * log(max(MIN_ERROR, a)));
                pos1 += cData._warpSize;
            }
        }
        else if constexpr (std::is_same<T, unsigned char>::value)
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
                error += w * (-t * log(max(MIN_ERROR, a)));
                pos1 += cData._warpSize;
            }
        }
        else
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = pSparseData[pos1];
                error += w * (-t * log(max(MIN_ERROR, a)));
                pos1 += cData._warpSize;
            }
        }
    }

    REDUCEERROR(error)
}


template<typename T>
float invokeIndexedSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks = CalculateBlocks(local_size);
    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    void* args[] = { &position, &batch, &stride, pLocalUnit[local_rank], pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
    cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseAnalogMultinomialCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseRawScaledMarginalCrossEntropyError_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    float error = (float)0.0;
    if (pos < size)
    {
        float w = cData._SMCE_zeroScale;
        if (pDataWeight != nullptr)
        {
            uint64_t dpos = pos / stride;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
            w *= pDataWeight[dpos];
        }
        float a = pUnit[pos];
        if (a > cData._SMCE_zeroTarget)
            error = -w * log(max(MIN_ERROR, (float)1.0 - a));
    }

    REDUCEERROR(error)
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._SMCE_oneScale * ((pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            if (a < cData._SMCE_oneTarget)
                error += -w * log(max(MIN_ERROR, a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            if (a > cData._SMCE_zeroTarget)
            {
                error += w * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
            }
            if (a < cData._SMCE_oneTarget)
            {
                error += -w * cData._SMCE_oneScale * log(max(MIN_ERROR, a));
            }
            pos1 += cData._warpSize;
        }


    }

    REDUCEERROR(error)
}

template<typename T>
float invokeSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks = CalculateBlocks(local_size);
    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    if (bSparseIgnoreZero) {
        void* args[] = { &position, &batch, &stride, pLocalUnit[local_rank], pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);
    }
    else {
        void* args[] = { &position, pDataWeight, pLocalUnit[local_rank], &stride, &local_size, pSparseStart, pSparseEnd, pSparseIndex };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);

        blocks = CalculateBlocks(batch * getGpu()._warpSize);
        void* args2[] = { &position, &batch, &stride, pLocalUnit[local_rank], pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseNonZeroScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args2, 0, streams[local_rank]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._SMCE_oneScale * ((pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            if (a < cData._SMCE_oneTarget)
                error += -w * log(max(MIN_ERROR, a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            if (a > cData._SMCE_zeroTarget)
            {
                error += w * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
            }
            if (a < cData._SMCE_oneTarget)
            {
                error += -w * cData._SMCE_oneScale * log(max(MIN_ERROR, a));
            }
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeIndexedSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks = CalculateBlocks(local_size);
    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    if (bSparseIgnoreZero) {
        void* args[] = { &position, &batch, &stride, pLocalUnit[local_rank], pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);
    }
    else {
        void* args[] = { &position, pDataWeight, pLocalUnit[local_rank], &stride, &local_size, pSparseStart, pSparseEnd, pSparseIndex };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);

        blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
        void* args2[] = { &position, &batch, &stride, pLocalUnit[local_rank], pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseNonZeroScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args2, 0, streams[local_rank]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseRawDataScaledMarginalCrossEntropyError_kernel(float* pUnit, uint64_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    float error = (float)0.0;
    if (pos < size)
    {
        float a = pUnit[pos];
        if (a > cData._SMCE_zeroTarget)
        {
            error = -cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
        }
    }

    REDUCEERROR(error)
}

template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeSparseNonZeroDataScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];

            if (a > cData._SMCE_zeroTarget)
            {
                error += cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
            }

            if (a < cData._SMCE_oneTarget)
            {
                error += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
            }
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks = CalculateBlocks(local_size);
    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    if (!bSparseIgnoreZero) {
        void* args[] = { pLocalUnit[local_rank], &local_size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawDataScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);
    }

    void* args2[] = { &position, &batch, &stride, pLocalUnit[local_rank], pSparseStart, pSparseEnd, pSparseIndex, pSparseData };
    cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseNonZeroDataScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args2, 0, streams[local_rank]);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}



template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseNonZeroDataScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];

            if (a > cData._SMCE_zeroTarget)
            {
                error += cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
            }

            if (a < cData._SMCE_oneTarget)
            {
                error += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
            }
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeIndexedSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t blocks = CalculateBlocks(local_size);
    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    if (!bSparseIgnoreZero) {
        void* args[] = { pLocalUnit[local_rank], &local_size };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseRawDataScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[local_rank]);
    }

    void* args2[] = { &position, &batch, &stride, pLocalUnit[local_rank], pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseData };
    cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseNonZeroDataScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args2, 0, streams[local_rank]);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos];
        uint64_t end = pSparseEnd[dpos];
        float w = cData._SMCE_oneScale * ((pDataWeight == nullptr) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos]);
        pos1 += threadIdx.x & cData._warpMask;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            if (a < cData._SMCE_oneTarget)
                error += -w * log(max(MIN_ERROR, a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pSparseData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
    dim3 gridSize(blocks, getGpu()._threadsPerBlock);
    dim3 blockSize(getGpu()._threadsPerBlock);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &batch, &stride, pUnit, pSparseData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseNonZeroScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pUnit, pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos];
        uint64_t end = pSparseEnd[dpos];
        float w = cData._SMCE_oneScale * ((pDataWeight == nullptr) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos]);
        pos1 += threadIdx.x & cData._warpMask;
        uint64_t offset = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            if (a < cData._SMCE_oneTarget)
                error += -w * log(max(MIN_ERROR, a));
            pos1 += cData._warpSize;
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeIndexedSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pDataWeight)
{

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
    dim3 gridSize(blocks, getGpu()._threadsPerBlock);
    dim3 blockSize(getGpu()._threadsPerBlock);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseNonZeroScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pUnit, pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._SMCE_oneScale * ((pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset = pos * stride;

        T t;
        if constexpr (std::is_same<T, char>::value)
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = pSparseData[pos1] * (float)(1.0 / 128.0);
                if (a < cData._SMCE_oneTarget)
                    error += -w * t * log(max(MIN_ERROR, a));
                pos1 += cData._warpSize;
            }
        }
        else if constexpr (std::is_same<T, unsigned char>::value)
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = pSparseData[pos1] * (float)(1.0 / 256.0);
                if (a < cData._SMCE_oneTarget)
                    error += -w * t * log(max(MIN_ERROR, a));
                pos1 += cData._warpSize;
            }
        }
        else
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = pSparseData[pos1];
                if (a < cData._SMCE_oneTarget)
                    error += -w * t * log(max(MIN_ERROR, a));
                pos1 += cData._warpSize;
            }
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
    dim3 gridSize(blocks, getGpu()._threadsPerBlock);
    dim3 blockSize(getGpu()._threadsPerBlock);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &batch, &stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pUnit, pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._SMCE_oneScale * ((pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset = pos * stride;

        T t;
        if constexpr (std::is_same<T, char>::value)
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = pSparseData[pos1] * (float)(1.0 / 128.0);
                if (a < cData._SMCE_oneTarget)
                    error += -w * t * log(max(MIN_ERROR, a));
                pos1 += cData._warpSize;
            }
        }
        else if constexpr (std::is_same<T, unsigned char>::value)
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = pSparseData[pos1] * (float)(1.0 / 256.0);
                if (a < cData._SMCE_oneTarget)
                    error += -w * t * log(max(MIN_ERROR, a));
                pos1 += cData._warpSize;
            }
        }
        else
        {
            while (pos1 < end)
            {
                uint64_t pos2 = offset + pSparseIndex[pos1];
                float a = pUnit[pos2];
                t = pSparseData[pos1];
                if (a < cData._SMCE_oneTarget)
                    error += -w * t * log(max(MIN_ERROR, a));
                pos1 += cData._warpSize;
            }
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
    dim3 gridSize(blocks, getGpu()._threadsPerBlock);
    dim3 blockSize(getGpu()._threadsPerBlock);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &batch, &stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel<T>), gridSize, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pUnit, pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != nullptr) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];

        if constexpr (std::is_same<T, unsigned char>::value)
        {
            float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
            error = w * fabsf(a - t);
        }
        else if constexpr (std::is_same<T, char>::value)
        {
            float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
            error = w * fabsf(a - t);
        }
        else
        {
            T t = pData[dOffset + pos];
            error = w * fabsf(a - static_cast<float>(t));
        }
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    dim3 blockSize(getGpu()._threadsPerBlock);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pUnit, pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeL1Error_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pUnit, pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeIndexedL1Error_kernel(
    uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }
    else if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }

    float error = w * fabsf(a - t_normalized);

    REDUCEERROR(error);
}

template<typename T>
float invokeIndexedL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    dim3 blockSize(getGpu()._threadsPerBlock);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pUnit, pIndex, pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedL1Error_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pUnit, pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeL2Error_kernel(
    uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    float w = 0.5f * ((pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f);
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }
    else if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }

    float error = w * (a - t_normalized) * (a - t_normalized);

    REDUCEERROR(error);
}

template<typename T>
float invokeL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    dim3 blockSize(getGpu()._threadsPerBlock);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pUnit, pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeL2Error_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pUnit, pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeIndexedL2Error_kernel(
    uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float w = 0.5f * ((pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f);
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }
    else if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }

    float error = w * (a - t_normalized) * (a - t_normalized);

    REDUCEERROR(error);
}

template<typename T>
float invokeIndexedL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    dim3 blockSize(getGpu()._threadsPerBlock);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pUnit, pIndex, pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedL2Error_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pUnit, pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeL2HingeError_kernel(
    uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    float w = 0.5f * ((pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f);
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }
    else if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }

    float diff = a - t_normalized;
    diff = (t_normalized > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
    float error = w * diff * diff;

    REDUCEERROR(error);
}

template<typename T>
float invokeL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    dim3 blockSize(getGpu()._threadsPerBlock);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pUnit, pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeL2HingeError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pUnit, pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeIndexedL2HingeError_kernel(
    uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float w = 0.5f * ((pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f);
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }
    else if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }

    float diff = a - t_normalized;
    diff = (t_normalized > 0.0f) ? min(0.0f, diff) : max(0.0f, diff);
    float error = w * diff * diff;

    REDUCEERROR(error);
}

template<typename T>
float invokeIndexedL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    dim3 blockSize(getGpu()._threadsPerBlock);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pUnit, pIndex, pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedL2HingeError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pUnit, pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pUnit, (void*)pUnit, batch, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeHingeError_kernel(
    uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = 0.0f;
    float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;

    while (pos < stride)
    {
        T raw_t = __ldg(&pData[pos]);
        float t_normalized = static_cast<float>(raw_t);

        if constexpr (std::is_same<T, unsigned char>::value)
        {
            t_normalized *= (1.0f / 128.0f);
        }
        else if constexpr (std::is_same<T, char>::value)
        {
            t_normalized *= (1.0f / 256.0f);
        }

        float y = __ldg(&pUnit[pos]);
        loss += w * max(0.0f, 1.0f - t_normalized * y);
        pos += blockDim.x;
    }

    REDUCEERROR(loss);
}

template<typename T>
float invokeHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{

    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long threads = max(32, min(stride, 128));
    dim3 grid(batch);
    dim3 blockSize(threads);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pLocalUnit[i], pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeHingeError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeIndexedHingeError_kernel(
    uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = 0.0f;
    float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;

    while (pos < stride)
    {
        T raw_t = __ldg(&pData[pos]);
        float t_normalized = static_cast<float>(raw_t);

        if constexpr (std::is_same<T, unsigned char>::value)
        {
            t_normalized *= (1.0f / 256.0f);
        }
        else if constexpr (std::is_same<T, char>::value)
        {
            t_normalized *= (1.0f / 128.0f);
        }

        float y = __ldg(&pUnit[pos]);
        loss += w * max(0.0f, 1.0f - t_normalized * y);
        pos += blockDim.x;
    }

    REDUCEERROR(loss);
}

template<typename T>
float invokeIndexedHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long threads = max(32, min(stride, 128));
    dim3 grid(batch);
    dim3 blockSize(threads);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pLocalUnit[i], pIndex, pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedHingeError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeCrossEntropyError_kernel(
    uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }
    else if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }

    float error = w * (-t_normalized * __logf(fmaxf(MIN_ERROR, a))
        - (1.0f - t_normalized) * __logf(fmaxf(MIN_ERROR, 1.0f - a)));

    REDUCEERROR(error);
}

template<typename T>
float invokeCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long threads = max(32, min(stride, 128));
    dim3 grid(batch);
    dim3 blockSize(threads);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pLocalUnit[i], pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeCrossEntropyError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeIndexedCrossEntropyError_kernel(
    uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }
    else if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }

    float error = w * (-t_normalized * __logf(fmaxf(MIN_ERROR, a))
        - (1.0f - t_normalized) * __logf(fmaxf(MIN_ERROR, 1.0f - a)));

    REDUCEERROR(error);
}

template<typename T>
float invokeIndexedCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long threads = max(32, min(stride, 128));
    dim3 grid(batch);
    dim3 blockSize(threads);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pLocalUnit[i], pIndex, pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedCrossEntropyError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeMultinomialCrossEntropyError_kernel(
    uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }
    else if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }

    float error = w * (-t_normalized * __logf(fmaxf(MIN_ERROR, a)));

    REDUCEERROR(error);
}

template<typename T>
float invokeMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long threads = max(32, min(stride, 128));
    dim3 grid(batch);
    dim3 blockSize(threads);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pLocalUnit[i], pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeMultinomialCrossEntropyError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


template<typename T>
__global__ void __launch_bounds__(256, 4) invokeIndexedMultinomialCrossEntropyError_kernel(
    uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }
    else if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }

    float error = 0.0f;
    error = w * (-t_normalized * __logf(fmaxf(MIN_ERROR, a)));

    REDUCEERROR(error);
}

template<typename T>
float invokeIndexedMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long threads = max(32, min(stride, 128));
    dim3 grid(batch);
    dim3 blockSize(threads);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pLocalUnit[i], pIndex, pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedMultinomialCrossEntropyError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeScaledMarginalCrossEntropyError_kernel(
    uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }
    else if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }

    float error = 0.0f;
    bool conditionOne = (t_normalized == 1.0f) && (a < cData._SMCE_oneTarget);
    bool conditionZero = (t_normalized == 0.0f) && (a > cData._SMCE_zeroTarget);

    if (conditionOne || conditionZero)
    {
        error = w * (-t_normalized * cData._SMCE_oneScale * __logf(fmaxf(MIN_ERROR, a))
            - (1.0f - t_normalized) * cData._SMCE_zeroScale * __logf(fmaxf(MIN_ERROR, 1.0f - a)));
    }

    REDUCEERROR(error);
}

template<typename T>
float invokeScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long threads = max(32, min(stride, 128));
    dim3 grid(batch);
    dim3 blockSize(threads);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pLocalUnit[i], pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeScaledMarginalCrossEntropyError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeIndexedScaledMarginalCrossEntropyError_kernel(
    uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }
    else if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }

    float error = 0.0f;
    bool conditionOne = (t_normalized == 1.0f) && (a < cData._SMCE_oneTarget);
    bool conditionZero = (t_normalized == 0.0f) && (a > cData._SMCE_zeroTarget);

    if (conditionOne || conditionZero)
    {
        error = w * (-t_normalized * cData._SMCE_oneScale * __logf(fmaxf(MIN_ERROR, a))
            - (1.0f - t_normalized) * cData._SMCE_zeroScale * __logf(fmaxf(MIN_ERROR, 1.0f - a)));
    }

    REDUCEERROR(error);
}

template<typename T>
float invokeIndexedScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long threads = max(32, min(stride, 128));
    dim3 grid(batch);
    dim3 blockSize(threads);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pLocalUnit[i], pIndex, pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedScaledMarginalCrossEntropyError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;

    if (pos >= stride) return;

    float error = 0.0f;
    uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;
    float a = __ldg(&pUnit[uOffset + pos]);

    T t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(t);

    if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }
    else if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }

    if ((t_normalized != 0.0f) && (a < cData._SMCE_oneTarget))
    {
        error = w * (-t_normalized * cData._SMCE_oneScale * logf(fmaxf(MIN_ERROR, a)));
    }

    REDUCEERROR(error)
}

template<typename T>
float invokeMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long threads = max(32, min(stride, 128));
    dim3 grid(batch);
    dim3 blockSize(threads);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pLocalUnit[i], pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeMultinomialScaledMarginalCrossEntropyError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void __launch_bounds__(256, 4) invokeIndexedMultinomialScaledMarginalCrossEntropyError_kernel(
    uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;

    if (pos >= stride) return;

    uint64_t uOffset = blockIdx.x * static_cast<uint64_t>(stride);
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float w = (pDataWeight != nullptr) ? __ldg(&pDataWeight[dpos]) : 1.0f;
    float a = __ldg(&pUnit[uOffset + pos]);

    T raw_t = __ldg(&pData[dOffset + pos]);
    float t_normalized = static_cast<float>(raw_t);

    if constexpr (std::is_same<T, char>::value)
    {
        t_normalized *= (1.0f / 128.0f);
    }
    else if constexpr (std::is_same<T, unsigned char>::value)
    {
        t_normalized *= (1.0f / 256.0f);
    }

    float error = 0.0f;
    if ((t_normalized != 0.0f) && (a < cData._SMCE_oneTarget))
    {
        error = w * (-t_normalized * cData._SMCE_oneScale * __logf(fmaxf(MIN_ERROR, a)));
    }

    REDUCEERROR(error);
}

template<typename T>
float invokeIndexedMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    MPI_Init(NULL, NULL);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId ncclId;
    ncclComm_t ncclComm[NUM_GPUS]{};
    int local_rank = rank % NUM_GPUS;

    if (local_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStream_t streams[NUM_GPUS]{};
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        ncclCommInitRank(&ncclComm[i], world_size, ncclId, rank * NUM_GPUS + i);
    }

    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint64_t local_size = size / world_size;

    std::vector<float*> pLocalUnit(NUM_GPUS);
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&pLocalUnit[i], local_size * sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    unsigned long threads = max(32, min(stride, 128));
    dim3 grid(batch);
    dim3 blockSize(threads);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        void* args[] = { &position, &stride, pLocalUnit[i], pIndex, pData, pDataWeight };
        cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedMultinomialScaledMarginalCrossEntropyError_kernel<T>), grid, blockSize, args, 0, streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(pLocalUnit[i], pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
    }

    ncclGroupStart();

    for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);

            if (collective == 0) {
                ncclAllReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
            else if (collective == 1) {
                ncclBroadcast((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 2) {
                ncclReduce((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
            }
            else if (collective == 3) {
                ncclAllGather((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclComm[i], streams[i]);
            }
            else if (collective == 4) {
                ncclReduceScatter((const void*)pLocalUnit[i], (void*)pLocalUnit[i], local_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
            }
        }
    }

    ncclGroupEnd();

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (rank == 0) {
        for (int i = 1; i < NUM_GPUS; i++) {
            cudaMemcpyAsync(&pUnit[local_size * i], pLocalUnit[i], local_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(pLocalUnit[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();

    return static_cast<float>(static_cast<double>(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// Explicitly instantiates various kernel functions for different data types.
/// </summary>
/// <remarks>
/// This macro template is used to instantiate kernel functions for error calculations,
/// such as L1 error, L2 error, Cross Entropy error, etc., for various data types.
/// </remarks>
/// <typeparam name="T">The data type for which kernel functions are instantiated.</typeparam>
#define EXPLICITLY_INSTANTIATE_KERNELS(T)                                                                                                                                                                  \
template float invokeL1Error<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);                                                                                                               \
template float invokeIndexedL1Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);                                                                                             \
template float invokeL2Error<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);                                                                                                               \
template float invokeIndexedL2Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);                                                                                             \
template float invokeL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);                                                                                                          \
template float invokeIndexedL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);                                                                                        \
template float invokeCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);                                                                                                     \
template float invokeIndexedCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);                                                                                   \
template float invokeScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);                                                                                       \
template float invokeIndexedScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);                                                                     \
template float invokeMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);                                                                                          \
template float invokeIndexedMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);                                                                        \
template float invokeMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);                                                                            \
template float invokeIndexedMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);                                                          \
template float invokeHingeError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);                                                                                                            \
template float invokeIndexedHingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);                                                                                          \
template float invokeSparseAnalogL1Error<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);                                                \
template float invokeIndexedSparseAnalogL1Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);                              \
template float invokeSparseAnalogL2Error<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);                                                \
template float invokeIndexedSparseAnalogL2Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);                              \
template float invokeSparseAnalogL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);                                           \
template float invokeIndexedSparseAnalogL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);                         \
template float invokeSparseAnalogMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*);                                 \
template float invokeIndexedSparseAnalogMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*);               \
template float invokeSparseAnalogMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*);                   \
template float invokeIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*); \
template float invokeSparseDataScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, T*, bool);                                                \
template float invokeIndexedSparseDataScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, T*, bool);

/// <summary>
/// Explicitly instantiates CUDA kernels for various error calculations with the float data type.
/// </summary>
EXPLICITLY_INSTANTIATE_KERNELS(float)

/// <summary>
/// Explicitly instantiates CUDA kernels for various error calculations with the double data type.
/// </summary>
EXPLICITLY_INSTANTIATE_KERNELS(double)

/// <summary>
/// Explicitly instantiates CUDA kernels for various error calculations with the unsigned char data type.
/// </summary>
EXPLICITLY_INSTANTIATE_KERNELS(unsigned char)

/// <summary>
/// Explicitly instantiates CUDA kernels for various error calculations with the char data type.
/// </summary>
EXPLICITLY_INSTANTIATE_KERNELS(char)

/// <summary>
/// Explicitly instantiates CUDA kernels for various error calculations with the uint32_t data type.
/// </summary>
EXPLICITLY_INSTANTIATE_KERNELS(uint32_t)

/// <summary>
/// Explicitly instantiates CUDA kernels for various error calculations with the uint64_t data type.
/// </summary>
EXPLICITLY_INSTANTIATE_KERNELS(uint64_t)

/// <summary>
/// Explicitly instantiates CUDA kernels for various error calculations with the int32_t data type.
/// </summary>
EXPLICITLY_INSTANTIATE_KERNELS(int32_t)

/// <summary>
/// Explicitly instantiates CUDA kernels for various error calculations with the int64_t data type.
/// </summary>
EXPLICITLY_INSTANTIATE_KERNELS(int64_t)