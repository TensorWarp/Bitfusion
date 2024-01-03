#include "gpuTypes.h"
#include "types.h"
#include <limits>
#include "bitonicSort.cuh"
#include "kernels.cuh"
#include "constants.h"
#include <math.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <vector_types.h>
#include <cooperative_groups.h>
#include <nccl.h>
#include <execution>
#include <cstdio>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <numeric>
#include <vector>
#include "mpi.h"
#include <thread>
#include <future>

namespace cg = cooperative_groups;

static __constant__ GpuData cData;

void SetsortingGpuData(const GpuData& data)
{
    cudaError_t status = cudaMemcpyToSymbol(cData, &data, sizeof(GpuData));
    if (status != cudaSuccess) {
        fprintf(stderr, "Error: cudaMemcpyToSymbol failed - %s\n", cudaGetErrorString(status));
        throw std::runtime_error("cudaMemcpyToSymbol failed");
    }
}

void GetsortingGpuData(GpuData& data)
{
    cudaError_t status = cudaMemcpyFromSymbol(&data, cData, sizeof(GpuData));
    if (status != cudaSuccess) {
        fprintf(stderr, "Error: cudaMemcpyFromSymbol failed - %s\n", cudaGetErrorString(status));
        throw std::runtime_error("cudaMemcpyFromSymbol failed");
    }
}

uint32_t CalculateBlocks(uint64_t size)
{
    return (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
}

__global__ void __launch_bounds__(256, 4) kScaleAndBias_kernel(float* pData, uint64_t size, float scale, float bias)
{
    uint64_t offset = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < size)
    {
        float value = pData[offset];
        pData[offset] = scale * value - bias;
    }
}

template <typename T>
void kScaleAndBias(float* pData, uint64_t size, float scale, float bias) {
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pData[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size, &scale, &bias };
            cudaLaunchKernel(reinterpret_cast<void*>(&kScaleAndBias_kernel<T>), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pData[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


__global__ void __launch_bounds__(256, 4) kClearUnit_kernel(float* pUnit, float* pBias, uint32_t stride, uint64_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    __shared__ float shared_bias[1024];

    cg::thread_block block = cg::this_thread_block();

    if (threadIdx.x < stride) {
        shared_bias[threadIdx.x] = pBias[threadIdx.x];
    }
    block.sync();

    if (pos < size)
    {
        pUnit[pos] = shared_bias[bpos];
    }
}


void kClearUnit(float* pUnit, float* pBias, uint32_t stride, uint32_t batch) {
    uint64_t size = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    std::vector<float*> pBiasPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBiasPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBiasPartitions[i], &pBias[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pUnitPartitions.begin(), pUnitPartitions.end(), rng);
        std::shuffle(pBiasPartitions.begin(), pBiasPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pUnitPartitions[task % NUM_PARTITIONS], &pBiasPartitions[task % NUM_PARTITIONS],
                             &stride, &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kClearUnit_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclAllReduce((const void*)pBiasPartitions[i % NUM_PARTITIONS], (void*)pBiasPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);

                    ncclBroadcast((const void*)pBiasPartitions[i % NUM_PARTITIONS], (void*)pBiasPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);

                    ncclReduce((const void*)pBiasPartitions[i % NUM_PARTITIONS], (void*)pBiasPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);

                    ncclAllGather((const void*)pBiasPartitions[i % NUM_PARTITIONS], (void*)pBiasPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclReduceScatter((const void*)pBiasPartitions[i % NUM_PARTITIONS], (void*)pBiasPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pUnitPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBias[rank * chunk_size + i * local_chunk_size], pBiasPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pBiasPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kClearDualSourceUnit_kernel(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    __shared__ volatile float shared_bias1[256 + 16];
    __shared__ volatile float shared_bias2[256 + 16];

    int laneId = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    for (int i = warpId; i < stride; i += tile.size() / warpSize)
    {
        shared_bias1[i * warpSize + laneId] = pBias1[i * warpSize + laneId];
        shared_bias2[i * warpSize + laneId] = pBias2[i * warpSize + laneId];
    }

    tile.sync();

    float result = 0.0f;

#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        int idx = bpos + i * 32;
        float product = __fmaf_rn(__half2float(__float2half(shared_bias1[idx])), __half2float(__float2half(shared_bias2[idx])), 0.0f);
        result += product;
    }

    if (pos < size)
    {
        pUnit[pos] = result;
    }
}

void kClearDualSourceUnit(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch) {
    uint64_t size = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    std::vector<float*> pBias1Partitions(NUM_PARTITIONS);
    std::vector<float*> pBias2Partitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBias1Partitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBias2Partitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBias1Partitions[i], &pBias1[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBias2Partitions[i], &pBias2[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pUnitPartitions.begin(), pUnitPartitions.end(), rng);
        std::shuffle(pBias1Partitions.begin(), pBias1Partitions.end(), rng);
        std::shuffle(pBias2Partitions.begin(), pBias2Partitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pUnitPartitions[task % NUM_PARTITIONS], &pBias1Partitions[task % NUM_PARTITIONS],
                             &pBias2Partitions[task % NUM_PARTITIONS], &stride, &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kClearDualSourceUnit_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclAllReduce((const void*)pBias1Partitions[i % NUM_PARTITIONS], (void*)pBias1Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclAllReduce((const void*)pBias2Partitions[i % NUM_PARTITIONS], (void*)pBias2Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);

                    ncclBroadcast((const void*)pBias1Partitions[i % NUM_PARTITIONS], (void*)pBias1Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);

                    ncclBroadcast((const void*)pBias2Partitions[i % NUM_PARTITIONS], (void*)pBias2Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);

                    ncclReduce((const void*)pBias1Partitions[i % NUM_PARTITIONS], (void*)pBias1Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);

                    ncclReduce((const void*)pBias2Partitions[i % NUM_PARTITIONS], (void*)pBias2Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);

                    ncclAllGather((const void*)pBias1Partitions[i % NUM_PARTITIONS], (void*)pBias1Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);

                    ncclAllGather((const void*)pBias2Partitions[i % NUM_PARTITIONS], (void*)pBias2Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclReduceScatter((const void*)pBias1Partitions[i % NUM_PARTITIONS], (void*)pBias1Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclReduceScatter((const void*)pBias2Partitions[i % NUM_PARTITIONS], (void*)pBias2Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pUnitPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBias1[rank * chunk_size + i * local_chunk_size], pBias1Partitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBias2[rank * chunk_size + i * local_chunk_size], pBias2Partitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pBias1Partitions[i]);
        cudaFree(pBias2Partitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kClearTripleSourceUnit_kernel(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    __shared__ float shared_bias1[1024];
    __shared__ float shared_bias2[1024];
    __shared__ float shared_bias3[1024];

    cg::thread_block block = cg::this_thread_block();

    if (threadIdx.x < stride) {
        shared_bias1[threadIdx.x] = pBias1[threadIdx.x];
        shared_bias2[threadIdx.x] = pBias2[threadIdx.x];
        shared_bias3[threadIdx.x] = pBias3[threadIdx.x];
    }
    block.sync();

    if (pos < size)
    {
        pUnit[pos] = shared_bias1[bpos] + shared_bias2[bpos] + shared_bias3[bpos];
    }
}

void kClearTripleSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch) {
    uint64_t size = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    std::vector<float*> pBias1Partitions(NUM_PARTITIONS);
    std::vector<float*> pBias2Partitions(NUM_PARTITIONS);
    std::vector<float*> pBias3Partitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBias1Partitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBias2Partitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBias3Partitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBias1Partitions[i], &pBias1[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBias2Partitions[i], &pBias2[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBias3Partitions[i], &pBias3[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pUnitPartitions.begin(), pUnitPartitions.end(), rng);
        std::shuffle(pBias1Partitions.begin(), pBias1Partitions.end(), rng);
        std::shuffle(pBias2Partitions.begin(), pBias2Partitions.end(), rng);
        std::shuffle(pBias3Partitions.begin(), pBias3Partitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pUnitPartitions[task % NUM_PARTITIONS], &pBias1Partitions[task % NUM_PARTITIONS],
                             &pBias2Partitions[task % NUM_PARTITIONS], &pBias3Partitions[task % NUM_PARTITIONS],
                             &stride, &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kClearTripleSourceUnit_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclAllReduce((const void*)pBias1Partitions[i % NUM_PARTITIONS], (void*)pBias1Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclAllReduce((const void*)pBias2Partitions[i % NUM_PARTITIONS], (void*)pBias2Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclAllReduce((const void*)pBias3Partitions[i % NUM_PARTITIONS], (void*)pBias3Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);

                    ncclBroadcast((const void*)pBias1Partitions[i % NUM_PARTITIONS], (void*)pBias1Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);

                    ncclBroadcast((const void*)pBias2Partitions[i % NUM_PARTITIONS], (void*)pBias2Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);

                    ncclBroadcast((const void*)pBias3Partitions[i % NUM_PARTITIONS], (void*)pBias3Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);

                    ncclReduce((const void*)pBias1Partitions[i % NUM_PARTITIONS], (void*)pBias1Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);

                    ncclReduce((const void*)pBias2Partitions[i % NUM_PARTITIONS], (void*)pBias2Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);

                    ncclReduce((const void*)pBias3Partitions[i % NUM_PARTITIONS], (void*)pBias3Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);

                    ncclAllGather((const void*)pBias1Partitions[i % NUM_PARTITIONS], (void*)pBias1Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);

                    ncclAllGather((const void*)pBias2Partitions[i % NUM_PARTITIONS], (void*)pBias2Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);

                    ncclAllGather((const void*)pBias3Partitions[i % NUM_PARTITIONS], (void*)pBias3Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclReduceScatter((const void*)pBias1Partitions[i % NUM_PARTITIONS], (void*)pBias1Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclReduceScatter((const void*)pBias2Partitions[i % NUM_PARTITIONS], (void*)pBias2Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);

                    ncclReduceScatter((const void*)pBias3Partitions[i % NUM_PARTITIONS], (void*)pBias3Partitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pUnitPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBias1[rank * chunk_size + i * local_chunk_size], pBias1Partitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBias2[rank * chunk_size + i * local_chunk_size], pBias2Partitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBias3[rank * chunk_size + i * local_chunk_size], pBias3Partitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pBias1Partitions[i]);
        cudaFree(pBias2Partitions[i]);
        cudaFree(pBias3Partitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kClearQuadSourceUnit_kernel(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    __shared__ float shared_bias1[1024];
    __shared__ float shared_bias2[1024];
    __shared__ float shared_bias3[1024];
    __shared__ float shared_bias4[1024];

    cg::thread_block block = cg::this_thread_block();

    if (threadIdx.x < stride) {
        shared_bias1[threadIdx.x] = pBias1[threadIdx.x];
        shared_bias2[threadIdx.x] = pBias2[threadIdx.x];
        shared_bias3[threadIdx.x] = pBias3[threadIdx.x];
        shared_bias4[threadIdx.x] = pBias4[threadIdx.x];
    }
    block.sync();

    if (pos < size)
    {
        pUnit[pos] = shared_bias1[bpos] + shared_bias2[bpos] + shared_bias3[bpos] + shared_bias4[bpos];
    }
}

template <typename T>
void kClearQuadSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch) {
    uint64_t size = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kClearQuadSourceUnit_kernel<T>), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kLoadSparseInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint32_t global_pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = global_pos / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1 = pos + position;
        pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1;
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        float w = (pDataWeight != nullptr) ? pDataWeight[pos1] : 1.0f;
        uint64_t offset = static_cast<uint64_t>(pos) * stride;

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            if (i < end)
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                if (pos2 < static_cast<unsigned long long>(batch) * stride)
                    pUnit[pos2] = w;
            }
        }
    }
}

void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight) {
    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS]{};
    cudaEvent_t events[NUM_GPUS]{};

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kLoadSparseInputUnit_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kLoadIndexedSparseInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint32_t global_pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = global_pos / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        float w = (pDataWeight != nullptr) ? pDataWeight[pos1] : 1.0f;
        uint64_t offset = static_cast<uint64_t>(pos) * stride;

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            if (i < end)
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                if (pos2 < static_cast<unsigned long long>(batch) * stride)
                    pUnit[pos2] = w;
            }
        }
    }
}

void kLoadIndexedSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight) {
    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS]{};
    cudaEvent_t events[NUM_GPUS]{};

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kLoadIndexedSparseInputUnit_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

template<typename T>
__global__ void __launch_bounds__(256, 4) kLoadSparseAnalogInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint32_t global_pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = global_pos / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        float w = (pDataWeight != nullptr) ? pDataWeight[pos1] : 1.0f;
        uint64_t offset = static_cast<uint64_t>(pos) * stride;

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            if (i < end)
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                if (pos2 < static_cast<unsigned long long>(batch) * stride)
                {
                    T data = pSparseData[i];
                    pUnit[pos2] = w * data;
                }
            }
        }
    }
}

template<typename T>
void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kLoadSparseAnalogInputUnit_kernel<T>), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


template<typename T>
__global__ void __launch_bounds__(256, 4) kLoadIndexedSparseAnalogInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint32_t global_pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = global_pos / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        float w = (pDataWeight != nullptr) ? pDataWeight[pos1] : 1.0f;
        uint64_t offset = static_cast<uint64_t>(pos) * stride;

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            if (i < end)
            {
                uint64_t pos2 = offset + pSparseIndex[i];
                if (pos2 < static_cast<unsigned long long>(batch) * stride)
                {
                    T data = pSparseData[i];
                    pUnit[pos2] = w * data;
                }
            }
        }
    }
}

template<typename T>
void kLoadIndexedSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kLoadIndexedSparseAnalogInputUnit_kernel<T>), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kLoadSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom)
{
    uint32_t global_pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = global_pos / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[pos1] : 1.0f);
        uint64_t offset = static_cast<uint64_t>(pos) * stride;

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            if (i < end)
            {
                float value = pRandom[i];
                uint64_t pos2 = offset + pSparseIndex[i];
                if (pos2 < static_cast<unsigned long long>(batch) * stride)
                {
                    if (value >= cData._denoising_p)
                        pUnit[pos2] = w;
                }
            }
        }
    }
}


void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom) {
    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kLoadSparseDenoisedInputUnit_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kLoadIndexedSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom)
{
    uint32_t global_pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = global_pos / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[pos1] : 1.0f);
        uint64_t offset = static_cast<uint64_t>(pos) * stride;

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            if (i < end)
            {
                float value = pRandom[i];
                uint64_t pos2 = offset + pSparseIndex[i];
                if (pos2 < static_cast<unsigned long long>(batch) * stride)
                {
                    if (value >= cData._denoising_p)
                        pUnit[pos2] = w;
                }
            }
        }
    }
}


void kLoadIndexedSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom) {
    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kLoadIndexedSparseDenoisedInputUnit_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

template<typename T>
__global__ void __launch_bounds__(256, 4) kLoadSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom)
{
    uint32_t global_pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = global_pos / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[pos1] : 1.0f);
        uint64_t offset = static_cast<uint64_t>(pos) * stride;

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            if (i < end)
            {
                float value = pRandom[i];
                uint64_t pos2 = offset + pSparseIndex[i];
                T data = pSparseData[i];
                if (pos2 < static_cast<unsigned long long>(batch) * stride)
                {
                    if (value >= cData._denoising_p)
                        pUnit[pos2] = w * data;
                }
            }
        }
    }
}

template<typename T>
void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom) {
    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kLoadSparseAnalogDenoisedInputUnit_kernel<T>), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

template<typename T>
__global__ void __launch_bounds__(256, 4) kLoadIndexedSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom)
{
    uint32_t global_pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = global_pos / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[pos1];
        float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[pos1] : 1.0f);
        uint64_t offset = static_cast<uint64_t>(pos) * stride;

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            if (i < end)
            {
                float value = pRandom[i];
                uint64_t pos2 = offset + pSparseIndex[i];
                T data = pSparseData[i];
                if (pos2 < static_cast<unsigned long long>(batch) * stride)
                {
                    if (value >= cData._denoising_p)
                        pUnit[pos2] = w * data;
                }
            }
        }
    }
}

template<typename T>
void kLoadIndexedSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom) {
    uint64_t size = (uint64_t)batch * (uint64_t)stride;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kLoadIndexedSparseAnalogDenoisedInputUnit_kernel<T>), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

template<typename T>
__global__ void __launch_bounds__(256, 4) kLoadInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset = static_cast<unsigned long long>(pos1) * stride + pos;
        uint64_t doffset = blockIdx.x * static_cast<unsigned long long>(stride) + pos;
        pUnit[doffset] = pData[soffset];
    }
}

template<typename T>
__global__ void __launch_bounds__(256, 4) kLoadNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset = static_cast<unsigned long long>(pos1) * stride + pos;
        uint64_t doffset = blockIdx.x * static_cast<unsigned long long>(stride) + pos;

        if constexpr (std::is_same<T, unsigned char>::value) {
            pUnit[doffset] = (float)pData[soffset] * (float)(1.0 / 256.0) - (float)0.5;
        }
        else if constexpr (std::is_same<T, char>::value) {
            pUnit[doffset] = (float)pData[soffset] * (float)(1.0 / 128.0);
        }
    }
}

template<typename T>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData)
{
    uint64_t size = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size, (void*)(uintptr_t)position, (void*)(uintptr_t)stride, (void*)(uintptr_t)pUnit, (void*)(uintptr_t)pData };

            if constexpr (std::is_same<T, unsigned char>::value) {
                cudaLaunchKernel(reinterpret_cast<void*>(&kLoadNormalizedInputUnit_kernel<T>), gridSize, blockSize, args);
            }
            else {
                cudaLaunchKernel(reinterpret_cast<void*>(&kLoadInputUnit_kernel<T>), gridSize, blockSize, args);
            }

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


template<typename T>
__global__ void __launch_bounds__(256, 4) kLoadIndexedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position];
        uint64_t soffset = static_cast<unsigned long long>(pos1) * stride + pos;
        uint64_t doffset = blockIdx.x * static_cast<unsigned long long>(stride) + pos;
        pUnit[doffset] = pData[soffset];
    }
}

template<typename T>
__global__ void __launch_bounds__(256, 4) kLoadIndexedNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1 = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position];
        uint64_t soffset = static_cast<unsigned long long>(pos1) * stride + pos;
        uint64_t doffset = blockIdx.x * static_cast<unsigned long long>(stride) + pos;

        if constexpr (std::is_same<T, unsigned char>::value) {
            pUnit[doffset] = (float)pData[soffset] * (float)(1.0 / 256.0) - (float)0.5;
        }
        else if constexpr (std::is_same<T, char>::value) {
            pUnit[doffset] = (float)pData[soffset] * (float)(1.0 / 128.0);
        }
    }
}

template<typename T>
void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData)
{
    uint64_t size = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size, pIndex, pData };
            if constexpr (std::is_same<T, unsigned char>::value) {
                cudaLaunchKernel(reinterpret_cast<void*>(&kLoadIndexedNormalizedInputUnit_kernel<T>), gridSize, blockSize, args);
            }
            else {
                cudaLaunchKernel(reinterpret_cast<void*>(&kLoadIndexedInputUnit_kernel<T>), gridSize, blockSize, args);
            }

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
kAddBias_kernel(float* pUnit, float* pBias, uint32_t stride, uint32_t size)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;
    if (pos < size)
    {
        pUnit[pos] += pBias[bpos];
    }
}


void kAddBias(float* pUnit, float* pBias, uint32_t stride, uint32_t batch) {
    uint64_t size = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAddBias_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


__global__ void __launch_bounds__(256, 4) kAddDualBias_kernel(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    __shared__ float shared_bias1[1024];
    __shared__ float shared_bias2[1024];

    cg::thread_block block = cg::this_thread_block();

    if (threadIdx.x < stride) {
        shared_bias1[threadIdx.x] = pBias1[threadIdx.x];
        shared_bias2[threadIdx.x] = pBias2[threadIdx.x];
    }
    block.sync();

    if (pos < size)
    {
        pUnit[pos] += shared_bias1[bpos] + shared_bias2[bpos];
    }
}

void kAddDualBias(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch) {
    uint64_t size = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAddDualBias_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kAddTripleBias_kernel(float* pUnit, const float* pBias1, const float* pBias2, const float* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    extern __shared__ float shared_biases[];
    float* shared_bias1 = &shared_biases[0];
    float* shared_bias2 = &shared_biases[stride];
    float* shared_bias3 = &shared_biases[2 * stride];

    cg::thread_block block = cg::this_thread_block();

#pragma unroll
    for (int i = threadIdx.x; i < stride; i += blockDim.x) {
        shared_bias1[i] = pBias1[i];
        shared_bias2[i] = pBias2[i];
        shared_bias3[i] = pBias3[i];
    }
    block.sync();

    if (pos < size && bpos < stride)
    {
        pUnit[pos] += shared_bias1[bpos] + shared_bias2[bpos] + shared_bias3[bpos];
    }
}

void kAddTripleBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch) {
    uint64_t size = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAddTripleBias_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kAddQuadBias_kernel(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    __shared__ float shared_bias1[1024];
    __shared__ float shared_bias2[1024];
    __shared__ float shared_bias3[1024];
    __shared__ float shared_bias4[1024];

    cg::thread_block block = cg::this_thread_block();

    if (threadIdx.x < stride) {
        shared_bias1[threadIdx.x] = pBias1[threadIdx.x];
        shared_bias2[threadIdx.x] = pBias2[threadIdx.x];
        shared_bias3[threadIdx.x] = pBias3[threadIdx.x];
        shared_bias4[threadIdx.x] = pBias4[threadIdx.x];
    }
    block.sync();

    if (pos < size)
    {
        pUnit[pos] += shared_bias1[bpos] + shared_bias2[bpos] + shared_bias3[bpos] + shared_bias4[bpos];
    }
}

void kAddQuadBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch) {
    uint64_t size = (uint64_t)stride * (uint64_t)batch;
    uint32_t blocks = (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pDataPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDataPartitions.begin(), pDataPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDataPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAddQuadBias_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDataPartitions[i % NUM_PARTITIONS], (void*)pDataPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {

                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size + i * local_chunk_size], pDataPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
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


__global__ void LAUNCH_BOUNDS256() invokeSparseZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSE];

    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != nullptr) ? pDataWeight[position] : 1.0f;
    pUnit += blockIdx.x * stride;

    cg::thread_block block = cg::this_thread_block();

    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = min(end - start, (uint64_t)MAXSPARSE);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;

        while (opos < sOpos)
        {
            float unit = (beta == 0.0f) ? 0.0f : (beta * pUnit[opos]);
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                unit += w * pWeight[offset + opos];
            }

            pUnit[opos] = unit;
            opos += blockDim.x;
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = 1.0f;
    }
}


void invokeSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta) {
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    dim3 gridSize(batch);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightPartitions[i], stride * sizeof(float));
        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[i * stride], stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint64_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pWeightPartitions.begin(), pWeightPartitions.end(), rng);
        std::shuffle(pUnitPartitions.begin(), pUnitPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &position, &stride, &pWeightPartitions[task % NUM_PARTITIONS], &pSparseStart, &pSparseEnd,
                             &pSparseIndex, &pDataWeight, &pUnitPartitions[task % NUM_PARTITIONS], &beta };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseZ_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size * stride + i * local_chunk_size * stride], pUnitPartitions[i % NUM_PARTITIONS],
            local_chunk_size * stride * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pWeightPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 5) invokeIndexedSparseZ_kernel(
    uint32_t position, uint32_t stride, float* pWeight,
    uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd,
    uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta)
{
    __shared__ uint32_t sOffset[MAXSPARSE];
    volatile __shared__ uint32_t sOpos;

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
    pUnit += blockIdx.x * stride;

    block.sync();

    float beta_pUnit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[threadIdx.x]);

    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSE);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        block.sync();

        float unit = beta_pUnit;

#pragma unroll
        for (uint32_t i = 0; i < inputs; i++)
        {
            uint32_t offset = sOffset[i];
            unit = __fmaf_rn(w, pWeight[offset + threadIdx.x], unit);
        }

        pUnit[threadIdx.x] = unit;

        start = tend;
        if (start < end)
        {
            block.sync();
        }
        beta = (float)1.0;
    }
}


void invokeIndexedSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    dim3 gridSize(batch);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightPartitions[i], stride * sizeof(float));
        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[i * stride], stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint64_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pWeightPartitions.begin(), pWeightPartitions.end(), rng);
        std::shuffle(pUnitPartitions.begin(), pUnitPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &position, &stride, &pWeightPartitions[task % NUM_PARTITIONS], &pIndex, &pSparseStart, &pSparseEnd,
                             &pSparseIndex, &pDataWeight, &pUnitPartitions[task % NUM_PARTITIONS], &beta };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseZ_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMemcpyAsync(&pUnit[rank * chunk_size * stride + i * local_chunk_size * stride], pUnitPartitions[i],
            local_chunk_size * stride * sizeof(float), cudaMemcpyDeviceToHost, streams[local_rank % NUM_GPUS]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pWeightPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

template<typename T>
__global__ void __launch_bounds__(256, 5) invokeSparseAnalogZ_kernel(
    uint32_t position,
    uint32_t stride,
    float* pWeight,
    uint64_t* pSparseStart,
    uint64_t* pSparseEnd,
    uint32_t* pSparseIndex,
    float* pDataWeight,
    T* pSparseData,
    float* pUnit,
    float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ T sValue[MAXSPARSEANALOG];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * pSparseData[tstart];
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

template<>
__global__ void
__launch_bounds__(256, 5)
invokeSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * ((float)pSparseData[tstart] * (float)(1.0 / 256.0));
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

template<>
__global__ void
__launch_bounds__(256, 5)
invokeSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * ((float)pSparseData[tstart] * (float)(1.0 / 256.0));
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

template<typename T>
void invokeSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    dim3 gridSize(batch);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightPartitions[i], stride * sizeof(float));
        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[i * stride], stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint64_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pWeightPartitions.begin(), pWeightPartitions.end(), rng);
        std::shuffle(pUnitPartitions.begin(), pUnitPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &position, &stride, &pWeightPartitions[task % NUM_PARTITIONS], &pSparseStart, &pSparseEnd,
                             &pSparseIndex, &pDataWeight, &pSparseData, &pUnitPartitions[task % NUM_PARTITIONS], &beta };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseAnalogZ_kernel<T>), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMemcpyAsync(&pUnit[rank * chunk_size * stride + i * local_chunk_size * stride], pUnitPartitions[i],
            local_chunk_size * stride * sizeof(float), cudaMemcpyDeviceToHost, streams[local_rank % NUM_GPUS]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pWeightPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


template<typename T>
__global__ void __launch_bounds__(256, 5) invokeIndexedSparseAnalogZ_kernel(
    uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex,
    uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex,
    float* pDataWeight, T* pSparseData, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
    float scalingFactor = (std::is_same<T, unsigned char>::value) ? (1.0 / 256.0) : (1.0 / 128.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * (static_cast<float>(pSparseData[tstart]) * scalingFactor);
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

template<typename T>
void invokeIndexedSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    dim3 gridSize(batch);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightPartitions[i], stride * sizeof(float));
        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[i * stride], stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint64_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pWeightPartitions.begin(), pWeightPartitions.end(), rng);
        std::shuffle(pUnitPartitions.begin(), pUnitPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &position, &stride, &pWeightPartitions[task % NUM_PARTITIONS], &pIndex, &pSparseStart, &pSparseEnd,
                             &pSparseIndex, &pDataWeight, &pSparseData, &pUnitPartitions[task % NUM_PARTITIONS], &beta };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseAnalogZ_kernel<T>), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMemcpyAsync(&pUnit[rank * chunk_size * stride + i * local_chunk_size * stride], pUnitPartitions[i],
            local_chunk_size * stride * sizeof(float), cudaMemcpyDeviceToHost, streams[local_rank % NUM_GPUS]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pWeightPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


__global__ void
LAUNCH_BOUNDS256()
invokeSparseDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSE];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos];
                }

                pUnit[opos] = w * unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

void invokeSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    dim3 gridSize(batch);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightPartitions[i], stride * sizeof(float));
        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[i * stride], stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint64_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<float*> pRandomPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pRandomPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pRandomPartitions[i], &pRandom[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint32_t beta_rounds = NUM_BETA_ROUNDS;
    float beta_increment = beta / beta_rounds;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int beta_iter = 0; beta_iter < beta_rounds; beta_iter++) {
            for (int task = 0; task < NUM_TASKS; task++) {
                int targetGPU = task % NUM_GPUS;
                cudaSetDevice(targetGPU);

                void* args[] = { &position, &stride, &pWeightPartitions[task % NUM_PARTITIONS], &pSparseStart, &pSparseEnd, &pSparseIndex,
                                 &pDataWeight, &pRandomPartitions[task % NUM_PARTITIONS], &pUnitPartitions[task % NUM_PARTITIONS],
                                 &beta_increment };
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseDenoisedZ_kernel), gridSize, blockSize, args);

                cudaEventRecord(events[targetGPU], streams[targetGPU]);
            }
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMemcpyAsync(&pUnit[rank * chunk_size * stride + i * local_chunk_size * stride], pUnitPartitions[i],
            local_chunk_size * stride * sizeof(float), cudaMemcpyDeviceToHost, streams[local_rank % NUM_GPUS]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pWeightPartitions[i]);
        cudaFree(pRandomPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
LAUNCH_BOUNDS256()
invokeIndexedSparseDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSE];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos];
                }

                pUnit[opos] = w * unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

void invokeIndexedSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    dim3 gridSize(batch);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightPartitions[i], stride * sizeof(float));
        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[i * stride], stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint64_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<float*> pRandomPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pRandomPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pRandomPartitions[i], &pRandom[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint32_t beta_rounds = NUM_BETA_ROUNDS;
    float beta_increment = beta / beta_rounds;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int beta_iter = 0; beta_iter < beta_rounds; beta_iter++) {
            for (int task = 0; task < NUM_TASKS; task++) {
                int targetGPU = task % NUM_GPUS;
                cudaSetDevice(targetGPU);

                void* args[] = { &position, &stride, &pWeightPartitions[task % NUM_PARTITIONS], &pIndex, &pSparseStart, &pSparseEnd,
                                 &pSparseIndex, &pDataWeight, &pRandomPartitions[task % NUM_PARTITIONS],
                                 &pUnitPartitions[task % NUM_PARTITIONS], &beta_increment };
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseDenoisedZ_kernel), gridSize, blockSize, args);

                cudaEventRecord(events[targetGPU], streams[targetGPU]);
            }
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMemcpyAsync(&pUnit[rank * chunk_size * stride + i * local_chunk_size * stride], pUnitPartitions[i],
            local_chunk_size * stride * sizeof(float), cudaMemcpyDeviceToHost, streams[local_rank % NUM_GPUS]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pWeightPartitions[i]);
        cudaFree(pRandomPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

template<typename T>
__global__ void
LAUNCH_BOUNDS256()
invokeSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ T sValue[MAXSPARSEANALOG];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = pSparseData[tstart] * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : pUnit[opos];
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
invokeSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ int32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;
        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = (float)pSparseData[tstart] * (float)(1.0 / 256.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
invokeSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;
        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = (float)pSparseData[tstart] * (float)(1.0 / 128.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

template<typename T>
void invokeSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    dim3 gridSize(batch);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightPartitions[i], stride * sizeof(float));
        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[i * stride], stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint64_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<float*> pRandomPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pRandomPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pRandomPartitions[i], &pRandom[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint32_t beta_rounds = NUM_BETA_ROUNDS;
    float beta_increment = beta / beta_rounds;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int beta_iter = 0; beta_iter < beta_rounds; beta_iter++) {
            for (int task = 0; task < NUM_TASKS; task++) {
                int targetGPU = task % NUM_GPUS;
                cudaSetDevice(targetGPU);

                void* args[] = { &position, &stride, &pWeightPartitions[task % NUM_PARTITIONS], &pSparseStart, &pSparseEnd,
                                 &pSparseIndex, &pDataWeight, &pSparseData, &pRandomPartitions[task % NUM_PARTITIONS],
                                 &pUnitPartitions[task % NUM_PARTITIONS], &beta_increment };
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseAnalogDenoisedZ_kernel<T>), gridSize, blockSize, args);

                cudaEventRecord(events[targetGPU], streams[targetGPU]);
            }
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 5) {
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);
                cudaStreamSynchronize(streams[i]);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMemcpyAsync(&pUnit[rank * chunk_size * stride + i * local_chunk_size * stride], pUnitPartitions[i],
            local_chunk_size * stride * sizeof(float), cudaMemcpyDeviceToHost, streams[local_rank % NUM_GPUS]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pWeightPartitions[i]);
        cudaFree(pRandomPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


template<typename T>
__global__ void
LAUNCH_BOUNDS256()
invokeIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ T sValue[MAXSPARSEANALOG];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = pSparseData[tstart] * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
invokeIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ int32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = (float)pSparseData[tstart] * (float)(1.0 / 256.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS256()
invokeIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    cg::thread_block block = cg::this_thread_block();

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = (float)pSparseData[tstart] * (float)(1.0 / 128.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
        }
        beta = (float)1.0;
    }
}

template<typename T>
void invokeIndexedSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = min(256, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    dim3 gridSize(batch);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightPartitions[i], stride * sizeof(float));
        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[i * stride], stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint64_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<float*> pRandomPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pRandomPartitions[i], local_chunk_size * stride * sizeof(float));
        cudaMemcpyAsync(pRandomPartitions[i], &pRandom[rank * chunk_size * stride + i * local_chunk_size * stride],
            local_chunk_size * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    uint32_t beta_rounds = NUM_BETA_ROUNDS;
    float beta_increment = beta / beta_rounds;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int beta_iter = 0; beta_iter < beta_rounds; beta_iter++) {
            for (int task = 0; task < NUM_TASKS; task++) {
                int targetGPU = task % NUM_GPUS;
                cudaSetDevice(targetGPU);

                void* args[] = { &position, &stride, &pWeightPartitions[task % NUM_PARTITIONS], &pIndex, &pSparseStart, &pSparseEnd,
                                 &pSparseIndex, &pDataWeight, &pSparseData, &pRandomPartitions[task % NUM_PARTITIONS],
                                 &pUnitPartitions[task % NUM_PARTITIONS], &beta_increment };
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseAnalogDenoisedZ_kernel<T>), gridSize, blockSize, args);

                cudaEventRecord(events[targetGPU], streams[targetGPU]);
            }
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size * stride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMemcpyAsync(&pUnit[rank * chunk_size * stride + i * local_chunk_size * stride], pUnitPartitions[i],
            local_chunk_size * stride * sizeof(float), cudaMemcpyDeviceToHost, streams[local_rank % NUM_GPUS]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pWeightPartitions[i]);
        cudaFree(pRandomPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


__global__ void
__launch_bounds__(256, 4)
invokeSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            start += cData._warpSize;
        }
    }
}

__global__ void
__launch_bounds__(256, 4)
invokeWeightedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = pDataWeight[position];
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            pSparseTransposedData[opos] = w;
            start += cData._warpSize;
        }
    }
}

void invokeSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);

    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = batch / (NUM_GPUS * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &position, &local_chunk_size, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight, &pSparseTransposedEnd, &pSparseTransposedIndex, &pSparseTransposedData };

            if (pDataWeight == nullptr) {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseTransposedMatrix_kernel), blocks, getGpu()._threadsPerBlock, args, 0, nullptr);
            }
            else {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeWeightedSparseTransposedMatrix_kernel), blocks, getGpu()._threadsPerBlock, args, 0, nullptr);
            }

            cudaEventRecord(events[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaEventSynchronize(events[i]);
        }

        ncclGroupStart();

        for (int i = 0; i < NUM_GPUS; i++) {
            ncclSend(pSparseTransposedData, local_chunk_size, ncclFloat, i, ncclComm[i], nullptr);
            ncclSend(pSparseTransposedEnd, local_chunk_size, ncclUint32, i, ncclComm[i], nullptr);
            ncclSend(pSparseTransposedIndex, local_chunk_size, ncclUint32, i, ncclComm[i], nullptr);
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(nullptr);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            start += cData._warpSize;
        }
    }
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedWeightedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = pDataWeight[position];
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            pSparseTransposedData[opos] = w;
            start += cData._warpSize;
        }
    }
}

void invokeIndexedSparseTransposedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch);

    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = batch / (NUM_GPUS * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &position, &local_chunk_size, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight, &pSparseTransposedEnd, &pSparseTransposedIndex, &pSparseTransposedData };

            if (pDataWeight == nullptr) {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseTransposedMatrix_kernel), blocks, getGpu()._warpSize, args, 0, nullptr);
            }
            else {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedWeightedSparseTransposedMatrix_kernel), blocks, getGpu()._warpSize, args, 0, nullptr);
            }

            cudaEventRecord(events[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaEventSynchronize(events[i]);
        }

        ncclGroupStart();

        for (int i = 0; i < NUM_GPUS; i++) {
            ncclSend(pSparseTransposedData, local_chunk_size, ncclFloat, i, ncclComm[i], nullptr);
            ncclSend(pSparseTransposedEnd, local_chunk_size, ncclUint32, i, ncclComm[i], nullptr);
            ncclSend(pSparseTransposedIndex, local_chunk_size, ncclUint32, i, ncclComm[i], nullptr);
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(nullptr);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
invokeSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        while (start < end)
        {
            float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
            }
            start += cData._warpSize;
        }
    }
}

__global__ void
__launch_bounds__(256, 4)
invokeWeightedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = cData._denoising_q * pDataWeight[position];
        while (start < end)
        {
            float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w;
            }
            start += cData._warpSize;
        }
    }
}

void invokeSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);

    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = batch / (NUM_GPUS * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &position, &local_chunk_size, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight, &pRandom, &pSparseTransposedEnd, &pSparseTransposedIndex, &pSparseTransposedData };

            if (pDataWeight == nullptr) {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseTransposedDenoisedMatrix_kernel), blocks, getGpu()._threadsPerBlock, args, 0, nullptr);
            }
            else {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeWeightedSparseTransposedDenoisedMatrix_kernel), blocks, getGpu()._threadsPerBlock, args, 0, nullptr);
            }

            cudaEventRecord(events[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaEventSynchronize(events[i]);
        }

        ncclGroupStart();

        for (int i = 0; i < NUM_GPUS; i++) {
            ncclSend(pSparseTransposedData, local_chunk_size, ncclFloat, i, ncclComm[i], nullptr);
            ncclSend(pSparseTransposedEnd, local_chunk_size, ncclUint32, i, ncclComm[i], nullptr);
            ncclSend(pSparseTransposedIndex, local_chunk_size, ncclUint32, i, ncclComm[i], nullptr);
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(nullptr);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        while (start < end)
        {
            float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
            }
            start += cData._warpSize;
        }
    }
}

__global__ void
__launch_bounds__(256, 4)
invokeIndexedWeightedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = cData._denoising_q * pDataWeight[position];
        while (start < end)
        {
            float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w;
            }
            start += cData._warpSize;
        }
    }
}

void invokeIndexedSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);

    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = batch * getGpu()._warpSize / (NUM_GPUS * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &position, &local_chunk_size, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight, &pRandom, &pSparseTransposedEnd, &pSparseTransposedIndex, &pSparseTransposedData };

            if (pDataWeight == nullptr) {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseTransposedDenoisedMatrix_kernel), blocks, getGpu()._threadsPerBlock, args, 0, nullptr);
            }
            else {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedWeightedSparseTransposedDenoisedMatrix_kernel), blocks, getGpu()._threadsPerBlock, args, 0, nullptr);
            }

            cudaEventRecord(events[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaEventSynchronize(events[i]);
        }

        ncclGroupStart();

        for (int i = 0; i < NUM_GPUS; i++) {
            ncclSend(pSparseTransposedData, local_chunk_size, ncclFloat, i, ncclComm[i], nullptr);
            ncclSend(pSparseTransposedEnd, local_chunk_size, ncclUint32, i, ncclComm[i], nullptr);
            ncclSend(pSparseTransposedIndex, local_chunk_size, ncclUint32, i, ncclComm[i], nullptr);
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(nullptr);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeSparseTransposedAnalogMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            T value = pSparseData[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            pSparseTransposedData[opos] = w * value;
            start += cData._warpSize;
        }
    }
}

template<typename T>
void invokeSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);
    invokeSparseTransposedAnalogMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);
}

template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseTransposedAnalogMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            uint32_t index = pSparseIndex[start];
            T value = pSparseData[start];
            uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos] = bpos;
            pSparseTransposedData[opos] = w * value;
            start += cData._warpSize;
        }
    }
}

template<typename T>
void invokeIndexedSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<uint32_t*> pIndexPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pIndexPartitions[i], local_chunk_size * sizeof(uint32_t));
        cudaMemcpyAsync(pIndexPartitions[i], &pIndex[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<uint32_t*> pSparseTransposedEndPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pSparseTransposedEndPartitions[i], local_chunk_size * sizeof(uint32_t));
        cudaMemcpyAsync(pSparseTransposedEndPartitions[i], &pSparseTransposedEnd[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<uint32_t*> pSparseTransposedIndexPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pSparseTransposedIndexPartitions[i], local_chunk_size * sizeof(uint32_t));
        cudaMemcpyAsync(pSparseTransposedIndexPartitions[i], &pSparseTransposedIndex[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<float*> pSparseTransposedDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pSparseTransposedDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pSparseTransposedDataPartitions[i], &pSparseTransposedData[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &position, &batch, &pIndexPartitions[task % NUM_PARTITIONS], &pSparseStart, &pSparseEnd,
                             &pSparseIndex, &pDataWeight, &pSparseData, &pSparseTransposedEndPartitions[task % NUM_PARTITIONS],
                             &pSparseTransposedIndexPartitions[task % NUM_PARTITIONS], &pSparseTransposedDataPartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseTransposedAnalogMatrix_kernel<T>), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pSparseTransposedDataPartitions[collective % NUM_PARTITIONS],
                        (void*)pSparseTransposedDataPartitions[collective % NUM_PARTITIONS], local_chunk_size,
                        ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMemcpyAsync(&pSparseTransposedData[rank * chunk_size + i * local_chunk_size],
            pSparseTransposedDataPartitions[i], local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost,
            streams[local_rank % NUM_GPUS]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pIndexPartitions[i]);
        cudaFree(pSparseTransposedEndPartitions[i]);
        cudaFree(pSparseTransposedIndexPartitions[i]);
        cudaFree(pSparseTransposedDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                T value = pSparseData[start];
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start += cData._warpSize;
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value = (float)pSparseData[start] * (float)(1.0 / 256.0);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start += cData._warpSize;
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value = (float)pSparseData[start] * (float)(1.0 / 128.0);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start += cData._warpSize;
        }
    }
}

template<typename T>
void invokeSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<uint32_t*> pSparseTransposedEndPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pSparseTransposedEndPartitions[i], local_chunk_size * sizeof(uint32_t));
        cudaMemcpyAsync(pSparseTransposedEndPartitions[i], &pSparseTransposedEnd[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<uint32_t*> pSparseTransposedIndexPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pSparseTransposedIndexPartitions[i], local_chunk_size * sizeof(uint32_t));
        cudaMemcpyAsync(pSparseTransposedIndexPartitions[i], &pSparseTransposedIndex[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<float*> pSparseTransposedDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pSparseTransposedDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pSparseTransposedDataPartitions[i], &pSparseTransposedData[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &position, &batch, &pSparseStart, &pSparseEnd,
                             &pSparseIndex, &pDataWeight, &pSparseData, &pRandom,
                             &pSparseTransposedEndPartitions[task % NUM_PARTITIONS],
                             &pSparseTransposedIndexPartitions[task % NUM_PARTITIONS],
                             &pSparseTransposedDataPartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseTransposedAnalogDenoisedMatrix_kernel<T>), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pSparseTransposedDataPartitions[collective % NUM_PARTITIONS],
                        (void*)pSparseTransposedDataPartitions[collective % NUM_PARTITIONS], local_chunk_size,
                        ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMemcpyAsync(&pSparseTransposedData[rank * chunk_size + i * local_chunk_size],
            pSparseTransposedDataPartitions[i], local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost,
            streams[local_rank % NUM_GPUS]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pSparseTransposedEndPartitions[i]);
        cudaFree(pSparseTransposedIndexPartitions[i]);
        cudaFree(pSparseTransposedDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

template<typename T>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                T value = pSparseData[start];
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start += cData._warpSize;
        }
    }
}
template<>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value = (float)pSparseData[start] * (float)(1.0 / 256.0);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start += cData._warpSize;
        }
    }
}

template<>
__global__ void
__launch_bounds__(256, 4)
invokeIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (bpos < batch)
    {
        position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start = pSparseStart[position] + tgx;
        uint64_t end = pSparseEnd[position];
        float w = (pDataWeight != nullptr) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd = pRandom[start];
            uint32_t index = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value = (float)pSparseData[start] * (float)(1.0 / 128.0);
                uint32_t opos = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos] = bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start += cData._warpSize;
        }
    }
}

template<typename T>
void invokeIndexedSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * getGpu()._warpSize);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<uint32_t*> pSparseTransposedEndPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pSparseTransposedEndPartitions[i], local_chunk_size * sizeof(uint32_t));
        cudaMemcpyAsync(pSparseTransposedEndPartitions[i], &pSparseTransposedEnd[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<uint32_t*> pSparseTransposedIndexPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pSparseTransposedIndexPartitions[i], local_chunk_size * sizeof(uint32_t));
        cudaMemcpyAsync(pSparseTransposedIndexPartitions[i], &pSparseTransposedIndex[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<float*> pSparseTransposedDataPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pSparseTransposedDataPartitions[i], local_chunk_size * sizeof(float));
        cudaMemcpyAsync(pSparseTransposedDataPartitions[i], &pSparseTransposedData[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &position, &batch, &pIndex, &pSparseStart,
                             &pSparseEnd, &pSparseIndex, &pDataWeight,
                             &pSparseData, &pRandom,
                             &pSparseTransposedEndPartitions[task % NUM_PARTITIONS],
                             &pSparseTransposedIndexPartitions[task % NUM_PARTITIONS],
                             &pSparseTransposedDataPartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeIndexedSparseTransposedAnalogDenoisedMatrix_kernel<T>), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int i = 0; i < NUM_GPUS; i++) {
            ncclSend(pSparseTransposedEndPartitions[i], local_chunk_size, ncclUint32, i, ncclComm[i], streams[i]);
            ncclSend(pSparseTransposedIndexPartitions[i], local_chunk_size, ncclUint32, i, ncclComm[i], streams[i]);
            ncclSend(pSparseTransposedDataPartitions[i], local_chunk_size, ncclFloat32, i, ncclComm[i], streams[i]);
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMemcpyAsync(&pSparseTransposedData[rank * chunk_size + i * local_chunk_size],
            pSparseTransposedDataPartitions[i], local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost,
            streams[local_rank % NUM_GPUS]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pSparseTransposedEndPartitions[i]);
        cudaFree(pSparseTransposedIndexPartitions[i]);
        cudaFree(pSparseTransposedDataPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


__global__ void
LAUNCH_BOUNDS256()
invokeSparseTransposedWeightGradient_kernel(float alpha, float beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pDelta, float* pWeightGradient)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSE];

    cg::thread_block block = cg::this_thread_block();

    uint64_t start = pSparseTransposedStart[blockIdx.x];
    uint64_t end = pSparseTransposedEnd[blockIdx.x];
    alpha *= cData._denoising_q;
    pWeightGradient += blockIdx.x * n;
    do
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSE);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseTransposedIndex[tstart] * n;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();

        uint32_t opos = threadIdx.x;
        uint32_t tgx = threadIdx.x & cData._warpMask;
        while (opos < n)
        {
            float oldgradient = (beta == (float)0.0) ? (float)0.0 : beta * pWeightGradient[opos];
            int64_t sum = 0;
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                sum += llrintf(ERRORSCALEF * pDelta[offset + opos]);
            }

            float fsum = alpha * (float)((double)sum * ONEOVERERRORSCALE);
            pWeightGradient[opos] = oldgradient + fsum;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
            opos += tgx;
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
            beta = (float)1.0;
        }
    } while (start < end);
}


void invokeSparseTransposedWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pDelta, float* pWeightGradient)
{
    uint32_t threads = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    dim3 gridSize(m);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = m / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &beta, &n, &pSparseTransposedStart, &pSparseTransposedEnd,
                             &pSparseTransposedIndex, &pDelta, &pWeightGradient,
                             &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseTransposedWeightGradient_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int i = 0; i < NUM_GPUS; i++) {
            ncclSend(pDelta, local_chunk_size * n, ncclFloat32, i, ncclComm[i], streams[i]);
            ncclRecv(pWeightGradient, local_chunk_size * n, ncclFloat32, i, ncclComm[i], streams[i]);
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
LAUNCH_BOUNDS256()
invokeSparseTransposedAnalogWeightGradient_kernel(float alpha, float beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData, float* pDelta, float* pWeightGradient)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    cg::thread_block block = cg::this_thread_block();

    uint64_t start = pSparseTransposedStart[blockIdx.x];
    uint64_t end = pSparseTransposedEnd[blockIdx.x];
    alpha *= cData._denoising_q;
    pWeightGradient += blockIdx.x * n;
    do
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseTransposedIndex[tstart] * n;
            sValue[pos] = pSparseTransposedData[start];
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        block.sync();


        uint32_t opos = threadIdx.x;
        uint32_t tgx = threadIdx.x & cData._warpMask;
        while (opos < n)
        {
            float oldgradient = (beta == (float)0.0) ? (float)0.0 : beta * pWeightGradient[opos];
            int64_t sum = 0;
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                float value = sValue[i];
                sum += llrintf(ERRORSCALEF * value * pDelta[offset + opos]);
            }

            float fsum = alpha * (float)((double)sum * ONEOVERERRORSCALE);
            pWeightGradient[opos] = oldgradient + fsum;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
            opos += tgx;
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            block.sync();
            beta = (float)1.0;
        }
    } while (start < end);
}

void invokeSparseTransposedAnalogWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData, float* pDelta, float* pWeightGradient)
{
    uint32_t threads = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    dim3 gridSize(m);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = m / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &beta, &n, &pSparseTransposedStart, &pSparseTransposedEnd,
                             &pSparseTransposedIndex, &pSparseTransposedData, &pDelta, &pWeightGradient,
                             &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeSparseTransposedAnalogWeightGradient_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int i = 0; i < NUM_GPUS; i++) {
            ncclSend(pDelta, local_chunk_size * n, ncclFloat32, i, ncclComm[i], streams[i]);
            ncclRecv(pWeightGradient, local_chunk_size * n, ncclFloat32, i, ncclComm[i], streams[i]);
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
kUpdateBiases_kernel(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum = (float)0.0;
        pDelta += pos;
        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }
        pBias[pos] -= alpha * sum;
    }
}

void kUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);

    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = batch / (NUM_GPUS * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &local_chunk_size, &width, &pDelta, &pBias };

            cudaLaunchKernel(reinterpret_cast<void*>(&kUpdateBiases_kernel), blocks, getGpu()._threadsPerBlock, args, 0, nullptr);

            cudaEventRecord(events[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaEventSynchronize(events[i]);
        }

        ncclGroupStart();

        for (int i = 0; i < NUM_GPUS; i++) {
            ncclSend(pBias, width, ncclFloat, i, ncclComm[i], nullptr);
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(nullptr);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
invokeRegularizationError_kernel(float* pWeight, uint64_t size, float lambda, float lambda1)
{
    uint64_t pos = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < size)
    {
        float w = pWeight[pos];
        error = lambda * w * w + lambda1 * abs(w);
    }

    REDUCEERROR(error)
}

float invokeRegularizationError(float lambda, float lambda1, float* pWeight, uint64_t size)
{
    uint32_t blocks = CalculateBlocks(size);
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &pWeight, &local_chunk_size, &lambda, &lambda1, &getGpu()._data._pAccumulator };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeRegularizationError_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int i = 0; i < NUM_GPUS; i++) {
            ncclSend(getGpu()._data._pAccumulator, sizeof(uint64_t), ncclUint64, i, ncclComm[i], streams[i]);
            ncclRecv(getGpu()._data._pAccumulator, sizeof(uint64_t), ncclUint64, i, ncclComm[i], streams[i]);
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
__launch_bounds__(256, 4)
kSGDUpdateWeights_kernel(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightGradient, float* pWeight)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g = pWeightGradient[pos];
        float w = pWeight[pos];
        pWeight[pos] = w + alpha * (g - lambda * w - lambda1 * sgn(w));
    }
}

void kSGDUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightGradient, float* pWeight)
{
    uint32_t blocks = CalculateBlocks(size);
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &lambda, &lambda1, &local_chunk_size, &pWeightGradient, &pWeight, &getGpu()._data._pAccumulator };
            cudaLaunchKernel(reinterpret_cast<void*>(&kSGDUpdateWeights_kernel), gridSize, blockSize, args, 0, nullptr);

            cudaEventRecord(events[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaEventSynchronize(events[i]);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pWeightGradient, (void*)pWeightGradient, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], nullptr);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pWeightGradient, (void*)pWeightGradient, local_chunk_size, ncclFloat, 0, ncclComm[i], nullptr);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pWeightGradient, (void*)pWeightGradient, local_chunk_size, ncclFloat, ncclSum, 0, ncclComm[i], nullptr);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pWeightGradient, (void*)pWeightGradient, local_chunk_size, ncclFloat, ncclComm[i], nullptr);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(nullptr);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
kSGDUpdateBiases_kernel(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }
        sum /= (float)batch;

        float bias = pBias[pos];
        pBias[pos] = bias - alpha * sum;
    }
}

void kSGDUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(float) * width);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint32_t chunk_size = width / (NUM_GPUS * NUM_NODES);
    uint32_t local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint32_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &batch, &local_chunk_size, &pDelta, &pBias, &getGpu()._data._pAccumulator };
            cudaLaunchKernel(reinterpret_cast<void*>(&kSGDUpdateBiases_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


__global__ void
__launch_bounds__(256, 4)
kMomentumUpdateWeights_kernel(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g = pWeightGradient[pos];
        float w = pWeight[pos];
        float v = pWeightVelocity[pos];
        v = mu * v + alpha * (g - lambda * w - lambda1 * sgn(w));
        pWeightVelocity[pos] = v;
        pWeight[pos] = w + v;
    }
}

void kMomentumUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint32_t blocks = CalculateBlocks(size);
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(float) * size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint32_t chunk_size = size / (NUM_GPUS * NUM_NODES);
    uint32_t local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint32_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &lambda, &lambda1, &mu, &local_chunk_size, &pWeightVelocity, &pWeightGradient, &pWeight, &getGpu()._data._pAccumulator };
            cudaLaunchKernel(reinterpret_cast<void*>(&kMomentumUpdateWeights_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
kMomentumUpdateBiases_kernel(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }
        sum /= (float)batch;

        float v = pBiasVelocity[pos];
        v = mu * v - alpha * sum;
        pBiasVelocity[pos] = v;
        pBias[pos] += v;
    }
}

void kMomentumUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(float) * width);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint32_t chunk_size = width / (NUM_GPUS * NUM_NODES);
    uint32_t local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint32_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &mu, &batch, &local_chunk_size, &pDelta, &pBiasVelocity, &pBias, &getGpu()._data._pAccumulator };
            cudaLaunchKernel(reinterpret_cast<void*>(&kMomentumUpdateBiases_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
kAdaGradUpdateWeights_kernel(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g = pWeightGradient[pos];
        float w = pWeight[pos];
        float v = pWeightVelocity[pos];
        g -= lambda * w + lambda1 * sgn(w);
        v += g * g;
        pWeightVelocity[pos] = v;
        pWeight[pos] = w + alpha * g * rsqrt(max(0.000000001f, v));
    }
}

void kAdaGradUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    unsigned long blocks = CalculateBlocks(size);
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(float) * size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &lambda, &lambda1, &local_chunk_size, &pWeightVelocity, &pWeightGradient, &pWeight, &getGpu()._data._pAccumulator };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAdaGradUpdateWeights_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kAdaGradUpdateBiases_kernel(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }
        sum /= __fdividef((float)batch, 1.0f);

        float v = pBiasVelocity[pos];
        v += sum * sum;
        pBiasVelocity[pos] = v;
        pBias[pos] -= alpha * sum * rsqrt(max(0.000000001f, v));
    }
}

void kAdaGradUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(float) * width);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = width / (NUM_GPUS * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &local_chunk_size, &batch, &width, &pDelta, &pBiasVelocity, &pBias, &getGpu()._data._pAccumulator };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAdaGradUpdateBiases_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
kAdaDeltaUpdateWeights_kernel(float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g = pWeightGradient[pos];
        float w = pWeight[pos];
        float v = pWeightVelocity[pos];
        float vg = pWeightGradientVelocity[pos];
        g -= lambda * w + lambda1 * sgn(w);
        vg = mu * vg + ((float)1.0 - mu) * g * g;
        float dw = sqrt(max((float)0.000000001, v) / max((float)0.000000001, vg)) * g;
        v = mu * v + ((float)1.0 - mu) * dw * dw;
        pWeightVelocity[pos] = v;
        pWeightGradientVelocity[pos] = vg;
        pWeight[pos] = w + dw;
    }
}

void kAdaDeltaUpdateWeights(float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight)
{
    unsigned long blocks = CalculateBlocks(size);
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(float) * size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &lambda, &lambda1, &mu, &local_chunk_size, &pWeightVelocity, &pWeightGradient, &pWeightGradientVelocity, &pWeight, &getGpu()._data._pAccumulator };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAdaDeltaUpdateWeights_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pWeight, (void*)pWeight, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kAdaDeltaUpdateBiases_kernel(float mu, uint32_t batch, uint32_t width, const float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    cg::thread_block block = cg::this_thread_block();

    if (pos < width)
    {
        __shared__ volatile float deltaSum[32];

        float sum = 0.0f;

        for (uint32_t i = threadIdx.x; i < batch; i += blockDim.x * 4)
        {
            float4 delta4_0 = __ldg(reinterpret_cast<const float4*>(pDelta) + i * width / 4 + pos / 4);
            float4 delta4_1 = __ldg(reinterpret_cast<const float4*>(pDelta) + (i + 1) * width / 4 + pos / 4);
            float4 delta4_2 = __ldg(reinterpret_cast<const float4*>(pDelta) + (i + 2) * width / 4 + pos / 4);
            float4 delta4_3 = __ldg(reinterpret_cast<const float4*>(pDelta) + (i + 3) * width / 4 + pos / 4);
            sum += delta4_0.x + delta4_0.y + delta4_0.z + delta4_0.w;
            sum += delta4_1.x + delta4_1.y + delta4_1.z + delta4_1.w;
            sum += delta4_2.x + delta4_2.y + delta4_2.z + delta4_2.w;
            sum += delta4_3.x + delta4_3.y + delta4_3.z + delta4_3.w;
        }

        for (int stride = warpSize / 2; stride > 0; stride >>= 1)
        {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, stride);
        }

        if (threadIdx.x % warpSize == 0)
        {
            deltaSum[threadIdx.x / warpSize] = sum;
        }

        block.sync();

        if (threadIdx.x < blockDim.x / warpSize)
        {
            sum = deltaSum[threadIdx.x];
#pragma unroll
            for (int i = 1; i < blockDim.x / warpSize; i++)
            {
                sum += deltaSum[i];
            }
        }

        if (threadIdx.x == 0)
        {
            sum /= static_cast<float>(batch);
        }

        float v = pBiasVelocity[pos];
        float vg = pBiasGradientVelocity[pos];
        vg = mu * vg + (1.0f - mu) * sum * sum;
        float dw = sqrtf(max(1e-9f, v) / max(1e-9f, vg)) * sum;
        v = mu * v + (1.0f - mu) * dw * dw;

        pBiasVelocity[pos] = v;
        pBiasGradientVelocity[pos] = vg;

        pBias[pos] -= dw;
    }
}

void kAdaDeltaUpdateBiases(float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias)
{
    uint32_t blocks = CalculateBlocks(width);
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(float) * width);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = width / (NUM_GPUS * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = task % NUM_GPUS;
            cudaSetDevice(targetGPU);

            void* args[] = { &mu, &local_chunk_size, &batch, &width, &pDelta, &pBiasVelocity, &pBiasGradientVelocity, &pBias, &getGpu()._data._pAccumulator };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAdaDeltaUpdateBiases_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pBias, (void*)pBias, local_chunk_size, ncclFloat, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
kAdamUpdateWeights_kernel(float alpha, float lambda, float lambda1, float beta1, float beta2, float t, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float dw = pWeightGradient[pos];
        float w = pWeight[pos];
        float vdw = pWeightVelocity[pos];
        float sdw = pWeightGradientVelocity[pos];
        dw -= lambda * w + lambda1 * sgn(w);
        vdw = beta1 * vdw + ((float)1.0 - beta1) * dw;
        sdw = beta2 * sdw + ((float)1.0 - beta2) * dw * dw;
        t += (float)1.0;
        pWeightVelocity[pos] = vdw;
        pWeightGradientVelocity[pos] = sdw;
        vdw /= (float)1.0 - pow(beta1, t);
        sdw /= (float)1.0 - pow(beta2, t);
        dw = alpha * vdw / (sqrt(sdw) + (float)1.0e-8);
        pWeight[pos] = w + dw;
    }
}

void kAdamUpdateWeights(float alpha, float lambda, float lambda1, float beta1, float beta2, float t, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight, cudaStream_t stream) {
    unsigned long blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pWeightVelocityPartitions(NUM_PARTITIONS);
    std::vector<float*> pWeightGradientPartitions(NUM_PARTITIONS);
    std::vector<float*> pWeightGradientVelocityPartitions(NUM_PARTITIONS);
    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightVelocityPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pWeightGradientPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pWeightGradientVelocityPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pWeightPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pWeightVelocityPartitions[i], &pWeightVelocity[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pWeightGradientPartitions[i], &pWeightGradient[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pWeightGradientVelocityPartitions[i], &pWeightGradientVelocity[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pWeightVelocityPartitions.begin(), pWeightVelocityPartitions.end(), rng);
        std::shuffle(pWeightGradientPartitions.begin(), pWeightGradientPartitions.end(), rng);
        std::shuffle(pWeightGradientVelocityPartitions.begin(), pWeightGradientVelocityPartitions.end(), rng);
        std::shuffle(pWeightPartitions.begin(), pWeightPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &lambda, &lambda1, &beta1, &beta2, &t, &local_chunk_size, &pWeightVelocityPartitions[task % NUM_PARTITIONS], &pWeightGradientPartitions[task % NUM_PARTITIONS], &pWeightGradientVelocityPartitions[task % NUM_PARTITIONS], &pWeightPartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAdamUpdateWeights_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pWeightVelocity[rank * chunk_size + i * local_chunk_size], pWeightVelocityPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pWeightGradient[rank * chunk_size + i * local_chunk_size], pWeightGradientPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pWeightGradientVelocity[rank * chunk_size + i * local_chunk_size], pWeightGradientVelocityPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pWeight[rank * chunk_size + i * local_chunk_size], pWeightPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pWeightVelocityPartitions[i]);
        cudaFree(pWeightGradientPartitions[i]);
        cudaFree(pWeightGradientVelocityPartitions[i]);
        cudaFree(pWeightPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kAdamUpdateBiases_kernel(
    float alpha,
    float beta1,
    float beta2,
    float t,
    uint32_t batch,
    uint32_t width,
    float* pDelta,
    float* pBiasVelocity,
    float* pBiasGradientVelocity,
    float* pBias
)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    cg::thread_block cta = cg::this_thread_block();

    if (pos < width)
    {
        __shared__ float sSum[32];
        sSum[threadIdx.x % 32] = 0.0f;

        pDelta += pos;
        float localSum = 0.0f;
        for (uint32_t i = threadIdx.x; i < batch * width; i += blockDim.x)
        {
            localSum += pDelta[i];
        }

        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            localSum += __shfl_down_sync(0xffffffff, localSum, offset);
        }

        if (threadIdx.x % 32 == 0)
        {
            sSum[threadIdx.x / 32] = localSum;
        }

        cg::sync(cta);

        if (threadIdx.x < 32)
        {
#pragma unroll
            for (int offset = 1; offset < 32; offset *= 2)
            {
                int idx = 2 * offset * threadIdx.x;
                if (idx < 32)
                {
                    sSum[idx] += sSum[idx + offset];
                }
            }

            if (threadIdx.x == 0)
            {
                sSum[0] /= static_cast<float>(batch);
                float vdw = pBiasVelocity[pos];
                float sdw = pBiasGradientVelocity[pos];

                vdw = beta1 * vdw + (1.0f - beta1) * sSum[0];
                sdw = beta2 * sdw + (1.0f - beta2) * sSum[0] * sSum[0];

                t += 1.0f;
                pBiasVelocity[pos] = vdw;
                pBiasGradientVelocity[pos] = sdw;

                float vdw_corrected = vdw / (1.0f - powf(beta1, t));
                float sdw_corrected = sdw / (1.0f - powf(beta2, t));
                float dw = alpha * vdw_corrected / (sqrtf(sdw_corrected) + 1.0e-8f);

                pBias[pos] -= dw;
            }
        }
    }
}

void kAdamUpdateBiases(float alpha, float mu, float mu1, float t, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias, cudaStream_t stream) {
    uint32_t blocks = CalculateBlocks(width);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = width / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDeltaPartitions(NUM_PARTITIONS);
    std::vector<float*> pBiasVelocityPartitions(NUM_PARTITIONS);
    std::vector<float*> pBiasGradientVelocityPartitions(NUM_PARTITIONS);
    std::vector<float*> pBiasPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDeltaPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBiasVelocityPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBiasGradientVelocityPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBiasPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pDeltaPartitions[i], &pDelta[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBiasVelocityPartitions[i], &pBiasVelocity[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBiasGradientVelocityPartitions[i], &pBiasGradientVelocity[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBiasPartitions[i], &pBias[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDeltaPartitions.begin(), pDeltaPartitions.end(), rng);
        std::shuffle(pBiasVelocityPartitions.begin(), pBiasVelocityPartitions.end(), rng);
        std::shuffle(pBiasGradientVelocityPartitions.begin(), pBiasGradientVelocityPartitions.end(), rng);
        std::shuffle(pBiasPartitions.begin(), pBiasPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &mu, &mu1, &t, &batch, &local_chunk_size, &pDeltaPartitions[task % NUM_PARTITIONS], &pBiasVelocityPartitions[task % NUM_PARTITIONS], &pBiasGradientVelocityPartitions[task % NUM_PARTITIONS], &pBiasPartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAdamUpdateBiases_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDeltaPartitions[i % NUM_PARTITIONS], (void*)pDeltaPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDeltaPartitions[i % NUM_PARTITIONS], (void*)pDeltaPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDeltaPartitions[i % NUM_PARTITIONS], (void*)pDeltaPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDeltaPartitions[i % NUM_PARTITIONS], (void*)pDeltaPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDeltaPartitions[i % NUM_PARTITIONS], (void*)pDeltaPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pDelta[rank * chunk_size + i * local_chunk_size], pDeltaPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBiasVelocity[rank * chunk_size + i * local_chunk_size], pBiasVelocityPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBiasGradientVelocity[rank * chunk_size + i * local_chunk_size], pBiasGradientVelocityPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBias[rank * chunk_size + i * local_chunk_size], pBiasPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDeltaPartitions[i]);
        cudaFree(pBiasVelocityPartitions[i]);
        cudaFree(pBiasGradientVelocityPartitions[i]);
        cudaFree(pBiasPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kNesterovUpdateWeights_kernel(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g = pWeightGradient[pos];
        float w = pWeight[pos];
        float vOld = pWeightVelocity[pos];
        float vNew = mu * vOld + alpha * (g - lambda * w - lambda1 * sgn(w));
        pWeightVelocity[pos] = vNew;
        w = w + vNew + mu * (vNew - vOld);
        pWeight[pos] = w;
    }
}

void kNesterovUpdateWeights(float alpha, float lambda, float lambda1, float mu,
    uint64_t size, float* pWeightVelocity, float* pWeightGradient,
    float* pWeight, cudaStream_t stream) {
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pWeightVelocityPartitions(NUM_PARTITIONS);
    std::vector<float*> pWeightGradientPartitions(NUM_PARTITIONS);
    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightVelocityPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pWeightGradientPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pWeightPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pWeightVelocityPartitions[i], &pWeightVelocity[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pWeightGradientPartitions[i], &pWeightGradient[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pWeightVelocityPartitions.begin(), pWeightVelocityPartitions.end(), rng);
        std::shuffle(pWeightGradientPartitions.begin(), pWeightGradientPartitions.end(), rng);
        std::shuffle(pWeightPartitions.begin(), pWeightPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &lambda, &lambda1, &mu, &local_chunk_size, &pWeightVelocityPartitions[task % NUM_PARTITIONS], &pWeightGradientPartitions[task % NUM_PARTITIONS], &pWeightPartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&kNesterovUpdateWeights_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pWeightVelocity[rank * chunk_size + i * local_chunk_size], pWeightVelocityPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pWeightGradient[rank * chunk_size + i * local_chunk_size], pWeightGradientPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pWeight[rank * chunk_size + i * local_chunk_size], pWeightPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pWeightVelocityPartitions[i]);
        cudaFree(pWeightGradientPartitions[i]);
        cudaFree(pWeightPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kNesterovUpdateBiases_kernel(float alpha, float mu, uint32_t batch, uint32_t width, const float* __restrict__ pDelta, float* __restrict__ pBiasVelocity, float* __restrict__ pBias)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum = 0.0f;

        const float* localDelta = pDelta + pos;

        for (uint32_t i = 0; i < batch; i += 4)
        {
            sum += localDelta[0];
            if (i + 1 < batch) sum += localDelta[width];
            if (i + 2 < batch) sum += localDelta[2 * width];
            if (i + 3 < batch) sum += localDelta[3 * width];

            localDelta += 4 * width;
        }

        sum /= static_cast<float>(batch);

        float vOld = pBiasVelocity[pos];
        float vNew = fmaf(mu, vOld, -alpha * sum);
        pBiasVelocity[pos] = vNew;
        pBias[pos] = fmaf(vNew + mu, vNew - vOld, pBias[pos]);
    }
}

void kNesterovUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias, cudaStream_t stream) {
    uint32_t blocks = CalculateBlocks(width);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = width / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDeltaPartitions(NUM_PARTITIONS);
    std::vector<float*> pBiasVelocityPartitions(NUM_PARTITIONS);
    std::vector<float*> pBiasPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDeltaPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBiasVelocityPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBiasPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pDeltaPartitions[i], &pDelta[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBiasVelocityPartitions[i], &pBiasVelocity[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBiasPartitions[i], &pBias[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDeltaPartitions.begin(), pDeltaPartitions.end(), rng);
        std::shuffle(pBiasVelocityPartitions.begin(), pBiasVelocityPartitions.end(), rng);
        std::shuffle(pBiasPartitions.begin(), pBiasPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &alpha, &mu, &batch, &local_chunk_size, &pDeltaPartitions[task % NUM_PARTITIONS], &pBiasVelocityPartitions[task % NUM_PARTITIONS], &pBiasPartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&kNesterovUpdateBiases_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDeltaPartitions[i % NUM_PARTITIONS], (void*)pDeltaPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDeltaPartitions[i % NUM_PARTITIONS], (void*)pDeltaPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDeltaPartitions[i % NUM_PARTITIONS], (void*)pDeltaPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDeltaPartitions[i % NUM_PARTITIONS], (void*)pDeltaPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDeltaPartitions[i % NUM_PARTITIONS], (void*)pDeltaPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pDelta[rank * chunk_size + i * local_chunk_size], pDeltaPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBiasVelocity[rank * chunk_size + i * local_chunk_size], pBiasVelocityPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBias[rank * chunk_size + i * local_chunk_size], pBiasPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDeltaPartitions[i]);
        cudaFree(pBiasVelocityPartitions[i]);
        cudaFree(pBiasPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kNesterovShiftWeights_kernel(float mu, uint64_t size, float* pWeightVelocity, float* pWeight)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float w = pWeight[pos];
        float v = pWeightVelocity[pos];
        pWeight[pos] = w + mu * v;
    }
}

void kNesterovShiftWeights(float mu, uint64_t size, float* pWeightVelocity, float* pWeight, cudaStream_t stream) {
    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pWeightVelocityPartitions(NUM_PARTITIONS);
    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightVelocityPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pWeightPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pWeightVelocityPartitions[i], &pWeightVelocity[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pWeightVelocityPartitions.begin(), pWeightVelocityPartitions.end(), rng);
        std::shuffle(pWeightPartitions.begin(), pWeightPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &mu, &local_chunk_size, &pWeightVelocityPartitions[task % NUM_PARTITIONS], &pWeightPartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&kNesterovShiftWeights_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pWeightVelocityPartitions[i % NUM_PARTITIONS], (void*)pWeightVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pWeightVelocity[rank * chunk_size + i * local_chunk_size], pWeightVelocityPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pWeight[rank * chunk_size + i * local_chunk_size], pWeightPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pWeightVelocityPartitions[i]);
        cudaFree(pWeightPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kNesterovShiftBiases_kernel(float mu, uint32_t width, float* pBiasVelocity, float* pBias)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float b = pBias[pos];
        float v = pBiasVelocity[pos];
        pBias[pos] = b + mu * v;
    }
}

void kNesterovShiftBiases(float mu, uint32_t width, float* pBiasVelocity, float* pBias, cudaStream_t stream) {
    uint32_t blocks = CalculateBlocks(width);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = width / (NUM_GPUS * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pBiasVelocityPartitions(NUM_PARTITIONS);
    std::vector<float*> pBiasPartitions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pBiasVelocityPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pBiasPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pBiasVelocityPartitions[i], &pBiasVelocity[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pBiasPartitions[i], &pBias[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pBiasVelocityPartitions.begin(), pBiasVelocityPartitions.end(), rng);
        std::shuffle(pBiasPartitions.begin(), pBiasPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &mu, &local_chunk_size, &pBiasVelocityPartitions[task % NUM_PARTITIONS], &pBiasPartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&kNesterovShiftBiases_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pBiasVelocityPartitions[i % NUM_PARTITIONS], (void*)pBiasVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pBiasVelocityPartitions[i % NUM_PARTITIONS], (void*)pBiasVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pBiasVelocityPartitions[i % NUM_PARTITIONS], (void*)pBiasVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pBiasVelocityPartitions[i % NUM_PARTITIONS], (void*)pBiasVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pBiasVelocityPartitions[i % NUM_PARTITIONS], (void*)pBiasVelocityPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pBiasVelocity[rank * chunk_size + i * local_chunk_size], pBiasVelocityPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        cudaMemcpyAsync(&pBias[rank * chunk_size + i * local_chunk_size], pBiasPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pBiasVelocityPartitions[i]);
        cudaFree(pBiasPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kRMSPropUpdateWeights_kernel(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g = pWeightGradient[pos];
        float w = pWeight[pos];
        float v = pWeightVelocity[pos];
        g -= lambda * w + lambda1 * sgn(w);
        v = mu * v + (1.0f - mu) * g * g;
        pWeightVelocity[pos] = v;
        pWeight[pos] = w + alpha * g * rsqrt(max(0.000000001f, v));
    }
}

void kRMSPropUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight, cudaStream_t stream) {
    if (size == 0)
        return;

    uint32_t blocks = CalculateBlocks(size);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* kernelArgs[] = { &alpha, &lambda, &lambda1, &mu, &size, &pWeightVelocity, &pWeightGradient, &pWeight };
            cudaLaunchKernel(reinterpret_cast<void*>(&kRMSPropUpdateWeights_kernel), blocks, getGpu()._threadsPerBlock, kernelArgs, 0, streams[targetGPU]);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pWeightVelocity, (void*)pWeightVelocity, size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pWeightVelocity, (void*)pWeightVelocity, size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pWeightVelocity, (void*)pWeightVelocity, size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pWeightVelocity, (void*)pWeightVelocity, size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pWeightVelocity, (void*)pWeightVelocity, size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kRMSPropUpdateBiases_kernel(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum = 0.0f;
        pDelta += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }
        sum /= (float)batch;

        float v = pBiasVelocity[pos];
        v = mu * v + (1.0f - mu) * sum * sum;
        pBiasVelocity[pos] = v;
        pBias[pos] -= alpha * sum * rsqrt(max(0.000000001f, v));
    }
}

void kRMSPropUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias, cudaStream_t stream) {
    if (batch == 0)
        return;

    uint32_t blocks = CalculateBlocks(width);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* kernelArgs[] = { &alpha, &mu, &batch, &width, &pDelta, &pBiasVelocity, &pBias };
            cudaLaunchKernel(reinterpret_cast<void*>(&kRMSPropUpdateBiases_kernel), blocks, getGpu()._threadsPerBlock, kernelArgs, 0, streams[targetGPU]);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pBiasVelocity, (void*)pBiasVelocity, width, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pBiasVelocity, (void*)pBiasVelocity, width, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pBiasVelocity, (void*)pBiasVelocity, width, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pBiasVelocity, (void*)pBiasVelocity, width, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pBiasVelocity, (void*)pBiasVelocity, width, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
invokeOutput_32_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    __shared__ volatile float sKey[64 * 4];
    __shared__ volatile uint32_t sValue[64 * 4];


    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        float* pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[64 * offset];
        volatile uint32_t* psValue = &sValue[64 * offset];

        float k0 = -MAX_VALUE;
        float k1 = -MAX_VALUE;
        uint32_t v0 = 0;
        uint32_t v1 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += cData._warpSize;

        float minValue = -MAX_VALUE;
        uint32_t rpos = 32;
        uint32_t bufferSize = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -MAX_VALUE;
            uint32_t value = wpos;
            if (wpos < width)
            {
                key = pOutput[wpos];
            }

            uint32_t count = __ballot_sync(0xFFFFFFFF, key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 32)
            {
                k1 = psKey[tgx];
                v1 = psValue[tgx];
                bool flag;
                BITONIC_SORT_64_LARGE();

                minValue = __shfl_sync(0xFFFFFFFF, k0, cData._warpSize - 1);

                bufferSize -= 32;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 32];
                    psValue[tgx] = psValue[tgx + 32];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 32))
        {
            k1 = -MAX_VALUE;
            v1 = 0;

            if (tgx < bufferSize)
            {
                k1 = psKey[tgx];
                v1 = psValue[tgx];
            }
            BITONIC_SORT_64();
        }

        float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
    }
}


__global__ void
__launch_bounds__(256, 4)
invokeOutput_64_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    __shared__ volatile float sKey[96 * 4];
    __shared__ volatile uint32_t sValue[96 * 4];


    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        float* pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[96 * offset];
        volatile uint32_t* psValue = &sValue[96 * offset];

        float k0 = -MAX_VALUE;
        float k1 = -MAX_VALUE;
        float k2 = -MAX_VALUE;
        float k3 = -MAX_VALUE;
        uint32_t v0 = 0;
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        uint32_t v3 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutput[wpos];
            v1 = wpos;
        }
        wpos += cData._warpSize;


        float minValue = -MAX_VALUE;
        uint32_t rpos = 64;
        uint32_t bufferSize = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -MAX_VALUE;
            uint32_t value = wpos;
            if (wpos < width)
            {
                key = pOutput[wpos];
            }

            uint32_t count = __ballot_sync(0xFFFFFFFF, key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 64)
            {
                k2 = psKey[tgx];
                v2 = psValue[tgx];
                k3 = psKey[tgx + cData._warpSize];
                v3 = psValue[tgx + cData._warpSize];
                bool flag;
                BITONICSORT128_128();

                minValue = __shfl_sync(0xFFFFFFFF, k1, cData._warpSize - 1);

                bufferSize -= 64;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 64];
                    psValue[tgx] = psValue[tgx + 64];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 64))
        {
            k2 = -MAX_VALUE;
            k3 = -MAX_VALUE;
            v2 = 0;
            v3 = 0;

            if (tgx < bufferSize)
            {
                k2 = psKey[tgx];
                v2 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k3 = psKey[tgx + cData._warpSize];
                v3 = psValue[tgx + cData._warpSize];
            }

            BITONICSORT128_128();
        }

        float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += cData._warpSize;
    }
}

__global__ void
__launch_bounds__(256, 4)
invokeOutput_128_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    __shared__ volatile float sKey[160 * 4];
    __shared__ volatile uint32_t sValue[160 * 4];


    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;


    if (pos < batch)
    {
        float* pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[160 * offset];
        volatile uint32_t* psValue = &sValue[160 * offset];

        float k0 = -MAX_VALUE;
        float k1 = -MAX_VALUE;
        float k2 = -MAX_VALUE;
        float k3 = -MAX_VALUE;
        float k4 = -MAX_VALUE;
        float k5 = -MAX_VALUE;
        float k6 = -MAX_VALUE;
        float k7 = -MAX_VALUE;
        uint32_t v0 = 0;
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        uint32_t v3 = 0;
        uint32_t v4 = 0;
        uint32_t v5 = 0;
        uint32_t v6 = 0;
        uint32_t v7 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutput[wpos];
            v1 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k2 = pOutput[wpos];
            v2 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k3 = pOutput[wpos];
            v3 = wpos;
        }

        float minValue = -MAX_VALUE;
        uint32_t rpos = 128;
        uint32_t bufferSize = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -MAX_VALUE;
            uint32_t value = wpos;
            if (wpos < width)
            {
                key = pOutput[wpos];
            }

            uint32_t count = __ballot_sync(0xFFFFFFFF, key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 128)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                minValue = __shfl_sync(0xFFFFFFFF, k3, cData._warpSize - 1);

                bufferSize -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 128];
                    psValue[tgx] = psValue[tgx + 128];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 128))
        {
            k4 = -MAX_VALUE;
            k5 = -MAX_VALUE;
            k6 = -MAX_VALUE;
            k7 = -MAX_VALUE;
            v4 = 0;
            v5 = 0;
            v6 = 0;
            v7 = 0;

            if (tgx < bufferSize)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
            }

            BITONICSORT256_256();
        }

        float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k2;
            pValue[wpos] = v2;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k3;
            pValue[wpos] = v3;
        }
    }
}

__global__ void __launch_bounds__(256, 4) invokeOutput_256_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    __shared__ volatile float sKey[288 * 4];
    __shared__ volatile uint32_t sValue[288 * 4];


    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;


    if (pos < batch)
    {
        float* pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[288 * offset];
        volatile uint32_t* psValue = &sValue[288 * offset];

        float k0 = -MAX_VALUE;
        float k1 = -MAX_VALUE;
        float k2 = -MAX_VALUE;
        float k3 = -MAX_VALUE;
        float k4 = -MAX_VALUE;
        float k5 = -MAX_VALUE;
        float k6 = -MAX_VALUE;
        float k7 = -MAX_VALUE;
        float k8 = -MAX_VALUE;
        float k9 = -MAX_VALUE;
        float k10 = -MAX_VALUE;
        float k11 = -MAX_VALUE;
        float k12 = -MAX_VALUE;
        float k13 = -MAX_VALUE;
        float k14 = -MAX_VALUE;
        float k15 = -MAX_VALUE;
        uint32_t v0 = 0;
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        uint32_t v3 = 0;
        uint32_t v4 = 0;
        uint32_t v5 = 0;
        uint32_t v6 = 0;
        uint32_t v7 = 0;
        uint32_t v8 = 0;
        uint32_t v9 = 0;
        uint32_t v10 = 0;
        uint32_t v11 = 0;
        uint32_t v12 = 0;
        uint32_t v13 = 0;
        uint32_t v14 = 0;
        uint32_t v15 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutput[wpos];
            v1 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k2 = pOutput[wpos];
            v2 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k3 = pOutput[wpos];
            v3 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k4 = pOutput[wpos];
            v4 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k5 = pOutput[wpos];
            v5 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k6 = pOutput[wpos];
            v6 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k7 = pOutput[wpos];
            v7 = wpos;
        }

        float minValue = -MAX_VALUE;
        uint32_t rpos = 256;
        uint32_t bufferSize = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -MAX_VALUE;
            uint32_t value = wpos;
            if (wpos < width)
            {
                key = pOutput[wpos];
            }

            uint32_t count = __ballot_sync(0xFFFFFFFF, key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 256)
            {
                k8 = psKey[tgx];
                v8 = psValue[tgx];
                k9 = psKey[tgx + cData._warpSize];
                v9 = psValue[tgx + cData._warpSize];
                k10 = psKey[tgx + 2 * cData._warpSize];
                v10 = psValue[tgx + 2 * cData._warpSize];
                k11 = psKey[tgx + 3 * cData._warpSize];
                v11 = psValue[tgx + 3 * cData._warpSize];
                k12 = psKey[tgx + 4 * cData._warpSize];
                v12 = psValue[tgx + 4 * cData._warpSize];
                k13 = psKey[tgx + 5 * cData._warpSize];
                v13 = psValue[tgx + 5 * cData._warpSize];
                k14 = psKey[tgx + 6 * cData._warpSize];
                v14 = psValue[tgx + 6 * cData._warpSize];
                k15 = psKey[tgx + 7 * cData._warpSize];
                v15 = psValue[tgx + 7 * cData._warpSize];
                bool flag;
                BITONICSORT512_512();

                minValue = __shfl_sync(0xFFFFFFFF, k7, cData._warpSize - 1);

                bufferSize -= 256;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 256];
                    psValue[tgx] = psValue[tgx + 256];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 256))
        {
            k8 = -MAX_VALUE;
            k9 = -MAX_VALUE;
            k10 = -MAX_VALUE;
            k11 = -MAX_VALUE;
            k12 = -MAX_VALUE;
            k13 = -MAX_VALUE;
            k14 = -MAX_VALUE;
            k15 = -MAX_VALUE;
            v8 = 0;
            v9 = 0;
            v10 = 0;
            v11 = 0;
            v12 = 0;
            v13 = 0;
            v14 = 0;
            v15 = 0;

            if (tgx < bufferSize)
            {
                k8 = psKey[tgx];
                v8 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k9 = psKey[tgx + cData._warpSize];
                v9 = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k10 = psKey[tgx + 2 * cData._warpSize];
                v10 = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k11 = psKey[tgx + 3 * cData._warpSize];
                v11 = psValue[tgx + 3 * cData._warpSize];
            }
            if (tgx + 4 * cData._warpSize < bufferSize)
            {
                k12 = psKey[tgx + 4 * cData._warpSize];
                v12 = psValue[tgx + 4 * cData._warpSize];
            }
            if (tgx + 5 * cData._warpSize < bufferSize)
            {
                k13 = psKey[tgx + 5 * cData._warpSize];
                v13 = psValue[tgx + 5 * cData._warpSize];
            }
            if (tgx + 6 * cData._warpSize < bufferSize)
            {
                k14 = psKey[tgx + 6 * cData._warpSize];
                v14 = psValue[tgx + 6 * cData._warpSize];
            }
            if (tgx + 7 * cData._warpSize < bufferSize)
            {
                k15 = psKey[tgx + 7 * cData._warpSize];
                v15 = psValue[tgx + 7 * cData._warpSize];
            }

            BITONICSORT512_512();
        }

        float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k8;
            pValue[wpos] = v8;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k9;
            pValue[wpos] = v9;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k10;
            pValue[wpos] = v10;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k11;
            pValue[wpos] = v11;
        }
        if (wpos < k)
        {
            pKey[wpos] = k12;
            pValue[wpos] = v12;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k13;
            pValue[wpos] = v13;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k14;
            pValue[wpos] = v14;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k15;
            pValue[wpos] = v15;
        }
    }
}

void invokeOutput(float* pOutput, float* pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    if (batch == 0)
        return;

    uint32_t blocks = (batch + 3) / 4;

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = batch / (NUM_GPUS * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pOutput, &pKey, &pValue, &batch, &width, &k };

            if (k <= 32) {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeOutput_32_kernel), blocks, 128, args, 0, streams[targetGPU]);
            }
            else if (k <= 64) {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeOutput_64_kernel), blocks, 128, args, 0, streams[targetGPU]);
            }
            else if (k <= 128) {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeOutput_128_kernel), blocks, 128, args, 0, streams[targetGPU]);
            }
            else {
                cudaLaunchKernel(reinterpret_cast<void*>(&invokeOutput_256_kernel), blocks, 128, args, 0, streams[targetGPU]);
            }

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pOutput, (void*)pOutput, width, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pOutput, (void*)pOutput, width, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pOutput, (void*)pOutput, width, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pOutput, (void*)pOutput, width, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pOutput, (void*)pOutput, width, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
invokeOutput_kernel(float* pOutputKey, float* pOutputValue, float* pKeyBuffer, float* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    __shared__ volatile float sKey[160 * 4];
    __shared__ volatile float sValue[160 * 4];


    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;


    if (pos < batch)
    {
        pOutputKey += pos * width;
        pOutputValue += pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[160 * offset];
        volatile float* psValue = &sValue[160 * offset];

        float k0 = -MAX_VALUE;
        float k1 = -MAX_VALUE;
        float k2 = -MAX_VALUE;
        float k3 = -MAX_VALUE;
        float k4 = -MAX_VALUE;
        float k5 = -MAX_VALUE;
        float k6 = -MAX_VALUE;
        float k7 = -MAX_VALUE;
        float v0 = 0.0f;
        float v1 = 0.0f;
        float v2 = 0.0f;
        float v3 = 0.0f;
        float v4 = 0.0f;
        float v5 = 0.0f;
        float v6 = 0.0f;
        float v7 = 0.0f;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutputKey[wpos];
            v0 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutputKey[wpos];
            v1 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k2 = pOutputKey[wpos];
            v2 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k3 = pOutputKey[wpos];
            v3 = pOutputValue[wpos];
        }

        float minValue = -MAX_VALUE;
        uint32_t rpos = 128;
        uint32_t bufferSize = 0;
        float key1, key2;
        float value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -MAX_VALUE;
            float value = 0.0f;
            if (wpos < width)
            {
                key = pOutputKey[wpos];
                value = pOutputValue[wpos];
            }

            uint32_t count = __ballot_sync(0xFFFFFFFF, key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 128)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                bufferSize -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 128];
                    psValue[tgx] = psValue[tgx + 128];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 128))
        {
            k4 = -MAX_VALUE;
            k5 = -MAX_VALUE;
            k6 = -MAX_VALUE;
            k7 = -MAX_VALUE;
            v4 = 0;
            v5 = 0;
            v6 = 0;
            v7 = 0;

            if (tgx < bufferSize)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
            }
            BITONICSORT256_256();
        }

        float* pKey = pKeyBuffer + pos * k;
        float* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k2;
            pValue[wpos] = v2;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k3;
            pValue[wpos] = v3;
        }
    }
}

template <typename T>
void invokeOutput(float* pOutputKey, float* pOutputValue, float* pKey, float* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    if (batch == 0)
        return;

    uint32_t blocks = (batch + 3) / 4;
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    dim3 gridSize(blocks);
    dim3 blockSize(128);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS]{};

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = batch / (NUM_GPUS * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pOutputKey, &pOutputValue, &pKey, &pValue, &batch, &width, &k, &local_chunk_size, &getGpu()._data._pAccumulator };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeOutput_kernel<T>), gridSize, blockSize, args, 0, streams[targetGPU]);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pOutputKey, (void*)pOutputKey, static_cast<size_t>(local_chunk_size) * width, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                    ncclAllReduce((const void*)pOutputValue, (void*)pOutputValue, static_cast<size_t>(local_chunk_size) * k, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pOutputKey, (void*)pOutputKey, static_cast<size_t>(local_chunk_size) * width, ncclFloat32, 0, ncclComm[i], streams[i]);
                    ncclBroadcast((const void*)pOutputValue, (void*)pOutputValue, static_cast<size_t>(local_chunk_size) * k, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pOutputKey, (void*)pOutputKey, static_cast<size_t>(local_chunk_size) * width, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                    ncclReduce((const void*)pOutputValue, (void*)pOutputValue, static_cast<size_t>(local_chunk_size) * k, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pOutputKey, (void*)pOutputKey, static_cast<size_t>(local_chunk_size) * width, ncclFloat32, ncclComm[i], streams[i]);
                    ncclAllGather((const void*)pOutputValue, (void*)pOutputValue, static_cast<size_t>(local_chunk_size) * k, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pOutputKey, (void*)pOutputKey, static_cast<size_t>(local_chunk_size) * width, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                    ncclReduceScatter((const void*)pOutputValue, (void*)pOutputValue, static_cast<size_t>(local_chunk_size) * k, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void
__launch_bounds__(256, 4)
invokeOutput_kernel(float* pOutputKey, uint32_t* pOutputValue, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    __shared__ volatile float sKey[160 * 4];
    __shared__ volatile uint32_t sValue[160 * 4];
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        pOutputKey += pos * width;
        pOutputValue += pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[160 * offset];
        volatile uint32_t* psValue = &sValue[160 * offset];

        float k0 = -MAX_VALUE;
        float k1 = -MAX_VALUE;
        float k2 = -MAX_VALUE;
        float k3 = -MAX_VALUE;
        float k4 = -MAX_VALUE;
        float k5 = -MAX_VALUE;
        float k6 = -MAX_VALUE;
        float k7 = -MAX_VALUE;
        uint32_t v0 = 0;
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        uint32_t v3 = 0;
        uint32_t v4 = 0;
        uint32_t v5 = 0;
        uint32_t v6 = 0;
        uint32_t v7 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutputKey[wpos];
            v0 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutputKey[wpos];
            v1 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k2 = pOutputKey[wpos];
            v2 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k3 = pOutputKey[wpos];
            v3 = pOutputValue[wpos];
        }

        float minValue = -MAX_VALUE;
        uint32_t rpos = 128;
        uint32_t bufferSize = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -MAX_VALUE;
            float value = 0.0f;
            if (wpos < width)
            {
                key = pOutputKey[wpos];
                value = pOutputValue[wpos];
            }

            uint32_t count = __ballot_sync(0xFFFFFFFF, key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 128)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                bufferSize -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 128];
                    psValue[tgx] = psValue[tgx + 128];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 128))
        {
            k4 = -MAX_VALUE;
            k5 = -MAX_VALUE;
            k6 = -MAX_VALUE;
            k7 = -MAX_VALUE;
            v4 = 0;
            v5 = 0;
            v6 = 0;
            v7 = 0;

            if (tgx < bufferSize)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
            }

            BITONICSORT256_256();
        }

        float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k2;
            pValue[wpos] = v2;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k3;
            pValue[wpos] = v3;
        }
    }
}

__global__ void __launch_bounds__(256, 4) kNormalizeWeights_kernel(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight)
{
    uint32_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        float r2 = 0.0f;
        float* pEnd = pWeight + outputStride * inputStride;
        pWeight += pos;
        float* p = pWeight;

        while (p < pEnd)
        {
            float x = *p;
            r2 += x * x;
            p += outputStride;
        }

        if (r2 > norm * norm)
        {
            norm *= rsqrt(r2);
            p = pWeight;
            while (p < pEnd)
            {
                *p *= norm;
                p += outputStride;
            }
        }
    }

}

void kNormalizeWeights(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight, cudaStream_t stream)
{
    if (outputStride == 0)
        return;

    uint32_t blocks = (outputStride + 127) / 128;
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    dim3 gridSize(blocks);
    dim3 blockSize(128);

    cudaStream_t streams[NUM_GPUS]{};
    cudaEvent_t events[NUM_GPUS]{};

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    unsigned long chunk_size = outputStride / (NUM_GPUS * NUM_NODES);
    unsigned long local_chunk_size = chunk_size;

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(unsigned long), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &norm, &outputStride, &inputStride, &pWeight, &local_chunk_size, &getGpu()._data._pAccumulator };
            cudaLaunchKernel(reinterpret_cast<void*>(&kNormalizeWeights_kernel), gridSize, blockSize, args, 0, streams[targetGPU]);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pWeight, (void*)pWeight, static_cast<size_t>(local_chunk_size) * inputStride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pWeight, (void*)pWeight, static_cast<size_t>(local_chunk_size) * inputStride, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pWeight, (void*)pWeight, static_cast<size_t>(local_chunk_size) * inputStride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pWeight, (void*)pWeight, static_cast<size_t>(local_chunk_size) * inputStride, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pWeight, (void*)pWeight, static_cast<size_t>(local_chunk_size) * inputStride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}


__global__ void __launch_bounds__(256, 4) invokeWeightMagnitudes_kernel(uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude)
{
    uint32_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        float r2 = 0.0f;
        float* pEnd = pWeight + outputStride * inputStride;
        pWeight += pos;
        float* p = pWeight;

        while (p < pEnd)
        {
            float x = *p;
            r2 += x * x;
            p += outputStride;
        }

        pMagnitude[pos] = r2;
    }

}

void invokeWeightMagnitudes(uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude, cudaStream_t stream) {
    if (outputStride == 0)
        return;

    uint32_t blocks = (outputStride + 127) / 128;

    dim3 gridSize(blocks);
    dim3 blockSize(128);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    size_t chunk_size = outputStride / (NUM_GPUS * NUM_NODES);
    size_t local_chunk_size = chunk_size;

    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    std::vector<float*> pMagnitudePartitions(NUM_PARTITIONS);

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pMagnitudePartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
        cudaMemcpyAsync(pMagnitudePartitions[i], &pMagnitude[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pWeightPartitions.begin(), pWeightPartitions.end(), rng);
        std::shuffle(pMagnitudePartitions.begin(), pMagnitudePartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &outputStride, &inputStride, &pWeightPartitions[task % NUM_PARTITIONS],
                &pMagnitudePartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeWeightMagnitudes_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pWeightPartitions[i % NUM_PARTITIONS], (void*)pWeightPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                    ncclAllReduce((const void*)pMagnitudePartitions[i % NUM_PARTITIONS], (void*)pMagnitudePartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pWeightPartitions[i % NUM_PARTITIONS], (void*)pWeightPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                    ncclBroadcast((const void*)pMagnitudePartitions[i % NUM_PARTITIONS], (void*)pMagnitudePartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pWeightPartitions[i % NUM_PARTITIONS], (void*)pWeightPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                    ncclReduce((const void*)pMagnitudePartitions[i % NUM_PARTITIONS], (void*)pMagnitudePartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pWeightPartitions[i % NUM_PARTITIONS], (void*)pWeightPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                    ncclAllGather((const void*)pMagnitudePartitions[i % NUM_PARTITIONS], (void*)pMagnitudePartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pWeightPartitions[i % NUM_PARTITIONS], (void*)pWeightPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                    ncclReduceScatter((const void*)pMagnitudePartitions[i % NUM_PARTITIONS], (void*)pMagnitudePartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pWeight[rank * chunk_size], pWeightPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(&pMagnitude[rank * chunk_size], pMagnitudePartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pWeightPartitions[i]);
        cudaFree(pMagnitudePartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kNormalizeWeightMagnitudes_kernel(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude)
{
    uint32_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        float r2 = pMagnitude[pos];
        float* pEnd = pWeight + outputStride * inputStride;
        pWeight += pos;
        float* p = pWeight;

        if (r2 > norm * norm)
        {
            norm *= rsqrt(r2);
            p = pWeight;
            while (p < pEnd)
            {
                *p *= norm;
                p += outputStride;
            }
        }
    }

}

void kNormalizeWeightMagnitudes(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude, cudaStream_t stream) {
    if (outputStride == 0)
        return;

    uint32_t blocks = (outputStride + 127) / 128;

    dim3 gridSize(blocks);
    dim3 blockSize(128);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    size_t chunk_size = outputStride / (NUM_GPUS * NUM_NODES);
    size_t local_chunk_size = chunk_size;

    std::vector<float*> pWeightPartitions(NUM_PARTITIONS);
    std::vector<float*> pMagnitudePartitions(NUM_PARTITIONS);

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pWeightPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pMagnitudePartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pWeightPartitions[i], &pWeight[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
        cudaMemcpyAsync(pMagnitudePartitions[i], &pMagnitude[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pWeightPartitions.begin(), pWeightPartitions.end(), rng);
        std::shuffle(pMagnitudePartitions.begin(), pMagnitudePartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &norm, &outputStride, &inputStride, &pWeightPartitions[task % NUM_PARTITIONS],
                &pMagnitudePartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&kNormalizeWeightMagnitudes_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pWeightPartitions[i % NUM_PARTITIONS], (void*)pWeightPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                    ncclAllReduce((const void*)pMagnitudePartitions[i % NUM_PARTITIONS], (void*)pMagnitudePartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pWeightPartitions[i % NUM_PARTITIONS], (void*)pWeightPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                    ncclBroadcast((const void*)pMagnitudePartitions[i % NUM_PARTITIONS], (void*)pMagnitudePartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pWeightPartitions[i % NUM_PARTITIONS], (void*)pWeightPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                    ncclReduce((const void*)pMagnitudePartitions[i % NUM_PARTITIONS], (void*)pMagnitudePartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pWeightPartitions[i % NUM_PARTITIONS], (void*)pWeightPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                    ncclAllGather((const void*)pMagnitudePartitions[i % NUM_PARTITIONS], (void*)pMagnitudePartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pWeightPartitions[i % NUM_PARTITIONS], (void*)pWeightPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                    ncclReduceScatter((const void*)pMagnitudePartitions[i % NUM_PARTITIONS], (void*)pMagnitudePartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pWeight[rank * chunk_size], pWeightPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(&pMagnitude[rank * chunk_size], pMagnitudePartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pWeightPartitions[i]);
        cudaFree(pMagnitudePartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) invokeScaledBiasedDropout_kernel(float* pUnit, float* pRandom, float p, float target, float a, float b, size_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float r = pRandom[pos];
        pUnit[pos] = (r < p) ? target : a * pUnit[pos] + b;
    }
}

void invokeScaledBiasedDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target, float a, float b, cudaStream_t stream) {
    curandGenerateUniform(getGpu()._RNG, pRandom, static_cast<size_t>(batch) * stride);
    if ((batch == 0) || (stride == 0))
        return;

    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * stride);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    size_t chunk_size = (static_cast<size_t>(batch) * stride) / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    size_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    std::vector<float*> pRandomPartitions(NUM_PARTITIONS);

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pRandomPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
        cudaMemcpyAsync(pRandomPartitions[i], &pRandom[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pUnitPartitions.begin(), pUnitPartitions.end(), rng);
        std::shuffle(pRandomPartitions.begin(), pRandomPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pUnitPartitions[task % NUM_PARTITIONS], &pRandomPartitions[task % NUM_PARTITIONS],
                &local_chunk_size, &p, &a, &b };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeScaledBiasedDropout_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                    ncclAllReduce((const void*)pRandomPartitions[i % NUM_PARTITIONS], (void*)pRandomPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                    ncclBroadcast((const void*)pRandomPartitions[i % NUM_PARTITIONS], (void*)pRandomPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                    ncclReduce((const void*)pRandomPartitions[i % NUM_PARTITIONS], (void*)pRandomPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                    ncclAllGather((const void*)pRandomPartitions[i % NUM_PARTITIONS], (void*)pRandomPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                    ncclReduceScatter((const void*)pRandomPartitions[i % NUM_PARTITIONS], (void*)pRandomPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size], pUnitPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(&pRandom[rank * chunk_size], pRandomPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pRandomPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) invokeDropout_kernel(float* pUnit, float* pRandom, float p, float scale, float target, size_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float r = pRandom[pos];
        pUnit[pos] = (r < p) ? target : scale * pUnit[pos];
    }
}

void invokeDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target, cudaStream_t stream) {
    curandGenerateUniform(getGpu()._RNG, pRandom, static_cast<size_t>(batch) * stride);
    if ((batch == 0) || (stride == 0))
        return;

    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(batch) * stride);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS]{};
    cudaEvent_t events[NUM_GPUS]{};

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    size_t chunk_size = (static_cast<size_t>(batch) * stride) / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    size_t local_chunk_size = chunk_size;

    std::vector<float*> pUnitPartitions(NUM_PARTITIONS);
    std::vector<float*> pRandomPartitions(NUM_PARTITIONS);

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pUnitPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pRandomPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pUnitPartitions[i], &pUnit[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
        cudaMemcpyAsync(pRandomPartitions[i], &pRandom[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pUnitPartitions.begin(), pUnitPartitions.end(), rng);
        std::shuffle(pRandomPartitions.begin(), pRandomPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pUnitPartitions[task % NUM_PARTITIONS], &pRandomPartitions[task % NUM_PARTITIONS],
                &local_chunk_size, &p, &target };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeDropout_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                    ncclAllReduce((const void*)pRandomPartitions[i % NUM_PARTITIONS], (void*)pRandomPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                    ncclBroadcast((const void*)pRandomPartitions[i % NUM_PARTITIONS], (void*)pRandomPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                    ncclReduce((const void*)pRandomPartitions[i % NUM_PARTITIONS], (void*)pRandomPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                    ncclAllGather((const void*)pRandomPartitions[i % NUM_PARTITIONS], (void*)pRandomPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pUnitPartitions[i % NUM_PARTITIONS], (void*)pUnitPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                    ncclReduceScatter((const void*)pRandomPartitions[i % NUM_PARTITIONS], (void*)pRandomPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pUnit[rank * chunk_size], pUnitPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(&pRandom[rank * chunk_size], pRandomPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pUnitPartitions[i]);
        cudaFree(pRandomPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) invokeMaxout_kernel(float* pSrc, size_t size, float* pDst)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float s = pSrc[pos];
        float d = pDst[pos];
        if (s > d)
            pDst[pos] = s;
    }
}

void invokeMaxout(float* pSrc, size_t size, float* pDst, cudaStream_t stream) {
    if (size == 0)
        return;

    uint32_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    size_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    size_t local_chunk_size = chunk_size;

    std::vector<float*> pSrcPartitions(NUM_PARTITIONS);
    std::vector<float*> pDstPartitions(NUM_PARTITIONS);

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pSrcPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pDstPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pSrcPartitions[i], &pSrc[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pSrcPartitions.begin(), pSrcPartitions.end(), rng);
        std::shuffle(pDstPartitions.begin(), pDstPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pSrcPartitions[task % NUM_PARTITIONS], &local_chunk_size, &pDstPartitions[task % NUM_PARTITIONS] };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeMaxout_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclMax, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclMax, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclMax, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pDst[rank * chunk_size], pDstPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pSrcPartitions[i]);
        cudaFree(pDstPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) invokeCosine_kernel(float* pVector1, float* pVector2, uint32_t stride, float* pDPOut, float* pAOut, float* pBOut, uint32_t outStride)
{
    __shared__ float sDP[64];
    __shared__ float sA[64];
    __shared__ float sB[64];

    cg::thread_block block = cg::this_thread_block();


    pVector1 += blockIdx.x * stride + threadIdx.x;
    pVector2 += blockIdx.x * stride + threadIdx.x;
    pDPOut += blockIdx.x * outStride;
    pAOut += blockIdx.x * outStride;
    pBOut += blockIdx.x * outStride;
    uint32_t pos = threadIdx.x;
    float dp = (float)0;
    float al = (float)0;
    float bl = (float)0;

    while (pos < stride)
    {
        float a = *pVector1;
        float b = *pVector2;
        dp += a * b;
        al += a * a;
        bl += b * b;
        pVector1 += blockDim.x;
        pVector2 += blockDim.x;
        pos += blockDim.x;
    }


    uint32_t tgx = threadIdx.x & cData._warpMask;
    dp += __shfl_sync(0xFFFFFFFF, dp, tgx ^ 1);
    al += __shfl_sync(0xFFFFFFFF, al, tgx ^ 1);
    bl += __shfl_sync(0xFFFFFFFF, bl, tgx ^ 1);
    dp += __shfl_sync(0xFFFFFFFF, dp, tgx ^ 2);
    al += __shfl_sync(0xFFFFFFFF, al, tgx ^ 2);
    bl += __shfl_sync(0xFFFFFFFF, bl, tgx ^ 2);
    dp += __shfl_sync(0xFFFFFFFF, dp, tgx ^ 4);
    al += __shfl_sync(0xFFFFFFFF, al, tgx ^ 4);
    bl += __shfl_sync(0xFFFFFFFF, bl, tgx ^ 4);
    dp += __shfl_sync(0xFFFFFFFF, dp, tgx ^ 8);
    al += __shfl_sync(0xFFFFFFFF, al, tgx ^ 8);
    bl += __shfl_sync(0xFFFFFFFF, bl, tgx ^ 8);
    dp += __shfl_sync(0xFFFFFFFF, dp, tgx ^ 16);
    al += __shfl_sync(0xFFFFFFFF, al, tgx ^ 16);
    bl += __shfl_sync(0xFFFFFFFF, bl, tgx ^ 16);
    if (tgx == 0)
    {
        uint32_t index = threadIdx.x >> cData._warpBits;
        sDP[index] = dp;
        sA[index] = al;
        sB[index] = bl;
    }
    block.sync();

    if (threadIdx.x < cData._warpSize)
    {
        uint32_t limit = (blockDim.x + cData._warpSize - 1) >> cData._warpBits;
        al = (threadIdx.x < limit) ? sA[threadIdx.x] : (float)0;
        bl = (threadIdx.x < limit) ? sB[threadIdx.x] : (float)0;
        dp = (threadIdx.x < limit) ? sDP[threadIdx.x] : (float)0;
        dp += __shfl_sync(0xFFFFFFFF, dp, tgx ^ 1);
        al += __shfl_sync(0xFFFFFFFF, al, tgx ^ 1);
        bl += __shfl_sync(0xFFFFFFFF, bl, tgx ^ 1);
        dp += __shfl_sync(0xFFFFFFFF, dp, tgx ^ 2);
        al += __shfl_sync(0xFFFFFFFF, al, tgx ^ 2);
        bl += __shfl_sync(0xFFFFFFFF, bl, tgx ^ 2);
        dp += __shfl_sync(0xFFFFFFFF, dp, tgx ^ 4);
        al += __shfl_sync(0xFFFFFFFF, al, tgx ^ 4);
        bl += __shfl_sync(0xFFFFFFFF, bl, tgx ^ 4);
        dp += __shfl_sync(0xFFFFFFFF, dp, tgx ^ 8);
        al += __shfl_sync(0xFFFFFFFF, al, tgx ^ 8);
        bl += __shfl_sync(0xFFFFFFFF, bl, tgx ^ 8);
        dp += __shfl_sync(0xFFFFFFFF, dp, tgx ^ 16);
        al += __shfl_sync(0xFFFFFFFF, al, tgx ^ 16);
        bl += __shfl_sync(0xFFFFFFFF, bl, tgx ^ 16);


        if (threadIdx.x == 0)
        {
            al = sqrt(al) + (float)1.0e-08;
            bl = sqrt(bl) + (float)1.0e-08;
            dp /= al * bl;
            *pAOut = al;
            *pBOut = bl;
            *pDPOut = dp;
        }
    }
}

void invokeCosine(float* pVector1In, float* pVector2In, uint32_t batch, uint32_t stride, float* pDPOut, float* pAOut, float* pBOut, uint32_t outStride, cudaStream_t stream) {
    if (batch == 0 || stride == 0)
        return;

    uint32_t blocks = batch;
    uint32_t threads = max(32, min(stride, getGpu()._threadsPerBlock));

    dim3 gridSize(blocks);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t events[NUM_GPUS];

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint32_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint32_t local_chunk_size = chunk_size;

    std::vector<float*> pVector1Partitions(NUM_PARTITIONS);
    std::vector<float*> pVector2Partitions(NUM_PARTITIONS);
    std::vector<float*> pDPOutPartitions(NUM_PARTITIONS);
    std::vector<float*> pAOutPartitions(NUM_PARTITIONS);
    std::vector<float*> pBOutPartitions(NUM_PARTITIONS);

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pVector1Partitions[i], static_cast<unsigned long long>(local_chunk_size) * stride * sizeof(float));
        cudaMalloc(&pVector2Partitions[i], static_cast<unsigned long long>(local_chunk_size) * stride * sizeof(float));
        cudaMalloc(&pDPOutPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pAOutPartitions[i], static_cast<unsigned long long>(local_chunk_size) * outStride * sizeof(float));
        cudaMalloc(&pBOutPartitions[i], static_cast<unsigned long long>(local_chunk_size) * outStride * sizeof(float));

        cudaMemcpyAsync(pVector1Partitions[i], &pVector1In[rank * chunk_size * stride + i * local_chunk_size * stride],
            static_cast<unsigned long long>(local_chunk_size) * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pVector2Partitions[i], &pVector2In[rank * chunk_size * stride + i * local_chunk_size * stride],
            static_cast<unsigned long long>(local_chunk_size) * stride * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint32_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pVector1Partitions.begin(), pVector1Partitions.end(), rng);
        std::shuffle(pVector2Partitions.begin(), pVector2Partitions.end(), rng);
        std::shuffle(pDPOutPartitions.begin(), pDPOutPartitions.end(), rng);
        std::shuffle(pAOutPartitions.begin(), pAOutPartitions.end(), rng);
        std::shuffle(pBOutPartitions.begin(), pBOutPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pVector1Partitions[task % NUM_PARTITIONS],
                             &pVector2Partitions[task % NUM_PARTITIONS], &stride,
                             &pDPOutPartitions[task % NUM_PARTITIONS], &pAOutPartitions[task % NUM_PARTITIONS],
                             &pBOutPartitions[task % NUM_PARTITIONS], &outStride, &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeCosine_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDPOutPartitions[i % NUM_PARTITIONS], (void*)pDPOutPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDPOutPartitions[i % NUM_PARTITIONS], (void*)pDPOutPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pAOutPartitions[i % NUM_PARTITIONS], (void*)pAOutPartitions[i % NUM_PARTITIONS],
                        static_cast<size_t>(local_chunk_size) * outStride, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pBOutPartitions[i % NUM_PARTITIONS], (void*)pBOutPartitions[i % NUM_PARTITIONS],
                        static_cast<size_t>(local_chunk_size) * outStride, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pAOutPartitions[i % NUM_PARTITIONS], (void*)pAOutPartitions[i % NUM_PARTITIONS],
                        static_cast<size_t>(local_chunk_size) * outStride, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pDPOut[rank * chunk_size], pDPOutPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(&pAOut[rank * chunk_size * outStride], pAOutPartitions[i % NUM_PARTITIONS],
            static_cast<unsigned long long>(local_chunk_size) * outStride * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(&pBOut[rank * chunk_size * outStride], pBOutPartitions[i % NUM_PARTITIONS],
            static_cast<unsigned long long>(local_chunk_size) * outStride * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pVector1Partitions[i]);
        cudaFree(pVector2Partitions[i]);
        cudaFree(pDPOutPartitions[i]);
        cudaFree(pAOutPartitions[i]);
        cudaFree(pBOutPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) invokeDotProduct_kernel(float* pVector1In, float* pVector2In, uint32_t strideIn, float* pDPOut, uint32_t strideOut)
{
    __shared__ float sDP[32];

    cg::thread_block block = cg::this_thread_block();

    pVector1In += blockIdx.x * strideIn + threadIdx.x;
    pVector2In += blockIdx.x * strideIn + threadIdx.x;
    pDPOut += blockIdx.x * strideOut;
    uint32_t pos = threadIdx.x;
    float dp = (float)0;


    while (pos < strideIn)
    {
        float a = *pVector1In;
        float b = *pVector2In;
        dp += a * b;
        pVector1In += blockDim.x;
        pVector2In += blockDim.x;
        pos += blockDim.x;
    }


    REDUCE(dp)
        uint32_t tgx = threadIdx.x & cData._warpMask;
    if (tgx == 0)
    {
        uint32_t index = threadIdx.x >> cData._warpBits;
        sDP[index] = dp;
    }
    block.sync();

    if (threadIdx.x < cData._warpSize)
    {
        uint32_t limit = (blockDim.x + cData._warpSize - 1) >> cData._warpBits;
        dp = (threadIdx.x < limit) ? sDP[threadIdx.x] : (float)0;
        REDUCE(dp)

            if (threadIdx.x == 0)
            {
                *pDPOut = dp;
            }
    }
}

void invokeDotProduct(float* pVector1In, float* pVector2In, uint32_t batch, uint32_t strideIn, float* pDPOut, uint32_t strideOut, cudaStream_t stream) {
    if (batch == 0 || strideIn == 0)
        return;

    uint32_t blocks = batch;
    uint32_t threads = max(32, min(strideIn, getGpu()._threadsPerBlock));

    dim3 gridSize(blocks);
    dim3 blockSize(threads);

    cudaStream_t streams[NUM_GPUS]{};
    cudaEvent_t events[NUM_GPUS]{};

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint32_t chunk_size = batch / (NUM_GPUS * NUM_NODES);
    uint32_t local_chunk_size = chunk_size;

    std::vector<float*> pVector1Partitions(NUM_PARTITIONS);
    std::vector<float*> pVector2Partitions(NUM_PARTITIONS);
    std::vector<float*> pDPOutPartitions(NUM_PARTITIONS);

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pVector1Partitions[i], static_cast<unsigned long long>(local_chunk_size) * strideIn * sizeof(float));
        cudaMalloc(&pVector2Partitions[i], static_cast<unsigned long long>(local_chunk_size) * strideIn * sizeof(float));
        cudaMalloc(&pDPOutPartitions[i], static_cast<unsigned long long>(local_chunk_size) * strideOut * sizeof(float));

        cudaMemcpyAsync(pVector1Partitions[i], &pVector1In[rank * chunk_size * strideIn + i * local_chunk_size * strideIn],
            static_cast<unsigned long long>(local_chunk_size) * strideIn * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pVector2Partitions[i], &pVector2In[rank * chunk_size * strideIn + i * local_chunk_size * strideIn],
            static_cast<unsigned long long>(local_chunk_size) * strideIn * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint32_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pVector1Partitions.begin(), pVector1Partitions.end(), rng);
        std::shuffle(pVector2Partitions.begin(), pVector2Partitions.end(), rng);
        std::shuffle(pDPOutPartitions.begin(), pDPOutPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pVector1Partitions[task % NUM_PARTITIONS],
                             &pVector2Partitions[task % NUM_PARTITIONS], &strideIn,
                             &pDPOutPartitions[task % NUM_PARTITIONS], &strideOut, &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&invokeDotProduct_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDPOutPartitions[i % NUM_PARTITIONS], (void*)pDPOutPartitions[i % NUM_PARTITIONS],
                        static_cast<size_t>(local_chunk_size) * strideOut, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDPOutPartitions[i % NUM_PARTITIONS], (void*)pDPOutPartitions[i % NUM_PARTITIONS],
                        static_cast<size_t>(local_chunk_size) * strideOut, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pDPOutPartitions[i % NUM_PARTITIONS], (void*)pDPOutPartitions[i % NUM_PARTITIONS],
                        static_cast<size_t>(local_chunk_size) * strideOut, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pDPOutPartitions[i % NUM_PARTITIONS], (void*)pDPOutPartitions[i % NUM_PARTITIONS],
                        static_cast<size_t>(local_chunk_size) * strideOut, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pDPOutPartitions[i % NUM_PARTITIONS], (void*)pDPOutPartitions[i % NUM_PARTITIONS],
                        static_cast<size_t>(local_chunk_size) * strideOut, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pDPOut[rank * chunk_size * strideOut + i * local_chunk_size * strideOut], pDPOutPartitions[i % NUM_PARTITIONS],
            static_cast<unsigned long long>(local_chunk_size)* strideOut * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pVector1Partitions[i]);
        cudaFree(pVector2Partitions[i]);
        cudaFree(pDPOutPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

#include <cub/cub.cuh>

template<typename KeyType, typename ValueType>
size_t kInitSort(uint32_t items, GpuBuffer<KeyType>* pbKey, GpuBuffer<ValueType>* pbValue)
{
    if (!pbKey || !pbValue) {
        throw std::invalid_argument("Null GpuBuffer pointers provided");
    }

    uint32_t itemStride = (items + ALIGNMENT_MASK) & ~ALIGNMENT_MASK;

    size_t tempBytes = 0;
    cub::DoubleBuffer<KeyType> d_keys(pbKey->_pDevData, pbKey->_pDevData + itemStride);
    cub::DoubleBuffer<ValueType> d_values(pbValue->_pDevData, pbValue->_pDevData + itemStride);

    cudaError_t err = cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, d_keys, d_values, items);
    if (err != cudaSuccess) {
        throw std::runtime_error("DeviceRadixSort failed: " + std::string(cudaGetErrorString(err)));
    }

    return tempBytes;
}

template<typename KeyType, typename ValueType>
bool kSort(uint32_t items, KeyType* pKey0, KeyType* pKey1, ValueType* pValue0, ValueType* pValue1, char* pTemp, size_t tempBytes)
{
    if (!pKey0 || !pKey1 || !pValue0 || !pValue1 || !pTemp) {
        return false;
    }

    cub::DoubleBuffer<KeyType> d_keys(pKey0, pKey1);
    cub::DoubleBuffer<ValueType> d_values(pValue0, pValue1);

    cudaError_t err = cub::DeviceRadixSort::SortPairs(pTemp, tempBytes, d_keys, d_values, items);
    if (err != cudaSuccess) {
        return false;
    }

    return true;
}

template size_t kInitSort<float, float>(uint32_t, GpuBuffer<float>*, GpuBuffer<float>*);
template size_t kInitSort<uint32_t, float>(uint32_t, GpuBuffer<uint32_t>*, GpuBuffer<float>*);
template size_t kInitSort<float, uint32_t>(uint32_t, GpuBuffer<float>*, GpuBuffer<uint32_t>*);
template size_t kInitSort<uint32_t, uint32_t>(uint32_t, GpuBuffer<uint32_t>*, GpuBuffer<uint32_t>*);

template bool kSort<float, float>(uint32_t, float*, float*, float*, float*, char*, size_t);
template bool kSort<float, uint32_t>(uint32_t, float*, float*, uint32_t*, uint32_t*, char*, size_t);
template bool kSort<uint32_t, float>(uint32_t, uint32_t*, uint32_t*, float*, float*, char*, size_t);
template bool kSort<uint32_t, uint32_t>(uint32_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, char*, size_t);


__global__ void __launch_bounds__(256, 4) kAddScaleBuffers_kernel(float* pDst, float* pSrc, float scale, uint64_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
        *(pDst + pos) += *(pSrc + pos) * scale;
}

void kAddScaleBuffers(float* pDst, float* pSrc, float scale, uint64_t size, cudaStream_t stream) {
    if (size == 0)
        return;

    uint64_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS]{};
    cudaEvent_t events[NUM_GPUS]{};

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDstPartitions(NUM_PARTITIONS);
    std::vector<float*> pSrcPartitions(NUM_PARTITIONS);

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDstPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pSrcPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pDstPartitions[i], &pDst[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pSrcPartitions[i], &pSrc[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDstPartitions.begin(), pDstPartitions.end(), rng);
        std::shuffle(pSrcPartitions.begin(), pSrcPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDstPartitions[task % NUM_PARTITIONS], &pSrcPartitions[task % NUM_PARTITIONS],
                             &scale, &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAddScaleBuffers_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pDst[rank * chunk_size + i * local_chunk_size], pDstPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDstPartitions[i]);
        cudaFree(pSrcPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kAddBuffers_kernel(float* pDst, float* pSrc, uint64_t size)
{
    uint64_t pos = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos < size)
        *(pDst + pos) += *(pSrc + pos);
}

void kAddBuffers(float* pDst, float* pSrc, uint64_t size, cudaStream_t stream) {
    if (size == 0)
        return;

    uint64_t blocks = CalculateBlocks(size);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS]{};
    cudaEvent_t events[NUM_GPUS]{};

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint64_t chunk_size = size / (static_cast<unsigned long long>(NUM_GPUS) * NUM_NODES);
    uint64_t local_chunk_size = chunk_size;

    std::vector<float*> pDstPartitions(NUM_PARTITIONS);
    std::vector<float*> pSrcPartitions(NUM_PARTITIONS);

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDstPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pSrcPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pDstPartitions[i], &pDst[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pSrcPartitions[i], &pSrc[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDstPartitions.begin(), pDstPartitions.end(), rng);
        std::shuffle(pSrcPartitions.begin(), pSrcPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDstPartitions[task % NUM_PARTITIONS],
                             &pSrcPartitions[task % NUM_PARTITIONS], &local_chunk_size };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAddBuffers_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pDst[rank * chunk_size + i * local_chunk_size], pDstPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDstPartitions[i]);
        cudaFree(pSrcPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kAddBuffers2D_kernel(float* pDst, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width)
{
    uint64_t yOffset = static_cast<uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (yOffset < width)
    {
        uint64_t dpos = blockIdx.x * static_cast<unsigned long long>(dpitch) + yOffset;
        uint64_t spos = blockIdx.x * static_cast<unsigned long long>(spitch) + yOffset;
        pDst[dpos] += pSrc[spos];
    }
}

void kAddBuffers2D(float* pDst, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream) {
    if ((height == 0) || (width == 0))
        return;

    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(height) * width);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS]{};
    cudaEvent_t events[NUM_GPUS]{};

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint32_t chunk_size = (height * width) / (NUM_GPUS * NUM_NODES);
    uint32_t local_chunk_size = chunk_size;

    std::vector<float*> pDstPartitions(NUM_PARTITIONS);
    std::vector<float*> pSrcPartitions(NUM_PARTITIONS);

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDstPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pSrcPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pDstPartitions[i], &pDst[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pSrcPartitions[i], &pSrc[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint32_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDstPartitions.begin(), pDstPartitions.end(), rng);
        std::shuffle(pSrcPartitions.begin(), pSrcPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDstPartitions[task % NUM_PARTITIONS], &dpitch,
                             &pSrcPartitions[task % NUM_PARTITIONS], &spitch, &width };
            cudaLaunchKernel(reinterpret_cast<void*>(&kAddBuffers2D_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pDst[rank * chunk_size + i * local_chunk_size], pDstPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDstPartitions[i]);
        cudaFree(pSrcPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

__global__ void __launch_bounds__(256, 4) kCopy2D_kernel(float* pDst, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width)
{
    const uint64_t globalIdx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const uint64_t dpos_base = blockIdx.y * static_cast<uint64_t>(dpitch);
    const uint64_t spos_base = blockIdx.y * static_cast<uint64_t>(spitch);

    if (globalIdx < width)
    {
        const uint64_t dpos = dpos_base + globalIdx;
        const uint64_t spos = spos_base + globalIdx;

        float4* pDstFloat4 = reinterpret_cast<float4*>(pDst + dpos);
        const float4* pSrcFloat4 = reinterpret_cast<const float4*>(pSrc + spos);

        *pDstFloat4 = *pSrcFloat4;
    }
}

void kCopy2D(float* pDst, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream) {
    if ((height == 0) || (width == 0))
        return;

    uint32_t blocks = CalculateBlocks(static_cast<uint64_t>(height) * width);

    dim3 gridSize(blocks);
    dim3 blockSize(getGpu()._threadsPerBlock);

    cudaStream_t streams[NUM_GPUS]{};
    cudaEvent_t events[NUM_GPUS]{};

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

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        ncclCommInitRank(&ncclComm[i], world_size / (NUM_GPUS * NUM_NODES), ncclId, local_rank);
    }

    uint32_t chunk_size = (height * width) / (NUM_GPUS * NUM_NODES);
    uint32_t local_chunk_size = chunk_size;

    std::vector<float*> pDstPartitions(NUM_PARTITIONS);
    std::vector<float*> pSrcPartitions(NUM_PARTITIONS);

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaMalloc(&pDstPartitions[i], local_chunk_size * sizeof(float));
        cudaMalloc(&pSrcPartitions[i], local_chunk_size * sizeof(float));

        cudaMemcpyAsync(pDstPartitions[i], &pDst[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);

        cudaMemcpyAsync(pSrcPartitions[i], &pSrc[rank * chunk_size + i * local_chunk_size],
            local_chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[local_rank % NUM_GPUS]);
    }

    std::vector<int> commGroup(NUM_GPUS);
    std::iota(commGroup.begin(), commGroup.end(), 0);

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        if (local_rank == 0) {
            local_chunk_size = chunk_size + (iteration * chunk_size / NUM_ITERATIONS);
        }
        MPI_Bcast(&local_chunk_size, sizeof(uint32_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::shuffle(commGroup.begin(), commGroup.end(), rng);
        std::shuffle(pDstPartitions.begin(), pDstPartitions.end(), rng);
        std::shuffle(pSrcPartitions.begin(), pSrcPartitions.end(), rng);

        for (int task = 0; task < NUM_TASKS; task++) {
            int targetGPU = commGroup[task % NUM_GPUS];
            cudaSetDevice(targetGPU);

            void* args[] = { &pDstPartitions[task % NUM_PARTITIONS], &dpitch,
                             &pSrcPartitions[task % NUM_PARTITIONS], &spitch, &width };
            cudaLaunchKernel(reinterpret_cast<void*>(&kCopy2D_kernel), gridSize, blockSize, args);

            cudaEventRecord(events[targetGPU], streams[targetGPU]);
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            cudaSetDevice(i);
            cudaStreamWaitEvent(streams[i], events[i], 0);
        }

        ncclGroupStart();

        for (int collective = 0; collective < NUM_COLLECTIVES; collective++) {
            for (int i = 0; i < NUM_GPUS; i++) {
                cudaSetDevice(i);

                if (collective == 0) {
                    ncclAllReduce((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
                else if (collective == 1) {
                    ncclBroadcast((const void*)pDstPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 2) {
                    ncclReduce((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, 0, ncclComm[i], streams[i]);
                }
                else if (collective == 3) {
                    ncclAllGather((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclComm[i], streams[i]);
                }
                else if (collective == 4) {
                    ncclReduceScatter((const void*)pSrcPartitions[i % NUM_PARTITIONS], (void*)pDstPartitions[i % NUM_PARTITIONS],
                        local_chunk_size, ncclFloat32, ncclSum, ncclComm[i], streams[i]);
                }
            }
        }

        ncclGroupEnd();

        for (int barrier = 0; barrier < NUM_BARRIERS; barrier++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (local_rank == 0) {
                std::cout << "Iteration " << iteration << ", Barrier " << barrier << " completed." << std::endl;
            }
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&pDst[rank * chunk_size + i * local_chunk_size], pDstPartitions[i % NUM_PARTITIONS],
            local_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_PARTITIONS; i++) {
        cudaSetDevice(local_rank % NUM_GPUS);
        cudaFree(pDstPartitions[i]);
        cudaFree(pSrcPartitions[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(ncclComm[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    MPI_Finalize();
}

#define EXPLICITLY_INSTANTIATE_KERNELS(T)                                                                                                                                                       \
template void kLoadSparseAnalogDenoisedInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*);                                           \
template void kLoadIndexedSparseAnalogDenoisedInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*);                         \
template void kLoadSparseAnalogInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*);                                                             \
template void kLoadIndexedSparseAnalogInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*);                                           \
template void invokeSparseAnalogZ<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float);                                             \
template void invokeIndexedSparseAnalogZ<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float);                           \
template void invokeSparseAnalogDenoisedZ<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float*, float);                           \
template void invokeIndexedSparseAnalogDenoisedZ<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float*, float);         \
template void invokeSparseTransposedAnalogMatrix<T>(uint32_t, uint32_t, uint64_t*, uint64_t*, uint32_t*, float*, T*, uint32_t*, uint32_t*, float*);                                     \
template void invokeIndexedSparseTransposedAnalogMatrix<T>(uint32_t, uint32_t, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, uint32_t*, uint32_t*, float*);                   \
template void invokeSparseTransposedAnalogDenoisedMatrix<T>(uint32_t, uint32_t, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, uint32_t*, uint32_t*, float*);                   \
template void invokeIndexedSparseTransposedAnalogDenoisedMatrix<T>(uint32_t, uint32_t, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, uint32_t*, uint32_t*, float*); \
template void kLoadInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, T*);                                                                                                                    \
template void kLoadIndexedInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*); \

EXPLICITLY_INSTANTIATE_KERNELS(float)
EXPLICITLY_INSTANTIATE_KERNELS(double)
EXPLICITLY_INSTANTIATE_KERNELS(unsigned char)
EXPLICITLY_INSTANTIATE_KERNELS(char)
EXPLICITLY_INSTANTIATE_KERNELS(uint32_t)
EXPLICITLY_INSTANTIATE_KERNELS(uint64_t)
EXPLICITLY_INSTANTIATE_KERNELS(int32_t)
EXPLICITLY_INSTANTIATE_KERNELS(int64_t)
