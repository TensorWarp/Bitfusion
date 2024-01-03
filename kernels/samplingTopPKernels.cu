#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/reduceKernelUtils.cuh"
#include "samplingTopPKernels.h"

using namespace bitfusion::common;

namespace bitfusion
{
    namespace kernels
    {
        __global__ void topPInitialize(
            int* topPIdValBuf, int* topPOffsetBuf, int* beginTopPOffsetBuf, const int batchSize, const int vocabSize)
        {
            int tid = threadIdx.x;
            int bid = blockIdx.x;

            if (bid == 0)
            {
                for (int i = tid; i < batchSize + 1; i += blockDim.x)
                {
                    topPOffsetBuf[i] = i * vocabSize;
                    beginTopPOffsetBuf[i] = topPOffsetBuf[i];
                }
            }

            int index = tid + bid * blockDim.x;

            while (index < batchSize * vocabSize)
            {
                topPIdValBuf[index] = index % vocabSize;
                index += blockDim.x * gridDim.x;
            }
        }

        void invokeTopPInitialize(int* topPIdValBuf, int* topPOffsetBuf, int* beginTopPOffsetBuf, const size_t batchSize,
            const int vocabSize, cudaStream_t stream)
        {
            topPInitialize << <32, 512, 0, stream >> > (topPIdValBuf, topPOffsetBuf, beginTopPOffsetBuf, batchSize, vocabSize);
        }

        template <typename T, int THREADBLOCK_SIZE>
        __launch_bounds__(THREADBLOCK_SIZE) __global__ void topPBeamTopKKernel(const T* logProbs,
            int* topKTmpIdBuf, T* topKTmpValBuf, const FinishedState* finishedInput, const int vocabSize, int* offsetBuf,
            int* beginOffsetBuf, const float topP, const float* topPs, const bool* skipDecode)
        {
            constexpr int MAX_K = 1;
            int threadId = threadIdx.x;
            int batchId = blockIdx.x;

            if ((skipDecode != nullptr && skipDecode[batchId])
                || (finishedInput != nullptr && finishedInput[batchId].isSkipDecoding()))
            {
                beginOffsetBuf[batchId] += vocabSize;
                return;
            }

            float pThreshold = (topPs != nullptr) ? topPs[batchId] : topP;

            typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            TopK<T, MAX_K> partial;

            const bool IS_FP16 = std::is_same<T, half>::value;
            const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
            for (int i = 0; i < MAX_K; ++i)
            {
                partial.p[i] = -1;
                partial.u[i] = -MAX_T_VAL;
            }

#pragma unroll
            for (int elemId = threadId; elemId < vocabSize; elemId += THREADBLOCK_SIZE)
            {
                int index = elemId + batchId * vocabSize;
                partial.insert(logProbs[index], elemId);
            }

            TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

            if (threadId == 0)
            {
                beginOffsetBuf[batchId] = offsetBuf[batchId];
                T sumProb = (T)(0.0f);

#pragma unroll
                for (int i = 0; i < MAX_K; i++)
                {
                    sumProb += total.u[i];
                }

                if ((float)sumProb >= pThreshold)
                {
                    beginOffsetBuf[batchId] += vocabSize;
                    int index = batchId * vocabSize;

#pragma unroll
                    for (int i = 0; i < MAX_K; ++i)
                    {
                        topKTmpIdBuf[index + i] = total.p[i];
                        topKTmpValBuf[index + i] = total.u[i];
                    }
                }
            }
        }

        struct BlockPrefixCallbackOp
        {
            float running_total;

            __device__ BlockPrefixCallbackOp(float running_total)
                : running_total(running_total)
            {
            }

            __device__ float operator()(float block_aggregate)
            {
                float old_prefix = running_total;
                running_total += block_aggregate;
                return old_prefix;
            }
        };

        template <typename T>
        __device__ void epilogue(int batchId, int currentStep, int offset, int** ids, int* sortedIdVals, T* sortedLogProbs,
            float* cumLogProbs, float* outputLogProbs, const int* endIds, int* sequenceLengths, FinishedState* finishedOutput)
        {
            ids[batchId][currentStep] = sortedIdVals[offset];

            if (cumLogProbs != nullptr || outputLogProbs != nullptr)
            {
                float lprob = logf(sortedLogProbs[offset]);
                if (cumLogProbs != nullptr)
                {
                    cumLogProbs[batchId] += lprob;
                }
                if (outputLogProbs != nullptr)
                {
                    outputLogProbs[batchId] = lprob;
                }
            }
            if (sequenceLengths != nullptr && finishedOutput != nullptr)
            {
                if (ids[batchId][currentStep] == endIds[batchId])
                {
                    finishedOutput[batchId].setFinishedEOS();
                }
                else
                {
                    sequenceLengths[batchId] += 1;
                }
            }
        }

        template <typename T, int blockSize>
        __global__ void topPSsampling(T* sortedLogProbs, int* sortedIdVals, int** ids, int* sequenceLength,
            const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
            const int* beginOffsetBuf, const int* offsetBuf, const int vocabSize, curandState_t* curandstate, const float topP,
            const float* topPs, const int* endIds, const int batchSize, const bool* skipDecode)
        {

            __shared__ float randNumS;

            const int tid = threadIdx.x;
            const int batchId = blockIdx.x;
            const FinishedState finishState = (finishedInput != nullptr) ? finishedInput[batchId] : FinishedState(); // Try to get from finishedInput or use a default state
            if ((skipDecode != nullptr && skipDecode[batchId]) || (finishState.isSkipDecoding()))
            {
                return;
            }

            if (finishState.isFinished())
            {
                if (finishedOutput != nullptr)
                {
                    finishedOutput[batchId] = finishState;
                }
                ids[batchId][sequenceLength[batchId]] = endIds[batchId];
                return;
            }

            constexpr int WARP_SIZE = 32;
            constexpr int NUM_WARPS = blockSize / WARP_SIZE;
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;
            const float probThreshold = (topPs != nullptr) ? topPs[batchId] : topP;
            const int currentStep = sequenceLength[batchId];

            if (threadIdx.x == 0)
            {
                randNumS = curand_uniform(curandstate + blockIdx.x) * probThreshold;
            }

            if (beginOffsetBuf[batchId] == offsetBuf[batchId])
            {
                if (tid == 0)
                {
                    int offset = batchId * vocabSize;
                    epilogue(batchId, currentStep, offset, ids, sortedIdVals, sortedLogProbs, cumLogProbs, outputLogProbs,
                        endIds, sequenceLength, finishedOutput);
                }
                return;
            }

            typedef cub::BlockScan<float, blockSize> BlockScan;
            __shared__ typename BlockScan::TempStorage tempStorage;
            __shared__ uint32_t selectedShared[NUM_WARPS];
            BlockPrefixCallbackOp prefixOp(0);

            if (laneId == 0)
            {
                selectedShared[warpId] = 0;
            }

            __syncthreads();

            int offset = batchId * vocabSize;
            ids[batchId][currentStep] = sortedIdVals[offset];
            int end = ((vocabSize + blockSize - 1) / blockSize) * blockSize;
            int selectedTokenId = 0;
            float threadOffset = 0;
            int count = 0;
            for (int vi = tid; vi < end; vi += blockSize)
            {
                float threadProb = (vi < vocabSize) ? (float)sortedLogProbs[offset + vi] : 0.f;
                BlockScan(tempStorage).InclusiveSum(threadProb, threadOffset, prefixOp);
                count = __syncthreads_count(randNumS <= threadOffset);
                selectedTokenId = vi;
                if (count != 0)
                {
                    break;
                }
            }

            if (threadIdx.x == min(blockDim.x - count, blockDim.x - 1))
            {
                epilogue(batchId, currentStep, offset + selectedTokenId, ids, sortedIdVals, sortedLogProbs, cumLogProbs,
                    outputLogProbs, endIds, sequenceLength, finishedOutput);
            }
        }

        template <typename T>
        void invokeBatchTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
            int* sequenceLength, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
            float* outputLogProbs, const T* logProbs, const int* idVals, int* offsetBuf, int* beginOffsetBuf,
            curandState_t* curandstate, const int batchSize, const size_t vocabSizePadded, const int* endIds,
            const float maxTopP, const float* topPs, cudaStream_t stream, const bool* skipDecode)
        {
            const int vocabSize = vocabSizePadded;

            size_t sortedLogProbBufSize = batchSize * vocabSize * sizeof(T);
            size_t sortedIdValsBufSize = batchSize * vocabSize * sizeof(int);
            sortedLogProbBufSize = divUp(sortedLogProbBufSize, 256) * 256;
            sortedIdValsBufSize = divUp(sortedIdValsBufSize, 256) * 256;

            void* cubTempStorage = workspace;
            T* sortedLogProbs = (T*)((char*)cubTempStorage + cubTempStorageSize);
            int* sortedIdVals = (int*)((char*)sortedLogProbs + sortedLogProbBufSize);

            if (workspace == nullptr)
            {
                check_cuda_error(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, cubTempStorageSize, logProbs,
                    (T*) nullptr, idVals, (int*) nullptr, vocabSize * batchSize, batchSize, beginOffsetBuf, offsetBuf + 1,
                    0,
                    sizeof(T) * 8,
                    stream));
                cubTempStorageSize = divUp(cubTempStorageSize, 256) * 256;
                workspaceSize = sortedLogProbBufSize + sortedIdValsBufSize + cubTempStorageSize;
                return;
            }

            constexpr int BLOCK_SIZE = 256;
            topPBeamTopKKernel<T, BLOCK_SIZE> << <batchSize, BLOCK_SIZE, 0, stream >> > (logProbs, sortedIdVals, sortedLogProbs,
                finishedInput, vocabSize, offsetBuf, beginOffsetBuf, maxTopP, topPs, skipDecode);

            check_cuda_error(cub::DeviceSegmentedRadixSort::SortPairsDescending(cubTempStorage, cubTempStorageSize, logProbs,
                sortedLogProbs, idVals, sortedIdVals, vocabSize * batchSize, batchSize, beginOffsetBuf, offsetBuf + 1,
                0,
                sizeof(T) * 8,
                stream));

            constexpr int SAMPLING_BLOCK_SIZE = 256;
            dim3 grid(batchSize);
            topPSsampling<T, SAMPLING_BLOCK_SIZE> << <grid, SAMPLING_BLOCK_SIZE, 0, stream >> > (sortedLogProbs, sortedIdVals,
                outputIds, sequenceLength, finishedInput, finishedOutput, cumLogProbs, outputLogProbs, beginOffsetBuf,
                offsetBuf + 1, vocabSize, curandstate, maxTopP, topPs, endIds, batchSize, skipDecode);
        }

        template void invokeBatchTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize,
            int** outputIds, int* sequenceLength, const FinishedState* finishedInput, FinishedState* finishedOutput,
            float* cumLogProbs, float* outputLogProbs, const float* logProbs, const int* idVals, int* offsetBuf,
            int* beginOffsetBuf, curandState_t* curandstate, const int batchSize, const size_t vocabSizePadded,
            const int* endIds, const float maxTopP, const float* topPs, cudaStream_t stream, const bool* skipDecode);

        template void invokeBatchTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize,
            int** outputIds, int* sequenceLength, const FinishedState* finishedInput, FinishedState* finishedOutput,
            float* cumLogProbs, float* outputLogProbs, const half* logProbs, const int* idVals, int* offsetBuf,
            int* beginOffsetBuf, curandState_t* curandstate, const int batchSize, const size_t vocabSizePadded,
            const int* endIds, const float maxTopP, const float* topPs, cudaStream_t stream, const bool* skipDecode);

        template <typename T>
        void invokeTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
            int* sequenceLength, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
            float* outputLogProbs, const T* logProbs, const int* idVals, int* offsetBuf, int* beginOffsetBuf,
            curandState_t* curandstate, const int batchSize, const size_t vocabSizePadded, const int* endIds, const float topP,
            cudaStream_t stream, const bool* skipDecode)
        {
            invokeBatchTopPSampling(workspace, workspaceSize, cubTempStorageSize, outputIds, sequenceLength, finishedInput,
                finishedOutput, cumLogProbs, outputLogProbs, logProbs, idVals, offsetBuf, beginOffsetBuf, curandstate,
                batchSize, vocabSizePadded, endIds, topP, nullptr, stream, skipDecode);
        }

        template void invokeTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
            int* sequenceLength, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
            float* outputLogProbs, const float* logProbs, const int* idVals, int* offsetBuf, int* beginOffsetBuf,
            curandState_t* curandstate, const int batchSize, const size_t vocabSizePadded, const int* endIds, const float topP,
            cudaStream_t stream, const bool* skipDecode);

        template void invokeTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
            int* sequenceLength, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
            float* outputLogProbs, const half* logProbs, const int* idVals, int* offsetBuf, int* beginOffsetBuf,
            curandState_t* curandstate, const int batchSize, const size_t vocabSizePadded, const int* endIds, const float topP,
            cudaStream_t stream, const bool* skipDecode);

        __global__ void computeToppDecay(float* runtimeTopP, const float* runtimeInitialTopP, const int** outputIds,
            const float* topPDecay, const float* topPMin, const int32_t* topPResetIds, const int* sequenceLengths)
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            const auto currentStep{ sequenceLengths[idx] };
            if (outputIds[idx][currentStep] == topPResetIds[idx])
            {
                runtimeTopP[idx] = runtimeInitialTopP[idx];
            }
            else
            {
                runtimeTopP[idx] = max(runtimeTopP[idx] * topPDecay[idx], topPMin[idx]);
            }
        }

        void invokeComputeToppDecay(float* runtimeTopP, const float* runtimeInitialTopP, const int** outputIds,
            const float* topPDecay, const float* topPMin, const int32_t* topPResetIds, const int* sequenceLengths,
            const int local_batchSize, cudaStream_t stream)
        {
            dim3 block(min(local_batchSize, 512));
            dim3 grid((local_batchSize + block.x - 1) / block.x);
            computeToppDecay << <grid, block, 0, stream >> > (
                runtimeTopP, runtimeInitialTopP, outputIds, topPDecay, topPMin, topPResetIds, sequenceLengths);
        }

    }
}