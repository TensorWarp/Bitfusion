#include "../common/cudaUtils.h"
#include "../common/reduceKernelUtils.cuh"
#include "stopCriteriaKernels.h"

using namespace bitfusion::common;

namespace bitfusion
{
    namespace kernels
    {
        __global__ void stopWordsCriterion(const int** outputIds, const int** parentIds, const int* stopWords,
            FinishedState* finished, const int* sequenceLengths, size_t stopWordsLen, int batchSize, int beamWidth,
            int maxSeqLen)
        {
            const int id = blockIdx.x * blockDim.x + threadIdx.x;
            const int batchIdx = blockIdx.y / beamWidth;
            const int beamIdx = blockIdx.y % beamWidth;

            const int* baseStopWords = stopWords + batchIdx * 2 * stopWordsLen;
            const int* baseOffsets = baseStopWords + stopWordsLen;

            if (id >= stopWordsLen || baseOffsets[id] < 0)
            {
                return;
            }

            const int itemEnd = baseOffsets[id];
            const int itemStart = (id > 0) ? baseOffsets[id - 1] : 0;
            const int itemSize = itemEnd - itemStart;

            bool shouldStop = false;

            const int currentStep = sequenceLengths[blockIdx.y] - 1;
            if (currentStep + 1 >= itemSize)
            {
                shouldStop = true;
                int parentId = beamIdx;
                const bool gatherBeam = beamWidth > 1;

                for (int tokenIdx = itemSize - 1; tokenIdx >= 0; tokenIdx--)
                {
                    const int previousToken
                        = outputIds[batchIdx][parentId * maxSeqLen + currentStep - (itemSize - 1) + tokenIdx];
                    if (previousToken != baseStopWords[itemStart + tokenIdx])
                    {
                        shouldStop = false;
                        break;
                    }
                    if (gatherBeam)
                    {
                        parentId = parentIds == nullptr
                            ? 0
                            : parentIds[batchIdx][parentId * maxSeqLen + currentStep - (itemSize - 1) + tokenIdx];

                        if (parentId < 0 || parentId >= beamWidth)
                        {
                            shouldStop = false;
                            break;
                        }
                    }
                }
            }

            if (shouldStop)
            {
                finished[batchIdx * beamWidth + beamIdx].setFinishedStopWords();
            }
        }

        void invokeStopWordsCriterion(const int** outputIds, const int** parentIds, const int* stopWords,
            FinishedState* finished, const int* sequenceLengths, size_t stopWordsLen, int batchSize, int beamWidth,
            int maxSeqLen, cudaStream_t stream)
        {
            dim3 block, grid;
            constexpr size_t maxBlockSize{ 256 };
            block.x = min(((stopWordsLen + 32 - 1) / 32) * 32, maxBlockSize);
            grid.x = (stopWordsLen + block.x - 1) / block.x;
            grid.y = batchSize * beamWidth;

            stopWordsCriterion << <grid, block, 0, stream >> > (
                outputIds, parentIds, stopWords, finished, sequenceLengths, stopWordsLen, batchSize, beamWidth, maxSeqLen);
            sync_check_cuda_error();
        }

        __global__ void lengthCriterion(FinishedState* finished, int* finishedSum, const uint32_t* sequenceLimitLength,
            const int* sequenceLengths, int batchSize, int beamWidth)
        {
            int threadFinishedCount = 0;
            for (int index = threadIdx.x; index < batchSize * beamWidth; index += blockDim.x)
            {
                const int batchIdx = index / beamWidth;

                auto finishState = finished[index];

                if (sequenceLengths[index] >= sequenceLimitLength[batchIdx])
                {
                    finishState.setFinishedMaxLength();
                }
                threadFinishedCount += finishState.isFinished() ? 1 : 0;
                finished[index] = finishState;
            }

            if (finishedSum)
            {
                int blockFinishedCount = 0;
                if (blockDim.x <= 32)
                {
                    blockFinishedCount = warpReduceSum(threadFinishedCount);
                }
                else
                {
                    blockFinishedCount = blockReduceSum(threadFinishedCount);
                }
                __syncthreads();

                if (threadIdx.x == 0)
                {
                    finishedSum[0] = blockFinishedCount;
                }
            }
        }

        void invokeLengthCriterion(FinishedState* finished, int* finishedSum, const uint32_t* sequenceLimitLength,
            const int* sequenceLengths, int batchSize, int beamWidth, cudaStream_t stream)
        {
            dim3 block{ min(512, uint32_t(batchSize * beamWidth)) };
            dim3 grid{ 1 };

            lengthCriterion << <grid, block, 0, stream >> > (
                finished, finishedSum, sequenceLimitLength, sequenceLengths, batchSize, beamWidth);
            sync_check_cuda_error();
        }

    }
}
