#pragma once

#include "decodingCommon.h"
#include <curand_kernel.h>

namespace bitfusion
{
    namespace kernels
    {

        void invokeTopPInitialize(int* topPIdValBuf, int* topPOffsetBuf, int* beginTopPOffsetBuf, const size_t batchSize,
            const int vocabSize, cudaStream_t stream);

        template <typename T>
        void invokeBatchTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
            int* sequenceLength, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
            float* outputLogProbs, const T* logProbs, const int* idVals, int* offsetBuf, int* beginOffsetBuf,
            curandState_t* curandstate, const int batchSize, const size_t vocabSizePadded, const int* endIds,
            const float maxTopP, const float* topPs, cudaStream_t stream, const bool* skipDecode);

        template <typename T>
        void invokeTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
            int* sequenceLength, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
            float* outputLogProbs, const T* logProbs, const int* idVals, int* offsetBuf, int* beginOffsetBuf,
            curandState_t* curandstate, const int batchSize, const size_t vocabSizePadded, const int* endIds, const float topPp,
            cudaStream_t stream, const bool* skipDecode);

        void invokeComputeToppDecay(float* runtimeTopP, const float* runtimeInitialTopP, const int** outputIds,
            const float* topPDecay, const float* topPMin, const int32_t* topPResetIds, const int* sequenceLengths,
            const int localBatchSize, cudaStream_t stream);

    }
}