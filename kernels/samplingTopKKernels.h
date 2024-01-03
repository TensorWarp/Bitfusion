#pragma once

#include "decodingCommon.h"
#include <curand_kernel.h>

namespace bitfusion
{
    namespace kernels
    {

        template <typename T>
        void invokeBatchTopKSampling(void* workspace, size_t& workspaceSize, const T* logProbs, int** ids, int* sequenceLengths,
            const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
            curandState_t* curandstate, const int maxTopK, const int* topKs, const float topP, const float* topPs,
            const int vocabSizePadded, const int* endIds, cudaStream_t stream, const int batchSize, const bool* skipDecode);

        template <typename T>
        void invokeTopKSampling(void* workspace, size_t& workspaceSize, const T* logProbs, int** outputIds, int* sequenceLength,
            const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
            curandState_t* curandstate, const int topK, const float topP, const int vocabSizePadded, const int* endIds,
            cudaStream_t stream, const int batchSize, const bool* skipDecode);

        template <typename T>
        void invokeAddBiasEndMask(T* logits, const T* bias, const int* endIds, const FinishedState* finished,
            const int batchSize, const int vocabSize, const int vocabSizePadded, cudaStream_t stream);

    }
}