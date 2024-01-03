#pragma once

#include <cuda_fp16.h>

#include "../common/cudaUtils.h"
#include "../kernels/penaltyTypes.h"

namespace bitfusion
{
    namespace kernels
    {

        template <typename T>
        void invokeBatchApplyRepetitionPenalty(T* logits, const float* penalties, const int** outputIds,
            const int* sequenceLengths, const int batchSize, const int vocabSize, const RepetitionPenaltyType penaltyType,
            int maxSeqLen, cudaStream_t stream);

        template <typename T>
        void invokeBatchApplyTemperaturePenalty(T* logits, const T* bias, const float* temperatures, const int batchSize,
            const int vocabSize, const int vocabSizePadded, cudaStream_t stream);

        template <typename T>
        void invokeApplyTemperaturePenalty(T* logits, const T* bias, const float temperature, const int batchSize,
            const int vocabSize, const int vocabSizePadded, cudaStream_t stream);

        template <typename T>
        void invokeMinLengthPenalty(T* logits, const int* minLengths, const int* endIds, const int* sequenceLengths,
            const int* contextLengths, const int batchSize, const int vocabSizePadded, cudaStream_t stream);

    }
}