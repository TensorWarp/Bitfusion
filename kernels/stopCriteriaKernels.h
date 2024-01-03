#pragma once

#include "decodingCommon.h"
#include <cuda_runtime.h>

namespace bitfusion
{
    namespace kernels
    {
        void invokeStopWordsCriterion(const int** outputIds, const int** parentIds, const int* stopWords,
            FinishedState* finished, const int* sequenceLengths, size_t stopWordsLen, int batchSize, int beamWidth,
            int maxSeqLen, cudaStream_t stream);

        void invokeLengthCriterion(FinishedState* finished, int* finishedSum, const uint32_t* sequenceLimitLength,
            const int* sequenceLengths, int batchSize, int beamWidth, cudaStream_t stream);
    }
}