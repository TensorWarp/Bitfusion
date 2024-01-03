#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bitfusion
{
    namespace kernels
    {

        enum class AttentionMaskType
        {
            PADDING = 0,
            CAUSAL = 1,
            BIDIRECTIONAL = 2,
            BIDIRECTIONALGLM = 3
        };

        enum class PositionEmbeddingType : int8_t
        {
            kLEARNED_ABSOLUTE = 0,
            kROPE_GPTJ = 1,
            kROPE_GPT_NEOX = 2,
            kALIBI = 3,
            kALIBI_WITH_SCALE = 4,
            kRELATIVE = 5
        };

        enum class RotaryScalingType : int8_t
        {
            kNONE = 0,
            kLINEAR = 1,
            kDYNAMIC = 2,
        };

        template <typename AttentionMaskDataType>
        struct BuildDecoderInfoParams
        {
            int* seqQOffsets;
            int* seqKVOffsets;
            int* paddingOffsets;

            AttentionMaskDataType* attentionMask;

            const int* seqQLengths;
            const int* seqKVLengths;

            int batchSize;
            int maxSeqLength;
            int attentionWindowSize;
            int numTokens;
            AttentionMaskType attentionMaskType;
        };

        template <typename T>
        void invokeBuildDecoderInfo(const BuildDecoderInfoParams<T>& params, cudaStream_t stream);

    }
}