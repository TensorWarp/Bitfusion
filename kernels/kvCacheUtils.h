#pragma once

#include "../common/assert.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

namespace bitfusion
{
    namespace kernels
    {

        enum class KVIdxType : int32_t
        {
            K_IDX = 0,
            V_IDX = 1
        };

        struct KVBlockArray
        {

            int32_t mMaxBlocksPerSeq;
            int32_t mMaxSeqs;
            int32_t mTokensPerBlock;
            int32_t mTokensPerBlockLog2;
            int64_t* data;

            KVBlockArray() {}

            KVBlockArray(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t sizePerToken)
                : mMaxSeqs(batchSize)
                , mMaxBlocksPerSeq(maxBlocksPerSeq)
                , mTokensPerBlock(tokensPerBlock)
                , data(nullptr)
            {
                const float tokensPerBlockSeqLog2 = log2(mTokensPerBlock);
                TLLM_CHECK_WITH_INFO(
                    ceil(tokensPerBlockSeqLog2) == floor(tokensPerBlockSeqLog2), "tokensPerBlock must be power of 2");
                TLLM_CHECK_WITH_INFO(static_cast<int64_t>(mMaxSeqs - 1) * mMaxBlocksPerSeq * 2 + maxBlocksPerSeq
                    <= std::numeric_limits<int32_t>::max(),
                    "kv cache is too large for gpt_attention_plugin");
                mTokensPerBlockLog2 = static_cast<int>(tokensPerBlockSeqLog2);
            }

            __host__ __device__ inline void** getRowPtr(KVIdxType kvIdx, int32_t seqIdx)
            {
                return reinterpret_cast<void**>(
                    data + seqIdx * mMaxBlocksPerSeq * 2 + static_cast<int32_t>(kvIdx) * mMaxBlocksPerSeq);
            }

            __host__ __device__ inline void* getBlockPtr(void** pointer, int32_t tokenIdx)
            {
                return pointer[tokenIdx >> mTokensPerBlockLog2];
            }

            __host__ __device__ inline void* getBlockPtr(int32_t seqIdx, int32_t tokenIdx, KVIdxType kvIdx)
            {
                return getBlockPtr(getRowPtr(kvIdx, seqIdx), tokenIdx);
            }

            __host__ __device__ inline void* getKBlockPtr(int32_t seqIdx, int32_t tokenIdx)
            {
                return getBlockPtr(seqIdx, tokenIdx, KVIdxType::K_IDX);
            }

            __host__ __device__ inline void* getVBlockPtr(int32_t seqIdx, int32_t tokenIdx)
            {
                return getBlockPtr(seqIdx, tokenIdx, KVIdxType::V_IDX);
            }

            __host__ __device__ inline int32_t getLocalIdx(int32_t globalIdx)
            {
                return globalIdx & ((1 << mTokensPerBlockLog2) - 1);
            }

            __host__ __device__ inline int32_t getKVLocalIdx(
                int32_t globalTokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx)
            {
                return headIdx * mTokensPerBlock * dimsPerHead + getLocalIdx(globalTokenIdx) * dimsPerHead + channelIdx;
            }
        };

        struct KVLinearBuffer
        {

            int32_t mMaxSeqs;
            int32_t mMaxSeqLen;
            int32_t mBytesPerSeq;
            int8_t* data;

            KVLinearBuffer() {}

            KVLinearBuffer(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t sizePerToken)
                : mMaxSeqs(batchSize)
                , mMaxSeqLen(tokensPerBlock)
                , mBytesPerSeq(tokensPerBlock* sizePerToken)
                , data(nullptr)
            {
                TLLM_CHECK_WITH_INFO(
                    static_cast<int64_t>(mMaxSeqs - 1) * mBytesPerSeq * 2 + mBytesPerSeq <= std::numeric_limits<int32_t>::max(),
                    "kv cache is too large for gpt_attention_plugin");
            }

            __host__ __device__ inline void** getRowPtr(KVIdxType kvIdx, int32_t seqIdx)
            {
                return reinterpret_cast<void**>(data + seqIdx * mBytesPerSeq * 2 + static_cast<int32_t>(kvIdx) * mBytesPerSeq);
            }

            __host__ __device__ inline void* getBlockPtr(void** pointer, int32_t tokenIdx)
            {
                return reinterpret_cast<void*>(pointer);
            }

            __host__ __device__ inline void* getKBlockPtr(int32_t seqIdx, int32_t)
            {
                return reinterpret_cast<void*>(getRowPtr(KVIdxType::K_IDX, seqIdx));
            }

            __host__ __device__ inline void* getVBlockPtr(int32_t seqIdx, int32_t)
            {
                return reinterpret_cast<void*>(getRowPtr(KVIdxType::V_IDX, seqIdx));
            }

            __host__ __device__ inline int32_t getKVLocalIdx(
                int32_t tokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx)
            {
                return headIdx * mMaxSeqLen * dimsPerHead + tokenIdx * dimsPerHead + channelIdx;
            }
        };

    }
}