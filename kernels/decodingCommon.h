#pragma once

#include <cstdint>
#include <curand_kernel.h>

namespace bitfusion
{
    namespace kernels
    {

        class FinishedState
        {
        public:
            static auto constexpr empty()
            {
                return FinishedState{ 0 };
            }

            static auto constexpr finished()
            {
                return FinishedState{ kFinished };
            }

            static auto constexpr skipDecoding()
            {
                return FinishedState{ kSkipDecoding };
            }

            static auto constexpr finishedEOS()
            {
                return FinishedState{ kFinishedEos };
            }

            static auto constexpr finishedMaxLength()
            {
                return FinishedState{ kFinishedMaxLength };
            }

            static auto constexpr finishedStopWords()
            {
                return FinishedState{ kFinishedStopWords };
            }

            __host__ __device__ void constexpr setFinishedEOS()
            {
                mState |= kFinishedEos;
            }

            __host__ __device__ bool constexpr isFinishedEOS()
            {
                return anyBitSet(kFinishedEos);
            }

            __host__ __device__ void constexpr setFinishedStopWords()
            {
                mState |= kFinishedStopWords;
            }

            __host__ __device__ bool constexpr isFinishedStopWords()
            {
                return anyBitSet(kFinishedStopWords);
            }

            __host__ __device__ void constexpr setFinishedMaxLength()
            {
                mState |= kFinishedMaxLength;
            }

            __host__ __device__ bool constexpr isFinishedMaxLength()
            {
                return anyBitSet(kFinishedMaxLength);
            }

            __host__ __device__ void constexpr setFinished()
            {
                mState |= kFinished;
            }

            __host__ __device__ bool constexpr isFinished() const
            {
                return anyBitSet(kFinished);
            }

            __host__ __device__ void constexpr setSkipDecoding()
            {
                mState = kSkipDecoding;
            }

            __host__ __device__ bool constexpr isSkipDecoding() const
            {
                return anyBitSet(kSkipDecoding);
            }

            using UnderlyingType = uint8_t;

        private:
            __host__ __device__ constexpr FinishedState(UnderlyingType state)
                : mState(state)
            {
            }

            static UnderlyingType constexpr kFinishedEos{ 1u << 0 };
            static UnderlyingType constexpr kFinishedStopWords{ 1u << 1 };
            static UnderlyingType constexpr kFinishedMaxLength{ 1u << 2 };
            static UnderlyingType constexpr kFinished{ kFinishedEos | kFinishedStopWords | kFinishedMaxLength };
            static UnderlyingType constexpr kSkipDecoding{ 1u << 3 };

            __host__ __device__ bool constexpr anyBitSet(UnderlyingType bits) const
            {
                return (mState & bits) != 0;
            }

            UnderlyingType mState{};
        };

        static_assert(!FinishedState::empty().isFinished());
        static_assert(!FinishedState::empty().isSkipDecoding());
        static_assert(FinishedState::finished().isFinished());
        static_assert(FinishedState::skipDecoding().isSkipDecoding());
        static_assert(FinishedState::finishedEOS().isFinishedEOS());
        static_assert(FinishedState::finishedStopWords().isFinishedStopWords());
        static_assert(FinishedState::finishedMaxLength().isFinishedMaxLength());

        void invokeCurandInitialize(
            curandState_t* state, const size_t batchSize, unsigned long long randomSeed, cudaStream_t stream);

        void invokeCurandBatchInitialize(
            curandState_t* states, const size_t batchSize, const unsigned long long* randomSeeds, cudaStream_t stream);

        template <typename T>
        void invokeAddBiasSoftMax(T* logits, T* probs, const T* bias, const int* endIds, const FinishedState* finished,
            const int batchSize, const int vocabSize, const int vocabSizePadded, cudaStream_t stream);

    }
}