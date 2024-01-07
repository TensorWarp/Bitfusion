#pragma once

#include "../common/cudaAllocator.h"
#include "../runtime/bufferManager.h"
#include "decodingInput.h"
#include "decodingOutput.h"
#include "samplingConfig.h"
#include <curand_kernel.h>

#include <cstdint>
#include <memory>

#include <NvInferRuntime.h>

namespace bitfusion
{
    namespace layers
    {
        template <typename T>
        class DynamicDecodeLayer;
    }

    namespace runtime
    {
        class IDecoder
        {
        public:
            virtual ~IDecoder() = default;

            virtual void setup(const SamplingConfig& samplingConfig, size_t batchSize, SizeType maxSequenceLength) = 0;

            virtual bool forward(DecodingOutput& output, const DecodingInput& input) = 0;

            virtual void forwardAsync(DecodingOutput& output, const DecodingInput& input) = 0;

            virtual void gatherTree(ITensor& finalOutputIds, const DecodingOutput& decodingOutput,
                const DecodingInput& decodingInput, const BufferManager& manager) = 0;

            virtual const SamplingConfig& getSamplingConfig() = 0;

            static void acceptDraftTokensByIds(const ITensor& targetTokenIds, const ITensor& draftTokenIds,
                const ITensor& contextLengths, const ITensor& numDraftTokens,
                ITensor& sequenceLengths, const ITensor& finishedVec, ITensor& finishedFinal,
                ITensor& finishedSum, const BufferManager::CudaStreamPtr& stream);

            static void acceptDraftTokensByLogits(ITensor& draftLogits, const ITensor& targetLogits, ITensor& draftProbs,
                ITensor& targetProbs, const ITensor& numDraftTokens, ITensor& finished,
                SizeType vocabSize, SizeType vocabSizePadded, bool useRandomAcceptThreshold,
                float randomAcceptThreshold, curandState_t* curandState,
                const BufferManager::CudaStreamPtr& stream);

            static std::unique_ptr<IDecoder> create(nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded,
                const BufferManager::CudaStreamPtr& stream);
        };

        template <typename T>
        class Decoder final : public IDecoder
        {
        public:
            using CudaStreamPtr = BufferManager::CudaStreamPtr;
            using TensorPtr = std::shared_ptr<ITensor>;

            Decoder(size_t vocabSize, size_t vocabSizePadded, const CudaStreamPtr& stream);

            void setup(const SamplingConfig& samplingConfig, size_t batchSize, SizeType maxSequenceLength) override;

            bool forward(DecodingOutput& output, const DecodingInput& input) override;

            void forwardAsync(DecodingOutput& output, const DecodingInput& input) override;

            void gatherTree(ITensor& finalOutputIds, const DecodingOutput& decodingOutput, const DecodingInput& decodingInput,
                const BufferManager& manager) override;

            const SamplingConfig& getSamplingConfig() override { return mSamplingConfig; }

        private:
            BufferManager mManager;
            common::CudaAllocator mAllocator;
            std::shared_ptr<bitfusion::layers::DynamicDecodeLayer<T>> mDynamicDecodeLayer;
            TensorPtr mLogProbsTiled;
            SamplingConfig mSamplingConfig;
        };

        std::unique_ptr<IDecoder> IDecoder::create(nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded,
            const BufferManager::CudaStreamPtr& stream)
        {
            switch (dtype)
            {
            case nvinfer1::DataType::kFLOAT:
                return std::make_unique<Decoder<float>>(vocabSize, vocabSizePadded, stream);
            case nvinfer1::DataType::kHALF:
                return std::make_unique<Decoder<half>>(vocabSize, vocabSizePadded, stream);
            default:
                return nullptr;
            }
        }
    }
}