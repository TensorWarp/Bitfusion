
#pragma once

#include "bufferManager.h"
#include "cudaEvent.h"
#include "cudaStream.h"
#include "gptDecoder.h"
#include "iStatefulGptDecoder.h"
#include "iTensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace bitfusion::runtime
{

class StatefulGptDecoder : public IStatefulGptDecoder
{
public:
    StatefulGptDecoder(std::size_t vocabSize, std::size_t vocabSizePadded, CudaStreamPtr stream);

    void setup(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxAttentionWindow, SizeType maxSequenceLength,
        SizeType maxTokensPerStep, nvinfer1::DataType dtype) override;

    void newBatch(
        GenerationInput const& input, GenerationOutput const& output, SamplingConfig const& samplingConfig) override;

    void forwardAsync(decoder::Output& output, decoder::Input const& input) override;

    void forwardSync() override;

    void finalize() const override;

    [[nodiscard]] TensorPtr getOutputIds() const override
    {
        return mDecodingOutput->ids;
    }

    [[nodiscard]] TensorPtr getCumLogProbs() const override
    {
        return mDecodingOutput->cumLogProbs;
    }

    [[nodiscard]] TensorPtr getLogProbs() const override
    {
        return mDecodingOutput->logProbs;
    }

    [[nodiscard]] TensorPtr getNewTokens(SizeType iter = 0) const override
    {
        TLLM_CHECK(iter == 0);
        return mDecodingOutput->newTokens;
    }

    [[nodiscard]] TensorPtr getAllNewTokens() const override
    {
        TensorPtr newTokens = std::move(ITensor::view(mDecodingOutput->newTokensSteps));
        newTokens->unsqueeze(0);
        return newTokens;
    }

    [[nodiscard]] TensorPtr getNbFinished() const override
    {
        return mDecodingOutput->finishedSum;
    }

private:
    void reshapeBuffers(
        SizeType batchSize, SizeType beamWidth, SizeType mMaxAttentionWindow, SizeType maxSequenceLength);

private:
    std::size_t const mVocabSize;
    std::size_t const mVocabSizePadded;
    CudaStreamPtr mStream;
    BufferManager mBufferManager;

    using GptDecoderPtr = std::unique_ptr<IGptDecoder>;
    GptDecoderPtr mDecoder;
    using DecodingInputPtr = std::unique_ptr<DecodingInput>;
    DecodingInputPtr mDecodingInput;
    using DecodingOutputPtr = std::unique_ptr<DecodingOutput>;
    DecodingOutputPtr mDecodingOutput;
    CudaEvent mDecodedEvent{};

    SizeType mNbSteps;
    SizeType mMaxSequenceLength{};
    SizeType mMaxAttentionWindow{};
};
}
