
#pragma once

#include "../common/cudaUtils.h"
#include "bufferManager.h"
#include "cudaEvent.h"
#include "cudaStream.h"
#include "generationOutput.h"
#include "gptDecoder.h"
#include "iGptDecoderBatch.h"
#include "iTensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace bitfusion::runtime
{

class GptDecoderBatch : public IGptDecoderBatch
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = ITensor::SharedPtr;

    GptDecoderBatch(std::size_t vocabSize, std::size_t vocabSizePadded, CudaStreamPtr stream);

    void setup(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxAttentionWindow, SizeType maxSequenceLength,
        SizeType maxTokensPerStep, nvinfer1::DataType dtype) override;

    void newRequest(
        SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig) override;

    void newBatch(
        GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig) override;

    TokenPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) override;

    void forwardSync(decoder_batch::Token const& e) override;

    void forwardAsync(decoder::Output& output, decoder::Input const& input) override;

    void forwardSync() override;

    [[nodiscard]] std::vector<bool> getFinished() const override
    {
        return {mFinished.begin(), mFinished.begin() + mActualBatchSize};
    }

    [[nodiscard]] TensorPtr getOutputIds(SizeType batchIdx) const override
    {
        auto tensor = ITensor::slice(mJointDecodingOutput->ids, batchIdx, 1);
        tensor->squeeze(0);
        return tensor;
    }

    [[nodiscard]] TensorPtr getOutputIds() const override
    {
        return ITensor::slice(mJointDecodingOutput->ids, 0, mActualBatchSize);
    }

    [[nodiscard]] CudaEvent finalize(SizeType batchIdx) const;

    void finalize() const override;

    [[nodiscard]] TensorPtr getParentIds() const override
    {
        return ITensor::slice(mJointDecodingOutput->parentIds, 0, mActualBatchSize);
    }

    [[nodiscard]] TensorPtr getCumLogProbs() const override
    {
        return ITensor::slice(mJointDecodingOutput->cumLogProbs, 0, mActualBatchSize);
    }

    [[nodiscard]] TensorPtr getCumLogProbs(SizeType batchIdx) const
    {
        auto tensor = ITensor::slice(mJointDecodingOutput->cumLogProbs, batchIdx, 1);
        tensor->squeeze(0);
        return tensor;
    }

    [[nodiscard]] TensorPtr getLogProbs() const override
    {
        return ITensor::slice(mJointDecodingOutput->logProbs, 0, mActualBatchSize);
    }

    [[nodiscard]] TensorPtr getLogProbs(SizeType batchIdx) const
    {
        auto tensor = ITensor::slice(mJointDecodingOutput->logProbs, batchIdx, 1);
        tensor->squeeze(0);
        return tensor;
    }

    [[nodiscard]] TensorPtr getAllNewTokens() const override
    {
        return mJointDecodingOutput->newTokensSteps;
    }

    [[nodiscard]] TensorPtr getNewTokens(SizeType iter = 0) const override
    {
        TensorPtr newTokensView = std::move(ITensor::slice(mJointDecodingOutput->newTokensSteps, iter, 1));
        newTokensView->squeeze(0);
        return ITensor::slice(newTokensView, 0, mActualBatchSize);
    }

    [[nodiscard]] std::vector<SizeType> getNbSteps() const override
    {
        return std::vector<SizeType>(mNbSteps.begin(), mNbSteps.begin() + mActualBatchSize);
    }

    [[nodiscard]] TensorPtr getNbFinished() const override
    {
        return mFinishedSum;
    }

private:
    CudaEvent postProcessRequest(SizeType batchIdx) const;

private:
    std::size_t const mVocabSize;
    std::size_t const mVocabSizePadded;
    CudaStreamPtr mStream;
    BufferManager mBufferManager;
    TokenPtr mForwardToken;
    CudaEvent mForwardEvent;

    std::vector<CudaStreamPtr> mStreams;
    using GptDecoderPtr = std::unique_ptr<IGptDecoder>;
    std::vector<GptDecoderPtr> mDecoders;
    using DecodingInputPtr = std::unique_ptr<DecodingInput>;
    std::vector<DecodingInputPtr> mDecodingInputs;
    using DecodingOutputPtr = std::unique_ptr<DecodingOutput>;
    std::vector<DecodingOutputPtr> mDecodingOutputs;

    DecodingInputPtr mJointDecodingInput;
    DecodingOutputPtr mJointDecodingOutput;

    std::vector<TensorPtr> mDraftTokenIds;
    std::vector<TensorPtr> mDraftLogits;
    std::vector<bool> mAcceptByLogits;
    TensorPtr mNumDraftTokens;
    TensorPtr mCurandStates;

    std::vector<SizeType> mNbSteps;
    std::vector<bool> mFinished;
    TensorPtr mFinishedSum;
    std::vector<SizeType> mMaxNewTokens;
    std::vector<SizeType> mBeamWidths;
    std::vector<SizeType> mGeneratedTokensPerStep;

    TensorPtr mFinishedSteps;
    TensorPtr mDraftProbs;
    TensorPtr mTargetProbs;
    SizeType mMaxSequenceLength{};
    SizeType mMaxAttentionWindow{};
    SizeType mActualBatchSize{};
    SizeType mMaxTokensPerStep{};
};
}
