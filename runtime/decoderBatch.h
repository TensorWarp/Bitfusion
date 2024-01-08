
#pragma once

#include "../common/cudaUtils.h"
#include "bufferManager.h"
#include "cudaEvent.h"
#include "cudaStream.h"
#include "generationOutput.h"
#include "Decoder.h"
#include "iDecoderBatch.h"
#include "iTensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace bitfusion::runtime
{
    /// <summary>
    /// Represents a batch of decoders for inference.
    /// </summary>
    class DecoderBatch : public IDecoderBatch
    {
public:
        /// <summary>
        /// Defines a shared pointer type for CUDA streams.
        /// </summary>
        using CudaStreamPtr = std::shared_ptr<CudaStream>;

        /// <summary>
        /// Defines a shared pointer type for ITensor objects.
        /// </summary>
        using TensorPtr = ITensor::SharedPtr;

        /// <summary>
        /// Constructor for the DecoderBatch class.
        /// </summary>
        /// <param name="vocabSize">The size of the vocabulary.</param>
        /// <param name="vocabSizePadded">The size of the padded vocabulary.</param>
        /// <param name="stream">The CUDA stream for asynchronous operations.</param>
        DecoderBatch(std::size_t vocabSize, std::size_t vocabSizePadded, CudaStreamPtr stream);

        /// <summary>
        /// Setup the DecoderBatch with configuration parameters.
        /// </summary>
        /// <param name="maxBatchSize">The maximum batch size.</param>
        /// <param name="maxBeamWidth">The maximum beam width.</param>
        /// <param name="maxAttentionWindow">The maximum attention window size.</param>
        /// <param name="maxSequenceLength">The maximum sequence length.</param>
        /// <param name="maxTokensPerStep">The maximum tokens per step.</param>
        /// <param name="dtype">The data type for inference.</param>
        void setup(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxAttentionWindow, SizeType maxSequenceLength,
            SizeType maxTokensPerStep, nvinfer1::DataType dtype) override;

        /// <summary>
        /// Create a new request for decoding.
        /// </summary>
        /// <param name="batchIdx">The index of the batch.</param>
        /// <param name="request">The decoding request.</param>
        /// <param name="samplingConfig">The sampling configuration.</param>
        void newRequest(
            SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig) override;

        /// <summary>
        /// Create a new batch for decoding.
        /// </summary>
        /// <param name="inputs">The generation inputs.</param>
        /// <param name="outputs">The generation outputs.</param>
        /// <param name="samplingConfig">The sampling configuration.</param>
        void newBatch(
            GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig) override;

        /// <summary>
        /// Perform asynchronous forward inference for a decoding request.
        /// </summary>
        /// <param name="output">The decoding output.</param>
        /// <param name="input">The decoding input.</param>
        /// <returns>The token pointer for the asynchronous inference.</returns>
        TokenPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) override;

        /// <summary>
        /// Perform synchronous forward inference for a decoding token.
        /// </summary>
        /// <param name="e">The decoding token.</param>
        void forwardSync(decoder_batch::Token const& e) override;

        /// <summary>
        /// Perform asynchronous forward inference for a decoding input.
        /// </summary>
        /// <param name="output">The decoding output.</param>
        /// <param name="input">The decoding input.</param>
        void forwardAsync(decoder::Output& output, decoder::Input const& input) override;

        /// <summary>
        /// Perform synchronous forward inference for all decoding inputs.
        /// </summary>
        void forwardSync() override;

        /// <summary>
        /// Get the vector of boolean flags indicating whether each batch is finished.
        /// </summary>
        /// <returns>The vector of boolean flags for batch completion status.</returns>
        [[nodiscard]] std::vector<bool> getFinished() const override
        {
            return { mFinished.begin(), mFinished.begin() + mActualBatchSize };
        }

        /// <summary>
        /// Get the tensor containing output IDs for a specific batch index.
        /// </summary>
        /// <param name="batchIdx">The index of the batch.</param>
        /// <returns>The tensor containing output IDs for the specified batch.</returns>
        [[nodiscard]] TensorPtr getOutputIds(SizeType batchIdx) const override
        {
            auto tensor = ITensor::slice(mJointDecodingOutput->ids, batchIdx, 1);
            tensor->squeeze(0);
            return tensor;
        }

        /// <summary>
        /// Get the tensor containing output IDs for the entire batch.
        /// </summary>
        /// <returns>The tensor containing output IDs for the entire batch.</returns>
        [[nodiscard]] TensorPtr getOutputIds() const override
        {
            return ITensor::slice(mJointDecodingOutput->ids, 0, mActualBatchSize);
        }

        /// <summary>
        /// Finalize processing for a specific batch index and return the associated CUDA event.
        /// </summary>
        /// <param name="batchIdx">The index of the batch to finalize.</param>
        /// <returns>The CUDA event associated with the finalization of the specified batch.</returns>
        [[nodiscard]] CudaEvent finalize(SizeType batchIdx) const;

        /// <summary>
        /// Finalize processing for the entire batch.
        /// </summary>
        void finalize() const override;

        /// <summary>
        /// Get the tensor containing parent IDs for the entire batch.
        /// </summary>
        /// <returns>The tensor containing parent IDs for the entire batch.</returns>
        [[nodiscard]] TensorPtr getParentIds() const override
        {
            return ITensor::slice(mJointDecodingOutput->parentIds, 0, mActualBatchSize);
        }

        /// <summary>
        /// Get the tensor containing cumulative log probabilities for the entire batch.
        /// </summary>
        /// <returns>The tensor containing cumulative log probabilities for the entire batch.</returns>
        [[nodiscard]] TensorPtr getCumLogProbs() const override
        {
            return ITensor::slice(mJointDecodingOutput->cumLogProbs, 0, mActualBatchSize);
        }

        /// <summary>
        /// Get the tensor containing cumulative log probabilities for a specific batch index.
        /// </summary>
        /// <param name="batchIdx">The index of the batch.</param>
        /// <returns>The tensor containing cumulative log probabilities for the specified batch.</returns>
        [[nodiscard]] TensorPtr getCumLogProbs(SizeType batchIdx) const
        {
            auto tensor = ITensor::slice(mJointDecodingOutput->cumLogProbs, batchIdx, 1);
            tensor->squeeze(0);
            return tensor;
        }

        /// <summary>
        /// Get the log probabilities tensor for the entire batch.
        /// </summary>
        /// <returns>The log probabilities tensor for the batch.</returns>
        [[nodiscard]] TensorPtr getLogProbs() const override
        {
            return ITensor::slice(mJointDecodingOutput->logProbs, 0, mActualBatchSize);
        }

        /// <summary>
        /// Get the log probabilities tensor for a specific batch index.
        /// </summary>
        /// <param name="batchIdx">The index of the batch.</param>
        /// <returns>The log probabilities tensor for the specified batch.</returns>
        [[nodiscard]] TensorPtr getLogProbs(SizeType batchIdx) const
        {
            auto tensor = ITensor::slice(mJointDecodingOutput->logProbs, batchIdx, 1);
            tensor->squeeze(0);
            return tensor;
        }

        /// <summary>
        /// Get the tensor containing all new tokens for the entire batch.
        /// </summary>
        /// <returns>The tensor containing all new tokens for the batch.</returns>
        [[nodiscard]] TensorPtr getAllNewTokens() const override
        {
            return mJointDecodingOutput->newTokensSteps;
        }

        /// <summary>
        /// Get the tensor containing new tokens for a specific iteration and batch index.
        /// </summary>
        /// <param name="iter">The iteration index (default is 0).</param>
        /// <returns>The tensor containing new tokens for the specified iteration and batch.</returns>
        [[nodiscard]] TensorPtr getNewTokens(SizeType iter = 0) const override
        {
            TensorPtr newTokensView = std::move(ITensor::slice(mJointDecodingOutput->newTokensSteps, iter, 1));
            newTokensView->squeeze(0);
            return ITensor::slice(newTokensView, 0, mActualBatchSize);
        }

        /// <summary>
        /// Get the vector of step counts for each batch.
        /// </summary>
        /// <returns>The vector of step counts for each batch.</returns>
        [[nodiscard]] std::vector<SizeType> getNbSteps() const override
        {
            return std::vector<SizeType>(mNbSteps.begin(), mNbSteps.begin() + mActualBatchSize);
        }

        /// <summary>
        /// Get the tensor containing the sum of finished steps for the entire batch.
        /// </summary>
        /// <returns>The tensor containing the sum of finished steps for the batch.</returns>
        [[nodiscard]] TensorPtr getNbFinished() const override
        {
            return mFinishedSum;
        }

private:
        /// <summary>
        /// Post-processes the request for a specific batch index.
        /// </summary>
        /// <param name="batchIdx">The index of the batch.</param>
        /// <returns>The CUDA event for post-processing.</returns>
        CudaEvent postProcessRequest(SizeType batchIdx) const;

private:
        /// <summary>
        /// The size of the vocabulary.
        /// </summary>
        std::size_t const mVocabSize;

        /// <summary>
        /// The size of the padded vocabulary.
        /// </summary>
        std::size_t const mVocabSizePadded;

        /// <summary>
        /// The CUDA stream for asynchronous operations.
        /// </summary>
        CudaStreamPtr mStream;

        /// <summary>
        /// Manages buffers used in the batch.
        /// </summary>
        BufferManager mBufferManager;

        /// <summary>
        /// A token used for forward processing.
        /// </summary>
        TokenPtr mForwardToken;

        /// <summary>
        /// The CUDA event for forward processing.
        /// </summary>
        CudaEvent mForwardEvent;

        /// <summary>
        /// A vector of CUDA streams for parallel processing.
        /// </summary>
        std::vector<CudaStreamPtr> mStreams;

        /// <summary>
        /// A unique pointer to a decoder interface.
        /// </summary>
        using DecoderPtr = std::unique_ptr<IDecoder>;

        /// <summary>
        /// A vector of decoder pointers.
        /// </summary>
        std::vector<DecoderPtr> mDecoders;

        /// <summary>
        /// A unique pointer to a decoding input.
        /// </summary>
        using DecodingInputPtr = std::unique_ptr<DecodingInput>;

        /// <summary>
        /// A vector of decoding input pointers.
        /// </summary>
        std::vector<DecodingInputPtr> mDecodingInputs;

        /// <summary>
        /// A unique pointer to a decoding output.
        /// </summary>
        using DecodingOutputPtr = std::unique_ptr<DecodingOutput>;

        /// <summary>
        /// A vector of decoding output pointers.
        /// </summary>
        std::vector<DecodingOutputPtr> mDecodingOutputs;

        /// <summary>
        /// A unique pointer to a joint decoding input.
        /// </summary>
        DecodingInputPtr mJointDecodingInput;

        /// <summary>
        /// A unique pointer to a joint decoding output.
        /// </summary>
        DecodingOutputPtr mJointDecodingOutput;

        /// <summary>
        /// A vector of token tensors used for draft generation.
        /// </summary>
        std::vector<TensorPtr> mDraftTokenIds;

        /// <summary>
        /// A vector of logit tensors used for draft generation.
        /// </summary>
        std::vector<TensorPtr> mDraftLogits;

        /// <summary>
        /// A vector of boolean flags indicating acceptance by logits.
        /// </summary>
        std::vector<bool> mAcceptByLogits;

        /// <summary>
        /// The number of draft tokens tensor.
        /// </summary>
        TensorPtr mNumDraftTokens;

        /// <summary>
        /// The CUDA random states tensor.
        /// </summary>
        TensorPtr mCurandStates;

        /// <summary>
        /// A vector of step counts for each batch.
        /// </summary>
        std::vector<SizeType> mNbSteps;

        /// <summary>
        /// A vector of boolean flags indicating whether each batch is finished.
        /// </summary>
        std::vector<bool> mFinished;

        /// <summary>
        /// The sum of finished steps tensor.
        /// </summary>
        TensorPtr mFinishedSum;

        /// <summary>
        /// A vector of maximum new tokens for each batch.
        /// </summary>
        std::vector<SizeType> mMaxNewTokens;

        /// <summary>
        /// A vector of beam widths for each batch.
        /// </summary>
        std::vector<SizeType> mBeamWidths;

        /// <summary>
        /// A vector of generated tokens per step for each batch.
        /// </summary>
        std::vector<SizeType> mGeneratedTokensPerStep;

        /// <summary>
        /// The tensor for finished steps.
        /// </summary>
        TensorPtr mFinishedSteps;

        /// <summary>
        /// The tensor for draft probabilities.
        /// </summary>
        TensorPtr mDraftProbs;

        /// <summary>
        /// The tensor for target probabilities.
        /// </summary>
        TensorPtr mTargetProbs;

        /// <summary>
        /// The maximum sequence length.
        /// </summary>
        SizeType mMaxSequenceLength{};

        /// <summary>
        /// The maximum attention window size.
        /// </summary>
        SizeType mMaxAttentionWindow{};

        /// <summary>
        /// The actual batch size.
        /// </summary>
        SizeType mActualBatchSize{};

        /// <summary>
        /// The maximum number of tokens per step.
        /// </summary>
        SizeType mMaxTokensPerStep{};
    };
}
