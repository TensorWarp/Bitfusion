#include "statefulDecoder.h"
#include "../common/cudaUtils.h"
#include "../common/memoryUtils.h"
#include "../kernels/decodingCommon.h"
#include "runtimeKernels.h"

namespace tc = bitfusion::common;
namespace tk = bitfusion::kernels;
using namespace bitfusion::runtime;

using TensorPtr = ITensor::SharedPtr;

/// <summary>
/// Constructor for the StatefulDecoder class.
/// </summary>
/// <param name="vocabSize">The vocabulary size.</param>
/// <param name="vocabSizePadded">The padded vocabulary size.</param>
/// <param name="stream">A pointer to the CUDA stream.</param>
StatefulDecoder::StatefulDecoder(std::size_t vocabSize, std::size_t vocabSizePadded, CudaStreamPtr stream)
    : mVocabSize{ vocabSize },
    mVocabSizePadded{ vocabSizePadded },
    mStream{ std::move(stream) },
    mBufferManager{ mStream },
    mDecodingInput(std::make_unique<DecodingInput>(0, 0, 0,
        mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<float>::value),
        mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<TokenIdType>::value))),
    mDecodingOutput(std::make_unique<DecodingOutput>(
        mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<TokenIdType>::value))),
    mNbSteps{ 0 }
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto& dInput = *mDecodingInput;
    auto& dOutput = *mDecodingOutput;

    dInput.sequenceLimitLength = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<SizeType>::value);
    dInput.lengths = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<SizeType>::value);

    auto outputIds = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<TokenIdType>::value);
    dOutput.newTokens = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<TokenIdType>::value);
    dOutput.parentIds = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<TokenIdType>::value);
    dOutput.finished = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);
    dOutput.finishedSum = BufferManager::pinned(ITensor::makeShape({ 1 }), TRTDataType<SizeType>::value);
    dOutput.lengths = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<SizeType>::value);
    dOutput.cumLogProbs = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<float>::value);
    dOutput.beamHypotheses.empty(mBufferManager);

    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

/// <summary>
/// Sets up the StatefulDecoder with the specified parameters.
/// </summary>
/// <param name="maxBatchSize">The maximum batch size.</param>
/// <param name="maxBeamWidth">The maximum beam width.</param>
/// <param name="maxAttentionWindow">The maximum attention window.</param>
/// <param name="maxSequenceLength">The maximum sequence length.</param>
/// <param name="maxTokensPerStep">The maximum tokens per step.</param>
/// <param name="dtype">The data type to use.</param>
void StatefulDecoder::setup(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxAttentionWindow, SizeType maxSequenceLength, SizeType maxTokensPerStep, nvinfer1::DataType dtype)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    CHECK(maxTokensPerStep == 1);
    mDecoder = IDecoder::create(dtype, mVocabSize, mVocabSizePadded, mStream);

    reshapeBuffers(maxBatchSize, maxBeamWidth, maxAttentionWindow, maxSequenceLength);
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

/// <summary>
/// Reshapes the buffers for the StatefulDecoder based on the specified parameters.
/// </summary>
/// <param name="batchSize">The batch size.</param>
/// <param name="beamWidth">The beam width.</param>
/// <param name="maxAttentionWindow">The maximum attention window.</param>
/// <param name="maxSequenceLength">The maximum sequence length.</param>
void StatefulDecoder::reshapeBuffers(SizeType batchSize, SizeType beamWidth, SizeType maxAttentionWindow, SizeType maxSequenceLength)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    CHECK(batchSize > 0);
    CHECK(beamWidth > 0);
    CHECK(maxSequenceLength > 0);

    mMaxSequenceLength = maxSequenceLength;
    mMaxAttentionWindow = maxAttentionWindow;

    auto const batchSizeShape = ITensor::makeShape({ batchSize });
    auto const batchSizeXbeamWidth = ITensor::makeShape({ batchSize, beamWidth });

    auto& dInput = *mDecodingInput;
    const_cast<ITensor&>(*dInput.endIds).reshape(batchSizeXbeamWidth);
    auto& sequenceLimitLength = const_cast<ITensor&>(*dInput.sequenceLimitLength);
    sequenceLimitLength.reshape(batchSizeShape);
    kernels::invokeFill(sequenceLimitLength, mMaxSequenceLength, *mStream);
    auto& inputLengths = const_cast<ITensor&>(*dInput.lengths);
    inputLengths.reshape(batchSizeXbeamWidth);
    mBufferManager.setZero(inputLengths);

    auto const outputIdsShape = ITensor::makeShape({ batchSize, beamWidth, maxSequenceLength });

    auto& dOutput = *mDecodingOutput;
    dOutput.ids->reshape(outputIdsShape);
    dOutput.newTokens->reshape(batchSizeXbeamWidth);
    mBufferManager.setZero(*dOutput.newTokens);
    dOutput.parentIds->reshape(outputIdsShape);
    dOutput.finished->reshape(batchSizeXbeamWidth);
    dInput.finished = ITensor::view(dOutput.finished);
    mBufferManager.setZero(*dOutput.finished);
    mBufferManager.setZero(*dOutput.finishedSum);

    if (beamWidth > 1)
    {
        dOutput.cumLogProbs->reshape(batchSizeXbeamWidth);
        mBufferManager.setZero(*dOutput.cumLogProbs);
        dOutput.beamHypotheses.reshape(batchSize, beamWidth, mMaxSequenceLength);
    }
    else
    {
        dOutput.beamHypotheses.release();
    }

    mNbSteps = 0;
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

/// <summary>
/// Sets up the StatefulDecoder with the specified parameters.
/// </summary>
/// <param name="inputs">The input data for generation.</param>
/// <param name="outputs">The output data for generation.</param>
/// <param name="samplingConfig">The configuration for sampling.</param>
void StatefulDecoder::newBatch(GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& manager = mBufferManager;
    auto& stream = mStream;

    auto const inputLengths = inputs.lengths;
    auto const& inputLengthsShape = inputLengths->getShape();
    auto const batchSize = inputLengthsShape.d[0];
    auto const beamWidth = samplingConfig.beamWidth;

    reshapeBuffers(batchSize, beamWidth, mMaxAttentionWindow, mMaxSequenceLength);
    mDecoder->setup(samplingConfig, batchSize, mMaxSequenceLength);

    auto const& outputIdsShape = mDecodingOutput->ids->getShape();
    auto const maxBatchSize = outputIdsShape.d[0];
    CHECK(batchSize == maxBatchSize);
    auto const maxBeamWidth = outputIdsShape.d[1];
    CHECK(beamWidth == maxBeamWidth);

    auto const& inputIds = inputs.ids;
    auto const inputLengthsHost = manager.copyFrom(*inputLengths, MemoryType::kCPU);
    auto const* inputLengthsData = bufferCast<SizeType>(*inputLengthsHost);
    SizeType const maxInputLength = *std::max_element(inputLengthsData, inputLengthsData + inputLengths->getSize());

    TensorPtr inputOffsets = manager.emptyTensor(MemoryType::kGPU, TRTDataType<SizeType>::value);
    if (inputs.packed)
    {
        inputOffsets->reshape(ITensor::makeShape({ batchSize + 1 }));
        manager.setZero(*inputOffsets);
        kernels::invokeInclusiveSum(*ITensor::slice(inputOffsets, 1), *inputLengths, manager, *stream);
    }

    CHECK(inputIds->getDataType() == TRTDataType<TokenIdType>::value);
    auto const endId = inputs.endId;
    auto const padId = inputs.padId;

    auto& dInput = *mDecodingInput;
    dInput.maxLength = maxInputLength;
    dInput.maxAttentionWindow = mMaxAttentionWindow;
    dInput.batchSize = batchSize;
    kernels::invokeFill(const_cast<ITensor&>(*dInput.endIds), endId, *stream);
    dInput.embeddingBias = inputs.embeddingBias;
    dInput.badWordsList = inputs.badWordsList;
    dInput.stopWordsList = inputs.stopWordsList;
    auto inputLengthsView = ITensor::view(dInput.lengths, ITensor::makeShape({ batchSize * beamWidth }));
    kernels::tileTensor(const_cast<ITensor&>(*inputLengthsView), *inputLengths, beamWidth, *stream);
    if (inputs.maxNewTokens)
    {
        auto const maxNewTokens = inputs.maxNewTokens.value();
        CHECK_WITH_INFO(maxInputLength + maxNewTokens <= mMaxSequenceLength,
            tc::fmtstr("Input length (%d) + max new tokens (%d) must be less than max sequence length (%d).",
                maxInputLength, maxNewTokens, mMaxSequenceLength));
        manager.copy(*inputLengths, const_cast<ITensor&>(*dInput.sequenceLimitLength));
        kernels::invokeAdd(const_cast<ITensor&>(*dInput.sequenceLimitLength), maxNewTokens, *stream);
    }
    else
    {
        kernels::invokeFill(const_cast<ITensor&>(*dInput.sequenceLimitLength), mMaxSequenceLength, *stream);
    }

    auto& dOutput = *mDecodingOutput;
    manager.setZero(*dOutput.newTokens);
    manager.setZero(*dOutput.finished);
    manager.setZero(*dOutput.finishedSum);

    if (outputs.cumLogProbs)
    {
        dOutput.cumLogProbs = outputs.cumLogProbs;
    }
    dOutput.logProbs = outputs.logProbs;

    if (dOutput.cumLogProbs)
        manager.setZero(*dOutput.cumLogProbs);

    if (dOutput.logProbs)
        manager.setZero(*dOutput.logProbs);

    if (beamWidth > 1)
    {
        std::vector<float> cumLogProbsHost(batchSize * beamWidth, DecodingOutput::kNegativeInfinity);
        for (SizeType i = 0; i < batchSize; ++i)
        {
            cumLogProbsHost[tc::flat_index2(i, 0, beamWidth)] = 0;
        }
        manager.copy(cumLogProbsHost.data(), *dOutput.cumLogProbs);

        manager.setZero(*dOutput.parentIds);
        dOutput.beamHypotheses.init(manager, endId);
    }
    else
    {
    }

    kernels::initOutputIds(
        *dOutput.ids, *inputIds, *inputLengths, *inputOffsets, padId, endId, maxInputLength, inputs.packed, *stream);

    mNbSteps = 0;
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

/// <summary>
/// Asynchronously performs the forward pass of the decoder.
/// </summary>
/// <param name="output">The output tensor to be filled by the decoder.</param>
/// <param name="input">The input tensor containing logits.</param>
void StatefulDecoder::forwardAsync(decoder::Output& output, decoder::Input const& input)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& logits = input.logits;
    auto const& logitsShape = logits->getShape();

    auto const& outputIdsShape = mDecodingOutput->ids->getShape();
    auto const batchSize = outputIdsShape.d[0];
    CHECK(logitsShape.d[0] == batchSize);
    auto const maxBeamWidth = outputIdsShape.d[1];
    CHECK(logitsShape.d[1] == maxBeamWidth);
    CHECK(static_cast<std::size_t>(logitsShape.d[2]) == mVocabSizePadded);

    auto& srcCacheIndirection = input.cacheIndirection;
    auto& tgtCacheIndirection = output.cacheIndirection;
    CHECK_WITH_INFO((srcCacheIndirection && tgtCacheIndirection) || (!srcCacheIndirection && !tgtCacheIndirection),
        "Specify both srcCacheIndirection and tgtCacheIndirection or neither.");
    CHECK(!srcCacheIndirection || srcCacheIndirection->getDataType() == TRTDataType<SizeType>::value);
    CHECK(!tgtCacheIndirection || tgtCacheIndirection->getDataType() == TRTDataType<SizeType>::value);

    auto& dInput = *mDecodingInput;
    auto& dOutput = *mDecodingOutput;
    dInput.logits = logits;
    if (srcCacheIndirection && tgtCacheIndirection)
    {
        dInput.cacheIndirection = srcCacheIndirection;
        dOutput.cacheIndirection = tgtCacheIndirection;
    }
    dOutput.lengths = output.sequenceLengths;

    mDecoder->forwardAsync(dOutput, dInput);
    mStream->record(mDecodedEvent.get());

    dInput.step += 1;
    mNbSteps += 1;
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

/// <summary>
/// Synchronizes the forward pass of the decoder, waiting for completion.
/// </summary>
void StatefulDecoder::forwardSync()
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    mDecodedEvent.synchronize();
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

/// <summary>
/// Finalizes the decoding process, gathering the output data.
/// </summary>
void StatefulDecoder::finalize() const
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& outputIds = mDecodingOutput->ids;
    auto finalOutputIds = mBufferManager.gpu(outputIds->getShape(), outputIds->getDataType());
    mDecoder->gatherTree(*finalOutputIds, *mDecodingOutput, *mDecodingInput, mBufferManager);
    mBufferManager.copy(*finalOutputIds, *outputIds);
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}
