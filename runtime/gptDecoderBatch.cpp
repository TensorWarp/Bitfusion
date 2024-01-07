#include "gptDecoderBatch.h"

#include "../common/assert.h"
#include "../kernels/decodingCommon.h"
#include "bufferManager.h"
#include "cudaEvent.h"
#include "runtimeKernels.h"

#include <algorithm>
#include <memory>

using namespace bitfusion::runtime;

namespace tc = bitfusion::common;
namespace tk = bitfusion::kernels;

namespace
{
SamplingConfig extractSamplingConfig(SamplingConfig const& batchSamplingConfig, SizeType batchIdx)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    SamplingConfig samplingConfig{batchSamplingConfig.beamWidth};

    auto extractOptional = [&batchIdx](auto& single, auto const& batch)
    {
        using T = typename std::remove_reference_t<decltype(batch)>::value_type;
        if (batch)
        {
            if (batch->size() > 1)
                single.emplace(T{batch->at(batchIdx)});
            else
                single.emplace(T{batch->at(0)});
        }
    };

    extractOptional(samplingConfig.temperature, batchSamplingConfig.temperature);
    extractOptional(samplingConfig.minLength, batchSamplingConfig.minLength);
    extractOptional(samplingConfig.repetitionPenalty, batchSamplingConfig.repetitionPenalty);
    extractOptional(samplingConfig.presencePenalty, batchSamplingConfig.presencePenalty);
    extractOptional(samplingConfig.topK, batchSamplingConfig.topK);
    extractOptional(samplingConfig.topP, batchSamplingConfig.topP);
    extractOptional(samplingConfig.randomSeed, batchSamplingConfig.randomSeed);
    extractOptional(samplingConfig.topPDecay, batchSamplingConfig.topPDecay);
    extractOptional(samplingConfig.topPMin, batchSamplingConfig.topPMin);
    extractOptional(samplingConfig.topPResetIds, batchSamplingConfig.topPResetIds);

    samplingConfig.beamSearchDiversityRate = batchSamplingConfig.beamSearchDiversityRate;
    samplingConfig.lengthPenalty = batchSamplingConfig.lengthPenalty;

    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return samplingConfig;
}

}

GptDecoderBatch::GptDecoderBatch(
    std::size_t vocabSize, std::size_t vocabSizePadded, GptDecoderBatch::CudaStreamPtr stream)
    : mVocabSize{vocabSize}
    , mVocabSizePadded{vocabSizePadded}
    , mStream{std::move(stream)}
    , mBufferManager{mStream}
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType>::value;
    auto constexpr nvFloatType = TRTDataType<float>::value;

    auto& dInput = mJointDecodingInput;
    auto dummyLogits = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    auto endIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dInput = std::make_unique<DecodingInput>(0, 0, 0, std::move(dummyLogits), std::move(endIds));

    dInput->sequenceLimitLength = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dInput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    auto& dOutput = mJointDecodingOutput;
    auto outputIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput = std::make_unique<DecodingOutput>(std::move(outputIds));

    dOutput->newTokensSteps = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->parentIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    mFinishedSteps
        = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);
    mDraftProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    mTargetProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->finishedSum = mBufferManager.emptyTensor(MemoryType::kPINNED, nvSizeType);
    mFinishedSum = mBufferManager.pinned(ITensor::makeShape({1}), nvSizeType);
    dOutput->cumLogProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->logProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->beamHypotheses.empty(mBufferManager);

    mNumDraftTokens = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    mCurandStates = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT8);

    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::setup(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxAttentionWindow,
    SizeType maxSequenceLength, SizeType maxTokensPerStep, nvinfer1::DataType dtype)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    CHECK(maxBatchSize > 0);
    CHECK(maxBeamWidth > 0);
    CHECK(maxTokensPerStep > 0);
    CHECK(maxSequenceLength > 0);
    mActualBatchSize = maxBatchSize;
    mGeneratedTokensPerStep.resize(maxBatchSize);
    mMaxSequenceLength = maxSequenceLength;
    mMaxAttentionWindow = maxAttentionWindow;
    mMaxTokensPerStep = maxTokensPerStep;

    auto const maxBatchSizeShape = ITensor::makeShape({maxBatchSize});
    auto const maxBatchSizeXmaxBeamWidth = ITensor::makeShape({maxBatchSize, maxBeamWidth});
    auto const maxBatchSizeXmaxTokensPerStepXmaxBeamWidth
        = ITensor::makeShape({maxBatchSize, maxTokensPerStep, maxBeamWidth});
    auto const maxTokensPerStepXmaxBatchSizeXmaxBeamWidth
        = ITensor::makeShape({maxTokensPerStep, maxBatchSize, maxBeamWidth});

    auto& dInput = *mJointDecodingInput;
    const_cast<ITensor&>(*dInput.endIds).reshape(maxBatchSizeXmaxBeamWidth);
    auto& sequenceLimitLength = const_cast<ITensor&>(*dInput.sequenceLimitLength);
    sequenceLimitLength.reshape(maxBatchSizeShape);
    kernels::invokeFill(sequenceLimitLength, mMaxSequenceLength, *mStream);
    auto& inputLengths = const_cast<ITensor&>(*dInput.lengths);
    inputLengths.reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(inputLengths);

    auto const jointOutputIdsShape = ITensor::makeShape({maxBatchSize, maxBeamWidth, maxSequenceLength});

    auto& dOutput = *mJointDecodingOutput;
    dOutput.ids->reshape(jointOutputIdsShape);

    dOutput.newTokensSteps->reshape(maxTokensPerStepXmaxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*dOutput.newTokensSteps);
    mFinishedSteps->reshape(maxBatchSizeXmaxTokensPerStepXmaxBeamWidth);
    mBufferManager.setZero(*mFinishedSteps);

    if (mMaxTokensPerStep > 1)
    {
        mDraftProbs->reshape(ITensor::makeShape(
            {maxBatchSize, maxTokensPerStep - 1, maxBeamWidth, static_cast<SizeType>(mVocabSizePadded)}));
        mTargetProbs->reshape(ITensor::makeShape(
            {maxBatchSize, maxTokensPerStep, maxBeamWidth, static_cast<SizeType>(mVocabSizePadded)}));
    }

    dOutput.parentIds->reshape(jointOutputIdsShape);
    dOutput.finishedSum->reshape(maxBatchSizeShape);
    mBufferManager.setZero(*dOutput.finishedSum);

    dOutput.cumLogProbs->reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*dOutput.cumLogProbs);

    dOutput.logProbs->reshape(ITensor::makeShape({maxBatchSize, maxBeamWidth, mMaxSequenceLength}));
    mBufferManager.setZero(*dOutput.logProbs);

    if (maxBeamWidth > 1)
    {
        dOutput.beamHypotheses.reshape(maxBatchSize, maxBeamWidth, mMaxSequenceLength);
    }
    else
    {
        dOutput.beamHypotheses.release();
    }

    mDraftTokenIds.resize(maxBatchSize);
    mDraftLogits.resize(maxBatchSize);
    mAcceptByLogits.resize(maxBatchSize);
    mNumDraftTokens->reshape(ITensor::makeShape({maxBatchSize, 1}));
    mCurandStates->reshape(ITensor::makeShape({maxBatchSize, sizeof(curandState_t)}));

    mStreams.resize(maxBatchSize);
    mDecoders.resize(maxBatchSize);
    mDecodingInputs.resize(maxBatchSize);
    mDecodingOutputs.resize(maxBatchSize);
    mNbSteps.resize(maxBatchSize);
    mFinished.resize(maxBatchSize);
    mMaxNewTokens.resize(maxBatchSize);
    mBeamWidths.resize(maxBatchSize);
    auto const device = mStream->getDevice();
    for (SizeType i = 0; i < maxBatchSize; ++i)
    {
        mStreams[i] = std::make_shared<CudaStream>();
        CHECK(mStreams[i]->getDevice() == device);
        mDecoders[i] = IGptDecoder::create(dtype, mVocabSize, mVocabSizePadded, mStreams[i]);
        mDecodingInputs[i].reset();
        mDecodingOutputs[i].reset();
        mNbSteps[i] = 0;
        mFinished[i] = true;
        mMaxNewTokens[i] = 0;
        mBeamWidths[i] = 0;
        mGeneratedTokensPerStep[i] = 0;
    }
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::newRequest(
    SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    CHECK(batchIdx >= 0);
    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const batchSize = jointOutputIdsShape.d[0];
    CHECK(0 <= batchSize && batchIdx < batchSize);
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    auto const beamWidth = samplingConfig.beamWidth;
    CHECK_WITH_INFO(beamWidth <= maxBeamWidth,
        tc::fmtstr("Beam width (%d) must be smaller than maxBeamWidth (%d) passed to decoder setup function.",
            beamWidth, maxBeamWidth));
    auto const& requestIds = request.ids;
    auto const inputLength = request.inputLen;
    auto const maxNewTokens = request.maxNewTokens.value_or(mMaxSequenceLength - inputLength);
    CHECK_WITH_INFO(inputLength + maxNewTokens <= mMaxSequenceLength,
        tc::fmtstr("Input length (%d) + max new tokens (%d) must be less than max sequence length (%d).", inputLength,
            maxNewTokens, mMaxSequenceLength));
    CHECK(requestIds->getDataType() == TRTDataType<TokenIdType>::value);
    auto const endId = request.endId.value_or(mVocabSize - 1);

    auto constexpr localBatchSize = 1;

    auto& stream = mStreams[batchIdx];
    BufferManager manager{stream};

    auto& dJointInput = *mJointDecodingInput;
    auto& dInput = mDecodingInputs.at(batchIdx);

    TensorPtr endIdTensorPtr{ITensor::slice(constPointerCast(dJointInput.endIds), batchIdx, localBatchSize)};
    kernels::invokeFill(*endIdTensorPtr, endId, *stream);
    dInput = std::make_unique<DecodingInput>(
        inputLength, mMaxAttentionWindow, localBatchSize, dJointInput.logits, endIdTensorPtr);

    if (request.embeddingBias)
    {
        TensorPtr biasView = ITensor::view(request.embeddingBias);
        biasView->unsqueeze(0);
        dInput->embeddingBias = biasView;
    }
    if (request.badWordsList)
    {
        TensorPtr badWordsView = ITensor::view(request.badWordsList);
        badWordsView->unsqueeze(0);
        dInput->badWordsList = badWordsView;
    }
    if (request.stopWordsList)
    {
        TensorPtr stopWordsView = ITensor::view(request.stopWordsList);
        stopWordsView->unsqueeze(0);
        dInput->stopWordsList = stopWordsView;
    }

    TensorPtr sequenceLimitLength{
        ITensor::slice(constPointerCast(dJointInput.sequenceLimitLength), batchIdx, localBatchSize)};
    kernels::invokeFill(*sequenceLimitLength, inputLength + maxNewTokens, *stream);
    dInput->sequenceLimitLength = std::move(sequenceLimitLength);
    TensorPtr inputLengths{ITensor::slice(constPointerCast(dJointInput.lengths), batchIdx, localBatchSize)};
    kernels::invokeFill(*inputLengths, inputLength, *stream);
    dInput->lengths = inputLengths;

    auto& dJointOutput = *mJointDecodingOutput;
    auto& dOutput = mDecodingOutputs.at(batchIdx);
    auto const outputIdsShape = ITensor::makeShape({localBatchSize, beamWidth, mMaxSequenceLength});

    TensorPtr outputIds = ITensor::slice(dJointOutput.ids, batchIdx, localBatchSize);
    outputIds->reshape(outputIdsShape);
    dOutput = std::make_unique<DecodingOutput>(outputIds);

    dOutput->finishedSum = ITensor::slice(dJointOutput.finishedSum, batchIdx, localBatchSize);
    manager.setZero(*dOutput->finishedSum);

    dOutput->newTokensVec.resize(mMaxTokensPerStep);
    for (SizeType ti = 0; ti < mMaxTokensPerStep; ++ti)
    {
        TensorPtr newTokensStepView = std::move(ITensor::slice(dJointOutput.newTokensSteps, ti, localBatchSize));
        newTokensStepView->squeeze(0);
        dOutput->newTokensVec[ti] = ITensor::slice(newTokensStepView, batchIdx, localBatchSize);
        manager.setZero(*dOutput->newTokensVec[ti]);
    }

    TensorPtr finishedSteps = ITensor::slice(mFinishedSteps, batchIdx, localBatchSize);
    manager.setZero(*finishedSteps);

    dOutput->cumLogProbs = nullptr;
    if (request.computeCumLogProbs || beamWidth > 1)
    {
        dOutput->cumLogProbs = ITensor::slice(dJointOutput.cumLogProbs, batchIdx, localBatchSize);
        manager.setZero(*dOutput->cumLogProbs);
    }

    dOutput->logProbs = nullptr;
    if (request.computeLogProbs)
    {
        dOutput->logProbs = ITensor::slice(dJointOutput.logProbs, batchIdx, localBatchSize);
        manager.setZero(*dOutput->logProbs);
    }

    if (beamWidth > 1)
    {
        kernels::invokeFill(
            *IBuffer::slice(dOutput->cumLogProbs, 1, beamWidth - 1), DecodingOutput::kNegativeInfinity, *stream);
        dOutput->parentIds = ITensor::slice(dJointOutput.parentIds, batchIdx, localBatchSize);
        dOutput->parentIds->reshape(outputIdsShape);
        manager.setZero(*dOutput->parentIds);
        dOutput->beamHypotheses = dJointOutput.beamHypotheses.slice(batchIdx, localBatchSize);
        dOutput->beamHypotheses.init(manager, endId);
    }

    auto generatedTokensPerStep = request.generatedTokensPerStep();
    if (generatedTokensPerStep > 1)
    {
        CHECK(beamWidth == 1);
        auto numDraftTokens = generatedTokensPerStep - 1;
        TensorPtr draftTokensView = ITensor::view(request.draftTokens, ITensor::makeShape({1, 1, numDraftTokens}));
        mDraftTokenIds[batchIdx] = draftTokensView;
        mAcceptByLogits[batchIdx] = false;
        if (request.draftLogits.has_value())
        {
            TensorPtr draftLogitsView = ITensor::view(request.draftLogits.value());
            mDraftLogits[batchIdx] = draftLogitsView;
            mAcceptByLogits[batchIdx] = true;
        }

        auto numDraftTokensView = ITensor::slice(mNumDraftTokens, batchIdx, localBatchSize);
        kernels::invokeFill(*numDraftTokensView, numDraftTokens, *stream);

        auto const curandStatesView = ITensor::slice(mCurandStates, batchIdx, localBatchSize);
        auto curandState = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*curandStatesView));
        if (samplingConfig.randomSeed.has_value())
        {
            tk::invokeCurandInitialize(
                curandState, localBatchSize, samplingConfig.randomSeed.value()[0], stream->get());
        }
        else
        {
            tk::invokeCurandInitialize(curandState, localBatchSize, 0, stream->get());
        }
    }

    mDecoders[batchIdx]->setup(samplingConfig, localBatchSize, mMaxSequenceLength);
    mBeamWidths[batchIdx] = beamWidth;
    mNbSteps[batchIdx] = 0;
    mFinished[batchIdx] = false;
    mMaxNewTokens[batchIdx] = maxNewTokens;
    mGeneratedTokensPerStep[batchIdx] = generatedTokensPerStep;

    auto const requestIdsShape = requestIds->getShape();
    auto inputIdsView = ITensor::view(requestIds, ITensor::makeShape({localBatchSize, requestIdsShape.d[0]}));
    auto outputIdsView = ITensor::view(outputIds, ITensor::makeShape({beamWidth, mMaxSequenceLength}));
    kernels::invokeFill(*outputIdsView, endId, *stream);
    kernels::tileTensor(*outputIdsView, *inputIdsView, beamWidth, *stream);
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

GptDecoderBatch::TokenPtr GptDecoderBatch::forwardAsync(
    decoder_batch::Output& output, decoder_batch::Input const& input)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& allTargetLogits = input.logits;

    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const maxBeamWidth = jointOutputIdsShape.d[1];

    auto& srcCacheIndirection = input.cacheIndirection;
    auto& tgtCacheIndirection = output.cacheIndirection;
    CHECK_WITH_INFO((srcCacheIndirection && tgtCacheIndirection) || (!srcCacheIndirection && !tgtCacheIndirection),
        "Specify both srcCacheIndirection and tgtCacheIndirection or neither.");
    CHECK(!srcCacheIndirection || srcCacheIndirection->getDataType() == TRTDataType<SizeType>::value);
    CHECK(!tgtCacheIndirection || tgtCacheIndirection->getDataType() == TRTDataType<SizeType>::value);

    CHECK(static_cast<SizeType>(output.sequenceLengths->getSize()) == mActualBatchSize * maxBeamWidth);
    TensorPtr sequenceLengths
        = ITensor::view(output.sequenceLengths, ITensor::makeShape({mActualBatchSize, maxBeamWidth}));
    CHECK(sequenceLengths);
    auto constexpr singleRequest = 1;

    CudaEvent eventStart{};
    mStream->record(eventStart);
    for (std::int32_t bi = 0; bi < mActualBatchSize; ++bi)
    {
        if (mFinished[bi] || !input.active.at(bi))
        {
            continue;
        }

        auto& targetLogits = allTargetLogits[bi];
        auto const& logitsShape = targetLogits->getShape();
        CHECK_WITH_INFO(logitsShape.d[0] == mGeneratedTokensPerStep[bi],
            tc::fmtstr(
                "First dim (%d) does not match generated tokens (%d)", logitsShape.d[0], mGeneratedTokensPerStep[bi]));
        CHECK_WITH_INFO(logitsShape.d[1] == mBeamWidths[bi],
            tc::fmtstr("Second dim (%d) does not match beam width (%d)", logitsShape.d[1], mBeamWidths[bi]));
        CHECK(static_cast<std::size_t>(logitsShape.d[2]) == mVocabSizePadded);

        auto& stream = mStreams[bi];
        stream->wait(eventStart.get());
        auto& dInput = *mDecodingInputs[bi];
        auto& dOutput = *mDecodingOutputs[bi];
        auto& decoder = *mDecoders[bi];

        TensorPtr finishedSteps = ITensor::slice(mFinishedSteps, bi, singleRequest);
        finishedSteps->squeeze(0);

        if (mGeneratedTokensPerStep[bi] > 1 && mAcceptByLogits[bi])
        {
            auto numDraftTokens = ITensor::slice(mNumDraftTokens, bi, singleRequest);
            auto const curandStatesView = ITensor::slice(mCurandStates, bi, singleRequest);
            auto curandState = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*curandStatesView));
            auto const& samplingConfig = decoder.getSamplingConfig();
            const bool useRandomAcceptanceThreshold = !samplingConfig.draftAcceptanceThreshold.has_value();
            const float randomAcceptanceThreshold
                = useRandomAcceptanceThreshold ? 0 : samplingConfig.draftAcceptanceThreshold.value()[0];

            TensorPtr draftProbs = ITensor::slice(mDraftProbs, bi, singleRequest);
            TensorPtr targetProbs = ITensor::slice(mTargetProbs, bi, singleRequest);
            draftProbs = ITensor::view(draftProbs,
                ITensor::makeShape(
                    {mMaxTokensPerStep - 1, singleRequest, mBeamWidths[bi], static_cast<SizeType>(mVocabSizePadded)}));
            targetProbs = ITensor::view(targetProbs,
                ITensor::makeShape(
                    {mMaxTokensPerStep, singleRequest, mBeamWidths[bi], static_cast<SizeType>(mVocabSizePadded)}));

            IGptDecoder::acceptDraftTokensByLogits(
 *mDraftLogits[bi],
 *targetLogits,
 *draftProbs,
 *targetProbs,
 *numDraftTokens,
 *finishedSteps, static_cast<SizeType>(mVocabSize),
                static_cast<SizeType>(mVocabSizePadded), useRandomAcceptanceThreshold, randomAcceptanceThreshold,
                curandState, stream);
        }

        if (srcCacheIndirection && tgtCacheIndirection)
        {
            auto srcView = std::shared_ptr(ITensor::slice(srcCacheIndirection, bi, singleRequest));
            auto tgtView = std::shared_ptr(ITensor::slice(tgtCacheIndirection, bi, singleRequest));
            dInput.cacheIndirection = ITensor::view(
                srcView, ITensor::makeShape({singleRequest, mBeamWidths[bi], srcView->getShape().d[2]}));
            dOutput.cacheIndirection = ITensor::view(
                tgtView, ITensor::makeShape({singleRequest, mBeamWidths[bi], tgtView->getShape().d[2]}));
        }

        auto sequenceLengthsView = std::shared_ptr(ITensor::slice(sequenceLengths, bi, singleRequest));
        dOutput.lengths = ITensor::view(sequenceLengthsView, ITensor::makeShape({singleRequest, mBeamWidths[bi]}));

        for (std::int32_t di = 0; di < mGeneratedTokensPerStep[bi]; ++di)
        {
            dInput.logits = ITensor::slice(targetLogits, di, singleRequest);
            dOutput.newTokens = ITensor::view(dOutput.newTokensVec[di]);
            dInput.finished = ITensor::slice(finishedSteps, di, 1);
            dOutput.finished = ITensor::slice(finishedSteps, std::min(di + 1, mGeneratedTokensPerStep[bi] - 1), 1);

            decoder.forwardAsync(dOutput, dInput);

            mNbSteps[bi] += 1;
            mFinished[bi] = mNbSteps[bi] >= mMaxNewTokens[bi];
            dInput.step += 1;
        }

        if (mGeneratedTokensPerStep[bi] > 1 && !mAcceptByLogits[bi])
        {
            auto draftTokenIds = mDraftTokenIds[bi];
            auto numDraftTokens = ITensor::slice(mNumDraftTokens, bi, singleRequest);
            auto finishedFinal = ITensor::slice(finishedSteps, 0, 1);
            IGptDecoder::acceptDraftTokensByIds(
 *dOutput.ids,
 *draftTokenIds,
 *dInput.lengths,
 *numDraftTokens,
 *dOutput.lengths,
 *finishedSteps,
 *finishedFinal,
 *dOutput.finishedSum, stream);
        }

        CudaEvent event{};
        stream->record(event);
        mStream->wait(event);
    }

    CudaEvent eventStop{};
    mStream->record(eventStop);
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return std::make_unique<decoder_batch::Token>(std::move(eventStop), input.active);
}

void GptDecoderBatch::forwardSync(decoder_batch::Token const& token)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    token.event.synchronize();

    for (std::int32_t i = 0; i < mActualBatchSize; ++i)
    {
        if (token.active[i] && !mFinished[i])
        {
            auto& dOutput = *mDecodingOutputs[i];
            mFinished[i] = mFinished[i]
                || *bufferCast<SizeType>(*dOutput.finishedSum) == static_cast<SizeType>(dOutput.lengths->getSize());
        }
    }
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

CudaEvent GptDecoderBatch::postProcessRequest(SizeType batchIdx) const
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& stream = mStreams[batchIdx];
    auto manager = BufferManager{stream};
    auto& decoder = *mDecoders[batchIdx];

    auto& dInput = *mDecodingInputs[batchIdx];
    auto& dOutput = *mDecodingOutputs[batchIdx];

    auto& outputIds = dOutput.ids;
    auto finalOutputIds = manager.gpu(outputIds->getShape(), outputIds->getDataType());
    decoder.gatherTree(*finalOutputIds, dOutput, dInput, manager);
    manager.copy(*finalOutputIds, *outputIds);

    CudaEvent event{};
    stream->record(event);
    mStream->wait(event);
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return event;
}

void GptDecoderBatch::newBatch(
    GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const& inputLengths = inputs.lengths;
    mActualBatchSize = inputLengths->getShape().d[0];
    mGeneratedTokensPerStep.resize(mActualBatchSize);

    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const maxBatchSize = jointOutputIdsShape.d[0];
    CHECK(mActualBatchSize <= maxBatchSize);
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    CHECK(samplingConfig.beamWidth <= maxBeamWidth);

    auto const inputIdsShape = inputs.ids->getShape();
    TensorPtr inputIdsFlatView = ITensor::view(inputs.ids);
    inputIdsFlatView->reshape(ITensor::makeShape({inputIdsShape.d[1]}));
    auto inputLengthsHost = mBufferManager.copyFrom(*inputLengths, MemoryType::kCPU);
    auto inputLengthsPtr = bufferCast<SizeType>(*inputLengthsHost);
    auto inputOffset = 0;
    for (auto batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        mGeneratedTokensPerStep[batchIdx] = 1;
        auto const inputLength = inputLengthsPtr[batchIdx];
        auto const inputShape = ITensor::makeShape({inputLength});
        TensorPtr inputView;
        if (inputs.packed)
        {
            inputView = ITensor::slice(inputIdsFlatView, inputOffset, inputLength);
            inputOffset += inputLength;
        }
        else
        {
            inputView = ITensor::slice(inputs.ids, batchIdx, 1);
            inputView->reshape(inputShape);
        }
        auto request = decoder_batch::Request{inputView, inputLength, inputs.maxNewTokens, inputs.endId};
        request.computeCumLogProbs = (outputs.cumLogProbs != nullptr);
        request.computeLogProbs = (outputs.logProbs != nullptr);

        if (inputs.embeddingBias)
        {
            THROW("newBatch doesn't support embeddingBias yet.");
        }
        if (inputs.badWordsList)
        {
            auto const& shape = inputs.badWordsList->getShape();
            if (shape.nbDims == 2)
            {
                request.badWordsList = inputs.badWordsList;
            }
            else
            {
                assert(shape.nbDims == 3);
                TensorPtr badWordsListView = ITensor::slice(inputs.badWordsList, batchIdx, 1);
                badWordsListView->squeeze(0);
                request.badWordsList = badWordsListView;
            }
        }
        if (inputs.stopWordsList)
        {
            TensorPtr stopWordsListView = ITensor::slice(inputs.stopWordsList, batchIdx, 1);
            stopWordsListView->squeeze(0);
            request.stopWordsList = stopWordsListView;
        }
        newRequest(batchIdx, request, extractSamplingConfig(samplingConfig, batchIdx));
    }
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::forwardAsync(decoder::Output& output, decoder::Input const& input)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto const& logitsShape = input.logits->getShape();
    auto const batchSize = logitsShape.d[0];
    auto constexpr singleRequest = 1;
    std::vector<ITensor::SharedConstPtr> logits;
    logits.reserve(batchSize);
    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        auto logitsSlice = std::shared_ptr(ITensor::slice(input.logits, batchIdx, singleRequest));
        logits.emplace_back(
            ITensor::view(logitsSlice, ITensor::makeShape({singleRequest, mBeamWidths[batchIdx], logitsShape.d[2]})));
    }

    decoder_batch::Input batchInput{logits};
    batchInput.cacheIndirection = input.cacheIndirection;

    decoder_batch::Output batchOutput;
    batchOutput.cacheIndirection = output.cacheIndirection;
    batchOutput.sequenceLengths = output.sequenceLengths;

    mForwardToken = forwardAsync(batchOutput, batchInput);
    mBufferManager.setZero(*mFinishedSum);
    kernels::reduce(*mFinishedSum, *ITensor::slice(mJointDecodingOutput->finishedSum, 0, mActualBatchSize), *mStream);
    mStream->record(mForwardEvent);

    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::forwardSync()
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    forwardSync(*mForwardToken);
    mForwardEvent.synchronize();
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::finalize() const
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    for (SizeType batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        postProcessRequest(batchIdx);
    }
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

CudaEvent GptDecoderBatch::finalize(SizeType batchIdx) const
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto event = postProcessRequest(batchIdx);
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return event;
}
