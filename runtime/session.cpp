
#include "Session.h"

#include "iBuffer.h"
#include "../batch_manager/kvCacheManager.h"
#include "../common/stringUtils.h"
#include "../kernels/decodingKernels.h"
#include "DecoderBatch.h"
#include "ipcUtils.h"
#include "ncclCommunicator.h"
#include "runtimeBuffers.h"
#include "runtimeKernels.h"
#include "statefulDecoder.h"
#include "logger.h"
#include "runtime.h"
#include "../runtime/utils/sessionUtils.h"

#include <algorithm>
#include <limits>
#include <memory>

using namespace bitfusion::runtime;

namespace tc = bitfusion::common;
namespace bmkv = bitfusion::batch_manager::kv_cache_manager;

Session::Session(Config const& sessionConfig, GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
    void const* engineBuffer, std::size_t engineSize, LoggerPtr logger)
    : mModelConfig{modelConfig}
    , mWorldConfig{worldConfig}
    , mDevice{utils::initDevice(worldConfig)}
    , mLogger{logger ? std::move(logger) : std::make_shared<Logger>()}
    , mRuntime{std::make_shared<Runtime>(engineBuffer, engineSize, *mLogger)}
    , mDecoders{}
    , mBuffers{}
    , mCudaGraphInstances{}
{
    if (mWorldConfig.isPipelineParallel())
    {
        mPipelineComm = NcclCommunicator::createPipelineComm(mWorldConfig);
        mCommStream = std::make_shared<CudaStream>();
    }

    CHECK_WITH_INFO(!(mModelConfig.usePromptTuning() && !mModelConfig.useGptAttentionPlugin()),
        "Prompt tuning is only enabled with GPT attention plugin.");


    setup(sessionConfig);
}

nvinfer1::ILogger& Session::getLogger() const
{
    return *mLogger;
}

BufferManager const& Session::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

void Session::createContexts(SizeType numCtxBatches, SizeType numGenBatches, bool useCudaGraphs)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    mRuntime->clearContexts();

    if (useCudaGraphs)
    {
        mCudaGraphInstances.resize(2 * numGenBatches);
    }

    auto const numProfiles = mRuntime->getNbProfiles();
    CHECK_WITH_INFO(
        numProfiles == 1 || numProfiles == 2, "GPT only expects one optimization profile or two optimization profiles");

    auto constexpr ctxContextId = 0;
    auto const genContextId = static_cast<std::int32_t>(numProfiles == 2);
    for (auto i = 0; i < 2 * numGenBatches; ++i)
        mRuntime->addContext(genContextId);
    for (auto i = 0; i < numCtxBatches; ++i)
        mRuntime->addContext(ctxContextId);

    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void Session::createBuffers(SizeType numMicroBatches)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    mBuffers.clear();

    for (SizeType i = 0; i < numMicroBatches; ++i)
    {
        mBuffers.emplace_back(std::make_shared<RuntimeBuffers>());
        mBuffers.back()->create(*mRuntime, mModelConfig, mWorldConfig);
    }
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void Session::createDecoders(SizeType batchSize, SizeType beamWidth, SizeType maxAttentionWindow,
    SizeType maxSequenceLength, nvinfer1::DataType logitsType, bool decoderPerRequest, SizeType numMicroBatches)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const vocabSize = mModelConfig.getVocabSize();
    auto const vocabSizePadded = mModelConfig.getVocabSizePadded(mWorldConfig.getSize());
    auto const& stream = mRuntime->getStreamPtr();

    mDecoders.clear();

    for (SizeType i = 0; i < numMicroBatches; ++i)
    {
        if (decoderPerRequest)
            mDecoders.emplace_back(std::make_shared<DecoderBatch>(vocabSize, vocabSizePadded, stream));
        else
            mDecoders.emplace_back(std::make_shared<StatefulDecoder>(vocabSize, vocabSizePadded, stream));
        constexpr SizeType maxTokensPerStep = 1;
        mDecoders.back()->setup(
            batchSize, beamWidth, maxAttentionWindow, maxSequenceLength, maxTokensPerStep, logitsType);
    }

    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void Session::createKvCacheManager(SizeType batchSize, SizeType beamWidth, SizeType maxAttentionWindow,
    SizeType maxSequenceLength, KvCacheConfig const& config)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const localNbLayers = mModelConfig.getNbLayers(mWorldConfig.getPipelineParallelism());
    auto const nbHeads = mModelConfig.getNbHeads();
    auto const nbKvHeads = mModelConfig.getNbKvHeads();
    auto const hiddenSize = mModelConfig.getHiddenSize();
    auto const tokensPerBlock = mModelConfig.getTokensPerBlock();

    nvinfer1::DataType kvDtype;
    if (mModelConfig.getQuantMode().hasFp8KvCache())
    {
        kvDtype = nvinfer1::DataType::kFP8;
    }
    else if (mModelConfig.getQuantMode().hasInt8KvCache())
    {
        kvDtype = nvinfer1::DataType::kINT8;
    }
    else
    {
        kvDtype = mModelConfig.getDataType();
    }

    auto const maxNumTokens
        = bmkv::KVCacheManager::getMaxNumTokens(config, kvDtype, mModelConfig, mWorldConfig, getBufferManager());
    LOG_INFO("Using %d tokens in paged KV cache.", maxNumTokens);
    auto const maxNumBlocks = tc::ceilDiv(maxNumTokens, tokensPerBlock);
    auto const maxBlocksPerSeq = tc::ceilDiv(std::min(maxSequenceLength, maxAttentionWindow), tokensPerBlock);

    mKvCacheManager
        = std::make_shared<bmkv::KVCacheManager>(localNbLayers, nbHeads, nbKvHeads, hiddenSize, tokensPerBlock,
            maxNumBlocks, batchSize, beamWidth, maxBlocksPerSeq, maxAttentionWindow, kvDtype, mRuntime->getStreamPtr());
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void Session::createCustomAllReduceWorkspace(
    SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength)
{
    setPeerAccess(mWorldConfig, true);

    mIpcMemoryHandles.clear();
    const std::size_t bufferSize = static_cast<std::size_t>(maxBatchSize) * maxBeamWidth * maxSequenceLength
        * mModelConfig.getHiddenSize() * mWorldConfig.getTensorParallelism() * sizeof(float);
    mIpcMemoryHandles.emplace_back(std::make_shared<IpcMemory>(mWorldConfig, bufferSize));
    mIpcMemoryHandles.emplace_back(std::make_shared<IpcMemory>(mWorldConfig, IpcMemory::FLAGS_SIZE * sizeof(int32_t)));
    mIpcMemoryHandles.emplace_back(std::make_shared<IpcMemory>(mWorldConfig, IpcMemory::FLAGS_SIZE * sizeof(int32_t)));

    auto& manager = mRuntime->getBufferManager();
    mCommPtrs = manager.cpu(
        ITensor::makeShape({static_cast<SizeType>(mIpcMemoryHandles.size()) * mWorldConfig.getTensorParallelism()}),
        nvinfer1::DataType::kINT64);
    const auto commPtrsData = bufferCast<void*>(*mCommPtrs);

    for (size_t memIdx = 0; memIdx < mIpcMemoryHandles.size(); memIdx++)
    {
        const auto& memCommPtrs = mIpcMemoryHandles[memIdx]->getCommPtrsTensor();
        for (SizeType tpIdx = 0; tpIdx < mWorldConfig.getTensorParallelism(); tpIdx++)
        {
            commPtrsData[memIdx * mWorldConfig.getTensorParallelism() + tpIdx] = memCommPtrs[tpIdx];
        }
    }
}

Session::MicroBatchConfig::MicroBatchConfig(SizeType maxBatchSize, SizeType pipelineParallelism,
    std::optional<SizeType> genMicroBatchSize, std::optional<SizeType> ctxMicroBatchSize)
{
    if (genMicroBatchSize || ctxMicroBatchSize)
    {
        genBatchSize = genMicroBatchSize.value_or(maxBatchSize);
        CHECK(genBatchSize <= maxBatchSize);
        ctxBatchSize = ctxMicroBatchSize.value_or(genBatchSize);
        CHECK_WITH_INFO(genBatchSize % ctxBatchSize == 0,
            "Generation batch size (%d) must be divisible by context batch size (%d)", genBatchSize, ctxBatchSize);
        numGenBatches = tc::ceilDiv(maxBatchSize, genBatchSize);
        numCtxBatches = numGenBatches * (genBatchSize / ctxBatchSize);
    }
    else
    {
        numCtxBatches = numGenBatches = pipelineParallelism;
        ctxBatchSize = genBatchSize = tc::ceilDiv(maxBatchSize, numGenBatches);
    }
}

void Session::setup(Config const& sessionConfig)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    mCudaGraphMode = sessionConfig.cudaGraphMode;

    auto const maxBatchSize = sessionConfig.maxBatchSize;
    auto const maxBeamWidth = sessionConfig.maxBeamWidth;
    auto const maxSequenceLength = sessionConfig.maxSequenceLength;
    auto const maxAttentionWindow = sessionConfig.kvCacheConfig.maxAttentionWindow.has_value()
        ? std::min(sessionConfig.kvCacheConfig.maxAttentionWindow.value(), maxSequenceLength)
        : maxSequenceLength;

    mMicroBatchConfig = MicroBatchConfig(maxBatchSize, mWorldConfig.getPipelineParallelism(),
        sessionConfig.genMicroBatchSize, sessionConfig.ctxMicroBatchSize);

    createContexts(mMicroBatchConfig.numCtxBatches, mMicroBatchConfig.numGenBatches, sessionConfig.cudaGraphMode);
    createBuffers(mMicroBatchConfig.numGenBatches);

    mDecoderMaxSequenceLength = maxSequenceLength;
    mDecoderMaxAttentionWindow = maxAttentionWindow;

    if (mModelConfig.usePagedKvCache())
    {
        createKvCacheManager(
            maxBatchSize, maxBeamWidth, maxAttentionWindow, maxSequenceLength, sessionConfig.kvCacheConfig);
    }

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto const logitsType = mRuntime->getEngine().getTensorDataType("logits");
        createDecoders(mMicroBatchConfig.genBatchSize, maxBeamWidth, maxAttentionWindow, maxSequenceLength, logitsType,
            sessionConfig.decoderPerRequest, mMicroBatchConfig.numGenBatches);
    }

    if (mWorldConfig.isPipelineParallel() || mMicroBatchConfig.numGenBatches > 1)
    {
        mReceivedEvents.clear();
        for (SizeType i = 0; i < mMicroBatchConfig.numGenBatches; ++i)
            mReceivedEvents.emplace_back();
    }

    if (mWorldConfig.isTensorParallel() && mModelConfig.useCustomAllReduce())
    {
        createCustomAllReduceWorkspace(mMicroBatchConfig.genBatchSize, maxBeamWidth, maxSequenceLength);
    }

    for (auto& buffers : mBuffers)
    {
        buffers->generationConfig = RuntimeBuffers::GenerationConfig{
            mMicroBatchConfig.genBatchSize, maxBeamWidth, 0, maxAttentionWindow, maxSequenceLength};
        buffers->reshape(mModelConfig, mWorldConfig);
    }
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void Session::kvCacheAddSequences(SizeType beamWidth, SizeType microBatchId, SizeType firstBatchIdx)
{
    if (mModelConfig.usePagedKvCache())
    {
        CHECK(mKvCacheManager);
        auto contextLengthsHost = mBuffers.at(microBatchId)->contextLengthsHost;
        CHECK(contextLengthsHost);
        auto const contextLengthsPtr = bufferCast<SizeType const>(*contextLengthsHost);
        auto const contextLengthsSize = static_cast<SizeType>(contextLengthsHost->getSize());
        for (SizeType batchIdx = 0; batchIdx < contextLengthsSize; ++batchIdx)
        {
            mKvCacheManager->addSequence(firstBatchIdx + batchIdx, contextLengthsPtr[batchIdx], beamWidth);
        }
    }
}

ITensor::SharedPtr Session::initDecoder(ITensor& outputIds, GenerationInput const& inputs,
    GenerationOutput const& outputs, SamplingConfig const& samplingConfig, SizeType microBatchId) const
{
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto& decoder = mDecoders.at(microBatchId);
        decoder->newBatch(inputs, outputs, samplingConfig);
        return decoder->getNewTokens();
    }
    else if (mWorldConfig.isFirstPipelineParallelRank())
    {
        auto& manager = mRuntime->getBufferManager();
        auto const& stream = mRuntime->getStreamPtr();

        auto const inputLengths = inputs.lengths;
        auto const batchSize = static_cast<SizeType>(inputLengths->getSize());

        auto const inputLengthsHost = manager.copyFrom(*inputLengths, MemoryType::kCPU);
        auto const* inputLengthsData = bufferCast<SizeType>(*inputLengthsHost);
        SizeType const maxInputLength = *std::max_element(inputLengthsData, inputLengthsData + inputLengths->getSize());

        ITensor::SharedPtr inputOffsets = manager.emptyTensor(MemoryType::kGPU, TRTDataType<SizeType>::value);
        if (inputs.packed)
        {
            inputOffsets->reshape(ITensor::makeShape({batchSize + 1}));
            manager.setZero(*inputOffsets);
            kernels::invokeInclusiveSum(*ITensor::slice(inputOffsets, 1), *inputLengths, manager, *stream);
        }

        kernels::initOutputIds(outputIds, *inputs.ids, *inputLengths, *inputOffsets, inputs.padId, inputs.endId,
            maxInputLength, inputs.packed, *stream);

        auto const beamWidth = samplingConfig.beamWidth;
        return manager.gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32);
    }
    else
    {
        return ITensor::SharedPtr{};
    }
}

namespace
{
std::tuple<std::vector<ITensor::SharedPtr>, std::vector<ITensor::SharedPtr>, std::vector<SizeType>> splitInputIds(
    GenerationInput const& inputs, SizeType microBatchSize, BufferManager& manager)
{
    auto const numRequests = inputs.lengths->getShape().d[0];

    std::vector<ITensor::SharedPtr> inputIds;
    std::vector<ITensor::SharedPtr> inputLengths;
    std::vector<SizeType> microBatchOffsets(1, 0);
    if (inputs.packed)
    {
        auto const contextLengthsHost = manager.copyFrom(*inputs.lengths, MemoryType::kCPU);
        ITensor::SharedPtr inputIdsView = ITensor::view(inputs.ids);
        inputIdsView->squeeze(0);
        auto const contextLengthsRange = BufferRange<SizeType>(*contextLengthsHost);

        auto tokensBegin = 0;
        for (auto offset = 0; offset < numRequests; offset += microBatchSize)
        {
            auto const batchSize = std::min(microBatchSize, numRequests - offset);
            auto const numTokens = std::accumulate(
                contextLengthsRange.begin() + offset, contextLengthsRange.begin() + offset + batchSize, 0);

            ITensor::SharedPtr batchInputs = ITensor::slice(inputIdsView, tokensBegin, numTokens);
            batchInputs->reshape(ITensor::makeShape({1, numTokens}));

            inputIds.emplace_back(std::move(batchInputs));
            inputLengths.emplace_back(ITensor::slice(inputs.lengths, offset, batchSize));
            microBatchOffsets.emplace_back(offset + batchSize);

            tokensBegin += numTokens;
        }
    }
    else
    {
        for (auto offset = 0; offset < numRequests; offset += microBatchSize)
        {
            auto const batchSize = std::min(microBatchSize, numRequests - offset);

            inputIds.emplace_back(ITensor::slice(inputs.ids, offset, batchSize));
            inputLengths.emplace_back(ITensor::slice(inputs.lengths, offset, batchSize));
            microBatchOffsets.emplace_back(offset + batchSize);
        }
    }

    return {inputIds, inputLengths, microBatchOffsets};
}

std::vector<GenerationInput> splitInputs(GenerationInput const& inputs, SizeType microBatchSize, BufferManager& manager)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto [inputIds, inputLengths, microBatchOffsets] = splitInputIds(inputs, microBatchSize, manager);

    std::vector<GenerationInput> inputBatches;
    for (std::size_t batchId = 0; batchId < inputIds.size(); ++batchId)
    {
        inputBatches.emplace_back(
            inputs.endId, inputs.padId, std::move(inputIds[batchId]), std::move(inputLengths[batchId]), inputs.packed);
    }

    for (std::size_t batchId = 0; batchId < inputBatches.size(); ++batchId)
    {
        auto& batch = inputBatches[batchId];
        auto const offset = microBatchOffsets[batchId];
        auto const batchSize = microBatchOffsets[batchId + 1] - offset;

        if (inputs.embeddingBias)
            batch.embeddingBias = inputs.embeddingBias;

        if (inputs.badWordsList)
        {
            auto const& shape = inputs.badWordsList->getShape();
            if (shape.nbDims == 2)
            {
                batch.badWordsList = inputs.badWordsList;
            }
            else
            {
                assert(shape.nbDims == 3);
                batch.badWordsList = ITensor::slice(inputs.badWordsList, offset, batchSize);
            }
        }
        if (inputs.stopWordsList)
        {
            batch.stopWordsList = ITensor::slice(inputs.stopWordsList, offset, batchSize);
        }
        if (inputs.maxNewTokens)
            batch.maxNewTokens = inputs.maxNewTokens;

        if (inputs.promptTuningParams.embeddingTable)
            batch.promptTuningParams.embeddingTable = inputs.promptTuningParams.embeddingTable;
        if (inputs.promptTuningParams.tasks)
            batch.promptTuningParams.tasks = ITensor::slice(inputs.promptTuningParams.tasks, offset, batchSize);
        if (inputs.promptTuningParams.vocabSize)
            batch.promptTuningParams.vocabSize = inputs.promptTuningParams.vocabSize;
    }

    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return inputBatches;
}

std::vector<GenerationOutput> splitOutputs(GenerationOutput& outputs, SizeType microBatchSize, BufferManager& manager)
{
    auto const numRequests = outputs.ids->getShape().d[0];

    std::vector<GenerationOutput> outputBatches;
    for (auto batchOffset = 0; batchOffset < numRequests; batchOffset += microBatchSize)
    {
        auto const batchSize = std::min(microBatchSize, numRequests - batchOffset);

        outputBatches.emplace_back(ITensor::slice(outputs.ids, batchOffset, batchSize),
            ITensor::slice(outputs.lengths, batchOffset, batchSize));

        if (outputs.cumLogProbs)
        {
            outputBatches.back().cumLogProbs = ITensor::slice(outputs.cumLogProbs, batchOffset, batchSize);
        }
        if (outputs.logProbs)
        {
            outputBatches.back().logProbs = ITensor::slice(outputs.logProbs, batchOffset, batchSize);
        }
        if (outputs.contextLogits)
        {
            outputBatches.back().contextLogits = ITensor::slice(outputs.contextLogits, batchOffset, batchSize);
        }
        if (outputs.generationLogits)
        {
            outputBatches.back().generationLogits = ITensor::slice(outputs.generationLogits, batchOffset, batchSize);
        }
    }

    return outputBatches;
}

void updateOutputIds(ITensor::SharedPtr const& outputIds, ITensor::SharedPtr const& newTokens, SizeType decoderStep,
    CudaStream const& stream)
{
    auto const& newTokensShape = newTokens->getShape();
    auto newTokensView = ITensor::view(newTokens, ITensor::makeShape({1, newTokensShape.d[0] * newTokensShape.d[1]}));
    auto const& outputIdsShape = outputIds->getShape();
    auto outputIdsView = ITensor::view(
        outputIds, ITensor::makeShape({outputIdsShape.d[0] * outputIdsShape.d[1], outputIdsShape.d[2]}));
    kernels::invokeTransposeWithOutputOffset(*outputIdsView, *newTokensView, decoderStep, stream);
    sync_check_cuda_error();
}
}

void Session::generate(
    GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    CHECK_WITH_INFO(inputs.packed == mModelConfig.usePackedInput(),
        "The chosen model requires a packed input tensor (did you set packed?).");
    auto const& inputLengths = inputs.lengths;
    CHECK_WITH_INFO(inputLengths->getShape().nbDims == 1, "Input lengths tensor must be one-dimensional.");

    auto& manager = mRuntime->getBufferManager();

    auto const batchSize = static_cast<SizeType>(inputLengths->getSize());
    auto const beamWidth = samplingConfig.beamWidth;
    outputs.ids->reshape(ITensor::makeShape({batchSize, beamWidth, mDecoderMaxSequenceLength}));
    outputs.lengths->reshape(ITensor::makeShape({batchSize, beamWidth}));
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        if (outputs.cumLogProbs)
        {
            CHECK_WITH_INFO(outputs.cumLogProbs,
                "outputs.cumLogProbs is nullptr. It must be allocated when computeLogProbs is true");
            outputs.cumLogProbs->reshape(ITensor::makeShape({batchSize, beamWidth}));
        }
        if (outputs.logProbs)
        {
            CHECK_WITH_INFO(
                outputs.logProbs, "outputs.logProbs is nullptr. It must be allocated when computeLogProbs is true");
            outputs.logProbs->reshape(ITensor::makeShape({batchSize, beamWidth, mDecoderMaxSequenceLength}));
        }
        if (mModelConfig.computeContextLogits() || mModelConfig.computeGenerationLogits())
        {
            CHECK_WITH_INFO(outputs.contextLogits,
                "outputs.contextLogits is nullptr. It must be allocated when computeContextLogits() is enabled.");
            auto const vocabSizePadded = mModelConfig.getVocabSizePadded(mWorldConfig.getSize());
            auto const inputLengthsHost = manager.copyFrom(*inputLengths, MemoryType::kCPU);
            auto const inputLengthsRange = BufferRange<SizeType>(*inputLengthsHost);
            auto const maxInputLength = *std::max_element(inputLengthsRange.begin(), inputLengthsRange.end());

            if (mModelConfig.computeContextLogits())
            {
                outputs.contextLogits->reshape(ITensor::makeShape({batchSize, maxInputLength, vocabSizePadded}));
            }

            if (mModelConfig.computeGenerationLogits())
            {
                SizeType maxNewTokens = 0;
                if (inputs.maxNewTokens)
                {
                    maxNewTokens = inputs.maxNewTokens.value();
                }
                else
                {
                    for (auto iter = inputLengthsRange.begin(); iter != inputLengthsRange.end(); iter++)
                    {
                        maxNewTokens = std::max(maxNewTokens, mDecoderMaxSequenceLength - *iter);
                    }
                }

                CHECK_WITH_INFO(maxNewTokens, "maxNewTokens is null");

                CHECK_WITH_INFO(outputs.generationLogits,
                    "outputs.generationLogits is nullptr. It must be allocated when computeGenerationLogits() is "
                    "enabled.");
                outputs.generationLogits->reshape(
                    ITensor::makeShape({batchSize, beamWidth, maxNewTokens - 1, vocabSizePadded}));
                auto const generationLogitsShape = outputs.generationLogits->getShape();
                CHECK_WITH_INFO(generationLogitsShape.d[0] == batchSize, "Invalid dim[0]");
                CHECK_WITH_INFO(generationLogitsShape.d[1] == beamWidth, "Invalid dim[1]");
                CHECK_WITH_INFO(generationLogitsShape.d[2] == maxNewTokens - 1, "Invalid dim[2]");
                CHECK_WITH_INFO(generationLogitsShape.d[3] == vocabSizePadded, "Invalid dim[3]");
            };
        }
    }

    auto const onTokenGenerated = createOnTokenGeneratedCallback(outputs);

    if (batchSize <= mMicroBatchConfig.genBatchSize)
    {
        std::vector<GenerationInput> microBatchesInputs{inputs};
        std::vector<GenerationOutput> microBatchesOutputs{outputs};
        generateBatched(microBatchesOutputs, microBatchesInputs, samplingConfig, onTokenGenerated);
    }
    else
    {
        auto const microBatchesInputs = splitInputs(inputs, mMicroBatchConfig.genBatchSize, manager);
        auto microBatchesOutputs = splitOutputs(outputs, mMicroBatchConfig.genBatchSize, manager);
        generateBatched(microBatchesOutputs, microBatchesInputs, samplingConfig, onTokenGenerated);
    }

    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

Session::TokenGeneratedCallback Session::createOnTokenGeneratedCallback(GenerationOutput& outputs)
{
    if (outputs.onTokenGenerated && mWorldConfig.isFirstPipelineParallelRank())
    {
        ITensor::SharedPtr outputIds{mWorldConfig.isPipelineParallel() || mMicroBatchConfig.numGenBatches > 1
                ? outputs.ids
                : mDecoders.front()->getOutputIds()};
        return [onTokenGenerated = outputs.onTokenGenerated, outputIds = std::move(outputIds)](
                   SizeType step, bool finished) { onTokenGenerated(outputIds, step, finished); };
    }
    else
    {
        return [](SizeType step, bool finished) {};
    }
}

void Session::generateBatched(std::vector<GenerationOutput>& microBatchesOutputs,
    std::vector<GenerationInput> const& microBatchesInputs, SamplingConfig const& samplingConfig,
    TokenGeneratedCallback const& onTokenGenerated)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto& manager = mRuntime->getBufferManager();
    CHECK(microBatchesInputs.size() == microBatchesOutputs.size());
    auto const numMicroBatches = static_cast<SizeType>(microBatchesInputs.size());
    CHECK(numMicroBatches > 0);
    CHECK(numMicroBatches <= mMicroBatchConfig.numGenBatches);
    SizeType const beamWidth{samplingConfig.beamWidth};

    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto const& microBatchInputs = microBatchesInputs.at(microBatchId);
        auto& buffers = *mBuffers.at(microBatchId);
        buffers.initFromInput(*microBatchInputs.ids, microBatchInputs.lengths, microBatchInputs.packed, beamWidth,
            mDecoderMaxAttentionWindow, mDecoderMaxSequenceLength, manager);
        buffers.reshape(mModelConfig, mWorldConfig);
        buffers.reset(manager);
    }

    std::vector<SizeType> microBatchOffsets(1, 0);
    microBatchOffsets.reserve(numMicroBatches + 1);
    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto const& generationConfig = mBuffers.at(microBatchId)->generationConfig;
        microBatchOffsets.emplace_back(microBatchOffsets.back() + generationConfig.batchSize);
    }

    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto& buffers = *mBuffers.at(microBatchId);
        auto const batchOffset = microBatchOffsets.at(microBatchId);
        kvCacheAddSequences(beamWidth, microBatchId, batchOffset);
        auto const& microBatchInputs = microBatchesInputs.at(microBatchId);
        auto& microBatchOutputs = microBatchesOutputs.at(microBatchId);
        buffers.outputIds = microBatchOutputs.ids;
        buffers.outputLengths = microBatchOutputs.lengths;
        buffers.newTokens
            = initDecoder(*buffers.outputIds, microBatchInputs, microBatchOutputs, samplingConfig, microBatchId);

        if (mWorldConfig.isLastPipelineParallelRank())
        {
            buffers.cumLogProbs = nullptr;
            if (microBatchOutputs.cumLogProbs)
            {
                buffers.cumLogProbs = microBatchOutputs.cumLogProbs;
            }
            buffers.logProbs = nullptr;
            if (microBatchOutputs.logProbs)
            {
                buffers.logProbs = microBatchOutputs.logProbs;
            }
            if (mModelConfig.computeContextLogits())
            {
                buffers.logits = microBatchOutputs.contextLogits;
            }
        }
        if (mModelConfig.usePromptTuning())
        {
            buffers.promptTuningParams = microBatchInputs.promptTuningParams;
        }
    }

    if (useCudaGraphs())
    {
        for (auto& instance : mCudaGraphInstances)
        {
            instance.clear();
        }
    }

    auto kvCacheManager = mModelConfig.usePagedKvCache() ? mKvCacheManager.get() : nullptr;

    executeContextStep(microBatchesInputs, microBatchOffsets, kvCacheManager);

    std::vector<bool> microBatchesFinished(numMicroBatches, false);
    SizeType numBatchesFinished{0};
    SizeType step{0};
    while (numBatchesFinished < numMicroBatches)
    {
        ++step;
        numBatchesFinished += executeGenerationStep(
            step, microBatchesInputs, microBatchesOutputs, microBatchOffsets, kvCacheManager, microBatchesFinished);

        onTokenGenerated(step - 1, numBatchesFinished == numMicroBatches);
    }

    for (auto microBatchId = 0; microBatchId < numMicroBatches; ++microBatchId)
    {
        auto const& generationConfig = mBuffers.at(microBatchId)->generationConfig;
        auto const microBatchSize = generationConfig.batchSize;

        auto const firstBatchIdx = microBatchOffsets.at(microBatchId);
        if (mModelConfig.usePagedKvCache())
        {
            for (auto batchIdx = firstBatchIdx; batchIdx < firstBatchIdx + microBatchSize; ++batchIdx)
            {
                kvCacheManager->removeSequence(batchIdx);
            }
        }

        if (beamWidth > 1)
        {
            finalize(microBatchId);
        }
        else if (!mWorldConfig.isPipelineParallel())
        {
            auto& buffers = *mBuffers.at(microBatchId);
            auto& decoder = *mDecoders.at(microBatchId);
            manager.copy(*decoder.getOutputIds(), *buffers.outputIds);

            auto& cumLogProbs = buffers.cumLogProbs;
            if (cumLogProbs)
                manager.copy(*decoder.getCumLogProbs(), *buffers.cumLogProbs);

            auto& logProbs = buffers.logProbs;
            if (logProbs)
                manager.copy(*decoder.getLogProbs(), *buffers.logProbs);
        }
    }

    manager.getStream().synchronize();
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void Session::executeContextStep(std::vector<GenerationInput> const& generationBatches,
    std::vector<SizeType> const& generationBatchOffsets, KvCacheManager const* kvCacheManager)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& manager = mRuntime->getBufferManager();

    auto const numGenerationBatches = static_cast<SizeType>(generationBatches.size());
    auto constexpr step = 0;
    for (auto generationBatchId = 0; generationBatchId < numGenerationBatches; ++generationBatchId)
    {
        auto const& generationBatchInputs = generationBatches.at(generationBatchId);
        auto& generationBuffers = *mBuffers.at(generationBatchId);

        auto const contextBatchSize = mMicroBatchConfig.ctxBatchSize;
        auto [inputIds, inputLengths, contextBatchOffsets]
            = splitInputIds(generationBatchInputs, contextBatchSize, manager);
        auto contextBuffers = generationBuffers.split(contextBatchSize, mModelConfig, mWorldConfig);
        CHECK(inputIds.size() == contextBuffers.size());
        auto const numContextBatches = static_cast<SizeType>(contextBuffers.size());

        for (auto contextBatchId = 0; contextBatchId < numContextBatches; ++contextBatchId)
        {
            auto batchOffset = generationBatchOffsets.at(generationBatchId) + contextBatchOffsets.at(contextBatchId);
            auto& buffers = contextBuffers.at(contextBatchId);
            auto& inputBuffer = buffers.inputBuffers[0];
            auto& outputBuffer = buffers.outputBuffers[0];

            auto const contextId = mMicroBatchConfig.getCtxContextId(generationBatchId, contextBatchId);

            buffers.prepareContextStep(inputIds.at(contextBatchId), generationBatchInputs.padId, manager,
                kvCacheManager, batchOffset, mModelConfig, mWorldConfig);
            buffers.getRuntimeBuffers(
                inputBuffer, outputBuffer, step, inputIds.at(contextBatchId), mCommPtrs, mModelConfig, mWorldConfig);
            mRuntime->setInputTensors(contextId, inputBuffer);
            mRuntime->setOutputTensors(contextId, outputBuffer);

            CHECK_WITH_INFO(mRuntime->executeContext(contextId), "Executing TRT engine in context step failed!");
            sync_check_cuda_error();
        }

        generationBuffers.postContextStep(contextBuffers, manager, mModelConfig, mWorldConfig);
        sync_check_cuda_error();

        std::swap(generationBuffers.cacheIndirectionDecoderInput, generationBuffers.cacheIndirectionDecoderOutput);

        auto const decoderStep = generationBuffers.generationConfig.maxInputLength + step;
        decoderStepAsync(decoderStep, generationBatchId);
    }
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

SizeType Session::executeGenerationStep(SizeType step, std::vector<GenerationInput> const& microBatchesInputs,
    std::vector<GenerationOutput>& microBatchesOutputs, std::vector<SizeType> const& microBatchOffsets,
    KvCacheManager* kvCacheManager, std::vector<bool>& microBatchesFinished)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    CHECK(microBatchesInputs.size() == microBatchesOutputs.size());
    auto& manager = mRuntime->getBufferManager();

    auto const numMicroBatches = static_cast<SizeType>(microBatchesInputs.size());
    SizeType numBatchesFinished{0};

    auto const flipFlopId = step % 2;
    for (auto generationBatchId = 0; generationBatchId < numMicroBatches; ++generationBatchId)
    {
        if (microBatchesFinished.at(generationBatchId))
            continue;

        auto& buffers = *mBuffers.at(generationBatchId);
        auto const& generationConfig = buffers.generationConfig;

        auto const contextId = mMicroBatchConfig.getGenContextId(flipFlopId, generationBatchId);
        auto& inputBuffer = buffers.inputBuffers[flipFlopId];
        auto& outputBuffer = buffers.outputBuffers[flipFlopId];

        auto nextInputIds = buffers.prepareNextStep(
            step - 1, manager, kvCacheManager, microBatchOffsets.at(generationBatchId), mModelConfig, mWorldConfig);
        buffers.getRuntimeBuffers(inputBuffer, outputBuffer, step, nextInputIds, mCommPtrs, mModelConfig, mWorldConfig);
        mRuntime->setInputTensors(contextId, inputBuffer);
        mRuntime->setOutputTensors(contextId, outputBuffer);

        if (useCudaGraphs())
        {
            mCudaGraphInstances.at(contextId).prepareNextGraph(*mRuntime, contextId);
        }

        if (shouldStopSync(generationConfig.batchSize, generationConfig.beamWidth, generationBatchId))
        {
            mLogger->log(nvinfer1::ILogger::Severity::kVERBOSE,
                tc::fmtstr("GPT decoding finished for step %d and microBatchId %d", step, generationBatchId).c_str());
            microBatchesFinished.at(generationBatchId) = true;
            numBatchesFinished += 1;
            continue;
        }

        if (useCudaGraphs())
        {
            auto& cudaGraphInstance = mCudaGraphInstances.at(contextId);
            CHECK(cudaGraphInstance.hasInstance());
            cudaGraphInstance.launch(mRuntime->getStream());
        }
        else
        {
            CHECK_WITH_INFO(
                mRuntime->executeContext(contextId), tc::fmtstr("Executing TRT engine in step %d failed!", step));
        }
        sync_check_cuda_error();

        if (mModelConfig.computeGenerationLogits())
        {
            auto& outputs = microBatchesOutputs.at(generationBatchId);
            auto const firstBatchSlotIdx = microBatchOffsets.at(generationBatchId);
            auto const microBatchSize = buffers.generationConfig.batchSize;
            auto const beamWidth = buffers.generationConfig.beamWidth;

            buffers.postEachGenerationStep(manager, outputs.generationLogits, step - 1, firstBatchSlotIdx,
                microBatchSize, beamWidth, mWorldConfig);
        }
        sync_check_cuda_error();

        std::swap(buffers.cacheIndirectionDecoderInput, buffers.cacheIndirectionDecoderOutput);

        auto const decoderStep = generationConfig.maxInputLength + step;
        decoderStepAsync(decoderStep, generationBatchId);
    }

    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return numBatchesFinished;
}

void Session::decoderStepAsync(SizeType decoderStep, SizeType microBatchId)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& stream = mRuntime->getStream();
    auto& buffers = *mBuffers.at(microBatchId);
    auto const& outputIds = buffers.outputIds;
    auto const& newTokens = buffers.newTokens;

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto& decoder = *mDecoders.at(microBatchId);

        decoder::Input decodingInput{buffers.logits};
        decoder::Output decodingOutput{};
        decodingInput.cacheIndirection = buffers.cacheIndirectionDecoderInput;
        decodingOutput.cacheIndirection = buffers.cacheIndirectionDecoderOutput;
        decodingOutput.sequenceLengths = buffers.sequenceLengths;

        decoder.forwardAsync(decodingOutput, decodingInput);
        if (mWorldConfig.isPipelineParallel())
        {
            stream.record(mCommEvent.get());
            mCommStream->wait(mCommEvent.get());
            auto const pipelineGroup = mWorldConfig.getPipelineParallelGroup();

            auto& cacheIndirection = *buffers.cacheIndirectionDecoderOutput;
            auto& sequenceLengths = *buffers.sequenceLengths;
            auto const beamWidth = cacheIndirection.getShape().d[1];
            for (auto peerIdx = 0; peerIdx < mWorldConfig.getPipelineParallelism() - 1; ++peerIdx)
            {
                mPipelineComm->send<SizeType>(*decoder.getNbFinished(), pipelineGroup[peerIdx], *mCommStream);
                if (beamWidth > 1)
                {
                    mPipelineComm->send<SizeType>(cacheIndirection, pipelineGroup[peerIdx], *mCommStream);
                }
                mPipelineComm->send<SizeType>(sequenceLengths, pipelineGroup[peerIdx], *mCommStream);
            }
            mPipelineComm->send<TokenIdType>(*decoder.getNewTokens(), pipelineGroup.front(), *mCommStream);
        }
    }
    else
    {
        stream.record(mCommEvent.get());
        mCommStream->wait(mCommEvent.get());
        auto const pipelineGroup = mWorldConfig.getPipelineParallelGroup();
        auto const peer = pipelineGroup.back();
        mPipelineComm->receive<SizeType>(*buffers.nbFinished, peer, *mCommStream);

        auto& cacheIndirection = *buffers.cacheIndirectionDecoderOutput;
        auto& sequenceLengths = *buffers.sequenceLengths;
        auto const beamWidth = cacheIndirection.getShape().d[1];
        if (beamWidth > 1)
        {
            mPipelineComm->receive<SizeType>(cacheIndirection, peer, *mCommStream);
        }
        mPipelineComm->receive<SizeType>(sequenceLengths, peer, *mCommStream);
        if (mWorldConfig.isFirstPipelineParallelRank())
        {
            mPipelineComm->receive<TokenIdType>(*newTokens, peer, *mCommStream);
            updateOutputIds(outputIds, newTokens, decoderStep, *mCommStream);
        }
        mCommStream->record(mReceivedEvents.at(microBatchId).get());
    }

    if (!mWorldConfig.isPipelineParallel() && mMicroBatchConfig.numGenBatches > 1)
    {
        updateOutputIds(outputIds, newTokens, decoderStep, stream);
        stream.record(mReceivedEvents.at(microBatchId).get());
    }

    sync_check_cuda_error();
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

bool Session::shouldStopSync(SizeType batchSize, SizeType beamWidth, SizeType microBatchId)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    SizeType nbFinished = 0;

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto& decoder = *mDecoders.at(microBatchId);
        decoder.forwardSync();
        nbFinished = *bufferCast<SizeType>(*decoder.getNbFinished());

        if (!mWorldConfig.isPipelineParallel() && mMicroBatchConfig.numGenBatches > 1)
        {
            mReceivedEvents.at(microBatchId).synchronize();
        }
    }
    else
    {
        mReceivedEvents.at(microBatchId).synchronize();
        nbFinished = *bufferCast<SizeType>(*mBuffers.at(microBatchId)->nbFinished);
    }
    sync_check_cuda_error();
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return nbFinished == batchSize * beamWidth;
}

void Session::finalize(SizeType microBatchId)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& manager = mRuntime->getBufferManager();
    auto& buffers = *mBuffers.at(microBatchId);
    auto& decoder = mDecoders.at(microBatchId);
    auto& outputIds = buffers.outputIds;
    auto& cumLogProbs = buffers.cumLogProbs;
    auto& logProbs = buffers.logProbs;
    auto& sequenceLengths = buffers.sequenceLengths;

    if (mWorldConfig.isPipelineParallel())
    {
        auto& stream = mRuntime->getStream();
        auto const pipelineGroup = mWorldConfig.getPipelineParallelGroup();

        if (mWorldConfig.isLastPipelineParallelRank())
        {
            auto const peer = pipelineGroup.front();
            decoder->finalize();
            auto finalOutputIds = decoder->getOutputIds();

            mPipelineComm->send<TokenIdType>(*finalOutputIds, peer, stream);
            mPipelineComm->send<SizeType>(*sequenceLengths, peer, stream);
            manager.copy(*finalOutputIds, *outputIds);

            if (cumLogProbs)
            {
                auto finalCumLogProbs = decoder->getCumLogProbs();
                mPipelineComm->send<float>(*finalCumLogProbs, peer, stream);
                manager.copy(*finalCumLogProbs, *cumLogProbs);
            }
            if (logProbs)
            {
                auto finalLogProbs = decoder->getLogProbs();
                mPipelineComm->send<float>(*finalLogProbs, peer, stream);
                manager.copy(*finalLogProbs, *logProbs);
            }
        }
        else if (mWorldConfig.isFirstPipelineParallelRank())
        {
            auto const peer = pipelineGroup.back();
            mPipelineComm->receive<TokenIdType>(*outputIds, peer, stream);
            mPipelineComm->receive<SizeType>(*sequenceLengths, peer, stream);
            if (cumLogProbs)
            {
                mPipelineComm->receive<float>(*cumLogProbs, peer, stream);
            }
            if (logProbs)
            {
                mPipelineComm->receive<float>(*logProbs, peer, stream);
            }
        }
    }
    else
    {
        decoder->finalize();
        auto finalOutputIds = decoder->getOutputIds();
        manager.copy(*finalOutputIds, *outputIds);
        if (cumLogProbs)
        {
            auto finalCumLogProbs = decoder->getCumLogProbs();
            manager.copy(*finalCumLogProbs, *cumLogProbs);
        }
        if (logProbs)
        {
            auto finalLogProbs = decoder->getLogProbs();
            manager.copy(*finalLogProbs, *logProbs);
        }
    }

    sync_check_cuda_error();
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void Session::CudaGraphExecutor::create(cudaGraph_t const& graph)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    assert(mInstance == nullptr);
    CUDA_CHECK(cudaGraphInstantiate(&mInstance, graph, nullptr, nullptr, 0));
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void Session::CudaGraphExecutor::uploadToStream(CudaStream const& stream)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    assert(hasInstance());
    CUDA_CHECK(cudaGraphUpload(mInstance, stream.get()));
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void Session::CudaGraphExecutor::launch(CudaStream const& stream)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    CUDA_CHECK(cudaGraphLaunch(mInstance, stream.get()));
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

bool Session::CudaGraphExecutor::update(cudaGraph_t const& graph)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    return cudaGraphExecUpdate(mInstance, graph, nullptr) != cudaSuccess;
}

void Session::CudaGraphExecutor::clear()
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    if (mInstance != nullptr)
    {
        CUDA_CHECK(cudaGraphExecDestroy(mInstance));
        mInstance = nullptr;
    }
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void Session::CudaGraphExecutor::prepareNextGraph(Runtime const& runtime, SizeType nextContextId)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& stream = runtime.getStream();

    cudaGraph_t nextGraph;
    CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    runtime.executeContext(nextContextId);
    CUDA_CHECK(cudaStreamEndCapture(stream.get(), &nextGraph));

    if (hasInstance())
    {
        if (update(nextGraph))
        {
            clear();
            create(nextGraph);
        }
    }
    else
    {
        create(nextGraph);
    }

    CUDA_CHECK(cudaGraphDestroy(nextGraph));
    uploadToStream(stream);
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}
