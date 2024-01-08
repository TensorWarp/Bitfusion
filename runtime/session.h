
#pragma once

#include "../batch_manager/kvCacheConfig.h"
#include "bufferManager.h"
#include "common.h"
#include "cudaEvent.h"
#include "generationInput.h"
#include "generationOutput.h"
#include "gptModelConfig.h"
#include "iTensor.h"
#include "samplingConfig.h"
#include "worldConfig.h"

#include <NvInferRuntime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace bitfusion::batch_manager
{
class TrtGptModelV1;
}

namespace bitfusion::batch_manager::kv_cache_manager
{
class KVCacheManager;
}

namespace bitfusion::runtime
{

namespace utils
{
std::vector<uint8_t> loadEngine(std::string const& enginePath);
}

class IpcMemory;
class IStatefulDecoder;
class NcclCommunicator;
class RuntimeBuffers;
class Runtime;

class Session
{
    using KvCacheManager = batch_manager::kv_cache_manager::KVCacheManager;
    using KvCacheConfig = batch_manager::kv_cache_manager::KvCacheConfig;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TokenGeneratedCallback = std::function<void(SizeType step, bool finished)>;

public:
    using LoggerPtr = std::shared_ptr<nvinfer1::ILogger>;

    class Config
    {
    public:
        Config(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength)
            : maxBatchSize{maxBatchSize}
            , maxBeamWidth{maxBeamWidth}
            , maxSequenceLength{maxSequenceLength}
        {
        }

        SizeType maxBatchSize;
        SizeType maxBeamWidth;
        SizeType maxSequenceLength;
        bool decoderPerRequest{false};
        bool cudaGraphMode{false};
        KvCacheConfig kvCacheConfig{};
        std::optional<SizeType> ctxMicroBatchSize = std::nullopt;
        std::optional<SizeType> genMicroBatchSize = std::nullopt;
    };

    Session(Config const& sessionConfig, GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
        void const* engineBuffer, std::size_t engineSize, LoggerPtr logger = nullptr);

    Session(Config const& sessionConfig, GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::vector<uint8_t> const& engineBuffer, LoggerPtr logger = nullptr)
        : Session(
            sessionConfig, modelConfig, worldConfig, engineBuffer.data(), engineBuffer.size(), std::move(logger))
    {
    }

    Session(Config const& sessionConfig, GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::string const& engineFile, LoggerPtr logger = nullptr)
        : Session(sessionConfig, modelConfig, worldConfig, utils::loadEngine(engineFile), std::move(logger))
    {
    }

    [[nodiscard]] nvinfer1::ILogger& getLogger() const;

    [[nodiscard]] BufferManager const& getBufferManager() const;

    [[nodiscard]] GptModelConfig const& getModelConfig() const
    {
        return mModelConfig;
    }

    [[nodiscard]] WorldConfig const& getWorldConfig() const
    {
        return mWorldConfig;
    }

    [[nodiscard]] int getDevice() const noexcept
    {
        return mDevice;
    }

    void generate(GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig);

private:
    [[nodiscard]] bool useCudaGraphs()
    {
        return !mCudaGraphInstances.empty();
    }

    void generateBatched(std::vector<GenerationOutput>& microBatchesOutputs,
        std::vector<GenerationInput> const& microBatchesInputs, SamplingConfig const& samplingConfig,
        TokenGeneratedCallback const& onTokenGenerated);

    void setup(Config const& sessionConfig);

    void createContexts(SizeType numBatchesCtx, SizeType numBatchesGen, bool useCudaGraphs);
    void createBuffers(SizeType numMicroBatches);
    void createDecoders(SizeType batchSize, SizeType beamWidth, SizeType maxAttentionWindow, SizeType maxSequenceLength,
        nvinfer1::DataType logitsType, bool decoderPerRequest, SizeType numMicroBatches);
    void createKvCacheManager(SizeType batchSize, SizeType beamWidth, SizeType maxAttentionWindow,
        SizeType maxSequenceLength, KvCacheConfig const& config);
    void createCustomAllReduceWorkspace(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength);

    void executeContextStep(std::vector<GenerationInput> const& microBatches,
        std::vector<SizeType> const& microBatchOffsets, KvCacheManager const* kvCacheManager);
    SizeType executeGenerationStep(SizeType step, std::vector<GenerationInput> const& microBatchesInputs,
        std::vector<GenerationOutput>& microBatchesOutputs, std::vector<SizeType> const& microBatchOffsets,
        KvCacheManager* kvCacheManager, std::vector<bool>& microBatchesFinished);

    void decoderStepAsync(SizeType decoderStep, SizeType microBatchId);

    bool shouldStopSync(SizeType batchSize, SizeType beamWidth, SizeType microBatchId);

    void finalize(SizeType microBatchId);

    void kvCacheAddSequences(SizeType beamWidth, SizeType microBatchId, SizeType firstBatchIdx);

    ITensor::SharedPtr initDecoder(ITensor& outputIds, GenerationInput const& inputs, GenerationOutput const& outputs,
        SamplingConfig const& samplingConfig, SizeType microBatchId) const;

    TokenGeneratedCallback createOnTokenGeneratedCallback(GenerationOutput& outputs);

    class CudaGraphExecutor
    {
    public:
        CudaGraphExecutor() = default;

        ~CudaGraphExecutor()
        {
            try
            {
                clear();
            }
            catch (std::exception& e)
            {
                LOG_EXCEPTION(e);
            }
        }

        bool hasInstance()
        {
            return mInstance != nullptr;
        }

        void clear();
        void prepareNextGraph(Runtime const& runtime, SizeType nextContextId);
        void launch(CudaStream const& stream);

    private:
        void create(cudaGraph_t const& graph);
        bool update(cudaGraph_t const& graph);
        void uploadToStream(CudaStream const& stream);

        cudaGraphExec_t mInstance;
    };

    class MicroBatchConfig
    {
    public:
        MicroBatchConfig()
            : numCtxBatches{1}
            , numGenBatches{1}
            , ctxBatchSize{0}
            , genBatchSize{0}
        {
        }

        explicit MicroBatchConfig(SizeType maxBatchSize, SizeType pipelineParallelism,
            std::optional<SizeType> genMicroBatchSize, std::optional<SizeType> ctxMicroBatchSize);

        constexpr SizeType numCtxPerGen() const
        {
            return numCtxBatches / numGenBatches;
        }

        constexpr SizeType getCtxContextId(SizeType generationBatchId, SizeType contextBatchId) const
        {
            return 2 * numGenBatches + generationBatchId * numCtxPerGen() + contextBatchId;
        }

        constexpr SizeType getGenContextId(SizeType flipFlopId, SizeType generationBatchId) const
        {
            return flipFlopId * numGenBatches + generationBatchId;
        }

        SizeType numCtxBatches;
        SizeType numGenBatches;
        SizeType ctxBatchSize;
        SizeType genBatchSize;
    };

    friend class batch_manager::TrtGptModelV1;

private:
    GptModelConfig const mModelConfig;
    WorldConfig const mWorldConfig;
    int mDevice{-1};
    std::shared_ptr<NcclCommunicator> mPipelineComm;
    std::shared_ptr<CudaStream> mCommStream;
    CudaEvent mCommEvent{};

    ITensor::SharedPtr mCommPtrs;
    std::vector<std::shared_ptr<IpcMemory>> mIpcMemoryHandles;

    SizeType mDecoderMaxSequenceLength{};
    SizeType mDecoderMaxAttentionWindow{};

    LoggerPtr mLogger;
    std::shared_ptr<Runtime> mRuntime;
    std::shared_ptr<KvCacheManager> mKvCacheManager;

    MicroBatchConfig mMicroBatchConfig;
    std::vector<std::shared_ptr<IStatefulDecoder>> mDecoders;
    std::vector<std::shared_ptr<RuntimeBuffers>> mBuffers;
    std::vector<CudaEvent> mReceivedEvents;

    bool mCudaGraphMode{false};
    std::vector<CudaGraphExecutor> mCudaGraphInstances;
};

}
