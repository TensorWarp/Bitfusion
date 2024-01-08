
#pragma once

#include "bufferManager.h"
#include "gptModelConfig.h"
#include "iTensor.h"
#include "promptTuningParams.h"
#include "worldConfig.h"

#include <array>
#include <vector>

namespace bitfusion::batch_manager::kv_cache_manager
{
class KVCacheManager;
}

namespace bitfusion::runtime
{
class Runtime;

class RuntimeBuffers
{
protected:
    using TensorPtr = ITensor::SharedPtr;
    using KvCacheManager = batch_manager::kv_cache_manager::KVCacheManager;

public:
    using TensorMap = StringPtrMap<ITensor>;

    class GenerationConfig
    {
    public:
        GenerationConfig() = default;

        explicit GenerationConfig(SizeType batchSize, SizeType beamWidth, SizeType maxInputLength,
            SizeType maxAttentionWindow, SizeType maxSeqLength, SizeType inputLengthSum = SizeType(0))
            : batchSize{batchSize}
            , beamWidth{beamWidth}
            , maxInputLength{maxInputLength}
            , maxAttentionWindow{maxAttentionWindow}
            , maxSeqLength{maxSeqLength}
            , inputLengthSum{inputLengthSum}
        {
        }

        SizeType batchSize{};
        SizeType beamWidth{};
        SizeType maxInputLength{};
        SizeType maxAttentionWindow{};
        SizeType maxSeqLength{};
        SizeType inputLengthSum{};

        static GenerationConfig fromInput(ITensor const& inputIds, ITensor const& inputLengths, bool inputPacked,
            SizeType beamWidth, SizeType maxAttentionWindow, SizeType maxSequenceLength);
    };

public:
    GenerationConfig generationConfig{};
    std::array<TensorMap, 2> inputBuffers{};
    std::array<TensorMap, 2> outputBuffers{};

    TensorPtr contextLengthsHost;
    TensorPtr contextLengthsDevice;

    TensorPtr logits;
    TensorPtr sequenceLengths;
    TensorPtr pastKeyValueLengths;
    TensorPtr attentionMask;
    TensorPtr positionIds;
    TensorPtr lastTokenIds;
    TensorPtr requestTypes;

    std::vector<TensorPtr> presentKeysVals;
    std::vector<TensorPtr> presentKeysValsAlt;
    std::vector<TensorPtr> maxAttentionWindows;
    TensorPtr kvCacheBlockPointersHost;
    TensorPtr kvCacheBlockPointersDevice;

    TensorPtr newTokens;
    TensorPtr outputIds;
    TensorPtr outputLengths;

    TensorPtr cacheIndirectionDecoderInput;
    TensorPtr cacheIndirectionDecoderOutput;

    TensorPtr nbFinished;

    TensorPtr cumLogProbs;
    TensorPtr logProbs;

    TensorPtr hiddenStates;

    PromptTuningParams promptTuningParams;
    TensorPtr promptTuningTasksHost;

    bool allocated{false};

public:
    void clear();
    void clearTensorMaps();

    void create(Runtime& runtime, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void initFromInput(ITensor const& inputIds, TensorPtr const& inputLengths, bool inputPacked, SizeType beamWidth,
        SizeType maxAttentionWindow, SizeType maxSequenceLength, BufferManager& manager);

    void reshape(GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void reset(BufferManager& manager);

    std::vector<RuntimeBuffers> split(
        SizeType contextBatchSize, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void postContextStep(std::vector<RuntimeBuffers> const& contextBuffers, BufferManager& manager,
        GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void postEachGenerationStep(BufferManager& manager, TensorPtr outputGenerationLogits, SizeType step,
        SizeType firstBatchSlotIdx, SizeType microBatchSize, SizeType beamWidth, WorldConfig const& worldConfig);

    void prepareContextStep(TensorPtr const& inputIds, TokenIdType padId, BufferManager& manager,
        KvCacheManager const* kvCacheManager, SizeType firstBatchSlotIdx, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig);
    TensorPtr prepareNextStep(SizeType step, BufferManager& manager, KvCacheManager* kvCacheManager,
        SizeType firstBatchSlotIdx, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void getRuntimeBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType const step,
        TensorPtr const& inputIds, TensorPtr const& commPtrs, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig) const;

private:
    void gatherLastTokenLogits(
        BufferManager& manager, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void copyAttentionMasks(std::vector<RuntimeBuffers> const& contextBatches, BufferManager& manager);

    void tile(BufferManager& manager, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    static std::vector<SizeType> getPositionIdsContextPhaseGlm(const SizeType& batchSize,
        const SizeType& maxInputLength, const SizeType* pInputLengths, const bool useGptAttentionPlugin,
        const bool usePackedInput);

    static std::vector<SizeType> getPositionIdsGenerationPhaseGlm(const SizeType& batchSize, const SizeType& beamSize,
        const SizeType& step, const SizeType* pInputLengths, const bool useGptAttentionPlugin,
        const bool usePackedInput);
};

}
