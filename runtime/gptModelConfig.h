
#pragma once

#include "../common/quantization.h"
#include "common.h"
#include <NvInferRuntime.h>

namespace bitfusion::runtime
{

class GptModelConfig
{
public:
    enum class ModelVariant : std::int32_t
    {
        kGpt = 0,
        kGlm = 1,
    };

    constexpr explicit GptModelConfig(
        SizeType vocabSize, SizeType nbLayers, SizeType nbHeads, SizeType hiddenSize, nvinfer1::DataType dtype)
        : mVocabSize(vocabSize)
        , mNbLayers(nbLayers)
        , mNbHeads(nbHeads)
        , mNbKvHeads(nbHeads)
        , mHiddenSize(hiddenSize)
        , mDataType(dtype)
        , mUseGptAttentionPlugin(false)
        , mInputPacked{false}
        , mPagedKvCache{false}
        , mTokensPerBlock{64}
        , mQuantMode{common::QuantMode::none()}
        , mMaxBatchSize(0)
        , mMaxBeamWidth(0)
        , mMaxInputLen(0)
        , mMaxOutputLen(0)
        , mMaxNumTokens(std::nullopt)
        , mComputeContextLogits(false)
        , mComputeGenerationLogits(false)
        , mModelVariant(ModelVariant::kGpt)
        , mUseCustomAllReduce(false)
        , mMaxPromptEmbeddingTableSize(0)
        , mMaxDraftLen(0)
        , mUseContextFMHAForGeneration(false)
        , mPagedContextFMHA(false)
    {
    }

    [[nodiscard]] SizeType constexpr getVocabSize() const noexcept
    {
        return mVocabSize;
    }

    [[nodiscard]] SizeType constexpr getVocabSizePadded(SizeType worldSize) const noexcept
    {
        return (mVocabSize + worldSize - 1) / worldSize * worldSize;
    }

    [[nodiscard]] SizeType constexpr getNbLayers(SizeType pipelineParallelism = 1) const
    {
        CHECK(mNbLayers % pipelineParallelism == 0);
        return mNbLayers / pipelineParallelism;
    }

    [[nodiscard]] SizeType constexpr getNbHeads() const noexcept
    {
        return mNbHeads;
    }

    [[nodiscard]] SizeType constexpr getNbKvHeads() const noexcept
    {
        return mNbKvHeads;
    }

    void constexpr setNbKvHeads(SizeType nbKvHeads) noexcept
    {
        mNbKvHeads = nbKvHeads;
    }

    [[nodiscard]] SizeType constexpr getHiddenSize() const noexcept
    {
        return mHiddenSize;
    }

    [[nodiscard]] SizeType constexpr getSizePerHead() const noexcept
    {
        return mHiddenSize / mNbHeads;
    }

    [[nodiscard]] nvinfer1::DataType constexpr getDataType() const noexcept
    {
        return mDataType;
    }

    [[nodiscard]] bool constexpr useGptAttentionPlugin() const noexcept
    {
        return mUseGptAttentionPlugin;
    }

    void constexpr useGptAttentionPlugin(bool useGptAttentionPlugin) noexcept
    {
        mUseGptAttentionPlugin = useGptAttentionPlugin;
    }

    [[nodiscard]] bool constexpr usePackedInput() const noexcept
    {
        return mInputPacked;
    }

    void constexpr usePackedInput(bool inputPacked) noexcept
    {
        mInputPacked = inputPacked;
    }

    [[nodiscard]] bool constexpr usePagedKvCache() const noexcept
    {
        return mPagedKvCache;
    }

    void constexpr usePagedKvCache(bool pagedKvCache) noexcept
    {
        mPagedKvCache = pagedKvCache;
    }

    [[nodiscard]] SizeType constexpr getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    void constexpr setTokensPerBlock(SizeType TokensPerBlock) noexcept
    {
        mTokensPerBlock = TokensPerBlock;
    }

    [[nodiscard]] common::QuantMode constexpr getQuantMode() const noexcept
    {
        return mQuantMode;
    }

    void constexpr setQuantMode(common::QuantMode QuantMode) noexcept
    {
        mQuantMode = QuantMode;
    }

    [[nodiscard]] bool constexpr supportsInflightBatching() const noexcept
    {
        return mUseGptAttentionPlugin && mInputPacked && mPagedKvCache;
    }

    [[nodiscard]] SizeType constexpr getMaxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    }

    void constexpr setMaxBatchSize(SizeType maxBatchSize) noexcept
    {
        mMaxBatchSize = maxBatchSize;
    }

    [[nodiscard]] SizeType constexpr getMaxBeamWidth() const noexcept
    {
        return mMaxBeamWidth;
    }

    void constexpr setMaxBeamWidth(SizeType maxBeamWidth) noexcept
    {
        mMaxBeamWidth = maxBeamWidth;
    }

    [[nodiscard]] SizeType constexpr getMaxInputLen() const noexcept
    {
        return mMaxInputLen;
    }

    void constexpr setMaxInputLen(SizeType maxInputLen) noexcept
    {
        mMaxInputLen = maxInputLen;
    }

    [[nodiscard]] SizeType constexpr getMaxOutputLen() const noexcept
    {
        return mMaxOutputLen;
    }

    void constexpr setMaxOutputLen(SizeType maxOutputLen) noexcept
    {
        mMaxOutputLen = maxOutputLen;
    }

    [[nodiscard]] std::optional<SizeType> constexpr getMaxNumTokens() const noexcept
    {
        return mMaxNumTokens;
    }

    void constexpr setMaxNumTokens(std::optional<SizeType> maxNumTokens) noexcept
    {
        mMaxNumTokens = maxNumTokens;
    }

    [[nodiscard]] bool constexpr usePromptTuning() const noexcept
    {
        return mMaxPromptEmbeddingTableSize > 0;
    }

    [[nodiscard]] SizeType constexpr getMaxPromptEmbeddingTableSize() const noexcept
    {
        return mMaxPromptEmbeddingTableSize;
    }

    void constexpr setMaxPromptEmbeddingTableSize(SizeType maxPromptEmbeddingTableSize) noexcept
    {
        mMaxPromptEmbeddingTableSize = maxPromptEmbeddingTableSize;
    }

    [[nodiscard]] bool constexpr computeContextLogits() const noexcept
    {
        return mComputeContextLogits;
    }

    void constexpr computeContextLogits(bool computeContextLogits) noexcept
    {
        mComputeContextLogits = computeContextLogits;
    }

    [[nodiscard]] bool constexpr computeGenerationLogits() const noexcept
    {
        return mComputeGenerationLogits;
    }

    void constexpr computeGenerationLogits(bool computeGenerationLogits) noexcept
    {
        mComputeGenerationLogits = computeGenerationLogits;
    }

    [[nodiscard]] ModelVariant getModelVariant() const
    {
        return mModelVariant;
    }

    void setModelVariant(ModelVariant modelVariant)
    {
        mModelVariant = modelVariant;
    }

    [[nodiscard]] bool constexpr useCustomAllReduce() const noexcept
    {
        return mUseCustomAllReduce;
    }

    void constexpr useCustomAllReduce(bool customAllReduce) noexcept
    {
        mUseCustomAllReduce = customAllReduce;
    }

    void constexpr setMaxDraftLen(SizeType maxDraftLen) noexcept
    {
        mMaxDraftLen = maxDraftLen;
    }

    [[nodiscard]] SizeType getMaxDraftLen() const
    {
        return mMaxDraftLen;
    }

    [[nodiscard]] SizeType constexpr getMaxTokensPerStep() const noexcept
    {
        return mMaxDraftLen + 1;
    }

    void constexpr setUseContextFMHAForGeneration(bool useContextFMHAForGeneration) noexcept
    {
        mUseContextFMHAForGeneration = useContextFMHAForGeneration;
    }

    [[nodiscard]] bool constexpr getContextFMHAForGeneration() const noexcept
    {
        return mUseContextFMHAForGeneration;
    }

    void constexpr setPagedContextFMHA(bool pagedContextFMHA) noexcept
    {
        mPagedContextFMHA = pagedContextFMHA;
    }

    [[nodiscard]] bool constexpr getPagedContextFMHA() const noexcept
    {
        return mPagedContextFMHA;
    }

private:
    SizeType mVocabSize;
    SizeType mNbLayers;
    SizeType mNbHeads;
    SizeType mNbKvHeads;
    SizeType mHiddenSize;
    nvinfer1::DataType mDataType;
    bool mUseGptAttentionPlugin;
    bool mInputPacked;
    bool mPagedKvCache;
    SizeType mTokensPerBlock;
    common::QuantMode mQuantMode;
    SizeType mMaxBatchSize;
    SizeType mMaxBeamWidth;
    SizeType mMaxInputLen;
    SizeType mMaxOutputLen;
    std::optional<SizeType> mMaxNumTokens;

    bool mComputeContextLogits;
    bool mComputeGenerationLogits;
    ModelVariant mModelVariant;
    bool mUseCustomAllReduce;

    SizeType mMaxPromptEmbeddingTableSize;
    SizeType mMaxDraftLen;

    bool mUseContextFMHAForGeneration;
    bool mPagedContextFMHA;
};

}
