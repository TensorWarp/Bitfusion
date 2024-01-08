#pragma once

#include "../common/quantization.h"
#include "common.h"
#include <NvInferRuntime.h>

namespace bitfusion::runtime
{

    class ModelConfig
    {
    public:
        enum class ModelVariant : std::int32_t
        {
            kGpt = 0,
            kGlm = 1,
        };

        constexpr explicit ModelConfig(
            SizeType vocabSize, SizeType nbLayers, SizeType nbHeads, SizeType hiddenSize, nvinfer1::DataType dtype)

            /// <summary>
            /// The vocabulary size of the model.
            /// </summary>
            : mVocabSize(vocabSize)

            /// <summary>
            /// The number of layers in the model.
            /// </summary>
            , mNbLayers(nbLayers)

            /// <summary>
            /// The number of attention heads in the model.
            /// </summary>
            , mNbHeads(nbHeads)

            /// <summary>
            /// The number of key and value attention heads in the model.
            /// </summary>
            , mNbKvHeads(nbHeads)
            
            /// <summary>
            /// The hidden size of the model.
            /// </summary>
            , mHiddenSize(hiddenSize)
            
            /// <summary>
            /// The data type used by the model.
            /// </summary>
            , mDataType(dtype)
            
            /// <summary>
            /// Flag indicating whether the attention plugin is used.
            /// </summary>
            , mUseGptAttentionPlugin(false)
            
            /// <summary>
            /// Flag indicating whether input is packed.
            /// </summary>
            , mInputPacked{ false }
            
            /// <summary>
            /// Flag indicating whether paged key-value cache is used.
            /// </summary>
            , mPagedKvCache{ false }
            
            /// <summary>
            /// The number of tokens per block.
            /// </summary>
            , mTokensPerBlock{ 64 }
            
            /// <summary>
            /// The quantization mode used by the model.
            /// </summary>
            , mQuantMode{ common::QuantMode::none() }
            
            /// <summary>
            /// The maximum batch size for the model.
            /// </summary>
            , mMaxBatchSize(0)
            
            /// <summary>
            /// The maximum beam width for the model.
            /// </summary>
            , mMaxBeamWidth(0)
            
            /// <summary>
            /// The maximum input length for the model.
            /// </summary>
            , mMaxInputLen(0)
            
            /// <summary>
            /// The maximum output length for the model.
            /// </summary>
            , mMaxOutputLen(0)
            
            /// <summary>
            /// The maximum number of tokens.
            /// </summary>
            , mMaxNumTokens(std::nullopt)
            
            /// <summary>
            /// Flag indicating whether to compute context logits.
            /// </summary>
            , mComputeContextLogits(false)
            
            /// <summary>
            /// Flag indicating whether to compute generation logits.
            /// </summary>
            , mComputeGenerationLogits(false)
            
            /// <summary>
            /// The model variant.
            /// </summary>
            , mModelVariant(ModelVariant::kGpt)
            
            /// <summary>
            /// Flag indicating whether custom all-reduce is used.
            /// </summary>
            , mUseCustomAllReduce(false)
            
            /// <summary>
            /// The maximum size of the prompt embedding table.
            /// </summary>
            , mMaxPromptEmbeddingTableSize(0)
            
            /// <summary>
            /// The maximum draft length for the model.
            /// </summary>  
            , mMaxDraftLen(0)
            
            /// <summary>
            /// Flag indicating whether context FMHA is used for generation.
            /// </summary> 
            , mUseContextFMHAForGeneration(false)
            
            /// <summary>
            /// Flag indicating whether paged context FMHA is used.
            /// </summary>   
            , mPagedContextFMHA(false)
        {
        }

        /// <summary>
        /// Get the vocabulary size of the model.
        /// </summary>
        /// <returns>The vocabulary size.</returns>
        [[nodiscard]] SizeType constexpr getVocabSize() const noexcept
        {
            return mVocabSize;
        }

        /// <summary>
        /// Get the padded vocabulary size based on the world size.
        /// </summary>
        /// <param name="worldSize">The world size for padding.</param>
        /// <returns>The padded vocabulary size.</returns>
        [[nodiscard]] SizeType constexpr getVocabSizePadded(SizeType worldSize) const noexcept
        {
            return (mVocabSize + worldSize - 1) / worldSize * worldSize;
        }

        /// <summary>
        /// Get the number of layers in the model, considering pipeline parallelism.
        /// </summary>
        /// <param name="pipelineParallelism">The pipeline parallelism factor.</param>
        /// <returns>The number of layers.</returns>
        [[nodiscard]] SizeType constexpr getNbLayers(SizeType pipelineParallelism = 1) const
        {
            CHECK(mNbLayers % pipelineParallelism == 0);
            return mNbLayers / pipelineParallelism;
        }

        /// <summary>
        /// Get the number of attention heads in the model.
        /// </summary>
        /// <returns>The number of attention heads.</returns>
        [[nodiscard]] SizeType constexpr getNbHeads() const noexcept
        {
            return mNbHeads;
        }

        /// <summary>
        /// Get the number of key and value attention heads in the model.
        /// </summary>
        /// <returns>The number of key and value attention heads.</returns>
        [[nodiscard]] SizeType constexpr getNbKvHeads() const noexcept
        {
            return mNbKvHeads;
        }

        /// <summary>
        /// Set the number of key and value attention heads in the model.
        /// </summary>
        /// <param name="nbKvHeads">The new number of key and value attention heads.</param>
        void constexpr setNbKvHeads(SizeType nbKvHeads) noexcept
        {
            mNbKvHeads = nbKvHeads;
        }


        /// <summary>
       /// Get the hidden size of the model.
       /// </summary>
       /// <returns>The hidden size.</returns>
        [[nodiscard]] SizeType constexpr getHiddenSize() const noexcept
        {
            return mHiddenSize;
        }

        /// <summary>
        /// Get the size per attention head.
        /// </summary>
        /// <returns>The size per attention head.</returns>
        [[nodiscard]] SizeType constexpr getSizePerHead() const noexcept
        {
            return mHiddenSize / mNbHeads;
        }

        /// <summary>
        /// Get the data type used by the model.
        /// </summary>
        /// <returns>The data type.</returns>
        [[nodiscard]] nvinfer1::DataType constexpr getDataType() const noexcept
        {
            return mDataType;
        }

        /// <summary>
        /// Check if the GPT attention plugin is used.
        /// </summary>
        /// <returns>True if the GPT attention plugin is used, false otherwise.</returns>
        [[nodiscard]] bool constexpr useGptAttentionPlugin() const noexcept
        {
            return mUseGptAttentionPlugin;
        }

        /// <summary>
        /// Set whether to use the GPT attention plugin.
        /// </summary>
        /// <param name="useGptAttentionPlugin">True to use the GPT attention plugin, false otherwise.</param>
        void constexpr useGptAttentionPlugin(bool useGptAttentionPlugin) noexcept
        {
            mUseGptAttentionPlugin = useGptAttentionPlugin;
        }

        /// <summary>
        /// Check if input is packed.
        /// </summary>
        /// <returns>True if input is packed, false otherwise.</returns>
        [[nodiscard]] bool constexpr usePackedInput() const noexcept
        {
            return mInputPacked;
        }

        /// <summary>
        /// Set whether input is packed.
        /// </summary>
        /// <param name="inputPacked">True to use packed input, false otherwise.</param>
        void constexpr usePackedInput(bool inputPacked) noexcept
        {
            mInputPacked = inputPacked;
        }

        /// <summary>
        /// Check if paged key-value cache is used.
        /// </summary>
        /// <returns>True if paged key-value cache is used, false otherwise.</returns>
        [[nodiscard]] bool constexpr usePagedKvCache() const noexcept
        {
            return mPagedKvCache;
        }

        /// <summary>
        /// Set whether to use paged key-value cache.
        /// </summary>
        /// <param name="pagedKvCache">True to use paged key-value cache, false otherwise.</param>
        void constexpr usePagedKvCache(bool pagedKvCache) noexcept
        {
            mPagedKvCache = pagedKvCache;
        }

        /// <summary>
        /// Get the number of tokens per block.
        /// </summary>
        /// <returns>The number of tokens per block.</returns>
        [[nodiscard]] SizeType constexpr getTokensPerBlock() const noexcept
        {
            return mTokensPerBlock;
        }

        /// <summary>
        /// Set the number of tokens per block.
        /// </summary>
        /// <param name="TokensPerBlock">The new number of tokens per block.</param>
        void constexpr setTokensPerBlock(SizeType TokensPerBlock) noexcept
        {
            mTokensPerBlock = TokensPerBlock;
        }

        /// <summary>
        /// Get the quantization mode used by the model.
        /// </summary>
        /// <returns>The quantization mode.</returns>
        [[nodiscard]] common::QuantMode constexpr getQuantMode() const noexcept
        {
            return mQuantMode;
        }

        /// <summary>
        /// Set the quantization mode used by the model.
        /// </summary>
        /// <param name="QuantMode">The new quantization mode.</param>
        void constexpr setQuantMode(common::QuantMode QuantMode) noexcept
        {
            mQuantMode = QuantMode;
        }

        /// <summary>
        /// Check if the model supports inflight batching.
        /// </summary>
        /// <returns>True if the model supports inflight batching, false otherwise.</returns>
        [[nodiscard]] bool constexpr supportsInflightBatching() const noexcept
        {
            return mUseGptAttentionPlugin && mInputPacked && mPagedKvCache;
        }

        /// <summary>
        /// Get the maximum batch size.
        /// </summary>
        /// <returns>The maximum batch size.</returns>
        [[nodiscard]] SizeType constexpr getMaxBatchSize() const noexcept
        {
            return mMaxBatchSize;
        }

        /// <summary>
        /// Set the maximum batch size.
        /// </summary>
        /// <param name="maxBatchSize">The new maximum batch size to set.</param>
        void constexpr setMaxBatchSize(SizeType maxBatchSize) noexcept
        {
            mMaxBatchSize = maxBatchSize;
        }

        /// <summary>
        /// Get the maximum beam width.
        /// </summary>
        /// <returns>The maximum beam width.</returns>
        [[nodiscard]] SizeType constexpr getMaxBeamWidth() const noexcept
        {
            return mMaxBeamWidth;
        }

        /// <summary>
        /// Set the maximum beam width.
        /// </summary>
        /// <param name="maxBeamWidth">The new maximum beam width to set.</param>
        void constexpr setMaxBeamWidth(SizeType maxBeamWidth) noexcept
        {
            mMaxBeamWidth = maxBeamWidth;
        }

        /// <summary>
        /// Get the maximum input length.
        /// </summary>
        /// <returns>The maximum input length.</returns>
        [[nodiscard]] SizeType constexpr getMaxInputLen() const noexcept
        {
            return mMaxInputLen;
        }

        /// <summary>
        /// Set the maximum input length.
        /// </summary>
        /// <param name="maxInputLen">The new maximum input length to set.</param>
        void constexpr setMaxInputLen(SizeType maxInputLen) noexcept
        {
            mMaxInputLen = maxInputLen;
        }

        /// <summary>
        /// Get the maximum output length.
        /// </summary>
        /// <returns>The maximum output length.</returns>
        [[nodiscard]] SizeType constexpr getMaxOutputLen() const noexcept
        {
            return mMaxOutputLen;
        }

        /// <summary>
        /// Set the maximum output length.
        /// </summary>
        /// <param name="maxOutputLen">The new maximum output length to set.</param>
        void constexpr setMaxOutputLen(SizeType maxOutputLen) noexcept
        {
            mMaxOutputLen = maxOutputLen;
        }

        /// <summary>
        /// Get the maximum number of tokens.
        /// </summary>
        /// <returns>The maximum number of tokens.</returns>
        [[nodiscard]] std::optional<SizeType> constexpr getMaxNumTokens() const noexcept
        {
            return mMaxNumTokens;
        }

        /// <summary>
        /// Set the maximum number of tokens.
        /// </summary>
        /// <param name="maxNumTokens">The new maximum number of tokens to set.</param>
        void constexpr setMaxNumTokens(std::optional<SizeType> maxNumTokens) noexcept
        {
            mMaxNumTokens = maxNumTokens;
        }

        /// <summary>
        /// Check if prompt tuning is enabled.
        /// </summary>
        /// <returns>True if prompt tuning is enabled, false otherwise.</returns>
        [[nodiscard]] bool constexpr usePromptTuning() const noexcept
        {
            return mMaxPromptEmbeddingTableSize > 0;
        }

        /// <summary>
        /// Get the maximum prompt embedding table size.
        /// </summary>
        /// <returns>The maximum prompt embedding table size.</returns>
        [[nodiscard]] SizeType constexpr getMaxPromptEmbeddingTableSize() const noexcept
        {
            return mMaxPromptEmbeddingTableSize;
        }

        /// <summary>
        /// Set the maximum prompt embedding table size.
        /// </summary>
        /// <param name="maxPromptEmbeddingTableSize">The new maximum prompt embedding table size to set.</param>
        void constexpr setMaxPromptEmbeddingTableSize(SizeType maxPromptEmbeddingTableSize) noexcept
        {
            mMaxPromptEmbeddingTableSize = maxPromptEmbeddingTableSize;
        }

        /// <summary>
        /// Check if context logits computation is enabled.
        /// </summary>
        /// <returns>True if context logits computation is enabled, false otherwise.</returns>
        [[nodiscard]] bool constexpr computeContextLogits() const noexcept
        {
            return mComputeContextLogits;
        }

        /// <summary>
        /// Set whether to compute context logits.
        /// </summary>
        /// <param name="computeContextLogits">True to compute context logits, false otherwise.</param>
        void constexpr computeContextLogits(bool computeContextLogits) noexcept
        {
            mComputeContextLogits = computeContextLogits;
        }

        /// <summary>
        /// Check if generation logits computation is enabled.
        /// </summary>
        /// <returns>True if generation logits computation is enabled, false otherwise.</returns>
        [[nodiscard]] bool constexpr computeGenerationLogits() const noexcept
        {
            return mComputeGenerationLogits;
        }

        /// <summary>
        /// Set whether to compute generation logits.
        /// </summary>
        /// <param name="computeGenerationLogits">True to compute generation logits, false otherwise.</param>
        void constexpr computeGenerationLogits(bool computeGenerationLogits) noexcept
        {
            mComputeGenerationLogits = computeGenerationLogits;
        }

        /// <summary>
        /// Get the model variant.
        /// </summary>
        /// <returns>The model variant.</returns>
        [[nodiscard]] ModelVariant getModelVariant() const
        {
            return mModelVariant;
        }

        /// <summary>
        /// Set the model variant.
        /// </summary>
        /// <param name="modelVariant">The new model variant to set.</param>
        void setModelVariant(ModelVariant modelVariant)
        {
            mModelVariant = modelVariant;
        }

        /// <summary>
        /// Check if custom all-reduce is used.
        /// </summary>
        /// <returns>True if custom all-reduce is used, false otherwise.</returns>
        [[nodiscard]] bool constexpr useCustomAllReduce() const noexcept
        {
            return mUseCustomAllReduce;
        }

        /// <summary>
        /// Set whether to use custom all-reduce.
        /// </summary>
        /// <param name="customAllReduce">True to use custom all-reduce, false otherwise.</param>
        void constexpr useCustomAllReduce(bool customAllReduce) noexcept
        {
            mUseCustomAllReduce = customAllReduce;
        }

        /// <summary>
        /// Set the maximum draft length.
        /// </summary>
        /// <param name="maxDraftLen">The new maximum draft length to set.</param>
        void constexpr setMaxDraftLen(SizeType maxDraftLen) noexcept
        {
            mMaxDraftLen = maxDraftLen;
        }

        /// <summary>
        /// Get the maximum draft length.
        /// </summary>
        /// <returns>The maximum draft length.</returns>
        [[nodiscard]] SizeType getMaxDraftLen() const
        {
            return mMaxDraftLen;
        }

        /// <summary>
        /// Get the maximum number of tokens per step, which is one more than the maximum draft length.
        /// </summary>
        /// <returns>The maximum number of tokens per step.</returns>
        [[nodiscard]] SizeType constexpr getMaxTokensPerStep() const noexcept
        {
            return mMaxDraftLen + 1;
        }

        /// <summary>
        /// Set whether to use context FMHA for generation.
        /// </summary>
        /// <param name="useContextFMHAForGeneration">True to use context FMHA for generation, false otherwise.</param>
        void constexpr setUseContextFMHAForGeneration(bool useContextFMHAForGeneration) noexcept
        {
            mUseContextFMHAForGeneration = useContextFMHAForGeneration;
        }

        /// <summary>
        /// Check if context FMHA is used for generation.
        /// </summary>
        /// <returns>True if context FMHA is used for generation, false otherwise.</returns>
        [[nodiscard]] bool constexpr getContextFMHAForGeneration() const noexcept
        {
            return mUseContextFMHAForGeneration;
        }

        /// <summary>
        /// Set whether to use paged context FMHA.
        /// </summary>
        /// <param name="pagedContextFMHA">True to use paged context FMHA, false otherwise.</param>
        void constexpr setPagedContextFMHA(bool pagedContextFMHA) noexcept
        {
            mPagedContextFMHA = pagedContextFMHA;
        }

        /// <summary>
        /// Check if paged context FMHA is used.
        /// </summary>
        /// <returns>True if paged context FMHA is used, false otherwise.</returns>
        [[nodiscard]] bool constexpr getPagedContextFMHA() const noexcept
        {
            return mPagedContextFMHA;
        }

    private:
        /// <summary>
        /// The vocabulary size of the model.
        /// </summary>
        SizeType mVocabSize;

        /// <summary>
        /// The number of layers in the model.
        /// </summary>
        SizeType mNbLayers;

        /// <summary>
        /// The number of attention heads in the model.
        /// </summary>
        SizeType mNbHeads;

        /// <summary>
        /// The number of key and value attention heads in the model.
        /// </summary>
        SizeType mNbKvHeads;

        /// <summary>
        /// The hidden size of the model.
        /// </summary>
        SizeType mHiddenSize;

        /// <summary>
        /// The data type used by the model.
        /// </summary>
        nvinfer1::DataType mDataType;

        /// <summary>
        /// Flag indicating whether GPT attention plugin is used.
        /// </summary>
        bool mUseGptAttentionPlugin;

        /// <summary>
        /// Flag indicating whether input is packed.
        /// </summary>
        bool mInputPacked;

        /// <summary>
        /// Flag indicating whether paged key-value cache is used.
        /// </summary>
        bool mPagedKvCache;

        /// <summary>
        /// The number of tokens per block.
        /// </summary>
        SizeType mTokensPerBlock;

        /// <summary>
        /// The quantization mode used by the model.
        /// </summary>
        common::QuantMode mQuantMode;

        /// <summary>
        /// The maximum batch size for the model.
        /// </summary>
        SizeType mMaxBatchSize;

        /// <summary>
        /// The maximum beam width for the model.
        /// </summary>
        SizeType mMaxBeamWidth;

        /// <summary>
        /// The maximum input length for the model.
        /// </summary>
        SizeType mMaxInputLen;

        /// <summary>
        /// The maximum output length for the model.
        /// </summary>
        SizeType mMaxOutputLen;

        /// <summary>
        /// The maximum number of tokens.
        /// </summary>
        std::optional<SizeType> mMaxNumTokens;

        /// <summary>
        /// Flag indicating whether to compute context logits.
        /// </summary>
        bool mComputeContextLogits;

        /// <summary>
        /// Flag indicating whether to compute generation logits.
        /// </summary>
        bool mComputeGenerationLogits;

        /// <summary>
        /// The model variant.
        /// </summary>
        ModelVariant mModelVariant;

        /// <summary>
        /// Flag indicating whether custom all-reduce is used.
        /// </summary>
        bool mUseCustomAllReduce;

        /// <summary>
        /// The maximum size of the prompt embedding table.
        /// </summary>
        SizeType mMaxPromptEmbeddingTableSize;

        /// <summary>
        /// The maximum draft length for the model.
        /// </summary>
        SizeType mMaxDraftLen;

        /// <summary>
        /// Flag indicating whether context FMHA is used for generation.
        /// </summary>
        bool mUseContextFMHAForGeneration;

        /// <summary>
        /// Flag indicating whether paged context FMHA is used.
        /// </summary>
        bool mPagedContextFMHA;
    };
}