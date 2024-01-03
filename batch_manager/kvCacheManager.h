#pragma once

#include "kvCacheConfig.h"
#include "llmRequest.h"
#include "../runtime/bufferManager.h"
#include "../runtime/common.h"
#include "../runtime/cudaStream.h"
#include "../runtime/gptModelConfig.h"
#include "../runtime/iTensor.h"
#include "../runtime/worldConfig.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace std
{

    template <>
    struct hash<vector<int32_t>>
    {
        size_t operator()(vector<int32_t> const& vec) const noexcept
        {
            size_t seed = vec.size();
            for (auto x : vec)
            {
                uint32_t y = static_cast<uint32_t>(x);
                y = ((y >> 16) ^ y) * 0x45d9f3b;
                y = ((y >> 16) ^ y) * 0x45d9f3b;
                y = (y >> 16) ^ y;
                seed ^= y + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

}

namespace bitfusion::batch_manager::kv_cache_manager
{

    class KVCacheBlock;

    using SizeType = bitfusion::runtime::SizeType;
    using TokenIdType = bitfusion::runtime::TokenIdType;
    using VecTokens = std::vector<TokenIdType>;
    using BeamTokens = std::vector<VecTokens>;
    using BlockPtr = std::shared_ptr<KVCacheBlock>;
    using FreeBlocksQueue = std::list<BlockPtr>;
    using NextBlockMap = std::unordered_map<VecTokens, BlockPtr>;

    struct KvCacheStats
    {
        SizeType maxNumBlocks;
        SizeType freeNumBlocks;
        SizeType usedNumBlocks;
        SizeType toksPerBlock;
    };

    class KVCacheBlock
    {
    public:
        explicit KVCacheBlock(SizeType blockIdx);

        void startScheduling();

        [[nodiscard]] SizeType getBlockIdx() const;

        void incRefCount();

        void decRefCount();

        void decSchedulingRefCount();

        [[nodiscard]] bool hasRefs() const;

        [[nodiscard]] bool hasSchedulingRefs() const;

        void setTokens(VecTokens& tokens, bool isFull);

        [[nodiscard]] VecTokens const& getTokens() const;

        void setFreeBlockIterator(FreeBlocksQueue::iterator freeBlockIterator);

        void resetFreeBlockIterator();

        [[nodiscard]] std::optional<FreeBlocksQueue::iterator> const& getFreeBlockIterator() const;

        void setPrevBlock(BlockPtr prevBlock);

        void addNextBlock(VecTokens const& tokens, BlockPtr block);

        void removeNextBlock(VecTokens const& tokens);

        [[nodiscard]] BlockPtr findMatchingBlock(VecTokens const& tokens) const;

        void freeLeafBlock();

        [[nodiscard]] bool isFull() const;

    private:
        SizeType mBlockIdx;

        SizeType mRefCount;

        SizeType mSchedulingRefCount;

        VecTokens mTokens;

        BlockPtr mPrevBlock;

        NextBlockMap mNextBlocks;

        std::optional<FreeBlocksQueue::iterator> mFreeBlockIterator;

        bool mIsFull;
    };

    class GenerationRequest
    {
    public:
        using SizeType = bitfusion::runtime::SizeType;
        using SharedPtr = std::shared_ptr<GenerationRequest>;

        explicit GenerationRequest(SizeType seqSlotIdx, SizeType numTokens, SizeType beamWidth)
            : mSeqSlotIdx(seqSlotIdx)
            , mNumTokens(numTokens)
            , mBeamWidth(beamWidth)
            , mCacheBlockIds(beamWidth)
        {
        }

        void addToken()
        {
            mNumTokens++;
        }

        [[nodiscard]] SizeType getSequenceSlotIdx() const
        {
            return mSeqSlotIdx;
        }

        [[nodiscard]] SizeType getNumTokens() const
        {
            return mNumTokens;
        }

        [[nodiscard]] SizeType getBeamWidth() const
        {
            return mBeamWidth;
        }

        [[nodiscard]] std::vector<std::vector<SizeType>> const& getCacheBlockIds() const
        {
            return mCacheBlockIds;
        }

        void addCacheBlock(SizeType beamIdx, SizeType blockIdx)
        {
            mCacheBlockIds.at(beamIdx).push_back(blockIdx);
        }

        void clearCacheBlocks()
        {
            for (auto& beamBlockIds : mCacheBlockIds)
            {
                beamBlockIds.clear();
            }
        }

        void setNumPrepopulatedTokens(std::vector<int> numPrepopulatedTokens)
        {
            mNumPrepopulatedTokens = std::move(numPrepopulatedTokens);
        }

        [[nodiscard]] std::vector<int> const& getNumPrepopulatedTokens() const
        {
            return mNumPrepopulatedTokens;
        }

    private:
        SizeType mSeqSlotIdx;
        SizeType mNumTokens;
        SizeType mBeamWidth;
        std::vector<std::vector<SizeType>> mCacheBlockIds;
        std::vector<int> mNumPrepopulatedTokens;
    };

    class BlockManager
    {
    public:
        using SizeType = bitfusion::runtime::SizeType;

        explicit BlockManager(SizeType blocksInPool, SizeType tokensPerBlock);

        ~BlockManager();

        void startScheduling();

        void addSequence(GenerationRequest& sequence, SizeType inputLength, std::shared_ptr<LlmRequest> const& llmRequest);

        void addSequence(GenerationRequest& sequence, SizeType inputLength, bool enableCyclicKvCache);

        void allocateBlock(GenerationRequest& sequence, bool shareAmongBeams = false);

        void releaseBlocks(GenerationRequest& sequence, std::shared_ptr<LlmRequest> const& llmRequest = nullptr);

        void schedulingReleaseBlocks(GenerationRequest& sequence);

        [[nodiscard]] SizeType getNumFreeBlocks() const
        {
            return mFreeBlocks.size();
        }

        [[nodiscard]] SizeType getNumAllocatedBlocks() const
        {
            return getMaxNumBlocks() - getNumFreeBlocks();
        }

        [[nodiscard]] bool hasFreeBlocks(SizeType numRequired = 1) const
        {
            return getNumFreeBlocks() >= numRequired;
        }

        [[nodiscard]] bool schedulingHasFreeBlocks(SizeType numRequired = 1) const
        {
            return mSchedulingNumFreeBlocks >= numRequired;
        }

        [[nodiscard]] SizeType getMaxNumBlocks() const
        {
            return static_cast<SizeType>(mAllBlocksByIdx.size());
        }

        [[nodiscard]] SizeType getTokensPerBlock() const
        {
            return mTokensPerBlock;
        }

    private:
        void addBlockToBeam(BlockPtr& block, GenerationRequest& sequence, SizeType beamIdx, SizeType seqSlotIdx);

        void storeBlocks(std::list<VecTokens> blockedTokens, std::vector<SizeType> const& blockIds);

        SizeType loadOrAllocateBlocks(
            std::list<VecTokens> blockedTokens, GenerationRequest& sequence, SizeType beamIdx, SizeType seqSlotIdx);

        [[nodiscard]] BlockPtr getFreeBlock();

        void claimBlock(KVCacheBlock& block);

        void claimLeafBlock(KVCacheBlock& block);

    private:
        FreeBlocksQueue mFreeBlocks;
        std::vector<std::vector<BlockPtr>> mAllocatedBlocksPerSeq;
        SizeType mSchedulingNumFreeBlocks;
        SizeType mTokensPerBlock;
        std::vector<BlockPtr> mAllBlocksByIdx;
        BlockPtr mCachedBlocksRoot;
        std::size_t mAllocTotalBlocks, mAllocNewBlocks, mReusedBlocks;
    };

    class KVCacheManager
    {
    public:
        using SizeType = bitfusion::runtime::SizeType;
        using SequencesPtr = GenerationRequest::SharedPtr;
        using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;

        KVCacheManager(SizeType numLayers, SizeType numHeads, SizeType numKvHeads, SizeType hiddenSize,
            SizeType tokensPerBlock, SizeType maxNumBlocks, SizeType maxNumSequences, SizeType maxBeamWidth,
            SizeType maxBlocksPerSeq, SizeType maxAttentionWindow, nvinfer1::DataType dtype, CudaStreamPtr stream,
            bool enableBlockReuse = false);

        void startScheduling();

        [[nodiscard]] SizeType getTokensPerBlock() const
        {
            return mBlockManager.getTokensPerBlock();
        }

        [[nodiscard]] SizeType getMaxNumBlocks() const
        {
            return mBlockManager.getMaxNumBlocks();
        }

        [[nodiscard]] SizeType getUsedNumBlocks() const
        {
            return mBlockManager.getNumAllocatedBlocks();
        }

        [[nodiscard]] SizeType getNumFreeBlocks() const
        {
            return mBlockManager.getNumFreeBlocks();
        }

        [[nodiscard]] KvCacheStats getKvCacheStats() const
        {
            KvCacheStats kvCacheStats;
            kvCacheStats.maxNumBlocks = getMaxNumBlocks();
            kvCacheStats.freeNumBlocks = getNumFreeBlocks();
            kvCacheStats.usedNumBlocks = getUsedNumBlocks();
            kvCacheStats.toksPerBlock = getTokensPerBlock();

            return kvCacheStats;
        }

        [[nodiscard]] SizeType getBlockSize() const
        {
            return mBlockSize;
        }

        [[nodiscard]] BlockManager const& getBlockManager() const
        {
            return mBlockManager;
        }

        SizeType getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const;

        SizeType getNeededBlocksToCompletion(LlmRequest const& req) const;

        [[nodiscard]] std::vector<runtime::ITensor::SharedPtr> const& getMemoryPools() const
        {
            return mPools;
        }

        void addToken(SizeType seqSlotIdx);

        void addSequence(SizeType seqSlotIdx, SizeType inputLength, SizeType beamWidth,
            std::shared_ptr<LlmRequest> const& llmRequest = nullptr);

        void removeSequence(SizeType seqSlotIdx, std::shared_ptr<LlmRequest> const& llmRequest = nullptr);

        void schedulingRemoveSequence(SizeType seqSlotIdx);

        void getBlockPointersOfBatch(
            runtime::ITensor& dstPointers, SizeType firstBatchSlotIdx, SizeType batchSize, SizeType beamWidth) const;

        void copyBlockPointers(
            runtime::ITensor& dstPointers, SizeType dstSlotOffset, SizeType seqSlotIdx, SizeType beamWidth) const;

        [[nodiscard]] static SizeType constexpr calculatePageSize(bitfusion::runtime::GptModelConfig const& modelConfig)
        {
            return 2 * modelConfig.getNbKvHeads() * modelConfig.getTokensPerBlock() * modelConfig.getSizePerHead();
        }

        [[nodiscard]] static SizeType constexpr calculateCacheSizePerToken(
            bitfusion::runtime::GptModelConfig const& modelConfig, bitfusion::runtime::WorldConfig const& worldConfig)
        {
            return modelConfig.getNbLayers(worldConfig.getPipelineParallelism()) * 2 * modelConfig.getNbKvHeads()
                * modelConfig.getSizePerHead();
        }

        [[nodiscard]] static SizeType getMaxNumTokens(KvCacheConfig const& config, nvinfer1::DataType dtype,
            bitfusion::runtime::GptModelConfig const& modelConfig, bitfusion::runtime::WorldConfig const& worldConfig,
            runtime::BufferManager const& bufferManager);

        [[nodiscard]] SizeType getNumPrepopulatedTokens(SizeType batchSlotIdx, SizeType beamIdx) const
        {
            auto const& prepopulatedTokens = mSequences.at(batchSlotIdx)->getNumPrepopulatedTokens();
            return prepopulatedTokens.size() > 0 ? prepopulatedTokens.at(beamIdx) : 0;
        }

        [[nodiscard]] bool isEnableBlockReuse() const
        {
            return mEnableBlockReuse;
        }

    private:
        void resetBlockPointers(SizeType seqSlotIdx, SizeType beamWidth);
        void cacheBlockPointers(GenerationRequest const& seq, SizeType seqSlotIdx);
        void cacheNewBlockPointers(GenerationRequest const& seq, SizeType seqSlotIdx);

    private:
        SizeType mBlockSize;
        SizeType mMaxNumSequences;
        SizeType mMaxBeamWidth;
        SizeType mMaxBlocksPerSeq;
        SizeType mMaxAttentionWindow;
        std::vector<runtime::ITensor::SharedPtr> mPools;
        BlockManager mBlockManager;
        std::vector<SequencesPtr> mSequences;
        runtime::ITensor::SharedPtr mSequenceBlockPointers;
        runtime::BufferManager mBufferManager;
        bool mEnableBlockReuse;
    };
}