#pragma once

#include "../runtime/bufferManager.h"
#include "../runtime/iTensor.h"
#include "../runtime/samplingConfig.h"

#include <assert.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace bitfusion::batch_manager
{

    enum LlmRequestState_t
    {
        REQUEST_STATE_UNKNOWN = 0,
        REQUEST_STATE_CONTEXT_INIT = 1,
        REQUEST_STATE_GENERATION_IN_PROGRESS = 2,
        REQUEST_STATE_GENERATION_COMPLETE = 3
    };

    template <typename TTensor>
    class GenericLlmRequest
    {
    public:
        using SizeType = runtime::SizeType;
        using TokenIdType = runtime::TokenIdType;
        using RequestIdType = std::uint64_t;
        using VecTokens = std::vector<TokenIdType>;
        using VecLogProbs = std::vector<float>;
        using BeamTokens = std::vector<VecTokens>;
        using TensorPtr = TTensor;

        GenericLlmRequest(RequestIdType requestId, SizeType maxNewTokens, std::shared_ptr<VecTokens> inputTokens,
            runtime::SamplingConfig samplingConfig, bool isStreaming, std::optional<SizeType> endId = std::nullopt,
            std::optional<SizeType> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
            std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
            std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
            std::optional<SizeType> promptVocabSize = std::nullopt, bool returnLogProbs = false,
            std::optional<std::shared_ptr<VecTokens>> draftTokens = std::nullopt,
            std::optional<TensorPtr> draftLogits = std::nullopt)
            : mRequestId(requestId)
            , mPromptLen(inputTokens->size())
            , mMaxNewTokens(maxNewTokens)
            , mSamplingConfig(samplingConfig)
            , mState(REQUEST_STATE_CONTEXT_INIT)
            , mIsStreaming(isStreaming)
            , mEndId(endId)
            , mPadId(padId)
            , mSeqSlot(-1)
            , mOrigPromptLen(inputTokens->size())
            , mEmbeddingBias(embeddingBias)
            , mBadWordsList(badWordsList)
            , mStopWordsList(stopWordsList)
            , mPromptEmbeddingTable(promptEmbeddingTable)
            , mPromptVocabSize(promptVocabSize)
            , mReturnLogProbs(returnLogProbs)
            , mLogProbs(samplingConfig.beamWidth)
            , mCumLogProbs(samplingConfig.beamWidth)
            , mDraftTokens(draftTokens.value_or(std::make_shared<VecTokens>()))
            , mDraftLogits(draftLogits)
        {
            mMaxSentTokenPos = mPromptLen - 1;
            mTokens = BeamTokens(mSamplingConfig.beamWidth, *inputTokens);

            if ((mPromptEmbeddingTable.has_value() && !mPromptVocabSize.has_value())
                || (!mPromptEmbeddingTable.has_value() && mPromptVocabSize.has_value()))
            {
                std::string errStr
                    = "Prompt embedding table and prompt vocab size tensors must both be provided for requests with prompt "
                    "tuning enabled.";
                LOG_ERROR(errStr);
                throw std::runtime_error(errStr);
            }

            if (draftLogits.has_value() && !draftTokens.has_value())
            {
                std::string errStr = "Draft tokens must be specified when draft logits are given.";
                LOG_ERROR(errStr);
                throw std::runtime_error(errStr);
            }
        }

        SizeType getNumTokens(SizeType beam) const
        {
            return mTokens.at(beam).size();
        }

        SizeType getMaxBeamNumTokens() const
        {
            SizeType maxTokens = 0;
            for (SizeType beam = 0; beam < mSamplingConfig.beamWidth; ++beam)
            {
                maxTokens = std::max(maxTokens, static_cast<SizeType>(mTokens.at(beam).size()));
            }
            return maxTokens;
        }

        TokenIdType getToken(SizeType beam, SizeType pos) const
        {
            return mTokens.at(beam).at(pos);
        }

        VecTokens const& getTokens(SizeType beam) const
        {
            return mTokens.at(beam);
        }

        BeamTokens const& getTokens() const
        {
            return mTokens;
        }

        std::shared_ptr<VecTokens> const& getDraftTokens() const
        {
            return mDraftTokens;
        }

        std::optional<TensorPtr> getDraftLogits() const
        {
            return mDraftLogits;
        }

        bool hasDraftTokens() const
        {
            return mDraftTokens && mDraftTokens->size() > 0;
        }

        SizeType getMaxNumGeneratedTokens() const
        {
            return getMaxBeamNumTokens() - mPromptLen;
        }

        void addNewToken(TokenIdType token, SizeType beam)
        {
            mTokens.at(beam).push_back(token);
        }

        void addNewTokens(VecTokens const& beamTokens)
        {
            assert(static_cast<size_t>(mSamplingConfig.beamWidth) == beamTokens.size());
            for (std::size_t beam = 0; beam < beamTokens.size(); ++beam)
            {
                const auto outputId = beamTokens[beam];
                mTokens.at(beam).push_back(outputId);
            }
        }

        void setGeneratedTokens(const BeamTokens& generatedBeamTokens)
        {
            assert(generatedBeamTokens.size() == static_cast<size_t>(mSamplingConfig.beamWidth));
            for (std::size_t beam = 0; beam < generatedBeamTokens.size(); ++beam)
            {
                auto& beamTokens = mTokens[beam];
                beamTokens.resize(mPromptLen);
                beamTokens.insert(beamTokens.end(), generatedBeamTokens[beam].begin(), generatedBeamTokens[beam].end());
            }
        }

        void pause(SizeType maxInputLen)
        {
            if (mSamplingConfig.beamWidth > 1)
            {
                for (std::size_t beam = 0; beam < mTokens.size(); ++beam)
                {
                    auto& beamTokens = mTokens.at(beam);
                    beamTokens.resize(mPromptLen);
                    if (mReturnLogProbs)
                    {
                        mLogProbs.at(beam).clear();
                    }
                }
            }
            else
            {
                SizeType newPromptLen = std::min(maxInputLen, mPromptLen + getMaxNumGeneratedTokens());
                for (std::size_t beam = 0; beam < mTokens.size(); ++beam)
                {
                    auto& beamTokens = mTokens.at(beam);
                    beamTokens.resize(newPromptLen);

                    if (mReturnLogProbs)
                    {
                        auto& logProb = mLogProbs.at(beam);
                        logProb.resize(newPromptLen - mPromptLen);
                    }
                }
                mMaxNewTokens -= (newPromptLen - mPromptLen);
                mPromptLen = newPromptLen;
            }
            mState = REQUEST_STATE_CONTEXT_INIT;
            mSeqSlot = -1;
        }

        SizeType getMaxSentTokenPos() const
        {
            return mMaxSentTokenPos;
        }

        void setMaxSentTokenPos(SizeType pos)
        {
            mMaxSentTokenPos = pos;
        }

        std::optional<TensorPtr> getPromptEmbeddingTable() const
        {
            return mPromptEmbeddingTable;
        }

        std::optional<SizeType> getPromptVocabSize() const
        {
            return mPromptVocabSize;
        }

        std::optional<TensorPtr> getEmbeddingBias() const
        {
            return mEmbeddingBias;
        }

        std::optional<TensorPtr> getBadWordsList() const
        {
            return mBadWordsList;
        }

        std::optional<TensorPtr> getStopWordsList() const
        {
            return mStopWordsList;
        }

        bool returnLogProbs() const
        {
            return mReturnLogProbs;
        }

        std::vector<VecLogProbs> const& getLogProbs() const
        {
            return mLogProbs;
        }

        VecLogProbs const& getLogProbs(SizeType beam) const
        {
            return mLogProbs.at(beam);
        }

        void setLogProbs(VecLogProbs const& logProbs, SizeType beam)
        {
            mLogProbs.at(beam).resize(mPromptLen - mOrigPromptLen);
            mLogProbs.at(beam).insert(mLogProbs.at(beam).end(), logProbs.begin(), logProbs.end());
        }

        VecLogProbs const& getCumLogProbs() const
        {
            return mCumLogProbs;
        }

        void setCumLogProb(float cumLogProb, SizeType beam)
        {
            mCumLogProbs.at(beam) = cumLogProb;
        }

        SizeType getOrigPromptLen() const
        {
            return mOrigPromptLen;
        }

        void setDraftTokens(const std::shared_ptr<VecTokens>& draftTokens)
        {
            mDraftTokens = draftTokens;
        }

        void setDraftLogits(const std::optional<TensorPtr>& draftLogits)
        {
            mDraftLogits = draftLogits;
        }

        RequestIdType mRequestId;
        SizeType mPromptLen;
        SizeType mMaxNewTokens;
        runtime::SamplingConfig mSamplingConfig;
        LlmRequestState_t mState;
        bool mIsStreaming;
        std::optional<SizeType> mEndId;
        std::optional<SizeType> mPadId;
        SizeType mSeqSlot;

    protected:
        SizeType mOrigPromptLen;
        BeamTokens mTokens;
        SizeType mMaxSentTokenPos;

        std::optional<TensorPtr> mEmbeddingBias;
        std::optional<TensorPtr> mBadWordsList;
        std::optional<TensorPtr> mStopWordsList;

        std::optional<TensorPtr> mPromptEmbeddingTable;
        std::optional<SizeType> mPromptVocabSize;

        bool mReturnLogProbs;

        std::vector<VecLogProbs> mLogProbs;
        VecLogProbs mCumLogProbs;
        std::shared_ptr<VecTokens> mDraftTokens;
        std::optional<TensorPtr> mDraftLogits;
    };

    class LlmRequest : public GenericLlmRequest<runtime::ITensor::SharedPtr>
    {
    public:
        using Base = GenericLlmRequest<runtime::ITensor::SharedPtr>;
        using TensorPtr = Base::TensorPtr;
        using SizeType = Base::SizeType;
        using TokenIdType = Base::TokenIdType;
        using RequestIdType = Base::RequestIdType;
        using VecLogProbs = Base::VecLogProbs;
        using BeamTokens = Base::BeamTokens;
        using VecTokens = Base::VecTokens;

        LlmRequest(RequestIdType requestId, SizeType maxNewTokens, std::shared_ptr<VecTokens> inputTokens,
            runtime::SamplingConfig samplingConfig, bool isStreaming, std::optional<SizeType> endId = std::nullopt,
            std::optional<SizeType> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
            std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
            std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
            std::optional<SizeType> promptVocabSize = std::nullopt, bool returnLogProbs = false,
            std::optional<std::shared_ptr<VecTokens>> draftTokens = std::nullopt,
            std::optional<TensorPtr> draftLogits = std::nullopt)
            : Base(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming, endId, padId, embeddingBias,
                badWordsList, stopWordsList, promptEmbeddingTable, promptVocabSize, returnLogProbs, draftTokens,
                draftLogits)
        {
        }

        void movePromptEmbeddingTableToGpu(runtime::BufferManager const& manager)
        {
            if (!mPromptEmbeddingTable.has_value()
                || mPromptEmbeddingTable.value()->getMemoryType() == runtime::MemoryType::kGPU)
            {
                return;
            }
            else
            {
                TensorPtr gpuPromptEmbeddingTable
                    = manager.copyFrom(*mPromptEmbeddingTable.value(), runtime::MemoryType::kGPU);
                mPromptEmbeddingTable = gpuPromptEmbeddingTable;
            }
        }
    };
}