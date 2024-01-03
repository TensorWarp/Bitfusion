#pragma once

#include "BatchManager.h"
#include "batchScheduler.h"
#include "callbacks.h"
#include "llmRequest.h"
#include "trtGptModelOptionalParams.h"
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>

namespace nvinfer1
{
    class ILogger;
}

namespace bitfusion::batch_manager
{

    class InferenceRequest;
    class TrtGptModel;

    class GptManager
    {
    public:
        using SizeType = bitfusion::runtime::SizeType;
        using TokenIdType = bitfusion::runtime::TokenIdType;
        using RequestList = std::list<std::shared_ptr<LlmRequest>>;
        using TensorPtr = runtime::ITensor::SharedPtr;

        GptManager(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType, SizeType maxBeamWidth,
            batch_scheduler::SchedulerPolicy schedulerPolicy, GetInferenceRequestsCallback getInferenceRequestsCb,
            SendResponseCallback sendResponseCb, PollStopSignalCallback pollStopSignalCb = nullptr,
            ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = nullptr,
            const TrtGptModelOptionalParams& optionalParams = TrtGptModelOptionalParams(),
            std::optional<uint64_t> terminateReqId = std::nullopt, std::optional<SizeType> maxDraftTokens = std::nullopt,
            bool excludeInputInOutput = false);

        BatchManagerErrorCode_t fetchNewRequests();

        BatchManagerErrorCode_t returnCompletedRequests();

        BatchManagerErrorCode_t pollStopSignals();

        BatchManagerErrorCode_t returnBatchManagerStats();

        BatchManagerErrorCode_t waitUntilTerminate();

        BatchManagerErrorCode_t shutdown();

        virtual ~GptManager();

    protected:
        virtual BatchManagerErrorCode_t step(RequestList& activeRequests, std::set<uint64_t>& activeRequestsIds);

    private:
        SizeType getMaxInputLen() const;
        SizeType getMaxOutputLen() const;
        SizeType getMaxNumSequences() const;

        void validateLlmRequest(LlmRequest& newReq) const;
        static std::shared_ptr<LlmRequest> fillLlmRequest(std::shared_ptr<InferenceRequest> newReq);
        static std::shared_ptr<std::vector<TokenIdType>> getReqInputTokens(std::shared_ptr<InferenceRequest> newReq);
        static SizeType getMaxNewTokens(std::shared_ptr<InferenceRequest> newReq);

        GetInferenceRequestsCallback mGetInferenceRequestsCb;
        SendResponseCallback mSendResponseCb;
        PollStopSignalCallback mPollStopSignalCb;
        ReturnBatchManagerStatsCallback mReturnBatchManagerStatsCb;

        std::shared_ptr<TrtGptModel> mTrtGptModel;
        std::optional<uint64_t> mTerminateReqId;
        std::optional<SizeType> mMaxDraftTokens;

        int64_t mIterationCounter;
        RequestList mActiveRequests;
        std::set<uint64_t> mActiveRequestsIds;
        bool mExcludeInputInOutput;

        std::atomic<bool> shutdown_requested_;
        void decoupled_execution_loop();
        std::shared_ptr<std::thread> worker_thread_;
        std::shared_ptr<nvinfer1::ILogger> mLogger{};
    };
}