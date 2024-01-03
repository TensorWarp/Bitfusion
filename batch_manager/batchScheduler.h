#pragma once

#include "kvCacheManager.h"
#include "llmRequest.h"
#include "../runtime/common.h"
#include <list>

namespace bitfusion::batch_manager::batch_scheduler
{

    enum class SchedulerPolicy
    {
        MAX_UTILIZATION,
        GUARANTEED_NO_EVICT,
    };

    class BatchScheduler
    {
    public:
        using RequestTable = std::map<uint64_t, std::shared_ptr<LlmRequest>>;
        using SizeType = bitfusion::runtime::SizeType;
        using RequestList = std::list<std::shared_ptr<LlmRequest>>;

        BatchScheduler(SizeType targetBatchSize, std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
            SchedulerPolicy schedulerPolicy, bool manyMicroBatches = false)
            : mTargetBatchSize(targetBatchSize)
            , mKvCacheManager(kvCacheManager)
            , mSchedulerPolicy(schedulerPolicy)
            , mManyMicroBatches(manyMicroBatches)
        {
        }

        std::tuple<RequestList, RequestTable> scheduleRequests(const RequestList& requestList);

    private:
        std::tuple<RequestList, RequestTable> scheduleRequestsMaxUtilization(const RequestList& requestList);

        bool trySchedulingRequestMaxUtilization(
            const LlmRequest& req, SizeType& numScheduledRequests, SizeType& numScheduledBlocks);

        std::tuple<RequestList, RequestTable> scheduleRequestsGuaranteedNoEvict(const RequestList& requestList);

        std::tuple<RequestList, RequestTable> scheduleTargetBatchSize(const RequestList& requestList);

        SizeType mTargetBatchSize;

        std::shared_ptr<kv_cache_manager::KVCacheManager> mKvCacheManager;

        SchedulerPolicy mSchedulerPolicy;

        bool mManyMicroBatches;
    };
}