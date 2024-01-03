#pragma once

#include <assert.h>
#include <chrono>
#include <iostream>
#include <list>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

namespace bitfusion::batch_manager
{
    enum class BatchManagerErrorCode_t
    {
        STATUS_SUCCESS = 0,
        STATUS_FAILED = 1,
        STATUS_NO_WORK = 2,
        STATUS_TERMINATE = 3
    };

    enum class TrtGptModelType
    {
        V1,
        InflightBatching,
        InflightFusedBatching
    };
}
