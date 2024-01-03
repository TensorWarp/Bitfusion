
#include "worldConfig.h"

#include "../common/assert.h"
#include "../common/logger.h"
#include "../common/stringUtils.h"
#include "tllmLogger.h"
#include "../runtime/utils/multiDeviceUtils.h"

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <mpi.h>
#include <mutex>
#include <numeric>
#include <set>

using namespace bitfusion::runtime;
namespace tc = bitfusion::common;

namespace
{

bool mpiInitialized = false;
std::mutex mpiMutex;

void initMpi(nvinfer1::ILogger& logger, int threadMode = MPI_THREAD_FUNNELED)
{
    std::lock_guard<std::mutex> lk(mpiMutex);
    if (mpiInitialized)
    {
        return;
    }

    int initialized = 0;
    TLLM_MPI_CHECK(MPI_Initialized(&initialized));
    if (!initialized)
    {
        logger.log(
            nvinfer1::ILogger::Severity::kINFO, tc::fmtstr("Initializing MPI with thread mode %d", threadMode).c_str());
        int providedMode;
        TLLM_MPI_CHECK(MPI_Init_thread(nullptr, nullptr, threadMode, &providedMode));
        TLLM_CHECK_WITH_INFO(providedMode >= threadMode, "MPI_Init_thread failed");
        std::atexit([]() { MPI_Finalize(); });

        auto previousHandler = std::signal(SIGABRT, [](int signal) { MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); });
        TLLM_CHECK_WITH_INFO(previousHandler != SIG_ERR, "Signal handler setup failed");
    }

    mpiInitialized = true;
}

}

WorldConfig::WorldConfig(SizeType tensorParallelism, SizeType pipelineParallelism, SizeType rank, SizeType gpusPerNode,
    std::optional<std::vector<SizeType>> const& deviceIds)
    : mTensorParallelism{tensorParallelism}
    , mPipelineParallelism{pipelineParallelism}
    , mRank{rank}
    , mGpusPerNode{gpusPerNode}
    , mDeviceIds{deviceIds.value_or(std::vector<SizeType>(mGpusPerNode))}
{
    auto const numDevices = mDeviceIds.size();
    TLLM_CHECK(numDevices > 0);

    if (!deviceIds.has_value())
    {
        mDeviceIds.resize(mGpusPerNode);
        std::iota(mDeviceIds.begin(), mDeviceIds.end(), 0);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(static_cast<SizeType>(numDevices) <= mGpusPerNode,
            "Number of device IDs %zu is greater than GPUs per node %d", numDevices, mGpusPerNode);

        TLLM_CHECK(*std::max_element(mDeviceIds.begin(), mDeviceIds.end()) < mGpusPerNode);
        TLLM_CHECK(*std::min_element(mDeviceIds.begin(), mDeviceIds.end()) >= 0);

        std::set<SizeType> const deviceIdSet(mDeviceIds.begin(), mDeviceIds.end());
        TLLM_CHECK_WITH_INFO(
            deviceIdSet.size() == numDevices, "Device IDs are not unique %zu != %zu", deviceIdSet.size(), numDevices);

        if (std::adjacent_find(deviceIdSet.begin(), deviceIdSet.end(), [](auto x, auto y) { return y - x != 1; })
            != deviceIdSet.end())
        {
            TLLM_LOG_WARNING("The user specified device IDs are not contiguous!");
        }
        TLLM_LOG_INFO("Using user-specified devices: %s", tc::arr2str(mDeviceIds.data(), numDevices).c_str());
    }

    TLLM_CHECK(mTensorParallelism > 0);
    TLLM_CHECK(mPipelineParallelism > 0);

    TLLM_CHECK_WITH_INFO(static_cast<SizeType>(numDevices) >= tensorParallelism * pipelineParallelism,
        "Number of GPUs per node %d must be at least as large as TP (%d) * PP (%d)", mGpusPerNode, mTensorParallelism,
        mPipelineParallelism);
}

bool WorldConfig::validConfig(nvinfer1::ILogger& logger, SizeType tensorParallelism, SizeType pipelineParallelism)
{
    initMpi(logger);

    int mpiSize;
    TLLM_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));
    return mpiSize == tensorParallelism * pipelineParallelism;
}

WorldConfig WorldConfig::mpi(nvinfer1::ILogger& logger, SizeType gpusPerNode, std::optional<SizeType> tensorParallelism,
    std::optional<SizeType> pipelineParallelism, std::optional<std::vector<SizeType>> const& deviceIds)
{
    initMpi(logger);

    int mpiSize, mpiRank;
    TLLM_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));
    TLLM_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
    logger.log(nvinfer1::ILogger::Severity::kINFO, tc::fmtstr("MPI size: %d, rank: %d", mpiSize, mpiRank).c_str());

    auto pp = pipelineParallelism.value_or(1);
    auto tp = tensorParallelism.value_or(mpiSize / pp);
    TLLM_CHECK(mpiSize == tp * pp);

    return WorldConfig{tp, pp, mpiRank, gpusPerNode, deviceIds};
}

WorldConfig WorldConfig::mpi(SizeType gpusPerNode, std::optional<SizeType> tensorParallelism,
    std::optional<SizeType> pipelineParallelism, std::optional<std::vector<SizeType>> const& deviceIds)
{
    TllmLogger logger{};
    return mpi(logger, gpusPerNode, tensorParallelism, pipelineParallelism, deviceIds);
}

std::vector<SizeType> WorldConfig::getPipelineParallelGroup() const
{
    auto const pp = getPipelineParallelism();
    auto const tp = getTensorParallelism();
    auto const worldSize = getSize();
    std::vector<SizeType> group;
    group.reserve(pp);
    for (SizeType idx = getTensorParallelRank(); idx < worldSize; idx += tp)
    {
        group.push_back(idx);
    }
    return group;
}
