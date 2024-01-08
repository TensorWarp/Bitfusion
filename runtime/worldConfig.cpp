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
#include <optional>
#include <vector>

using namespace bitfusion::runtime;
namespace tc = bitfusion::common;

namespace
{
    std::mutex mpiMutex;

    bool mpiInitialized = false;

    /// <summary>
    /// Initializes MPI if it has not been initialized yet.
    /// </summary>
    /// <param name="logger">An instance of the nvinfer1::ILogger for logging.</param>
    /// <param name="threadMode">Thread mode for MPI initialization.</param>
    void initMpi(nvinfer1::ILogger& logger, int threadMode = MPI_THREAD_FUNNELED)
    {
        std::lock_guard<std::mutex> lk(mpiMutex);
        if (mpiInitialized)
        {
            return;
        }

        int initialized = 0;
        MPI_CHECK(MPI_Initialized(&initialized));
        if (!initialized)
        {
            logger.log(
                nvinfer1::ILogger::Severity::kINFO, tc::fmtstr("Initializing MPI with thread mode %d", threadMode).c_str());
            int providedMode;
            MPI_CHECK(MPI_Init_thread(nullptr, nullptr, threadMode, &providedMode));
            CHECK_WITH_INFO(providedMode >= threadMode, "MPI_Init_thread failed");
            std::atexit([]() { MPI_Finalize(); });

            auto previousHandler = std::signal(SIGABRT, [](int signal) { MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); });
            CHECK_WITH_INFO(previousHandler != SIG_ERR, "Signal handler setup failed");
        }

        mpiInitialized = true;
    }
}

/// <summary>
/// Constructor for WorldConfig.
/// </summary>
/// <param name="tensorParallelism">Size of tensor parallelism.</param>
/// <param name="pipelineParallelism">Size of pipeline parallelism.</param>
/// <param name="rank">Rank of the instance.</param>
/// <param name="gpusPerNode">Number of GPUs per node.</param>
/// <param name="deviceIds">Optional vector of device IDs.</param>
WorldConfig::WorldConfig(SizeType tensorParallelism, SizeType pipelineParallelism, SizeType rank, SizeType gpusPerNode,
    std::optional<std::vector<SizeType>> const& deviceIds)
    : mTensorParallelism{ tensorParallelism },
    mPipelineParallelism{ pipelineParallelism },
    mRank{ rank },
    mGpusPerNode{ gpusPerNode },
    mDeviceIds{ deviceIds.value_or(std::vector<SizeType>(mGpusPerNode)) }
{
    auto const numDevices = mDeviceIds.size();
    CHECK(numDevices > 0);

    if (!deviceIds.has_value())
    {
        mDeviceIds.resize(mGpusPerNode);
        std::iota(mDeviceIds.begin(), mDeviceIds.end(), 0);
    }
    else
    {
        CHECK_WITH_INFO(static_cast<SizeType>(numDevices) <= mGpusPerNode,
            "Number of device IDs %zu is greater than GPUs per node %d", numDevices, mGpusPerNode);

        CHECK(*std::max_element(mDeviceIds.begin(), mDeviceIds.end()) < mGpusPerNode);
        CHECK(*std::min_element(mDeviceIds.begin(), mDeviceIds.end()) >= 0);

        std::set<SizeType> const deviceIdSet(mDeviceIds.begin(), mDeviceIds.end());
        CHECK_WITH_INFO(deviceIdSet.size() == numDevices, "Device IDs are not unique %zu != %zu", deviceIdSet.size(),
            numDevices);

        if (std::adjacent_find(deviceIdSet.begin(), deviceIdSet.end(), [](auto x, auto y) { return y - x != 1; }) !=
            deviceIdSet.end())
        {
            LOG_WARNING("The user-specified device IDs are not contiguous!");
        }
        LOG_INFO("Using user-specified devices: %s", tc::arr2str(mDeviceIds.data(), numDevices).c_str());
    }

    CHECK(mTensorParallelism > 0);
    CHECK(mPipelineParallelism > 0);

    CHECK_WITH_INFO(static_cast<SizeType>(numDevices) >= tensorParallelism * pipelineParallelism,
        "Number of GPUs per node %d must be at least as large as TP (%d) * PP (%d)", mGpusPerNode,
        mTensorParallelism, mPipelineParallelism);
}

/// <summary>
/// Validates the WorldConfig.
/// </summary>
/// <param name="logger">An instance of the nvinfer1::ILogger for logging.</param>
/// <param name="tensorParallelism">Size of tensor parallelism.</param>
/// <param name="pipelineParallelism">Size of pipeline parallelism.</param>
/// <returns>True if the configuration is valid, false otherwise.</returns>
bool WorldConfig::validConfig(nvinfer1::ILogger& logger, SizeType tensorParallelism, SizeType pipelineParallelism)
{
    initMpi(logger);

    int mpiSize;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));
    return mpiSize == tensorParallelism * pipelineParallelism;
}

/// <summary>
/// Creates an instance of WorldConfig for MPI usage.
/// </summary>
/// <param name="logger">An instance of the nvinfer1::ILogger for logging.</param>
/// <param name="gpusPerNode">Number of GPUs per node.</param>
/// <param name="tensorParallelism">Optional tensor parallelism value.</param>
/// <param name="pipelineParallelism">Optional pipeline parallelism value.</param>
/// <param name="deviceIds">Optional vector of device IDs.</param>
/// <returns>An instance of WorldConfig.</returns>
WorldConfig WorldConfig::mpi(nvinfer1::ILogger& logger, SizeType gpusPerNode, std::optional<SizeType> tensorParallelism,
    std::optional<SizeType> pipelineParallelism, std::optional<std::vector<SizeType>> const& deviceIds)
{
    initMpi(logger);

    int mpiSize, mpiRank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
    logger.log(nvinfer1::ILogger::Severity::kINFO, tc::fmtstr("MPI size: %d, rank: %d", mpiSize, mpiRank).c_str());

    auto pp = pipelineParallelism.value_or(1);
    auto tp = tensorParallelism.value_or(mpiSize / pp);
    CHECK(mpiSize == tp * pp);

    return WorldConfig{ tp, pp, mpiRank, gpusPerNode, deviceIds };
}

/// <summary>
/// Creates an instance of WorldConfig for MPI usage with a default logger.
/// </summary>
/// <param name="gpusPerNode">Number of GPUs per node.</param>
/// <param name="tensorParallelism">Optional tensor parallelism value.</param>
/// <param name="pipelineParallelism">Optional pipeline parallelism value.</param>
/// <param name="deviceIds">Optional vector of device IDs.</param>
/// <returns>An instance of WorldConfig.</returns>
WorldConfig WorldConfig::mpi(SizeType gpusPerNode, std::optional<SizeType> tensorParallelism,
    std::optional<SizeType> pipelineParallelism, std::optional<std::vector<SizeType>> const& deviceIds)
{
    TllmLogger logger{};
    return mpi(logger, gpusPerNode, tensorParallelism, pipelineParallelism, deviceIds);
}

/// <summary>
/// Gets the pipeline parallel group.
/// </summary>
/// <returns>A vector containing the pipeline parallel group.</returns>
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
