
#pragma once

#include "common.h"

#include <NvInferRuntime.h>
#include <optional>
#include <vector>

namespace bitfusion::runtime
{
class WorldConfig
{
public:
    static SizeType constexpr kDefaultGpusPerNode = 8;

    explicit WorldConfig(SizeType tensorParallelism = 1, SizeType pipelineParallelism = 1, SizeType rank = 0,
        SizeType gpusPerNode = kDefaultGpusPerNode,
        std::optional<std::vector<SizeType>> const& deviceIds = std::nullopt);

    [[nodiscard]] SizeType constexpr getSize() const noexcept
    {
        return mTensorParallelism * mPipelineParallelism;
    }

    [[nodiscard]] SizeType constexpr getTensorParallelism() const noexcept
    {
        return mTensorParallelism;
    }

    [[nodiscard]] bool constexpr isTensorParallel() const noexcept
    {
        return mTensorParallelism > 1;
    }

    [[nodiscard]] SizeType constexpr getPipelineParallelism() const noexcept
    {
        return mPipelineParallelism;
    }

    [[nodiscard]] bool constexpr isPipelineParallel() const noexcept
    {
        return mPipelineParallelism > 1;
    }

    [[nodiscard]] SizeType constexpr getRank() const noexcept
    {
        return mRank;
    }

    [[nodiscard]] SizeType constexpr getGpusPerNode() const noexcept
    {
        return mGpusPerNode;
    }

    [[nodiscard]] SizeType getGpusPerGroup() const noexcept
    {
        return static_cast<SizeType>(mDeviceIds.size());
    }

    [[nodiscard]] SizeType getDevice() const noexcept
    {
        return mDeviceIds[mRank % getGpusPerGroup()];
    }

    [[nodiscard]] SizeType constexpr getPipelineParallelRank() const noexcept
    {
        return mRank / mTensorParallelism;
    }

    [[nodiscard]] SizeType constexpr getTensorParallelRank() const noexcept
    {
        return mRank % mTensorParallelism;
    }

    [[nodiscard]] bool constexpr isFirstPipelineParallelRank() const noexcept
    {
        return getPipelineParallelRank() == 0;
    }

    [[nodiscard]] bool constexpr isLastPipelineParallelRank() const noexcept
    {
        return getPipelineParallelRank() == getPipelineParallelism() - 1;
    }

    [[nodiscard]] SizeType constexpr getLastRank() const noexcept
    {
        return getSize() - 1;
    }

    [[nodiscard]] std::vector<SizeType> getPipelineParallelGroup() const;

    static bool validConfig(nvinfer1::ILogger& logger, SizeType tensorParallelism, SizeType pipelineParallelism);

    static WorldConfig mpi(nvinfer1::ILogger& logger, SizeType gpusPerNode = kDefaultGpusPerNode,
        std::optional<SizeType> tensorParallelism = std::nullopt,
        std::optional<SizeType> pipelineParallelism = std::nullopt,
        std::optional<std::vector<SizeType>> const& deviceIds = std::nullopt);

    static WorldConfig mpi(SizeType gpusPerNode = kDefaultGpusPerNode,
        std::optional<SizeType> tensorParallelism = std::nullopt,
        std::optional<SizeType> pipelineParallelism = std::nullopt,
        std::optional<std::vector<SizeType>> const& deviceIds = std::nullopt);

private:
    SizeType mTensorParallelism;
    SizeType mPipelineParallelism;
    SizeType mRank;
    SizeType mGpusPerNode;
    std::vector<SizeType> mDeviceIds;
};

}
