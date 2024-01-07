
#include <gtest/gtest.h>

#include "../../runtime/worldConfig.h"

#include "../../common/tllmException.h"

namespace tr = bitfusion::runtime;
namespace tc = bitfusion::common;

TEST(WorldConfig, DeviceIds)
{
    auto constexpr tensorParallelism = 2;
    auto constexpr pipelineParallelism = 3;
    auto constexpr rank = 1;
    auto constexpr gpusPerNode = 8;
    EXPECT_NO_THROW(tr::WorldConfig(tensorParallelism, pipelineParallelism, rank, gpusPerNode));

    EXPECT_NO_THROW(
        tr::WorldConfig(tensorParallelism, pipelineParallelism, rank, gpusPerNode, std::vector{0, 1, 2, 3, 4, 5}));

    EXPECT_THROW(tr::WorldConfig(tensorParallelism, pipelineParallelism, rank, gpusPerNode, std::vector{0, 1, 2, 3, 4}),
        tc::TllmException);

    EXPECT_THROW(tr::WorldConfig(
                     tensorParallelism, pipelineParallelism, rank, gpusPerNode, std::vector{0, 1, 2, 3, 4, 5, 6, 7, 7}),
        tc::TllmException);

    EXPECT_THROW(
        tr::WorldConfig(tensorParallelism, pipelineParallelism, rank, gpusPerNode, std::vector{0, 1, 2, 3, 4, -1}),
        tc::TllmException);
    EXPECT_THROW(
        tr::WorldConfig(tensorParallelism, pipelineParallelism, rank, gpusPerNode, std::vector{0, 1, 2, 3, 4, 8}),
        tc::TllmException);

    EXPECT_THROW(
        tr::WorldConfig(tensorParallelism, pipelineParallelism, rank, gpusPerNode, std::vector{0, 1, 5, 3, 4, 5}),
        tc::TllmException);

    EXPECT_NO_THROW(
        tr::WorldConfig(tensorParallelism, pipelineParallelism, rank, gpusPerNode, std::vector{0, 1, 2, 3, 4, 6}));
}
