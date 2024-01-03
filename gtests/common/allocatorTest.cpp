#include <gtest/gtest.h>

#include "../../common/cudaAllocator.h"
#include "../../runtime/bufferManager.h"
#include "../../runtime/cudaStream.h"

#include <memory>

namespace tc = bitfusion::common;
namespace tr = bitfusion::runtime;

TEST(Allocator, DeviceDestruction)
{
    auto streamPtr = std::make_shared<tr::CudaStream>();
    {
        auto allocator = std::make_unique<tc::CudaAllocator>(tr::BufferManager(streamPtr));

        auto constexpr sizeBytes = 1024 * 1024;
        void* ptr{};
        ptr = allocator->reMalloc(ptr, sizeBytes, false);
        EXPECT_NE(ptr, nullptr);
        allocator->free(&ptr);
        EXPECT_EQ(ptr, nullptr);
        ptr = allocator->reMalloc(ptr, sizeBytes, true);
        EXPECT_NE(ptr, nullptr);
        ptr = allocator->reMalloc(ptr, sizeBytes / 2, true);
        EXPECT_NE(ptr, nullptr);
        ptr = allocator->reMalloc(ptr, sizeBytes * 2, true);
        EXPECT_NE(ptr, nullptr);
        ptr = allocator->reMalloc(ptr, sizeBytes, false);
        EXPECT_NE(ptr, nullptr);
    }
    streamPtr->synchronize();
}
