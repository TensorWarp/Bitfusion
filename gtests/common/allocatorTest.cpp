#include <gtest/gtest.h>
#include <memory>
#include <format>
#include <source_location>

#include "../../common/cudaAllocator.h"
#include "../../runtime/bufferManager.h"
#include "../../runtime/cudaStream.h"

namespace tc = bitfusion::common;
namespace tr = bitfusion::runtime;

/// <summary>
/// Logs an error message with an optional source location.
/// </summary>
/// <param name="message">The error message to log.</param>
/// <param name="location">The source location where the error occurred (optional).</param>
void logError(const std::string& message,
    const std::source_location& location = std::source_location::current()) {
    std::cerr << std::format("Error at {}:{}: {}", location.file_name(), location.line(), message) << std::endl;
}

/// <summary>
/// Test case for Allocator and DeviceDestruction.
/// </summary>
TEST(Allocator, DeviceDestruction)
{
    auto streamPtr = std::make_shared<tr::CudaStream>();

    {
        auto allocator = std::make_unique<tc::CudaAllocator>(tr::BufferManager(streamPtr));
        constexpr auto sizeBytes = 1024 * 1024;

        void* ptr = nullptr;

        auto testReMalloc = [&allocator, &ptr](size_t size, bool expectNonNull) {
            ptr = allocator->reMalloc(ptr, size, expectNonNull);
            if (expectNonNull) {
                ASSERT_NE(ptr, nullptr) << std::format("Allocation failed for size: {}", size);
            }
            else {
                ASSERT_EQ(ptr, nullptr) << std::format("Allocation unexpectedly succeeded for size: {}", size);
            }
            };

        testReMalloc(sizeBytes, false);
        allocator->free(&ptr);
        ASSERT_EQ(ptr, nullptr);

        for (const auto& newSize : { sizeBytes, sizeBytes / 2, sizeBytes * 2, sizeBytes })
        {
            testReMalloc(newSize, true);
        }
    }

    streamPtr->synchronize();
}