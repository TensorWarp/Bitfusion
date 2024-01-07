#include "memoryCounters.h"
#include "../common/stringUtils.h"
#include <array>
#include <cmath>
#include <format>
#include <iostream>
#include <string>

namespace tc = bitfusion::common;

namespace
{
    /// <summary>
    /// Array of byte unit names.
    /// </summary>
    constexpr std::array<const char*, 7> kByteUnits = { "B", "KB", "MB", "GB", "TB", "PB", "EB" };

    /// <summary>
    /// Convert bytes to a formatted string with a specified precision.
    /// </summary>
    /// <param name="bytes">The number of bytes.</param>
    /// <param name="precision">The decimal precision for formatting.</param>
    /// <returns>A formatted string representing the bytes.</returns>
    std::string doubleBytesToString(double bytes, int precision)
    {
        std::uint32_t unitIdx = 0;

        while (std::abs(bytes) >= 1024.0 && unitIdx < kByteUnits.size() - 1)
        {
            bytes /= 1024.0;
            ++unitIdx;
        }

        return std::format("{:.{}f} {}", bytes, precision, kByteUnits[unitIdx]);
    }
}

namespace bitfusion::runtime
{
    thread_local MemoryCounters MemoryCounters::mInstance;

    /// <summary>
    /// Convert bytes to a formatted string with a specified precision.
    /// </summary>
    /// <param name="bytes">The number of bytes.</param>
    /// <param name="precision">The decimal precision for formatting.</param>
    /// <returns>A formatted string representing the bytes.</returns>
    std::string MemoryCounters::bytesToString(SizeType bytes, int precision)
    {
        return doubleBytesToString(static_cast<double>(bytes), precision);
    }

    /// <summary>
    /// Convert bytes to a formatted string with a specified precision.
    /// </summary>
    /// <param name="bytes">The number of bytes.</param>
    /// <param name="precision">The decimal precision for formatting.</param>
    /// <returns>A formatted string representing the bytes.</returns>
    std::string MemoryCounters::bytesToString(DiffType bytes, int precision)
    {
        return doubleBytesToString(static_cast<double>(bytes), precision);
    }

    /// <summary>
    /// Get a string representation of memory counters.
    /// </summary>
    /// <returns>A formatted string with memory counter information.</returns>
    std::string MemoryCounters::toString() const
    {
        return std::format("[MemUsage] GPU {}, CPU {}, Pinned {}",
            bytesToString(this->getGpu()), bytesToString(this->getCpu()), bytesToString(this->getPinned()));
    }

    /// <summary>
    /// Allocate memory of the specified type and size.
    /// </summary>
    /// <param name="memoryType">The type of memory to allocate.</param>
    /// <param name="size">The size of memory to allocate.</param>
    void MemoryCounters::allocate(MemoryType memoryType, MemoryCounters::SizeType size)
    {
        switch (memoryType)
        {
        case MemoryType::kGPU: [[likely]] allocate<MemoryType::kGPU>(size); break;
        case MemoryType::kCPU: [[likely]] allocate<MemoryType::kCPU>(size); break;
        case MemoryType::kPINNED: [[likely]] allocate<MemoryType::kPINNED>(size); break;
        default: THROW("Unknown memory type");
        }
    }

    /// <summary>
    /// Deallocate memory of the specified type and size.
    /// </summary>
    /// <param name="memoryType">The type of memory to deallocate.</param>
    /// <param name="size">The size of memory to deallocate.</param>
    void MemoryCounters::deallocate(MemoryType memoryType, MemoryCounters::SizeType size)
    {
        switch (memoryType)
        {
        case MemoryType::kGPU: [[likely]] deallocate<MemoryType::kGPU>(size); break;
        case MemoryType::kCPU: [[likely]] deallocate<MemoryType::kCPU>(size); break;
        case MemoryType::kPINNED: [[likely]] deallocate<MemoryType::kPINNED>(size); break;
        default: THROW("Unknown memory type");
        }
    }
}
