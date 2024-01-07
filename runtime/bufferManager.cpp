#include "bufferManager.h"
#include "../common/assert.h"
#include "tllmBuffers.h"

#include <cstring>
#include <cuda_runtime_api.h>
#include <memory>
#include <unordered_set>
#include <span>

using namespace bitfusion::runtime;

namespace tc = bitfusion::common;

/// <summary>
/// Constructor for BufferManager.
/// </summary>
/// <param name="stream">A pointer to a CudaStream.</param>
BufferManager::BufferManager(CudaStreamPtr stream)
    : mStream{ std::move(stream) }
{
    CHECK_WITH_INFO(static_cast<bool>(mStream), "Undefined CUDA stream");
    thread_local static std::unordered_set<int> initializedDevices(8);
    auto const device = mStream->getDevice();
    if (initializedDevices.find(device) == initializedDevices.end())
    {
        initializedDevices.insert(device);
        initMemoryPool(device);
    }
}

/// <summary>
/// Create a GPU buffer.
/// </summary>
/// <param name="size">The size of the buffer.</param>
/// <param name="type">The data type of the buffer.</param>
/// <returns>A unique pointer to the created IBuffer.</returns>
auto BufferManager::gpu(std::size_t size, nvinfer1::DataType type) const -> IBufferPtr
{
    return std::make_unique<DeviceBuffer>(size, type, CudaAllocatorAsync{ mStream });
}

/// <summary>
/// Create a GPU tensor.
/// </summary>
/// <param name="dims">The dimensions of the tensor.</param>
/// <param name="type">The data type of the tensor.</param>
/// <returns>A unique pointer to the created ITensor.</returns>
auto BufferManager::gpu(nvinfer1::Dims dims, nvinfer1::DataType type) const -> ITensorPtr
{
    return std::make_unique<DeviceTensor>(dims, type, CudaAllocatorAsync{ mStream });
}

/// <summary>
/// Create a CPU buffer.
/// </summary>
/// <param name="size">The size of the buffer.</param>
/// <param name="type">The data type of the buffer.</param>
/// <returns>A unique pointer to the created IBuffer.</returns>
auto BufferManager::cpu(std::size_t size, nvinfer1::DataType type) -> IBufferPtr
{
    return std::make_unique<HostBuffer>(size, type);
}

/// <summary>
/// Create a CPU tensor.
/// </summary>
/// <param name="dims">The dimensions of the tensor.</param>
/// <param name="type">The data type of the tensor.</param>
/// <returns>A unique pointer to the created ITensor.</returns>
auto BufferManager::cpu(nvinfer1::Dims dims, nvinfer1::DataType type) -> ITensorPtr
{
    return std::make_unique<HostTensor>(dims, type);
}

/// <summary>
/// Create a pinned buffer.
/// </summary>
/// <param name="size">The size of the buffer.</param>
/// <param name="type">The data type of the buffer.</param>
/// <returns>A unique pointer to the created IBuffer.</returns>
auto BufferManager::pinned(std::size_t size, nvinfer1::DataType type) -> IBufferPtr
{
    return std::make_unique<PinnedBuffer>(size, type);
}

/// <summary>
/// Create a pinned tensor.
/// </summary>
/// <param name="dims">The dimensions of the tensor.</param>
/// <param name="type">The data type of the tensor.</param>
/// <returns>A unique pointer to the created ITensor.</returns>
auto BufferManager::pinned(nvinfer1::Dims dims, nvinfer1::DataType type) -> ITensorPtr
{
    return std::make_unique<PinnedTensor>(dims, type);
}

/// <summary>
/// Set all elements of a buffer to zero.
/// </summary>
/// <param name="buffer">The buffer to set to zero.</param>
void BufferManager::setZero(IBuffer& buffer) const
{
    if (buffer.getMemoryType() == MemoryType::kGPU)
    {
        CUDA_CHECK(cudaMemsetAsync(buffer.data(), 0, buffer.getSizeInBytes(), mStream->get()));
    }
    else
    {
        std::memset(buffer.data(), 0, buffer.getSizeInBytes());
    }
}

/// <summary>
/// Copy data from a source buffer to a destination buffer.
/// </summary>
/// <param name="src">A pointer to the source data.</param>
/// <param name="dst">The destination buffer.</param>
/// <param name="srcType">The memory type of the source data.</param>
void BufferManager::copy(const void* src, IBuffer& dst, MemoryType srcType) const
{
    if (dst.getSizeInBytes() > 0)
    {
        if (srcType != MemoryType::kGPU && dst.getMemoryType() != MemoryType::kGPU)
        {
            std::memcpy(dst.data(), src, dst.getSizeInBytes());
        }
        else
        {
            CUDA_CHECK(cudaMemcpyAsync(dst.data(), src, dst.getSizeInBytes(), cudaMemcpyDefault, mStream->get()));
        }
    }
}

/// <summary>
/// Copy data from a source buffer to a destination pointer.
/// </summary>
/// <param name="src">The source buffer.</param>
/// <param name="dst">A pointer to the destination data.</param>
/// <param name="dstType">The memory type of the destination data.</param>
void BufferManager::copy(const IBuffer& src, void* dst, MemoryType dstType) const
{
    if (src.getSizeInBytes() > 0)
    {
        if (src.getMemoryType() != MemoryType::kGPU && dstType != MemoryType::kGPU)
        {
            std::memcpy(dst, src.data(), src.getSizeInBytes());
        }
        else
        {
            CUDA_CHECK(cudaMemcpyAsync(dst, src.data(), src.getSizeInBytes(), cudaMemcpyDefault, mStream->get()));
        }
    }
}

/// <summary>
/// Copy data from a source buffer to a destination buffer.
/// </summary>
/// <param name="src">The source buffer.</param>
/// <param name="dst">The destination buffer.</param>
void BufferManager::copy(const IBuffer& src, IBuffer& dst) const
{
    CHECK_WITH_INFO(src.getDataType() == dst.getDataType(),
        tc::fmtstr("Incompatible data types: %s != %s", src.getDataTypeName(), dst.getDataTypeName()));
    CHECK_WITH_INFO(src.getSizeInBytes() == dst.getSizeInBytes(),
        tc::fmtstr("Incompatible buffer sizes: %lu != %lu", src.getSizeInBytes(), dst.getSizeInBytes()));
    copy(src, dst.data(), dst.getMemoryType());
}

/// <summary>
/// Allocate a buffer with the specified memory type, size, and data type.
/// </summary>
/// <param name="memoryType">The memory type for the allocation.</param>
/// <param name="size">The size of the buffer.</param>
/// <param name="type">The data type of the buffer.</param>
/// <returns>A unique pointer to the allocated IBuffer.</returns>
auto BufferManager::allocate(MemoryType memoryType, std::size_t size, nvinfer1::DataType type) const -> IBufferPtr
{
    switch (memoryType)
    {
    case MemoryType::kCPU: return cpu(size, type);
    case MemoryType::kGPU: return gpu(size, type);
    case MemoryType::kPINNED: return pinned(size, type);
    default: THROW("Unknown memory type");
    }
}

/// <summary>
/// Allocate a tensor with the specified memory type, dimensions, and data type.
/// </summary>
/// <param name="memoryType">The memory type for the allocation.</param>
/// <param name="dims">The dimensions of the tensor.</param>
/// <param name="type">The data type of the tensor.</param>
/// <returns>A unique pointer to the allocated ITensor.</returns>
auto BufferManager::allocate(MemoryType memoryType, nvinfer1::Dims dims, nvinfer1::DataType type) const -> ITensorPtr
{
    switch (memoryType)
    {
    case MemoryType::kCPU: return cpu(dims, type);
    case MemoryType::kGPU: return gpu(dims, type);
    case MemoryType::kPINNED: return pinned(dims, type);
    default: THROW("Unknown memory type");
    }
}

/// <summary>
/// Copy data from a source buffer to a newly allocated buffer with the specified memory type.
/// </summary>
/// <param name="src">The source buffer.</param>
/// <param name="memoryType">The memory type for the destination buffer.</param>
/// <returns>A unique pointer to the allocated and copied IBuffer.</returns>
auto BufferManager::copyFrom(const IBuffer& src, MemoryType memoryType) const -> IBufferPtr
{
    auto dst = allocate(memoryType, src.getSize(), src.getDataType());
    copy(src, *dst);
    return dst;
}

/// <summary>
/// Copy data from a source tensor to a newly allocated tensor with the specified memory type.
/// </summary>
/// <param name="src">The source tensor.</param>
/// <param name="memoryType">The memory type for the destination tensor.</param>
/// <returns>A unique pointer to the allocated and copied ITensor.</returns>
auto BufferManager::copyFrom(const ITensor& src, MemoryType memoryType) const -> ITensorPtr
{
    auto dst = allocate(memoryType, src.getShape(), src.getDataType());
    copy(src, *dst);
    return dst;
}

/// <summary>
/// Get the CUDA stream associated with this BufferManager.
/// </summary>
/// <returns>A reference to the CudaStream.</returns>
const CudaStream& BufferManager::getStream() const
{
    return *mStream;
}

/// <summary>
/// Initialize the CUDA memory pool for the specified device.
/// </summary>
/// <param name="device">The device for which to initialize the memory pool.</param>
void BufferManager::initMemoryPool(int device)
{
    auto const deviceCount = tc::getDeviceCount();
    ::cudaMemPool_t memPool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, device));
    for (auto peerDevice = 0; peerDevice < deviceCount; ++peerDevice)
    {
        if (peerDevice == device)
        {
            continue;
        }
        int peerAccessAvailable = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&peerAccessAvailable, device, peerDevice));
        if (!peerAccessAvailable)
        {
            LOG_WARNING("Device " + std::to_string(device) + " peer access Device " + std::to_string(peerDevice)
                + " is not available.");
            continue;
        }
        ::cudaMemAccessDesc desc{};
        desc.location.type = cudaMemLocationTypeDevice;
        desc.location.id = peerDevice;
        desc.flags = cudaMemAccessFlagsProtReadWrite;
        CUDA_CHECK(cudaMemPoolSetAccess(memPool, &desc, 1));
    }
    auto maxThreshold = std::numeric_limits<std::uint64_t>::max();
    CUDA_CHECK(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &maxThreshold));
}

/// <summary>
/// Get the current reserved memory size in the CUDA memory pool for the specified device.
/// </summary>
/// <param name="device">The device for which to get the reserved memory size.</param>
/// <returns>The current reserved memory size.</returns>
std::size_t BufferManager::memoryPoolReserved(int device)
{
    ::cudaMemPool_t memPool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, device));
    std::size_t reserved = 0;
    CUDA_CHECK(cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, &reserved));
    return reserved;
}

/// <summary>
/// Get the current used memory size in the CUDA memory pool for the specified device.
/// </summary>
/// <param name="device">The device for which to get the used memory size.</param>
/// <returns>The current used memory size.</returns>
std::size_t BufferManager::memoryPoolUsed(int device)
{
    ::cudaMemPool_t memPool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, device));
    std::size_t used = 0;
    CUDA_CHECK(cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, &used));
    return used;
}

/// <summary>
/// Trim the CUDA memory pool for the specified device to the given size.
/// </summary>
/// <param name="device">The device for which to trim the memory pool.</param>
/// <param name="size">The target size to which the memory pool should be trimmed.</param>
void BufferManager::memoryPoolTrimTo(int device, std::size_t size)
{
    ::cudaMemPool_t memPool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, device));
    CUDA_CHECK(cudaMemPoolTrimTo(memPool, size));
}

/// <summary>
/// Get the current reserved memory size in the CUDA memory pool for the associated device.
/// </summary>
/// <returns>The current reserved memory size.</returns>
std::size_t BufferManager::memoryPoolReserved() const
{
    return memoryPoolReserved(mStream->getDevice());
}

/// <summary>
/// Get the current used memory size in the CUDA memory pool for the associated device.
/// </summary>
/// <returns>The current used memory size.</returns>
std::size_t BufferManager::memoryPoolUsed() const
{
    return memoryPoolUsed(mStream->getDevice());
}

/// <summary>
/// Get the current free memory size in the CUDA memory pool for the associated device.
/// </summary>
/// <returns>The current free memory size.</returns>
std::size_t BufferManager::memoryPoolFree() const
{
    return memoryPoolFree(mStream->getDevice());
}

/// <summary>
/// Trim the CUDA memory pool for the associated device to the given size.
/// </summary>
/// <param name="size">The target size to which the memory pool should be trimmed.</param>
void BufferManager::memoryPoolTrimTo(std::size_t size)
{
    mStream->synchronize();
    memoryPoolTrimTo(mStream->getDevice(), size);
}
