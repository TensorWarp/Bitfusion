#include "bufferManager.h"
#include "../common/assert.h"
#include "tllmBuffers.h"

#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <memory>
#include <unordered_set>

using namespace bitfusion::runtime;

namespace tc = bitfusion::common;

BufferManager::BufferManager(CudaStreamPtr stream)
    : mStream{std::move(stream)}
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

BufferManager::IBufferPtr BufferManager::gpu(std::size_t size, nvinfer1::DataType type) const
{
    return std::make_unique<DeviceBuffer>(size, type, CudaAllocatorAsync{mStream});
}

BufferManager::ITensorPtr BufferManager::gpu(nvinfer1::Dims dims, nvinfer1::DataType type) const
{
    return std::make_unique<DeviceTensor>(dims, type, CudaAllocatorAsync{mStream});
}

BufferManager::IBufferPtr BufferManager::cpu(std::size_t size, nvinfer1::DataType type)
{
    return std::make_unique<HostBuffer>(size, type);
}

BufferManager::ITensorPtr BufferManager::cpu(nvinfer1::Dims dims, nvinfer1::DataType type)
{
    return std::make_unique<HostTensor>(dims, type);
}

BufferManager::IBufferPtr BufferManager::pinned(std::size_t size, nvinfer1::DataType type)
{
    return std::make_unique<PinnedBuffer>(size, type);
}

BufferManager::ITensorPtr BufferManager::pinned(nvinfer1::Dims dims, nvinfer1::DataType type)
{
    return std::make_unique<PinnedTensor>(dims, type);
}

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

void BufferManager::copy(void const* src, IBuffer& dst, MemoryType srcType) const
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

void BufferManager::copy(IBuffer const& src, void* dst, MemoryType dstType) const
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

void BufferManager::copy(IBuffer const& src, IBuffer& dst) const
{
    CHECK_WITH_INFO(src.getDataType() == dst.getDataType(),
        tc::fmtstr("Incompatible data types: %s != %s", src.getDataTypeName(), dst.getDataTypeName()));
    CHECK_WITH_INFO(src.getSizeInBytes() == dst.getSizeInBytes(),
        tc::fmtstr("Incompatible buffer sizes: %lu != %lu", src.getSizeInBytes(), dst.getSizeInBytes()));
    copy(src, dst.data(), dst.getMemoryType());
}

BufferManager::IBufferPtr BufferManager::allocate(
    MemoryType memoryType, std::size_t size, nvinfer1::DataType type) const
{
    switch (memoryType)
    {
    case MemoryType::kCPU: return cpu(size, type);
    case MemoryType::kGPU: return gpu(size, type);
    case MemoryType::kPINNED: return pinned(size, type);
    default: THROW("Unknown memory type");
    }
}

BufferManager::ITensorPtr BufferManager::allocate(
    MemoryType memoryType, nvinfer1::Dims dims, nvinfer1::DataType type) const
{
    switch (memoryType)
    {
    case MemoryType::kCPU: return cpu(dims, type);
    case MemoryType::kGPU: return gpu(dims, type);
    case MemoryType::kPINNED: return pinned(dims, type);
    default: THROW("Unknown memory type");
    }
}

BufferManager::IBufferPtr BufferManager::copyFrom(IBuffer const& src, MemoryType memoryType) const
{
    auto dst = allocate(memoryType, src.getSize(), src.getDataType());
    copy(src, *dst);
    return dst;
}

BufferManager::ITensorPtr BufferManager::copyFrom(ITensor const& src, MemoryType memoryType) const
{
    auto dst = allocate(memoryType, src.getShape(), src.getDataType());
    copy(src, *dst);
    return dst;
}

CudaStream const& BufferManager::getStream() const
{
    return *mStream;
}

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

std::size_t BufferManager::memoryPoolReserved(int device)
{
    ::cudaMemPool_t memPool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, device));
    std::size_t reserved = 0;
    CUDA_CHECK(cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, &reserved));
    return reserved;
}

std::size_t BufferManager::memoryPoolUsed(int device)
{
    ::cudaMemPool_t memPool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, device));
    std::size_t used = 0;
    CUDA_CHECK(cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, &used));
    return used;
}

void BufferManager::memoryPoolTrimTo(int device, std::size_t size)
{
    ::cudaMemPool_t memPool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, device));
    CUDA_CHECK(cudaMemPoolTrimTo(memPool, size));
}

std::size_t BufferManager::memoryPoolReserved() const
{
    return memoryPoolReserved(mStream->getDevice());
}

std::size_t BufferManager::memoryPoolUsed() const
{
    return memoryPoolUsed(mStream->getDevice());
}

std::size_t BufferManager::memoryPoolFree() const
{
    return memoryPoolFree(mStream->getDevice());
}

void BufferManager::memoryPoolTrimTo(std::size_t size)
{
    mStream->synchronize();
    memoryPoolTrimTo(mStream->getDevice(), size);
}
