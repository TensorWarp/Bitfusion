
#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/logger.h"
#include "common.h"
#include "cudaStream.h"
#include "iBuffer.h"
#include "iTensor.h"
#include "memoryCounters.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <limits>
#include <memory>
#include <type_traits>

namespace bitfusion::runtime
{

template <typename TDerived, MemoryType memoryType>
class BaseAllocator
{
public:
    using ValueType = void;
    using PointerType = ValueType*;
    using SizeType = std::size_t;

    PointerType allocate(SizeType n)
    {
        PointerType ptr{};
        static_cast<TDerived*>(this)->allocateImpl(&ptr, n);
        MemoryCounters::getInstance().allocate<memoryType>(n);
        return ptr;
    }

    void deallocate(PointerType ptr, SizeType n)
    {
        if (ptr)
        {
            static_cast<TDerived*>(this)->deallocateImpl(ptr, n);
            MemoryCounters::getInstance().deallocate<memoryType>(n);
        }
    }

    [[nodiscard]] MemoryType constexpr getMemoryType() const
    {
        return memoryType;
    }
};

class CudaAllocator : public BaseAllocator<CudaAllocator, MemoryType::kGPU>
{
    friend class BaseAllocator<CudaAllocator, MemoryType::kGPU>;

public:
    CudaAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, SizeType n)
    {
        CUDA_CHECK(::cudaMalloc(ptr, n));
    }

    void deallocateImpl(
        PointerType ptr, [[gnu::unused]] SizeType n)
    {
        CUDA_CHECK(::cudaFree(ptr));
    }
};

class CudaAllocatorAsync : public BaseAllocator<CudaAllocatorAsync, MemoryType::kGPU>
{
    friend class BaseAllocator<CudaAllocatorAsync, MemoryType::kGPU>;

public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;

    explicit CudaAllocatorAsync(CudaStreamPtr stream)
        : mCudaStream(std::move(stream))
    {
        CHECK_WITH_INFO(static_cast<bool>(mCudaStream), "Undefined CUDA stream");
    }

    [[nodiscard]] CudaStreamPtr getCudaStream() const
    {
        return mCudaStream;
    }

protected:
    void allocateImpl(PointerType* ptr, SizeType n)
    {
        CUDA_CHECK(::cudaMallocAsync(ptr, n, mCudaStream->get()));
    }

    void deallocateImpl(PointerType ptr, [[gnu::unused]] SizeType n)
    {
        CUDA_CHECK(::cudaFreeAsync(ptr, mCudaStream->get()));
    }

private:
    CudaStreamPtr mCudaStream;
};

class PinnedAllocator : public BaseAllocator<PinnedAllocator, MemoryType::kPINNED>
{
    friend class BaseAllocator<PinnedAllocator, MemoryType::kPINNED>;

public:
    PinnedAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, SizeType n)
    {
        CUDA_CHECK(::cudaHostAlloc(ptr, n, cudaHostAllocDefault));
    }

    void deallocateImpl(
        PointerType ptr, [[gnu::unused]] SizeType n)
    {
        CUDA_CHECK(::cudaFreeHost(ptr));
    }
};

class HostAllocator : public BaseAllocator<HostAllocator, MemoryType::kCPU>
{
    friend class BaseAllocator<HostAllocator, MemoryType::kCPU>;

public:
    HostAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, SizeType n)
    {
        *ptr = std::malloc(n);
        if (*ptr == nullptr)
        {
            throw std::bad_alloc();
        }
    }

    void deallocateImpl(
        PointerType ptr, [[gnu::unused]] SizeType n)
    {
        std::free(ptr);
    }
};

template <MemoryType memoryType>
class BorrowingAllocator : public BaseAllocator<BorrowingAllocator<memoryType>, memoryType>
{
    friend class BaseAllocator<BorrowingAllocator<memoryType>, memoryType>;

public:
    using Base = BaseAllocator<BorrowingAllocator<memoryType>, memoryType>;
    using typename Base::PointerType;
    using typename Base::SizeType;

    BorrowingAllocator(void* ptr, SizeType capacity)
        : mPtr(ptr)
        , mCapacity(capacity)
    {
        CHECK_WITH_INFO(capacity == 0 || static_cast<bool>(mPtr), "Undefined pointer");
        CHECK_WITH_INFO(mCapacity >= 0, "Capacity must be non-negative");
    }

protected:
    void allocateImpl(PointerType* ptr, SizeType n)
    {
        if (n <= mCapacity)
        {
            *ptr = mPtr;
        }
        else
        {
            throw std::bad_alloc();
        }
    }

    void deallocateImpl(
        [[gnu::unused]] PointerType ptr, [[gnu::unused]] SizeType n)
    {
    }

private:
    typename Base::PointerType mPtr;
    typename Base::SizeType mCapacity;
};

using CpuBorrowingAllocator = BorrowingAllocator<MemoryType::kCPU>;
using GpuBorrowingAllocator = BorrowingAllocator<MemoryType::kGPU>;
using PinnedBorrowingAllocator = BorrowingAllocator<MemoryType::kPINNED>;


template <typename TAllocator>
class GenericBuffer : virtual public IBuffer
{
public:
    using AllocatorType = TAllocator;

    explicit GenericBuffer(nvinfer1::DataType type, TAllocator allocator = {})
        : GenericBuffer{0, type, std::move(allocator)} {};

    explicit GenericBuffer(
        std::size_t size, nvinfer1::DataType type, TAllocator allocator = {})
        : GenericBuffer{size, size, type, std::move(allocator)} {};

    GenericBuffer(GenericBuffer&& buf) noexcept
        : mSize{buf.mSize}
        , mCapacity{buf.mCapacity}
        , mType{buf.mType}
        , mAllocator{std::move(buf.mAllocator)}
        , mBuffer{buf.mBuffer}
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf) noexcept
    {
        if (this != &buf)
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mAllocator = std::move(buf.mAllocator);
            mBuffer = buf.mBuffer;
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    void* data() override
    {
        return LIKELY(mSize > 0) ? mBuffer : nullptr;
    }

    [[nodiscard]] void const* data() const override
    {
        return LIKELY(mSize > 0) ? mBuffer : nullptr;
    }

    [[nodiscard]] std::size_t getSize() const override
    {
        return mSize;
    }

    [[nodiscard]] std::size_t getCapacity() const override
    {
        return mCapacity;
    }

    [[nodiscard]] nvinfer1::DataType getDataType() const override
    {
        return mType;
    }

    [[nodiscard]] MemoryType getMemoryType() const override
    {
        return mAllocator.getMemoryType();
    }

    void resize(std::size_t newSize) override
    {
        if (mCapacity < newSize)
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
            mBuffer = mAllocator.allocate(toBytes(newSize));
            mCapacity = newSize;
        }
        mSize = newSize;
    }

    void release() override
    {
        mAllocator.deallocate(mBuffer, toBytes(mCapacity));
        mSize = 0;
        mCapacity = 0;
        mBuffer = nullptr;
    }

    ~GenericBuffer() override
    {
        try
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
        }
        catch (std::exception& e)
        {
            LOG_EXCEPTION(e);
        }
    }

protected:
    explicit GenericBuffer(std::size_t size, std::size_t capacity, nvinfer1::DataType type, TAllocator allocator = {})
        : mSize{size}
        , mCapacity{capacity}
        , mType{type}
        , mAllocator{std::move(allocator)}
        , mBuffer{capacity > 0 ? mAllocator.allocate(toBytes(capacity)) : nullptr}
    {
        CHECK(size <= capacity);
        CHECK(capacity == 0 || size > 0);
    }

private:
    std::size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    TAllocator mAllocator;
    void* mBuffer;
};

using DeviceBuffer = GenericBuffer<CudaAllocatorAsync>;
using HostBuffer = GenericBuffer<HostAllocator>;
using PinnedBuffer = GenericBuffer<PinnedAllocator>;

template <typename T>
typename std::make_unsigned<T>::type nonNegative(T value)
{
    CHECK_WITH_INFO(value >= 0, "Value must be non-negative");
    return static_cast<typename std::make_unsigned<T>::type>(value);
}

template <typename TAllocator>
class GenericTensor : virtual public ITensor, public GenericBuffer<TAllocator>
{
public:
    using Base = GenericBuffer<TAllocator>;

    explicit GenericTensor(nvinfer1::DataType type, TAllocator allocator = {})
        : Base{type, std::move(allocator)}
    {
        mDims.nbDims = 0;
    }

    explicit GenericTensor(nvinfer1::Dims const& dims, nvinfer1::DataType type, TAllocator allocator = {})
        : Base{nonNegative(volume(dims)), type, std::move(allocator)}
        , mDims{dims}
    {
    }

    explicit GenericTensor(
        nvinfer1::Dims const& dims, std::size_t capacity, nvinfer1::DataType type, TAllocator allocator = {})
        : Base{nonNegative(volume(dims)), capacity, type, std::move(allocator)}
        , mDims{dims}
    {
    }

    GenericTensor(GenericTensor&& tensor) noexcept
        : Base{std::move(tensor)}
        , mDims{tensor.dims}
    {
        tensor.mDims.nbDims = 0;
    }

    GenericTensor& operator=(GenericTensor&& tensor) noexcept
    {
        if (this != &tensor)
        {
            Base::operator=(std::move(tensor));
            mDims = tensor.dims;
            tensor.mDims.nbDims = 0;
        }
        return *this;
    }

    [[nodiscard]] nvinfer1::Dims const& getShape() const override
    {
        return mDims;
    }

    void reshape(nvinfer1::Dims const& dims) override
    {
        Base::resize(nonNegative(volume(dims)));
        mDims = dims;
    }

    void resize(std::size_t newSize) override
    {
        ITensor::resize(newSize);
    }

    void release() override
    {
        Base::release();
        mDims.nbDims = 0;
    }

private:
    nvinfer1::Dims mDims{};
};

using DeviceTensor = GenericTensor<CudaAllocatorAsync>;
using HostTensor = GenericTensor<HostAllocator>;
using PinnedTensor = GenericTensor<PinnedAllocator>;

}
