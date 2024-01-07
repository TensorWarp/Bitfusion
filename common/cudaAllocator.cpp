#include "cudaAllocator.h"
#include "cudaUtils.h"

#include <unordered_map>
#include <utility>

namespace bitfusion::common {
    namespace tr = bitfusion::runtime;

    /// <summary>
    /// Constructor for the CudaAllocator class.
    /// </summary>
    /// <param name="bufferManager">A BufferManager instance to manage GPU buffers.</param>
    CudaAllocator::CudaAllocator(tr::BufferManager bufferManager)
        : mBufferManager(std::move(bufferManager))
    {
    }

    /// <summary>
    /// Determines the type of reallocation needed based on the current and requested sizes.
    /// </summary>
    /// <param name="ptr">Pointer to the current memory location.</param>
    /// <param name="size">Requested size of memory.</param>
    /// <returns>The type of reallocation needed (INCREASE, REUSE, or DECREASE).</returns>
    ReallocType CudaAllocator::reallocType(const void* ptr, size_t size) const
    {
        CHECK(contains(ptr));
        const auto currentSize = mPointerMapping.at(ptr)->getSize();
        LOG_DEBUG("current_buffer_size: %zu, original buffer: %p, new buffer: %zu", currentSize, ptr, size);
        return (currentSize < size) ? ReallocType::INCREASE :
            (currentSize == size) ? ReallocType::REUSE :
            ReallocType::DECREASE;
    }

    /// <summary>
    /// Allocates GPU memory and optionally sets it to zero.
    /// </summary>
    /// <param name="size">Size of memory to allocate.</param>
    /// <param name="setZero">Flag indicating whether to initialize the memory to zero.</param>
    /// <returns>A pointer to the allocated GPU memory.</returns>
    void* CudaAllocator::malloc(size_t size, bool setZero)
    {
        LOG_TRACE(__PRETTY_FUNCTION__);
        auto bufferPtr = mBufferManager.gpu(size);
        if (setZero)
        {
            mBufferManager.setZero(*bufferPtr);
        }
        void* ptr = bufferPtr->data();
        LOG_DEBUG("malloc buffer %p with size %zu", ptr, size);
        mPointerMapping.emplace(ptr, std::move(bufferPtr));
        return ptr;
    }

    /// <summary>
    /// Frees GPU memory associated with the provided pointer.
    /// </summary>
    /// <param name="ptr">Pointer to the GPU memory to free.</param>
    void CudaAllocator::free(void** ptr)
    {
        LOG_TRACE(__PRETTY_FUNCTION__);
        mPointerMapping.erase(*ptr);
        *ptr = nullptr;
    }

    /// <summary>
    /// Sets GPU memory to a specified value asynchronously.
    /// </summary>
    /// <param name="ptr">Pointer to the GPU memory to set.</param>
    /// <param name="val">Value to set in the memory.</param>
    /// <param name="size">Size of the memory to set.</param>
    void CudaAllocator::memSet(void* ptr, int val, size_t size)
    {
        check_cuda_error(cudaMemsetAsync(ptr, val, size, mBufferManager.getStream().get()));
    }
}