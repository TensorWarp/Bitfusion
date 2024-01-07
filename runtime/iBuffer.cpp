#include "iBuffer.h"
#include "iTensor.h"
#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "bufferView.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <iostream>
#include <span>
#include <format>

namespace bitfusion::runtime {

    /// <summary>
    /// Determines the memory type of the provided data pointer.
    /// </summary>
    /// <param name="data">The data pointer to analyze.</param>
    /// <returns>The memory type as MemoryType enum.</returns>
    MemoryType IBuffer::memoryType(const void* data) {
        cudaPointerAttributes attributes{};
        CUDA_CHECK(::cudaPointerGetAttributes(&attributes, data));
        switch (attributes.type) {
        case cudaMemoryTypeHost: return MemoryType::kPINNED;
        case cudaMemoryTypeDevice:
        case cudaMemoryTypeManaged: return MemoryType::kGPU;
        case cudaMemoryTypeUnregistered: return MemoryType::kCPU;
        default: THROW("Unsupported memory type");
        }
    }

    /// <summary>
    /// Creates a new BufferView from the given buffer with the specified offset and size.
    /// </summary>
    /// <param name="buffer">The shared pointer to the original buffer.</param>
    /// <param name="offset">The offset within the original buffer.</param>
    /// <param name="size">The size of the new BufferView.</param>
    /// <returns>A unique pointer to the created BufferView.</returns>
    IBuffer::UniquePtr IBuffer::slice(IBuffer::SharedPtr buffer, std::size_t offset, std::size_t size) {
        return std::make_unique<BufferView>(std::move(buffer), offset, size);
    }

    /// <summary>
    /// Wraps a data pointer with a buffer of the specified type, size, and capacity.
    /// </summary>
    /// <param name="data">The data pointer to wrap.</param>
    /// <param name="type">The data type of the buffer.</param>
    /// <param name="size">The initial size of the buffer.</param>
    /// <param name="capacity">The capacity of the buffer.</param>
    /// <returns>A unique pointer to the wrapped buffer.</returns>
    IBuffer::UniquePtr IBuffer::wrap(void* data, nvinfer1::DataType type, std::size_t size, std::size_t capacity) {
        CHECK_WITH_INFO(size <= capacity, "Requested size is larger than capacity");
        auto memoryType = IBuffer::memoryType(data);

        IBuffer::UniquePtr result;
        auto const capacityInBytes = capacity * BufferDataType(type).getSize();
        switch (memoryType) {
        case MemoryType::kPINNED:
            result = std::make_unique<GenericBuffer<PinnedBorrowingAllocator>>(
                capacity, type, PinnedBorrowingAllocator(data, capacityInBytes));
            break;
        case MemoryType::kCPU:
            result = std::make_unique<GenericBuffer<CpuBorrowingAllocator>>(
                capacity, type, CpuBorrowingAllocator(data, capacityInBytes));
            break;
        case MemoryType::kGPU:
            result = std::make_unique<GenericBuffer<GpuBorrowingAllocator>>(
                capacity, type, GpuBorrowingAllocator(data, capacityInBytes));
            break;
        default: THROW("Unknown memory type");
        }
        result->resize(size);
        return result;
    }

    /// <summary>
    /// Overloaded operator for printing IBuffer objects to an ostream.
    /// </summary>
    /// <param name="output">The ostream to output to.</param>
    /// <param name="buffer">The IBuffer object to print.</param>
    /// <returns>The modified ostream.</returns>
    std::ostream& operator<<(std::ostream& output, const IBuffer& buffer) {
        auto data = const_cast<IBuffer&>(buffer).data();
        auto tensor = ITensor::wrap(data, buffer.getDataType(),
            ITensor::makeShape({ static_cast<SizeType>(buffer.getSize()) }), buffer.getCapacity());
        return output << *tensor;
    }

    /// <summary>
    /// Gets the name of the data type associated with this buffer.
    /// </summary>
    /// <returns>The name of the data type as a string.</returns>
    const char* IBuffer::getDataTypeName() const {
        switch (getDataType()) {
        case nvinfer1::DataType::kINT64: return DataTypeTraits<nvinfer1::DataType::kINT64>::name;
        case nvinfer1::DataType::kINT32: return DataTypeTraits<nvinfer1::DataType::kINT32>::name;
        case nvinfer1::DataType::kFLOAT: return DataTypeTraits<nvinfer1::DataType::kFLOAT>::name;
        case nvinfer1::DataType::kHALF: return DataTypeTraits<nvinfer1::DataType::kHALF>::name;
        case nvinfer1::DataType::kBOOL: return DataTypeTraits<nvinfer1::DataType::kBOOL>::name;
        case nvinfer1::DataType::kUINT8: return DataTypeTraits<nvinfer1::DataType::kUINT8>::name;
        case nvinfer1::DataType::kINT8: return DataTypeTraits<nvinfer1::DataType::kINT8>::name;
        case nvinfer1::DataType::kFP8: return "kFP8";
        case nvinfer1::DataType::kBF16: return "kBF16";
        }
        THROW("Unknown data type");
    }

    /// <summary>
    /// Gets the name of the memory type associated with this buffer.
    /// </summary>
    /// <returns>The name of the memory type as a string.</returns>
    const char* IBuffer::getMemoryTypeName() const {
        switch (getMemoryType()) {
        case MemoryType::kPINNED: return MemoryTypeString<MemoryType::kPINNED>::value;
        case MemoryType::kCPU: return MemoryTypeString<MemoryType::kCPU>::value;
        case MemoryType::kGPU: return MemoryTypeString<MemoryType::kGPU>::value;
        }
        THROW("Unknown memory type");
    }
}