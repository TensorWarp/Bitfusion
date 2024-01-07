
#include "iBuffer.h"
#include "iTensor.h"

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "bufferView.h"

#include <cuda_runtime_api.h>

#include <memory>

using namespace bitfusion::runtime;

MemoryType IBuffer::memoryType(void const* data)
{
    cudaPointerAttributes attributes{};
    CUDA_CHECK(::cudaPointerGetAttributes(&attributes, data));
    switch (attributes.type)
    {
    case cudaMemoryTypeHost: return MemoryType::kPINNED;
    case cudaMemoryTypeDevice:
    case cudaMemoryTypeManaged: return MemoryType::kGPU;
    case cudaMemoryTypeUnregistered: return MemoryType::kCPU;
    default: THROW("Unsupported memory type");
    }
}

IBuffer::UniquePtr IBuffer::slice(IBuffer::SharedPtr buffer, std::size_t offset, std::size_t size)
{
    return std::make_unique<BufferView>(std::move(buffer), offset, size);
}

IBuffer::UniquePtr IBuffer::wrap(void* data, nvinfer1::DataType type, std::size_t size, std::size_t capacity)
{
    CHECK_WITH_INFO(size <= capacity, "Requested size is larger than capacity");
    auto memoryType = IBuffer::memoryType(data);

    IBuffer::UniquePtr result;
    auto const capacityInBytes = capacity * BufferDataType(type).getSize();
    switch (memoryType)
    {
    case MemoryType::kPINNED:
        result.reset(new GenericBuffer<PinnedBorrowingAllocator>(
            capacity, type, PinnedBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kCPU:
        result.reset(
            new GenericBuffer<CpuBorrowingAllocator>(capacity, type, CpuBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kGPU:
        result.reset(
            new GenericBuffer<GpuBorrowingAllocator>(capacity, type, GpuBorrowingAllocator(data, capacityInBytes)));
        break;
    default: THROW("Unknown memory type");
    }
    result->resize(size);
    return result;
}

std::ostream& bitfusion::runtime::operator<<(std::ostream& output, IBuffer const& buffer)
{
    auto data = const_cast<IBuffer&>(buffer).data();
    auto tensor = ITensor::wrap(data, buffer.getDataType(),
        ITensor::makeShape({static_cast<SizeType>(buffer.getSize())}), buffer.getCapacity());
    return output << *tensor;
}

char const* IBuffer::getDataTypeName() const
{
    switch (getDataType())
    {
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

char const* IBuffer::getMemoryTypeName() const
{
    switch (getMemoryType())
    {
    case MemoryType::kPINNED: return MemoryTypeString<MemoryType::kPINNED>::value;
    case MemoryType::kCPU: return MemoryTypeString<MemoryType::kCPU>::value;
    case MemoryType::kGPU: return MemoryTypeString<MemoryType::kGPU>::value;
    }
    THROW("Unknown memory type");
}
