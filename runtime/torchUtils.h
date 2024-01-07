
#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "cudaStream.h"
#include "iTensor.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceType.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <algorithm>
#include <initializer_list>
#include <type_traits>
#include <vector>

namespace bitfusion::runtime
{

class TorchUtils
{
public:
    using SizeType = at::IntArrayRef::value_type;

    static std::vector<SizeType> shape(ITensor::Shape const& dims)
    {
        CHECK(dims.nbDims >= 0);
        std::vector<SizeType> shape{};
        shape.reserve(dims.nbDims);
        std::transform(
            dims.d, dims.d + dims.nbDims, std::back_inserter(shape), [](auto x) { return static_cast<SizeType>(x); });
        return shape;
    }

    static ITensor::Shape shape(at::IntArrayRef const& sizes)
    {
        CHECK(sizes.size() <= ITensor::Shape::MAX_DIMS);
        ITensor::Shape shape{static_cast<runtime::SizeType>(sizes.size())};
        using dimType = std::remove_reference_t<decltype(shape.d[0])>;
        for (std::size_t i = 0; i < sizes.size(); ++i)
        {
            CHECK(sizes[i] <= std::numeric_limits<dimType>::max());
            shape.d[i] = static_cast<dimType>(sizes[i]);
        }
        return shape;
    }

    static std::vector<SizeType> makeShape(std::initializer_list<runtime::SizeType> sizes)
    {
        return shape(ITensor::makeShape(sizes));
    }

    static at::Device device(void const* ptr)
    {
        ::cudaPointerAttributes attr{};
        CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
        auto const memoryType = attr.type;
        return (memoryType == ::cudaMemoryTypeDevice || memoryType == ::cudaMemoryTypeManaged)
            ? at::Device{at::kCUDA, static_cast<at::DeviceIndex>(attr.device)}
            : at::Device{at::kCPU};
    }

    static at::ScalarType dataType(IBuffer::DataType dataType)
    {
        switch (dataType)
        {
        case IBuffer::DataType::kFLOAT: return at::ScalarType::Float;
        case IBuffer::DataType::kHALF: return at::ScalarType::Half;
        case IBuffer::DataType::kINT8: return torch::kInt8;
        case IBuffer::DataType::kUINT8: return torch::kUInt8;
        case IBuffer::DataType::kINT32: return torch::kInt32;
        case IBuffer::DataType::kINT64: return torch::kInt64;
        case IBuffer::DataType::kBOOL: return at::ScalarType::Bool;
        case IBuffer::DataType::kFP8: return at::ScalarType::Bits8;
        case IBuffer::DataType::kBF16: return at::ScalarType::BFloat16;
        default: THROW("unsupported data type");
        }
    }

    static IBuffer::DataType dataType(at::ScalarType scalarType)
    {
        switch (scalarType)
        {
        case at::ScalarType::Float: return IBuffer::DataType::kFLOAT;
        case at::ScalarType::Half: return IBuffer::DataType::kHALF;
        case torch::kInt8: return IBuffer::DataType::kINT8;
        case torch::kUInt8: return IBuffer::DataType::kUINT8;
        case torch::kInt32: return IBuffer::DataType::kINT32;
        case torch::kInt64: return IBuffer::DataType::kINT64;
        case at::ScalarType::Bool: return IBuffer::DataType::kBOOL;
        case at::ScalarType::Bits8: return IBuffer::DataType::kFP8;
        case at::ScalarType::BFloat16: return IBuffer::DataType::kBF16;
        default: THROW("unsupported data type");
        }
    }

    static at::DeviceType deviceType(runtime::MemoryType memoryType)
    {
        switch (memoryType)
        {
        case runtime::MemoryType::kGPU: return c10::kCUDA;
        case runtime::MemoryType::kCPU: [[fallthrough]];
        case runtime::MemoryType::kPINNED: [[fallthrough]];
        default: return c10::kCPU;
        }
    }

    static at::cuda::CUDAStream stream(runtime::CudaStream& cudaStream)
    {
        return at::cuda::getStreamFromExternal(cudaStream.get(), static_cast<at::DeviceIndex>(cudaStream.getDevice()));
    }

private:
    TorchUtils() = default;
};

}
