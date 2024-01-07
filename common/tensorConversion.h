#pragma once

#include "tensor.h"
#include "../runtime/iTensor.h"
#include <NvInferRuntime.h>

#include <algorithm>

namespace bitfusion::common::conversion
{

    inline DataType toTllmDataType(nvinfer1::DataType type)
    {
        switch (type)
        {
        case nvinfer1::DataType::kFLOAT: return DataType::TYPE_FP32;
        case nvinfer1::DataType::kHALF: return DataType::TYPE_FP16;
        case nvinfer1::DataType::kBF16: return DataType::TYPE_BF16;
        case nvinfer1::DataType::kFP8: return DataType::TYPE_FP8_E4M3;
        case nvinfer1::DataType::kINT8: return DataType::TYPE_INT8;
        case nvinfer1::DataType::kUINT8: return DataType::TYPE_UINT8;
        case nvinfer1::DataType::kINT32: return DataType::TYPE_INT32;
        case nvinfer1::DataType::kINT64: return DataType::TYPE_INT64;
        case nvinfer1::DataType::kBOOL: return DataType::TYPE_BOOL;
        default: THROW("Unsupported data type: %d", static_cast<int>(type));
        }
    }

    inline MemoryType toTllmMemoryType(runtime::MemoryType type)
    {
        switch (type)
        {
        case runtime::MemoryType::kGPU: return MemoryType::MEMORY_GPU;
        case runtime::MemoryType::kCPU: return MemoryType::MEMORY_CPU;
        case runtime::MemoryType::kPINNED: return MemoryType::MEMORY_CPU_PINNED;
        default: THROW("Unsupported memory type: %d", static_cast<int>(type));
        }
    }

    inline Tensor toTllmTensor(runtime::ITensor const& tensor)
    {
        MemoryType memoryType = toTllmMemoryType(tensor.getMemoryType());
        DataType dataType = toTllmDataType(tensor.getDataType());

        auto const& dims = tensor.getShape();
        std::vector<std::size_t> shape(dims.d, dims.d + dims.nbDims);

        auto* data = tensor.data();

        return Tensor(memoryType, dataType, shape, data);
    }

    inline Tensor toTllmTensor(runtime::IBuffer const& buffer)
    {
        MemoryType memoryType = toTllmMemoryType(buffer.getMemoryType());
        DataType dataType = toTllmDataType(buffer.getDataType());
        std::vector<std::size_t> shape{ buffer.getSize() };
        auto* data = buffer.data();

        return Tensor(memoryType, dataType, shape, data);
    }

    template <typename T>
    Tensor toTllmTensor(MemoryType memoryType, std::vector<std::size_t> const& shape, T* data)
    {
        return Tensor{ memoryType, getTensorType<T>(), shape, data };
    }

    template <typename T>
    Tensor toTllmTensor(std::vector<T> const& data)
    {
        return Tensor{ MemoryType::MEMORY_CPU, getTensorType<T>(), {data.size()}, data.data() };
    }

    template <typename T>
    Tensor scalarToTllmTensor(T& data)
    {
        return Tensor{ MemoryType::MEMORY_CPU, getTensorType<T>(), {1}, &data };
    }

}
