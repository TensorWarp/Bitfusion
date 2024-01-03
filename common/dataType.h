#pragma once

#include <NvInferRuntime.h>

namespace bitfusion::common
{

    constexpr static size_t getDTypeSize(nvinfer1::DataType type)
    {
        switch (type)
        {
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kINT32: [[fallthrough]];
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kBF16: [[fallthrough]];
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL: [[fallthrough]];
        case nvinfer1::DataType::kUINT8: [[fallthrough]];
        case nvinfer1::DataType::kINT8: [[fallthrough]];
        case nvinfer1::DataType::kFP8: return 1;
        }
        return 0;
    }

}