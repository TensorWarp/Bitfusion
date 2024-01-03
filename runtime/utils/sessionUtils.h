
#pragma once

#include "../../runtime/bufferManager.h"
#include "../../runtime/worldConfig.h"

#include <NvInferRuntime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

namespace bitfusion::runtime
{
class TllmRuntime;

namespace utils
{

int initDevice(WorldConfig const& worldConfig);

std::vector<uint8_t> loadEngine(std::string const& enginePath);

template <typename TInputContainer, typename TFunc>
auto transformVector(TInputContainer const& input, TFunc func)
    -> std::vector<std::remove_reference_t<decltype(func(input.front()))>>
{
    std::vector<std::remove_reference_t<decltype(func(input.front()))>> output{};
    output.reserve(input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(output), func);
    return output;
}

std::vector<ITensor::SharedPtr> createBufferVector(TllmRuntime const& runtime, SizeType indexOffset,
    SizeType numBuffers, std::string const& prefix, MemoryType memType);

std::vector<ITensor::SharedPtr> createBufferVector(
    TllmRuntime const& runtime, SizeType numBuffers, MemoryType memType, nvinfer1::DataType dtype);

void reshapeBufferVector(std::vector<ITensor::SharedPtr>& vector, nvinfer1::Dims const& shape);

std::vector<ITensor::SharedPtr> sliceBufferVector(
    std::vector<ITensor::SharedPtr> const& vector, SizeType offset, SizeType size);

void insertTensorVector(StringPtrMap<ITensor>& map, std::string const& key, std::vector<ITensor::SharedPtr> const& vec,
    SizeType indexOffset);

void insertTensorSlices(
    StringPtrMap<ITensor>& map, std::string const& key, ITensor::SharedPtr const& tensor, SizeType indexOffset);

void setRawPointers(ITensor& pointers, ITensor::SharedPtr const& input, int32_t pointersSlot, int32_t inputSlot);

void setRawPointers(ITensor& pointers, ITensor::SharedPtr const& input);

void scatterBufferReplace(ITensor::SharedPtr& tensor, SizeType beamWidth, BufferManager& manager);

void tileBufferReplace(ITensor::SharedPtr& tensor, SizeType beamWidth, BufferManager& manager);

void tileCpuBufferReplace(ITensor::SharedPtr& tensor, SizeType beamWidth, BufferManager& manager);

}
}
