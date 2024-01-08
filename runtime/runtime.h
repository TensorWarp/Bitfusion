#pragma once

#include "bufferManager.h"
#include "common.h"
#include "iTensor.h"
#include <NvInferRuntime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace bitfusion::runtime
{
class Runtime
{
public:
    using TensorMap = StringPtrMap<ITensor>;

    explicit Runtime(void const* engineData, std::size_t engineSize, nvinfer1::ILogger& logger);

    explicit Runtime(nvinfer1::IHostMemory const& engineBuffer, nvinfer1::ILogger& logger)
        : Runtime{engineBuffer.data(), engineBuffer.size(), logger}
    {
    }

    explicit Runtime(void const* engineData, std::size_t engineSize);

    explicit Runtime(nvinfer1::IHostMemory const& engineBuffer)
        : Runtime{engineBuffer.data(), engineBuffer.size()}
    {
    }

    SizeType getNbContexts() const
    {
        return static_cast<SizeType>(mContexts.size());
    }

    nvinfer1::IExecutionContext& getContext(SizeType contextIndex) const
    {
        return *mContexts.at(contextIndex);
    }

    SizeType getNbProfiles() const
    {
        return static_cast<SizeType>(mEngine->getNbOptimizationProfiles());
    }

    nvinfer1::IExecutionContext& addContext(std::int32_t profileIndex);

    void clearContexts();

    void setInputTensors(SizeType contextIndex, TensorMap const& tensorMap);

    void setOutputTensors(SizeType contextIndex, TensorMap& tensorMap);

    bool executeContext(SizeType contextIndex) const;

    CudaStream const& getStream() const;

    BufferManager::CudaStreamPtr getStreamPtr()
    {
        return mStream;
    }

    nvinfer1::ICudaEngine& getEngine()
    {
        return *mEngine;
    }

    nvinfer1::ICudaEngine const& getEngine() const
    {
        return *mEngine;
    }

    BufferManager& getBufferManager()
    {
        return mBufferManager;
    }

    BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

private:
    BufferManager::CudaStreamPtr mStream;
    BufferManager mBufferManager;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    BufferManager::IBufferPtr mEngineBuffer;
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> mContexts;
    std::unique_ptr<ITensor> mDummyTensor;
};
}
