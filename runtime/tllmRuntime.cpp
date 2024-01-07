#include "tllmRuntime.h"
#include "../common/cudaUtils.h"
#include "../common/nvtxUtils.h"
#include "../common/stringUtils.h"
#include "tllmBuffers.h"
#include "tllmLogger.h"

#include <limits>
#include <type_traits>

using namespace bitfusion::runtime;

namespace tc = bitfusion::common;

namespace
{
using DimType = std::remove_reference_t<decltype(std::declval<nvinfer1::Dims>().d[0])>;
static_assert(sizeof(SizeType) >= sizeof(DimType), "SizeType is too small");
static_assert(std::is_signed<SizeType>::value, "SizeType must be signed");

nvinfer1::Dims shapeToDims(std::vector<std::size_t> const& shape)
{
    CHECK(shape.size() <= nvinfer1::Dims::MAX_DIMS);
    nvinfer1::Dims dims;
    auto constexpr dim_max = std::numeric_limits<DimType>::max();
    dims.nbDims = static_cast<std::int32_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        CHECK(shape[i] <= static_cast<std::size_t>(dim_max));
        dims.d[i] = static_cast<DimType>(shape[i]);
    }
    return dims;
}

std::vector<std::size_t> dimsToShape(nvinfer1::Dims const& dims)
{
    CHECK(dims.nbDims >= 0);
    std::vector<std::size_t> shape(dims.nbDims);
    for (std::int32_t i = 0; i < dims.nbDims; ++i)
    {
        CHECK(dims.d[i] >= 0);
        shape[i] = static_cast<std::size_t>(dims.d[i]);
    }
    return shape;
}

bitfusion::runtime::TllmLogger defaultLogger{};

}

TllmRuntime::TllmRuntime(void const* engineData, std::size_t engineSize, nvinfer1::ILogger& logger)
    : mStream(std::make_shared<CudaStream>())
    , mBufferManager{mStream}
    , mRuntime{nvinfer1::createInferRuntime(logger)}
    , mEngine{mRuntime->deserializeCudaEngine(engineData, engineSize)}
{
    CHECK_WITH_INFO(mEngine != nullptr, "Failed to deserialize cuda engine");
    auto const devMemorySize = mEngine->getDeviceMemorySize();
    mEngineBuffer = mBufferManager.gpu(devMemorySize);
}

TllmRuntime::TllmRuntime(void const* engineData, std::size_t engineSize)
    : TllmRuntime{engineData, engineSize, defaultLogger}
{
}

nvinfer1::IExecutionContext& TllmRuntime::addContext(std::int32_t profileIndex)
{
    CHECK(0 <= profileIndex && profileIndex < mEngine->getNbOptimizationProfiles());
    mContexts.emplace_back(mEngine->createExecutionContextWithoutDeviceMemory());
    auto& context = *mContexts.back();
    context.setDeviceMemory(mEngineBuffer->data());
    context.setOptimizationProfileAsync(profileIndex, mStream->get());
    return context;
}

void TllmRuntime::clearContexts()
{
    for (auto& context : mContexts)
    {
        context.reset();
    }
    mContexts.clear();
}

bool TllmRuntime::executeContext(SizeType contextIndex) const
{
    NVTX3_FUNC_RANGE();
    auto& context = getContext(contextIndex);
    return context.enqueueV3(mStream->get());
}

void TllmRuntime::setInputTensors(SizeType contextIndex, TensorMap const& tensorMap)
{
    NVTX3_FUNC_RANGE();
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& context = getContext(contextIndex);
    for (std::int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        auto const name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            NVTX3_SCOPED_RANGE(input_tensor);
            auto pos = tensorMap.find(name);
            if (pos == tensorMap.end())
            {
                auto expectedShape = mEngine->getTensorShape(name);
                THROW(
                    "Input tensor '%s' not found; expected shape: %s", name, ITensor::toString(expectedShape).c_str());
            }
            auto const& tensor = pos->second;
            auto const tensorDtype = tensor->getDataType();
            auto const engineDtype = mEngine->getTensorDataType(name);
            CHECK_WITH_INFO(tensorDtype == engineDtype
                    || (tensorDtype == nvinfer1::DataType::kFP8 && engineDtype == nvinfer1::DataType::kHALF),
                "%s: expected type %d, provided type %d", name, static_cast<std::int32_t>(engineDtype),
                static_cast<std::int32_t>(tensorDtype));

            auto const shapeExpected = mEngine->getTensorShape(name);
            auto const shapeProvided = tensor->getShape();
            CHECK_WITH_INFO(shapeExpected.nbDims == shapeProvided.nbDims, "%s: expected %d dims, provided %d dims",
                name, shapeExpected.nbDims, shapeProvided.nbDims);
            for (SizeType j = 0; j < shapeExpected.nbDims; ++j)
            {
                auto const dimExpected = shapeExpected.d[j];
                auto const dimProvided = shapeProvided.d[j];
                if (dimExpected >= 0 && dimExpected != dimProvided)
                {
                    LOG_WARNING(
                        "%s: expected dim[%d] = %d, provided dim[%d] = %d", name, j, dimExpected, j, dimProvided);
                }
            }
            CHECK_WITH_INFO(context.setInputShape(name, shapeProvided), "Tensor '%s' has invalid shape %s", name,
                ITensor::toString(shapeProvided).c_str());
            auto* const data = tensor->data();
            if (data)
            {
                context.setInputTensorAddress(name, data);
            }
            else
            {
                CHECK_WITH_INFO(tensor->getSize() == 0, std::string("Invalid data for tensor: ") + name);
                if (!mDummyTensor)
                {
                    mDummyTensor = mBufferManager.gpu(ITensor::makeShape({1}));
                }
                context.setInputTensorAddress(name, mDummyTensor->data());
            }
        }
    }

    {
        NVTX3_SCOPED_RANGE(infer_shapes);
        char const* missing;
        auto const nbMissing = context.inferShapes(1, &missing);
        if (nbMissing > 0)
        {
            THROW("Input shape not specified: %s", missing);
        }
        else if (nbMissing < 0)
        {
            THROW("Invalid input shape");
        }
    }

    {
        NVTX3_SCOPED_RANGE(final_checks);
        CHECK_WITH_INFO(context.allInputDimensionsSpecified(), "Input dimensions not specified");
        CHECK_WITH_INFO(context.allInputShapesSpecified(), "Input shapes not specified");
    }
}

void TllmRuntime::setOutputTensors(SizeType contextIndex, TensorMap& tensorMap)
{
    NVTX3_FUNC_RANGE();
    auto& context = getContext(contextIndex);
    for (std::int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        auto const name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            NVTX3_SCOPED_RANGE(output_tensor);
            auto const dims = context.getTensorShape(name);
            auto const engineDtype = mEngine->getTensorDataType(name);
            auto pos = tensorMap.find(name);
            if (pos != tensorMap.end())
            {
                auto const& tensor = pos->second;
                auto const tensorDtype = tensor->getDataType();
                CHECK_WITH_INFO(tensorDtype == engineDtype
                        || (tensorDtype == nvinfer1::DataType::kFP8 && engineDtype == nvinfer1::DataType::kHALF),
                    "%s: expected type %d, provided type %d", name, static_cast<std::int32_t>(engineDtype),
                    static_cast<std::int32_t>(tensorDtype));

                tensor->reshape(dims);
                context.setTensorAddress(name, tensor->data());
            }
            else
            {
                auto tensor = ITensor::SharedPtr(mBufferManager.gpu(dims, engineDtype));
                tensorMap.insert(pos, std::make_pair(name, tensor));
                context.setTensorAddress(name, tensor->data());
            }
        }
    }
}

CudaStream const& TllmRuntime::getStream() const
{
    return *mStream;
}
