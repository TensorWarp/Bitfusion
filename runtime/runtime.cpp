#include "runtime.h"
#include "../common/cudaUtils.h"
#include "../common/nvtxUtils.h"
#include "../common/stringUtils.h"
#include "tllmBuffers.h"
#include "tllmLogger.h"
#include <limits>
#include <type_traits>
#include <memory>
#include <ranges>

using namespace bitfusion::runtime;
namespace tc = bitfusion::common;

namespace
{
    // Define a type alias for convenience in extracting dimension values.
    using DimType = std::remove_reference_t<decltype(std::declval<nvinfer1::Dims>().d[0])>;

    // Static assertions to ensure the compatibility of SizeType and DimType.
    static_assert(sizeof(SizeType) >= sizeof(DimType), "SizeType is too small");
    static_assert(std::is_signed_v<SizeType>, "SizeType must be signed");

    /// <summary>
    /// Converts a vector of size_t values to an nvinfer1::Dims object.
    /// </summary>
    /// <param name="shape">The vector of size_t values representing tensor dimensions.</param>
    /// <returns>An nvinfer1::Dims object representing the tensor dimensions.</returns>
    nvinfer1::Dims shapeToDims(const std::vector<std::size_t>& shape)
    {
        CHECK(shape.size() <= nvinfer1::Dims::MAX_DIMS);
        nvinfer1::Dims dims;
        constexpr DimType dim_max = std::numeric_limits<DimType>::max();
        dims.nbDims = static_cast<std::int32_t>(shape.size());
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            CHECK(shape[i] <= static_cast<std::size_t>(dim_max));
            dims.d[i] = static_cast<DimType>(shape[i]);
        }
        return dims;
    }

    /// <summary>
    /// Converts an nvinfer1::Dims object to a vector of size_t values representing tensor dimensions.
    /// </summary>
    /// <param name="dims">The nvinfer1::Dims object representing tensor dimensions.</param>
    /// <returns>A vector of size_t values representing tensor dimensions.</returns>
    std::vector<std::size_t> dimsToShape(const nvinfer1::Dims& dims)
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

    // Create an instance of the default logger for the bitfusion::runtime::TllmLogger class.
    bitfusion::runtime::TllmLogger defaultLogger{};
}

/// <summary>
/// Constructor for the Runtime class that takes engine data, engine size, and a custom logger.
/// </summary>
/// <param name="engineData">Pointer to the engine data.</param>
/// <param name="engineSize">Size of the engine data.</param>
/// <param name="logger">A custom logger object.</param>
Runtime::Runtime(const void* engineData, std::size_t engineSize, nvinfer1::ILogger& logger)
    : mStream(std::make_shared<CudaStream>()),
    mBufferManager(mStream),
    mRuntime(nvinfer1::createInferRuntime(logger)),
    mEngine(mRuntime->deserializeCudaEngine(engineData, engineSize))
{
    CHECK_WITH_INFO(mEngine != nullptr, "Failed to deserialize cuda engine");
    const auto devMemorySize = mEngine->getDeviceMemorySize();
    mEngineBuffer = mBufferManager.gpu(devMemorySize);
}

/// <summary>
/// Constructor for the Runtime class that takes engine data and engine size,
/// using the default logger.
/// </summary>
/// <param name="engineData">Pointer to the engine data.</param>
/// <param name="engineSize">Size of the engine data.</param>
Runtime::Runtime(const void* engineData, std::size_t engineSize)
    : Runtime(engineData, engineSize, defaultLogger)
{
}

/// <summary>
/// Adds a new execution context for a specific optimization profile to the Runtime.
/// </summary>
/// <param name="profileIndex">The index of the optimization profile to use.</param>
/// <returns>The created execution context.</returns>
nvinfer1::IExecutionContext& Runtime::addContext(std::int32_t profileIndex)
{
    CHECK(0 <= profileIndex && profileIndex < mEngine->getNbOptimizationProfiles());
    mContexts.emplace_back(mEngine->createExecutionContextWithoutDeviceMemory());
    auto& context = *mContexts.back();
    context.setDeviceMemory(mEngineBuffer->data());
    context.setOptimizationProfileAsync(profileIndex, mStream->get());
    return context;
}

/// <summary>
/// Clears all execution contexts associated with the Runtime.
/// </summary>
void Runtime::clearContexts()
{
    mContexts.clear();
}

/// <summary>
/// Executes the specified execution context on the associated stream.
/// </summary>
/// <param name="contextIndex">The index of the execution context to execute.</param>
/// <returns>True if execution is successful; otherwise, false.</returns>
bool Runtime::executeContext(SizeType contextIndex) const
{
    NVTX3_FUNC_RANGE();
    auto& context = getContext(contextIndex);
    return context.enqueueV3(mStream->get());
}

/// <summary>
/// Sets input tensors for a specified execution context using a tensor map.
/// </summary>
/// <param name="contextIndex">The index of the execution context to set input tensors for.</param>
/// <param name="tensorMap">A map of tensor names to their respective tensors.</param>
void Runtime::setInputTensors(SizeType contextIndex, const TensorMap& tensorMap)
{
    NVTX3_FUNC_RANGE();
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& context = getContext(contextIndex);

    // Loop through the input tensors of the engine and set their values based on the provided tensor map.
    for (const std::int32_t i : std::views::iota(0, mEngine->getNbIOTensors()))
    {
        const auto name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            NVTX3_SCOPED_RANGE(input_tensor);
            auto pos = tensorMap.find(name);
            if (pos == tensorMap.end())
            {
                const auto expectedShape = mEngine->getTensorShape(name);
                THROW("Input tensor '%s' not found; expected shape: %s", name,
                    ITensor::toString(expectedShape).c_str());
            }
            const auto& tensor = pos->second;
            const auto tensorDtype = tensor->getDataType();
            const auto engineDtype = mEngine->getTensorDataType(name);
            CHECK_WITH_INFO(tensorDtype == engineDtype ||
                (tensorDtype == nvinfer1::DataType::kFP8 && engineDtype == nvinfer1::DataType::kHALF),
                "%s: expected type %d, provided type %d", name, static_cast<std::int32_t>(engineDtype),
                static_cast<std::int32_t>(tensorDtype));

            const auto shapeExpected = mEngine->getTensorShape(name);
            const auto shapeProvided = tensor->getShape();
            CHECK_WITH_INFO(shapeExpected.nbDims == shapeProvided.nbDims, "%s: expected %d dims, provided %d dims",
                name, shapeExpected.nbDims, shapeProvided.nbDims);

            for (SizeType j = 0; j < shapeExpected.nbDims; ++j)
            {
                const auto dimExpected = shapeExpected.d[j];
                const auto dimProvided = shapeProvided.d[j];
                if (dimExpected >= 0 && dimExpected != dimProvided)
                {
                    LOG_WARNING("%s: expected dim[%d] = %d, provided dim[%d] = %d", name, j, dimExpected, j,
                        dimProvided);
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
                    mDummyTensor = mBufferManager.gpu(ITensor::makeShape({ 1 }));
                }
                context.setInputTensorAddress(name, mDummyTensor->data());
            }
        }
    }

    {
        NVTX3_SCOPED_RANGE(infer_shapes);
        const char* missing;
        const auto nbMissing = context.inferShapes(1, &missing);
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

/// <summary>
/// Sets the output tensors for a specified execution context using a tensor map.
/// </summary>
/// <param name="contextIndex">The index of the execution context to set output tensors for.</param>
/// <param name="tensorMap">A map of tensor names to their respective tensors.</param>
void Runtime::setOutputTensors(SizeType contextIndex, TensorMap& tensorMap)
{
    NVTX3_FUNC_RANGE();
    auto& context = getContext(contextIndex);

    // Loop through the output tensors of the engine and set their values based on the provided tensor map.
    for (const std::int32_t i : std::views::iota(0, mEngine->getNbIOTensors()))
    {
        const auto name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            NVTX3_SCOPED_RANGE(output_tensor);
            const auto dims = context.getTensorShape(name);
            const auto engineDtype = mEngine->getTensorDataType(name);
            auto pos = tensorMap.find(name);

            if (pos != tensorMap.end())
            {
                const auto& tensor = pos->second;
                const auto tensorDtype = tensor->getDataType();
                CHECK_WITH_INFO(tensorDtype == engineDtype ||
                    (tensorDtype == nvinfer1::DataType::kFP8 && engineDtype == nvinfer1::DataType::kHALF),
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

/// <summary>
/// Gets a reference to the CUDA stream associated with the Runtime.
/// </summary>
/// <returns>A reference to the CUDA stream.</returns>
const CudaStream& Runtime::getStream() const
{
    return *mStream;
}
