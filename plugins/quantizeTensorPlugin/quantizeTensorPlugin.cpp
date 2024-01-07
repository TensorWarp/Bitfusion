#include "quantizeTensorPlugin.h"
#include "../../kernels/quantization.h"

using namespace nvinfer1;
using namespace bitfusion::kernels;
using bitfusion::plugins::QuantizeTensorPluginCreator;
using bitfusion::plugins::QuantizeTensorPlugin;

static const char* QUANTIZE_TENSOR_PLUGIN_VERSION{"1"};
static const char* QUANTIZE_TENSOR_PLUGIN_NAME{"QuantizeTensor"};
PluginFieldCollection QuantizeTensorPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> QuantizeTensorPluginCreator::mPluginAttributes;

QuantizeTensorPlugin::QuantizeTensorPlugin() {}

QuantizeTensorPlugin::QuantizeTensorPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    CHECK(d == a + length);
}

nvinfer1::IPluginV2DynamicExt* QuantizeTensorPlugin::clone() const noexcept
{
    return new QuantizeTensorPlugin(*this);
}

nvinfer1::DimsExprs QuantizeTensorPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        CHECK(nbInputs == 2);
        CHECK(outputIndex < 1);
        return inputs[0];
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool QuantizeTensorPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
            && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        CHECK(false);
        return false;
    }
}

void QuantizeTensorPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t QuantizeTensorPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int QuantizeTensorPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    int64_t numElts = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims; ++ii)
    {
        numElts *= inputDesc[0].dims.d[ii];
    }

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        invokeQuantization<float>(reinterpret_cast<int8_t*>(outputs[0]), reinterpret_cast<const float*>(inputs[0]),
            numElts, reinterpret_cast<const float*>(inputs[1]), stream, mProp.maxGridSize[0]);
    }
    else
    {
        invokeQuantization<half>(reinterpret_cast<int8_t*>(outputs[0]), reinterpret_cast<const half*>(inputs[0]),
            numElts, reinterpret_cast<const float*>(inputs[1]), stream, mProp.maxGridSize[0]);
    }

    return 0;
}

nvinfer1::DataType QuantizeTensorPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    CHECK(nbInputs == 2);
    CHECK(index == 0);
    return nvinfer1::DataType::kINT8;
}


const char* QuantizeTensorPlugin::getPluginType() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_NAME;
}

const char* QuantizeTensorPlugin::getPluginVersion() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_VERSION;
}

int QuantizeTensorPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int QuantizeTensorPlugin::initialize() noexcept
{
    int deviceId = 0;
    bitfusion::common::check_cuda_error(cudaGetDevice(&deviceId));
    bitfusion::common::check_cuda_error(cudaGetDeviceProperties(&mProp, deviceId));
    return 0;
}

void QuantizeTensorPlugin::terminate() noexcept {}

size_t QuantizeTensorPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void QuantizeTensorPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    assert(d == a + getSerializationSize());
}

void QuantizeTensorPlugin::destroy() noexcept
{
    delete this;
}


QuantizeTensorPluginCreator::QuantizeTensorPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QuantizeTensorPluginCreator::getPluginName() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_NAME;
}

const char* QuantizeTensorPluginCreator::getPluginVersion() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_VERSION;
}

const PluginFieldCollection* QuantizeTensorPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QuantizeTensorPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        auto* obj = new QuantizeTensorPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QuantizeTensorPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new QuantizeTensorPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
