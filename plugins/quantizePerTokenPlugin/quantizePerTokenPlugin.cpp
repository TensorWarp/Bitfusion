#include "quantizePerTokenPlugin.h"
#include "../../kernels/quantization.h"

using namespace nvinfer1;
using namespace bitfusion::kernels;
using bitfusion::plugins::QuantizePerTokenPluginCreator;
using bitfusion::plugins::QuantizePerTokenPlugin;

static const char* QUANTIZE_PER_TOKEN_PLUGIN_VERSION{"1"};
static const char* QUANTIZE_PER_TOKEN_PLUGIN_NAME{"QuantizePerToken"};
PluginFieldCollection QuantizePerTokenPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> QuantizePerTokenPluginCreator::mPluginAttributes;

QuantizePerTokenPlugin::QuantizePerTokenPlugin() {}

QuantizePerTokenPlugin::QuantizePerTokenPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    CHECK(d == a + length);
}

nvinfer1::IPluginV2DynamicExt* QuantizePerTokenPlugin::clone() const noexcept
{
    auto* plugin = new QuantizePerTokenPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs QuantizePerTokenPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        CHECK(nbInputs == 1);
        CHECK(outputIndex < 2);
        if (outputIndex == 0)
        {
            return inputs[0];
        }

        DimsExprs ret;
        ret.nbDims = inputs[0].nbDims;
        for (int ii = 0; ii < ret.nbDims - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        ret.d[ret.nbDims - 1] = exprBuilder.constant(1);
        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool QuantizePerTokenPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
            && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        assert(false);
        return false;
    }
}

void QuantizePerTokenPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t QuantizePerTokenPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int QuantizePerTokenPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    int64_t m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    const int64_t k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        invokePerTokenQuantization<float>(reinterpret_cast<int8_t*>(outputs[0]),
            reinterpret_cast<const float*>(inputs[0]), m, k, reinterpret_cast<float*>(outputs[1]), stream);
    }
    else
    {
        invokePerTokenQuantization<half>(reinterpret_cast<int8_t*>(outputs[0]),
            reinterpret_cast<const half*>(inputs[0]), m, k, reinterpret_cast<float*>(outputs[1]), stream);
    }

    return 0;
}

nvinfer1::DataType QuantizePerTokenPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    CHECK(nbInputs == 1);
    CHECK(index < 2);
    return index == 0 ? nvinfer1::DataType::kINT8 : nvinfer1::DataType::kFLOAT;
}


const char* QuantizePerTokenPlugin::getPluginType() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_NAME;
}

const char* QuantizePerTokenPlugin::getPluginVersion() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_VERSION;
}

int QuantizePerTokenPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int QuantizePerTokenPlugin::initialize() noexcept
{
    return 0;
}

void QuantizePerTokenPlugin::terminate() noexcept {}

size_t QuantizePerTokenPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void QuantizePerTokenPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    assert(d == a + getSerializationSize());
}

void QuantizePerTokenPlugin::destroy() noexcept
{
    delete this;
}


QuantizePerTokenPluginCreator::QuantizePerTokenPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QuantizePerTokenPluginCreator::getPluginName() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_NAME;
}

const char* QuantizePerTokenPluginCreator::getPluginVersion() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_VERSION;
}

const PluginFieldCollection* QuantizePerTokenPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QuantizePerTokenPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        auto* obj = new QuantizePerTokenPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QuantizePerTokenPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new QuantizePerTokenPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
