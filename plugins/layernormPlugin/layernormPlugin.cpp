
#include "layernormPlugin.h"
#include "../../common/assert.h"
#include "../../kernels/layernormKernels.h"

using namespace nvinfer1;
using namespace bitfusion::kernels;
using namespace bitfusion::common;
using bitfusion::plugins::LayernormPluginCreator;
using bitfusion::plugins::LayernormPlugin;

static const char* LAYERNORM_PLUGIN_VERSION{"1"};
static const char* LAYERNORM_PLUGIN_NAME{"Layernorm"};
PluginFieldCollection LayernormPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LayernormPluginCreator::mPluginAttributes;

LayernormPlugin::LayernormPlugin(float eps, bool useDiffOfSquares, nvinfer1::DataType type)
    : mEps(eps)
    , mUseDiffOfSquares(useDiffOfSquares)
    , mType(type)
{
    CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
}

LayernormPlugin::LayernormPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mEps);
    read(d, mUseDiffOfSquares);
    read(d, mType);
    CHECK(d == a + length);
    CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16), "Unsupported data type");
}

nvinfer1::IPluginV2DynamicExt* LayernormPlugin::clone() const noexcept
{
    auto* plugin = new LayernormPlugin(mEps, mUseDiffOfSquares, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs LayernormPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[outputIndex];
}

bool LayernormPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    CHECK(0 <= pos && pos < 5);
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void LayernormPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t LayernormPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int LayernormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int m = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m *= inputDesc[0].dims.d[i];
    }
    const int n = inputDesc[1].dims.d[0];

    if (mType == DataType::kHALF)
    {
        const half* input = reinterpret_cast<const half*>(inputs[0]);
        const half* weight = reinterpret_cast<const half*>(inputs[1]);
        const half* bias = reinterpret_cast<const half*>(inputs[2]);
        half* output = reinterpret_cast<half*>(outputs[0]);
        invokeGeneralLayerNorm(output, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares);
    }
    else if (mType == DataType::kFLOAT)
    {
        const float* input = reinterpret_cast<const float*>(inputs[0]);
        const float* weight = reinterpret_cast<const float*>(inputs[1]);
        const float* bias = reinterpret_cast<const float*>(inputs[2]);
        float* output = reinterpret_cast<float*>(outputs[0]);
        invokeGeneralLayerNorm(output, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        const __nv_bfloat16* input = reinterpret_cast<const __nv_bfloat16*>(inputs[0]);
        const __nv_bfloat16* weight = reinterpret_cast<const __nv_bfloat16*>(inputs[1]);
        const __nv_bfloat16* bias = reinterpret_cast<const __nv_bfloat16*>(inputs[2]);
        __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(outputs[0]);
        invokeGeneralLayerNorm(output, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares);
    }
#endif

    return 0;
}

nvinfer1::DataType LayernormPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}


const char* LayernormPlugin::getPluginType() const noexcept
{
    return LAYERNORM_PLUGIN_NAME;
}

const char* LayernormPlugin::getPluginVersion() const noexcept
{
    return LAYERNORM_PLUGIN_VERSION;
}

int LayernormPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int LayernormPlugin::initialize() noexcept
{
    return 0;
}

void LayernormPlugin::terminate() noexcept {}

size_t LayernormPlugin::getSerializationSize() const noexcept
{
    return sizeof(mEps) + sizeof(mUseDiffOfSquares) + sizeof(mType);
}

void LayernormPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mEps);
    write(d, mUseDiffOfSquares);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void LayernormPlugin::destroy() noexcept
{
    delete this;
}


LayernormPluginCreator::LayernormPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1e-5f));
    mPluginAttributes.emplace_back(PluginField("use_diff_of_squares", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LayernormPluginCreator::getPluginName() const noexcept
{
    return LAYERNORM_PLUGIN_NAME;
}

const char* LayernormPluginCreator::getPluginVersion() const noexcept
{
    return LAYERNORM_PLUGIN_VERSION;
}

const PluginFieldCollection* LayernormPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LayernormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    float eps;
    bool useDiffOfSquares;
    nvinfer1::DataType type;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "eps"))
        {
            CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            eps = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "use_diff_of_squares"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            useDiffOfSquares = static_cast<bool>(*(static_cast<const bool*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new LayernormPlugin(eps, useDiffOfSquares, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LayernormPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new LayernormPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
