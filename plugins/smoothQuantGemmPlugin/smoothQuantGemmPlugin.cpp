#include "smoothQuantGemmPlugin.h"
#include <numeric>

using namespace nvinfer1;
using namespace bitfusion::common;
using namespace bitfusion::kernels::cutlass_kernels;
using bitfusion::plugins::SmoothQuantGemmPluginCreator;
using bitfusion::plugins::SmoothQuantGemmPlugin;
using bitfusion::plugins::SmoothQuantGemmPluginProfiler;
using bitfusion::plugins::read;
using bitfusion::plugins::write;

static const char* SQ_GEMM_PLUGIN_VERSION{"1"};
static const char* SQ_GEMM_PLUGIN_NAME{"SmoothQuantGemm"};
PluginFieldCollection SmoothQuantGemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SmoothQuantGemmPluginCreator::mPluginAttributes;

void SmoothQuantGemmPluginProfiler::runTactic(int m, int n, int k, const SmoothQuantGemmPluginProfiler::Config& tactic,
    char* workspace, const cudaStream_t& stream)
{
    int8_t* aTmp = reinterpret_cast<int8_t*>(workspace);
    int8_t* bTmp = nextWorkspacePtr(aTmp, m * k * sizeof(int8_t));
    void* cTmp = reinterpret_cast<void*>(nextWorkspacePtr(bTmp, n * k * sizeof(int8_t)));
    float* alphaRowTmp = reinterpret_cast<float*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(cTmp), m * n * (mType == nvinfer1::DataType::kFLOAT ? 4 : 2)));
    float* alphaColTmp
        = reinterpret_cast<float*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(alphaRowTmp), m * sizeof(float)));
    char* workspaceTmp
        = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(alphaColTmp), n * sizeof(float)));

    const int wsSize = mRunner->getWorkspaceSize(m, n, k);

    mRunner->gemm(
        aTmp, bTmp, mQuantMode, alphaColTmp, alphaRowTmp, cTmp, m, n, k, tactic, workspaceTmp, wsSize, stream);
}

void SmoothQuantGemmPluginProfiler::computeTmpSize(int maxM, int n, int k)
{
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(int8_t),
        n * k * sizeof(int8_t),
        maxM * n * (mType == nvinfer1::DataType::kFLOAT ? 4u : 2u),
        maxM * sizeof(float),
        n * sizeof(float),
        mRunner->getWorkspaceSize(maxM, n, k)
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<SmoothQuantGemmPluginProfiler::Config> SmoothQuantGemmPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

SmoothQuantGemmPlugin::SmoothQuantGemmPlugin(
    QuantMode quantMode, nvinfer1::DataType type, const SmoothQuantGemmPlugin::PluginProfilerPtr& pluginProfiler)
    : mQuantMode(quantMode)
    , mPluginProfiler(pluginProfiler)
{
    init(type);
}

SmoothQuantGemmPlugin::SmoothQuantGemmPlugin(
    const void* data, size_t length, const SmoothQuantGemmPlugin::PluginProfilerPtr& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    bool perChannelScaling = false, perTokenScaling = false;
    nvinfer1::DataType type;
    unsigned int quantMode;
    read(d, quantMode);
    read(d, type);
    read(d, mDims);

    mQuantMode = QuantMode(quantMode);

    init(type);

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK(d == a + length);
}

void SmoothQuantGemmPlugin::init(nvinfer1::DataType type)
{
    mType = type;
    if (mType == nvinfer1::DataType::kHALF)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<half>>();
    }
    else if (mType == nvinfer1::DataType::kFLOAT)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<float>>();
    }
    else if (mType == nvinfer1::DataType::kINT32)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<int32_t>>();
    }
    else
    {
        TLLM_THROW("Support for bf16 is missing");
    }

    mPluginProfiler->setQuantMode(mQuantMode);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

nvinfer1::IPluginV2DynamicExt* SmoothQuantGemmPlugin::clone() const noexcept
{
    auto* plugin = new SmoothQuantGemmPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs SmoothQuantGemmPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 4);
        TLLM_CHECK(outputIndex == 0);
        const int nbDimsA = inputs[0].nbDims;
        TLLM_CHECK(nbDimsA >= 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        ret.d[nbDimsA - 1] = inputs[1].d[0];
        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool SmoothQuantGemmPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
    case 3:
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    case 4:
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        assert(false);
        return false;
    }
}

void SmoothQuantGemmPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    const auto minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    const auto maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    const int maxK = in[0].max.d[in[0].max.nbDims - 1];
    const int maxN = in[1].max.d[0];
    const int minK = in[0].min.d[in[0].min.nbDims - 1];
    const int minN = in[1].min.d[0];

    TLLM_CHECK_WITH_INFO(minN == maxN, "Variable out channels is not allowed");
    TLLM_CHECK_WITH_INFO(minK == maxK, "Variable in channels is not allowed");

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, maxN, maxK};
    }
    mGemmId = {maxN, maxK, mType};

    m_workspaceMaxSize = m_sqGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t SmoothQuantGemmPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int SmoothQuantGemmPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    const int n = inputDesc[1].dims.d[0];
    const int k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    const int wsSize = m_sqGemmRunner->getWorkspaceSize(m, n, k);

    const auto& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
    TLLM_CHECK_WITH_INFO(bestTactic, "No valid SQ GEMM tactic");
    m_sqGemmRunner->gemm(reinterpret_cast<const int8_t*>(inputs[0]), reinterpret_cast<const int8_t*>(inputs[1]),
        mQuantMode, reinterpret_cast<const float*>(inputs[3]), reinterpret_cast<const float*>(inputs[2]),
        reinterpret_cast<void*>(outputs[0]), m, n, k, *bestTactic, reinterpret_cast<char*>(workspace), wsSize, stream);

    return 0;
}

nvinfer1::DataType SmoothQuantGemmPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}


const char* SmoothQuantGemmPlugin::getPluginType() const noexcept
{
    return SQ_GEMM_PLUGIN_NAME;
}

const char* SmoothQuantGemmPlugin::getPluginVersion() const noexcept
{
    return SQ_GEMM_PLUGIN_VERSION;
}

int SmoothQuantGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int SmoothQuantGemmPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void SmoothQuantGemmPlugin::terminate() noexcept {}

size_t SmoothQuantGemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(unsigned int) +
        sizeof(nvinfer1::DataType) +
        sizeof(mDims) +
        mPluginProfiler->getSerializationSize(mGemmId);
}

void SmoothQuantGemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mQuantMode.value());
    write(d, mType);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    assert(d == a + getSerializationSize());
}

void SmoothQuantGemmPlugin::destroy() noexcept
{
    delete this;
}

void SmoothQuantGemmPlugin::configGemm()
{
    mPluginProfiler->profileTactics(m_sqGemmRunner, mType, mDims, mGemmId);
}


SmoothQuantGemmPluginCreator::SmoothQuantGemmPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("has_per_channel_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_per_token_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SmoothQuantGemmPluginCreator::getPluginName() const noexcept
{
    return SQ_GEMM_PLUGIN_NAME;
}

const char* SmoothQuantGemmPluginCreator::getPluginVersion() const noexcept
{
    return SQ_GEMM_PLUGIN_VERSION;
}

const PluginFieldCollection* SmoothQuantGemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SmoothQuantGemmPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    bool perTokenScaling, perChannelScaling;
    nvinfer1::DataType type;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "has_per_channel_scaling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            perChannelScaling = static_cast<bool>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "has_per_token_scaling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            perTokenScaling = static_cast<bool>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler( false);
        QuantMode quantMode = QuantMode::fromDescription(true, true, perTokenScaling, perChannelScaling);
        auto* obj = new SmoothQuantGemmPlugin(quantMode, type, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SmoothQuantGemmPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler( true);
        auto* obj = new SmoothQuantGemmPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
