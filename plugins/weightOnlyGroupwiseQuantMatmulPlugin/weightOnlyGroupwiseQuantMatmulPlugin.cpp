#include "weightOnlyGroupwiseQuantMatmulPlugin.h"
#include "../../kernels/weightOnlyBatchedGemv/enabled.h"

using namespace nvinfer1;
using namespace bitfusion::common;
using namespace bitfusion::kernels::cutlass_kernels;
using bitfusion::plugins::WeightOnlyGroupwiseQuantMatmulPluginCreator;
using bitfusion::plugins::WeightOnlyGroupwiseQuantMatmulPlugin;
using bitfusion::plugins::WeightOnlyGroupwiseQuantGemmPluginProfiler;

static constexpr int BIAS = int(1) << 0;
static constexpr int ZERO = int(1) << 1;
static constexpr int PRE_QUANT_SCALE = int(1) << 2;
using bitfusion::plugins::read;
using bitfusion::plugins::write;

static const char* WOQ_GROUPWISE_MATMUL_PLUGIN_VERSION{"1"};
static const char* WOQ_GROUPWISE_MATMUL_PLUGIN_NAME{"WeightOnlyGroupwiseQuantMatmul"};
PluginFieldCollection WeightOnlyGroupwiseQuantMatmulPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> WeightOnlyGroupwiseQuantMatmulPluginCreator::mPluginAttributes;

void WeightOnlyGroupwiseQuantGemmPluginProfiler::runTactic(int m, int n, int k,
    const WeightOnlyGroupwiseQuantGemmPluginProfiler::Config& tactic, char* workspace, const cudaStream_t& stream)
{
    const int originalN = n * INT8_INT4_RATIO;
    half* actPtr = reinterpret_cast<half*>(workspace);
    cutlass::uint4b_t* weightPtr = reinterpret_cast<cutlass::uint4b_t*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), m * k * sizeof(half)));
    half* inputScalesPtr
        = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), n * k * sizeof(float)));
    half* zerosPtr = reinterpret_cast<half*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(inputScalesPtr), k * originalN * sizeof(half) / mGroupSize));
    half* biasesPtr = reinterpret_cast<half*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(zerosPtr), k * originalN * sizeof(half) / mGroupSize));
    half* outputPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(biasesPtr), m * sizeof(half)));
    char* workspacePtr
        = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), m * originalN * sizeof(half)));

    if ((mQuantAlgo & ZERO) == 0)
    {
        zerosPtr = nullptr;
    }

    if ((mQuantAlgo & BIAS) == 0)
    {
        biasesPtr = nullptr;
    }

    const int wsSize = mRunner->getWorkspaceSize(m, n, k);

    mRunner->gemm(actPtr, weightPtr, inputScalesPtr, zerosPtr, biasesPtr, outputPtr, m, originalN, k, mGroupSize,
        tactic, workspacePtr, wsSize, stream);
}

void WeightOnlyGroupwiseQuantGemmPluginProfiler::computeTmpSize(int maxM, int n, int k)
{
    const int originalN = n * INT8_INT4_RATIO;
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(half),
        k * n * sizeof(float),
        k * originalN * sizeof(half) / mGroupSize,
        k * originalN * sizeof(half) / mGroupSize,
        maxM * sizeof(half),
        maxM * originalN * sizeof(half),
        mRunner->getWorkspaceSize(maxM, n, k)
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<WeightOnlyGroupwiseQuantGemmPluginProfiler::Config> WeightOnlyGroupwiseQuantGemmPluginProfiler::getTactics(
    int m, int n, int k) const
{
    return mRunner->getConfigs();
}

WeightOnlyGroupwiseQuantMatmulPlugin::WeightOnlyGroupwiseQuantMatmulPlugin(nvinfer1::DataType type, int quant_algo,
    int group_size, const WeightOnlyGroupwiseQuantMatmulPlugin::PluginProfilerPtr& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    init(type, quant_algo, group_size);
}

WeightOnlyGroupwiseQuantMatmulPlugin::WeightOnlyGroupwiseQuantMatmulPlugin(
    const void* data, size_t length, const WeightOnlyGroupwiseQuantMatmulPlugin::PluginProfilerPtr& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    nvinfer1::DataType type;
    int quant_algo = 0;
    int group_size = 0;
    read(d, type);
    read(d, quant_algo);
    read(d, group_size);
    read(d, mDims);

    init(type, quant_algo, group_size);

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    CHECK(d == a + length);
}

void WeightOnlyGroupwiseQuantMatmulPlugin::init(nvinfer1::DataType type, int quant_algo, int group_size)
{
    mType = type;
    mQuantAlgo = quant_algo;
    mGroupSize = group_size;

    mPreQuantScaleInputIdx = (quant_algo & PRE_QUANT_SCALE) ? 1 : 0;
    mWeightInputIdx = mPreQuantScaleInputIdx + 1;
    mScalesInputIdx = mWeightInputIdx + 1;
    mZerosInputIdx = (quant_algo & ZERO) ? mScalesInputIdx + 1 : mScalesInputIdx;
    mBiasesInputIdx = (quant_algo & BIAS) ? mZerosInputIdx + 1 : mZerosInputIdx;

    if (mType == nvinfer1::DataType::kHALF)
    {
        if (quant_algo & ZERO)
        {
            m_weightOnlyGroupwiseGemmRunner
                = std::make_shared<bitfusion::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<half,
                    cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
        }
        else
        {
            m_weightOnlyGroupwiseGemmRunner
                = std::make_shared<bitfusion::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<half,
                    cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
        }
    }
#if defined(ENABLE_BF16)
    else if (mType == nvinfer1::DataType::kBF16)
    {
        if (quant_algo & ZERO)
        {
            m_weightOnlyGroupwiseGemmRunner
                = std::make_shared<bitfusion::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_bfloat16,
                    cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
        }
        else
        {
            m_weightOnlyGroupwiseGemmRunner
                = std::make_shared<bitfusion::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_bfloat16,
                    cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
        }
    }
#endif
    else
    {
        THROW("Unsupported data type");
    }
    mCudaKernelEnabled
        = bitfusion::kernels::isWeightOnlyBatchedGemvEnabled(bitfusion::kernels::WeightOnlyQuantType::Int4b);
    mPluginProfiler->setQuantAlgo(mQuantAlgo);
    mPluginProfiler->setGroupSize(mGroupSize);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

nvinfer1::IPluginV2DynamicExt* WeightOnlyGroupwiseQuantMatmulPlugin::clone() const noexcept
{
    auto* plugin = new WeightOnlyGroupwiseQuantMatmulPlugin(*this);
    return plugin;
}

void WeightOnlyGroupwiseQuantMatmulPlugin::configGemm()
{
    mPluginProfiler->profileTactics(m_weightOnlyGroupwiseGemmRunner, mType, mDims, mGemmId);
}

nvinfer1::DimsExprs WeightOnlyGroupwiseQuantMatmulPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{


    try
    {
        CHECK(nbInputs == mBiasesInputIdx + 1);
        CHECK(outputIndex == 0);
        const int nbDimsA = inputs[0].nbDims;
        const int nbDimsB = inputs[mWeightInputIdx].nbDims;
        CHECK(nbDimsA >= 2);
        CHECK(nbDimsB == 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }

        ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[mWeightInputIdx].d[1]->getConstantValue() * INT8_INT4_RATIO);

        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool WeightOnlyGroupwiseQuantMatmulPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos < mBiasesInputIdx + 2)
    {
        if (pos == mWeightInputIdx)
        {
            return inOut[mWeightInputIdx].type == nvinfer1::DataType::kINT8
                && inOut[mWeightInputIdx].format == TensorFormat::kLINEAR;
        }
        else
        {
            return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
        }
    }
    else
    {
        assert(false);
        return false;
    }
}

void WeightOnlyGroupwiseQuantMatmulPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    const auto minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    const auto maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    const int maxK = in[0].max.d[in[0].max.nbDims - 1];
    const int maxN = in[mWeightInputIdx].max.d[1] * INT8_INT4_RATIO;

    const auto K = maxK;
    const auto N = maxN / INT8_INT4_RATIO;

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, N, K};
    }
    mGemmId = {N, K, mType};

    size_t smoothedActSize = static_cast<size_t>(maxM) * static_cast<size_t>(maxK)
        * (in[0].desc.type == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(half));
    m_workspaceMaxSize = smoothedActSize + m_weightOnlyGroupwiseGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t WeightOnlyGroupwiseQuantMatmulPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int WeightOnlyGroupwiseQuantMatmulPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    int m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    const int n = inputDesc[mWeightInputIdx].dims.d[1];
    const int k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    bool use_cuda_kernel = m < SMALL_M_FAST_PATH && mCudaKernelEnabled;
    bool use_pre_quant_scale = mQuantAlgo & PRE_QUANT_SCALE;

    const half* zeros_ptr = (mQuantAlgo & ZERO) ? reinterpret_cast<const half*>(inputs[mZerosInputIdx]) : nullptr;
    const half* biases_ptr = (mQuantAlgo & BIAS) ? reinterpret_cast<const half*>(inputs[mBiasesInputIdx]) : nullptr;
    const half* act_ptr = reinterpret_cast<const half*>(inputs[0]);

    if (use_pre_quant_scale && !use_cuda_kernel)
    {
        act_ptr = reinterpret_cast<const half*>(workspace);
        if (mType == nvinfer1::DataType::kHALF)
        {
            bitfusion::kernels::apply_per_channel_scale_kernel_launcher<half>(reinterpret_cast<half*>(workspace),
                reinterpret_cast<const half*>(inputs[0]), reinterpret_cast<const half*>(inputs[mPreQuantScaleInputIdx]),
                m, k, stream);
        }
#if defined(ENABLE_BF16)
        else if (mType == nvinfer1::DataType::kBF16)
        {
            bitfusion::kernels::apply_per_channel_scale_kernel_launcher<__nv_bfloat16>(
                reinterpret_cast<__nv_bfloat16*>(workspace), reinterpret_cast<const __nv_bfloat16*>(inputs[0]),
                reinterpret_cast<const __nv_bfloat16*>(inputs[mPreQuantScaleInputIdx]), m, k, stream);
        }
#endif
    }

#if defined(ENABLE_BF16)
    CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16,
        "No valid weightOnlyGropwiseQuantMatmul configuration");
#else
    CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF, "No valid weightOnlyGropwiseQuantMatmul configuration");
#endif

    bitfusion::kernels::WeightOnlyActivationType weight_only_act_type;
    int real_n = n * INT8_INT4_RATIO;
    if (mType == nvinfer1::DataType::kHALF)
    {
        weight_only_act_type = bitfusion::kernels::WeightOnlyActivationType::FP16;
    }
    else if (mType == nvinfer1::DataType::kBF16)
    {
        weight_only_act_type = bitfusion::kernels::WeightOnlyActivationType::BF16;
    }
    if (use_cuda_kernel)
    {
        const void* pre_quant_scale = nullptr;
        if (use_pre_quant_scale)
            pre_quant_scale = inputs[mPreQuantScaleInputIdx];
        bitfusion::kernels::WeightOnlyParams params{reinterpret_cast<const uint8_t*>(inputs[mWeightInputIdx]),
            inputs[mScalesInputIdx], zeros_ptr, act_ptr, pre_quant_scale, biases_ptr, outputs[0], m, real_n, k,
            mGroupSize, bitfusion::kernels::WeightOnlyQuantType::Int4b,
            bitfusion::kernels::WeightOnlyType::GroupWise,
            bitfusion::kernels::WeightOnlyActivationFunctionType::Identity, weight_only_act_type};
        bitfusion::kernels::weight_only_batched_gemv_launcher(params, stream);
    }
    else
    {
        const int ws_bytes = m_weightOnlyGroupwiseGemmRunner->getWorkspaceSize(m, n, k);

        int32_t* weight_ptr = const_cast<int32_t*>(reinterpret_cast<const int32_t*>(inputs[mWeightInputIdx]));

        const auto& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
        CHECK_WITH_INFO(bestTactic,
            "No valid weight only groupwise GEMM tactic(It is usually caused by the failure to execute all candidate "
            "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
            "engine.)");
        m_weightOnlyGroupwiseGemmRunner->gemm(act_ptr, weight_ptr, inputs[mScalesInputIdx], zeros_ptr, biases_ptr,
            outputs[0], m, real_n, k, mGroupSize, *bestTactic,
            reinterpret_cast<char*>(workspace) + m * k * sizeof(half), ws_bytes, stream);
    }

    return 0;
}

nvinfer1::DataType WeightOnlyGroupwiseQuantMatmulPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    CHECK(index == 0);
    return mType;
}


const char* WeightOnlyGroupwiseQuantMatmulPlugin::getPluginType() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_NAME;
}

const char* WeightOnlyGroupwiseQuantMatmulPlugin::getPluginVersion() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_VERSION;
}

int WeightOnlyGroupwiseQuantMatmulPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int WeightOnlyGroupwiseQuantMatmulPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void WeightOnlyGroupwiseQuantMatmulPlugin::terminate() noexcept {}

size_t WeightOnlyGroupwiseQuantMatmulPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) +
        sizeof(int) +
        sizeof(nvinfer1::DataType) +
        sizeof(mDims) +
        mPluginProfiler->getSerializationSize(mGemmId);
}

void WeightOnlyGroupwiseQuantMatmulPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mQuantAlgo);
    write(d, mGroupSize);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    assert(d == a + getSerializationSize());
}

void WeightOnlyGroupwiseQuantMatmulPlugin::destroy() noexcept
{
    delete this;
}


WeightOnlyGroupwiseQuantMatmulPluginCreator::WeightOnlyGroupwiseQuantMatmulPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("quant_algo", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* WeightOnlyGroupwiseQuantMatmulPluginCreator::getPluginName() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_NAME;
}

const char* WeightOnlyGroupwiseQuantMatmulPluginCreator::getPluginVersion() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_VERSION;
}

const PluginFieldCollection* WeightOnlyGroupwiseQuantMatmulPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* WeightOnlyGroupwiseQuantMatmulPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    nvinfer1::DataType type;
    int QuantAlgo;
    int GroupSize;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "quant_algo"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            QuantAlgo = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "group_size"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            GroupSize = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler( false);
        auto* obj = new WeightOnlyGroupwiseQuantMatmulPlugin(type, QuantAlgo, GroupSize, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* WeightOnlyGroupwiseQuantMatmulPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler( true);
        auto* obj = new WeightOnlyGroupwiseQuantMatmulPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
