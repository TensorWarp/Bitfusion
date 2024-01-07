
#include "../../plugins/common/gemmPluginProfiler.h"
#include "../../common/cublasMMWrapper.h"
#include "../../kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "../../kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "../../plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"

namespace bitfusion::plugins
{

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::GemmPluginProfiler()
{
    mMNKProfileMap = std::make_shared<MNKProfileMap>();

    const auto skipEnv = std::getenv("SKIP_GEMM_PLUGIN_PROFILINGS");
    mSkip = (skipEnv != NULL && std::stoi(skipEnv));
    if (mSkip)
    {
        LOG_DEBUG(
            "SKIP_GEMM_PLUGIN_PROFILINGS is set. Skipping GEMM plugin profilings. It could result in runtime error "
            "if default tactic is not defined.");
    }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::serialize(
    char*& buffer, const GemmIdType& gemmId) const
{
    auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

    write(buffer, static_cast<int>(mProfileMap->size()));
    for (const auto& pair : *mProfileMap)
    {
        write(buffer, pair);
    }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::deserialize(
    const char*& data, GemmDims& dims, const GemmIdType& gemmId)
{
    writer_lock lock(mMNKProfileMap->mutex);

    mDims = dims;

    if (!mMNKProfileMap->existsMProfileMap(gemmId))
    {
        mMNKProfileMap->createMProfileMap(gemmId);
    }
    auto profileMap = mMNKProfileMap->getMProfileMap(gemmId);
    int selectedMapSize;
    read(data, selectedMapSize);
    for (int ii = 0; ii < selectedMapSize; ++ii)
    {
        std::pair<int, std::optional<Config>> config;
        read(data, config);
        profileMap->insert(config);
    }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
size_t GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getSerializationSize(
    const GemmIdType& gemmId) const
{
    reader_lock lock(mMNKProfileMap->mutex);
    return sizeof(int) +
        mMNKProfileMap->getMProfileMap(gemmId)->size()
        * sizeof(std::pair<int, std::optional<Config>>);
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTactics(
    const RunnerPtr& runner, const nvinfer1::DataType& type, const GemmDims& dims, const GemmIdType& gemmId)
{
    writer_lock lock(mMNKProfileMap->mutex);

    if (!dims.isInitialized())
    {
        return;
    }

    mRunner = runner;
    mType = type;

    const int maxM = std::min(nextPowerOfTwo(dims.maxM), MAX_PROFILE_M);
    computeTmpSize(maxM, dims.n, dims.k);

    if (!mMNKProfileMap->existsMProfileMap(gemmId))
    {
        mMNKProfileMap->createMProfileMap(gemmId);
    }

    if (mSkip)
    {
        return;
    }

    auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

    auto profileTactics = [&mProfileMap, this](int m, int n, int k)
    {
        if (mProfileMap->count(m) == 0)
        {
            initTmpData(m, n, k, mWorkspaceTmp, mTmpWorkspaceSizeInBytes, cudaStreamDefault);
            const auto tactics = this->getTactics(m, n, k);
            mProfileMap->insert({m, this->profileTacticsForProblem(m, n, k, tactics)});
        }
    };

    allocateTmpData();

    const int startMinMRounded = nextPowerOfTwo(dims.minM);
    for (int m = startMinMRounded; m < maxM; m *= 2)
    {
        profileTactics(m, dims.n, dims.k);
    }

    profileTactics(maxM, dims.n, dims.k);
    freeTmpData();
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getBestConfig(
    int m, const GemmIdType& gemmId) const
{
    reader_lock lock(mMNKProfileMap->mutex);

    if (mSkip)
    {
        return std::nullopt;
    }

    const int mRounded = std::min(nextPowerOfTwo(m), MAX_PROFILE_M);
    return mMNKProfileMap->getMProfileMap(gemmId)->at(mRounded);
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::allocateTmpData()
{
    CHECK_WITH_INFO(mTmpWorkspaceSizeInBytes > 0, "tmpWorkspaceSizeInBytes must be larger than 0");
    const auto status = cudaMalloc(&mWorkspaceTmp, mTmpWorkspaceSizeInBytes);
    CHECK_WITH_INFO(status == cudaSuccess, "Can't allocate tmp workspace for GEMM tactics profiling.");
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::freeTmpData()
{
    const auto status = cudaFree(mWorkspaceTmp);
    CHECK_WITH_INFO(status == cudaSuccess, "Can't free tmp workspace for GEMM tactics profiling.");
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticsForProblem(
    int m, int n, int k, const std::vector<Config>& tactics)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);

    float bestTime = std::numeric_limits<float>::max();
    Config bestConfig;
    bool foundOne = false;

    for (int ii = 0; ii < tactics.size(); ++ii)
    {
        const Config& candidateConfig = tactics[ii];
        float time = std::numeric_limits<float>::max();
        try
        {
            if (!checkTactic(m, n, k, candidateConfig))
            {
                continue;
            }
            time = profileTacticForProblem(m, n, k, candidateConfig);
            foundOne = true;
        }
        catch (const std::exception& e)
        {
            std::ostringstream msg;
            msg << "Cannot profile configuration " << ii << " (for"
                << " m=" << m << ", n=" << n << ", k=" << k << ")"
                << ", reason: \"" << e.what() << "\". Skipped";
            LOG_WARNING(msg.str());
            continue;
        }

        if (time < bestTime)
        {
            bestConfig = candidateConfig;
            bestTime = time;
        }
    }

    if (!foundOne)
    {
        std::ostringstream msg;
        msg << "Have not found any valid GEMM config for shape ("
            << "m=" << m << ", n=" << n << ", k=" << k << "). Will try to use default or fail at runtime";
        LOG_WARNING(msg.str());
        return std::nullopt;
    }
    return {bestConfig};
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
float GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticForProblem(
    int m, int n, int k, const Config& tactic)
{
    constexpr int warmup = 5;
    constexpr int runs = 10;

    cudaStream_t stream = cudaStreamDefault;
    for (int i = 0; i < warmup; ++i)
    {
        runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    for (int i = 0; i < runs; ++i)
    {
        runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
    }

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed / runs;
}

template class GemmPluginProfiler<bitfusion::cutlass_extensions::CutlassGemmConfig,
    std::shared_ptr<bitfusion::kernels::cutlass_kernels::CutlassInt8GemmRunnerInterface>, GemmIdCore,
    GemmIdCoreHash>;

template class GemmPluginProfiler<bitfusion::cutlass_extensions::CutlassGemmConfig,
    std::shared_ptr<bitfusion::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>, GemmIdCore,
    GemmIdCoreHash>;

template class GemmPluginProfiler<cublasLtMatmulHeuristicResult_t,
    std::shared_ptr<bitfusion::common::CublasMMWrapper>, GemmIdCublas, GemmIdCublasHash>;

template class GemmPluginProfiler<bitfusion::cutlass_extensions::CutlassGemmConfig, MixtureOfExpertsPlugin*,
    GemmIDMoe, GemmIDMoeHash>;

}
