#pragma once

#include "../../common/quantization.h"
#include "../../kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "../../kernels/weightOnlyBatchedGemv/kernelLauncher.h"
#include "../../plugins/common/gemmPluginProfiler.h"
#include "../../plugins/common/plugin.h"

#include <cassert>
#include <cutlass/numeric_types.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cutlass/integer_subbyte.h"

namespace bitfusion::plugins
{
enum class WeightTypeId
{
    INT8 = 1,
    INT4 = 2,
};

constexpr int32_t INT8_BITS = 8;
constexpr int32_t INT4_BITS = 4;
constexpr int32_t INT8_INT4_RATIO = INT8_BITS / INT4_BITS;

inline int32_t getWeightTypeMultiplier(WeightTypeId weightTypeId)
{
    return weightTypeId == WeightTypeId::INT8 ? 1 : INT8_INT4_RATIO;
}

using WeightOnlyGemmRunner = bitfusion::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class WeightOnlyQuantGemmPluginProfiler : public GemmPluginProfiler<bitfusion::cutlass_extensions::CutlassGemmConfig,
                                              WeightOnlyGemmRunnerPtr, GemmIdCore, GemmIdCoreHash>
{
public:
    using Config = bitfusion::cutlass_extensions::CutlassGemmConfig;

    void setWeightTypeId(WeightTypeId weightId)
    {
        mWeightTypeId = weightId;
    }

protected:
    void runTactic(int m, int n, int k, const Config& tactic, char* workspace, const cudaStream_t& stream) override;

    void computeTmpSize(int maxM, int n, int k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    WeightTypeId mWeightTypeId;
};

class WeightOnlyQuantMatmulPlugin : public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<WeightOnlyQuantGemmPluginProfiler>;
    WeightOnlyQuantMatmulPlugin() = delete;

    WeightOnlyQuantMatmulPlugin(nvinfer1::DataType type, WeightTypeId weightTypeId, const PluginProfilerPtr& profiler);

    WeightOnlyQuantMatmulPlugin(const void* data, size_t length, const PluginProfilerPtr& profiler);

    ~WeightOnlyQuantMatmulPlugin() override = default;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

private:
    void init(nvinfer1::DataType type, WeightTypeId weightTypeId);

    void configGemm();

private:
    const std::string mLayerName;

    WeightOnlyGemmRunnerPtr m_weightOnlyGemmRunner;
    size_t m_workspaceMaxSize;
    nvinfer1::DataType mType;
    WeightTypeId mWeightTypeId;
    bool mCudaKernelEnabled;

    static constexpr int SMALL_M_FAST_PATH = 5;

    GemmDims mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

class WeightOnlyQuantMatmulPluginCreator : public BaseCreator
{
public:
    WeightOnlyQuantMatmulPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<WeightOnlyQuantGemmPluginProfiler> gemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}
