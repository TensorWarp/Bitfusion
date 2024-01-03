#pragma once

#include "../../common/quantization.h"
#include "../../kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "../../kernels/preQuantScaleKernel.h"
#include "../../kernels/weightOnlyBatchedGemv/kernelLauncher.h"
#include "../../plugins/common/gemmPluginProfiler.h"
#include "../../plugins/common/plugin.h"
#include "../../plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"

#include <cutlass/numeric_types.h>

#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cutlass/integer_subbyte.h"

namespace bitfusion::plugins
{

using WeightOnlyGemmRunner = bitfusion::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class WeightOnlyGroupwiseQuantGemmPluginProfiler
    : public GemmPluginProfiler<bitfusion::cutlass_extensions::CutlassGemmConfig, WeightOnlyGemmRunnerPtr,
          GemmIdCore, GemmIdCoreHash>
{
public:
    using Config = bitfusion::cutlass_extensions::CutlassGemmConfig;

    void setQuantAlgo(int quantAlgo)
    {
        mQuantAlgo = quantAlgo;
    }

    void setGroupSize(int groupSize)
    {
        mGroupSize = groupSize;
    }

protected:
    void runTactic(int m, int n, int k, const Config& tactic, char* workspace, const cudaStream_t& stream) override;

    void computeTmpSize(int maxM, int n, int k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    int mQuantAlgo;
    int mGroupSize;
};

class WeightOnlyGroupwiseQuantMatmulPlugin : public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<WeightOnlyGroupwiseQuantGemmPluginProfiler>;

    WeightOnlyGroupwiseQuantMatmulPlugin() = delete;

    WeightOnlyGroupwiseQuantMatmulPlugin(
        nvinfer1::DataType type, int quant_algo, int group_size, const PluginProfilerPtr& profiler);

    WeightOnlyGroupwiseQuantMatmulPlugin(const void* data, size_t length, const PluginProfilerPtr& profiler);

    ~WeightOnlyGroupwiseQuantMatmulPlugin() override = default;

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
    void init(nvinfer1::DataType type, int quant_algo, int group_size);

    void configGemm();

private:
    const std::string mLayerName;

    WeightOnlyGemmRunnerPtr m_weightOnlyGroupwiseGemmRunner;
    size_t m_workspaceMaxSize;
    nvinfer1::DataType mType;
    bool mCudaKernelEnabled;

    static constexpr int SMALL_M_FAST_PATH = 5;

    int mQuantAlgo;

    int mGroupSize;

    int mPreQuantScaleInputIdx;
    int mWeightInputIdx;
    int mScalesInputIdx;
    int mZerosInputIdx;
    int mBiasesInputIdx;

    GemmDims mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

class WeightOnlyGroupwiseQuantMatmulPluginCreator : public BaseCreator
{
public:
    WeightOnlyGroupwiseQuantMatmulPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<WeightOnlyGroupwiseQuantGemmPluginProfiler> gemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}
