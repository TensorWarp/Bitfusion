#pragma once

#include "../../common/quantization.h"
#include "../../kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "../../plugins/common/gemmPluginProfiler.h"
#include "../../plugins/common/plugin.h"
#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace bitfusion::plugins
{

using perfMapType = std::unordered_map<int, bitfusion::cutlass_extensions::CutlassGemmConfig>;
using SqGemmRunnerPtr = std::shared_ptr<bitfusion::kernels::cutlass_kernels::CutlassInt8GemmRunnerInterface>;

class SmoothQuantGemmPluginProfiler : public GemmPluginProfiler<bitfusion::cutlass_extensions::CutlassGemmConfig,
                                          SqGemmRunnerPtr, GemmIdCore, GemmIdCoreHash>
{
public:
    using Config = bitfusion::cutlass_extensions::CutlassGemmConfig;

    void setQuantMode(const bitfusion::common::QuantMode& quantMode)
    {
        mQuantMode = quantMode;
    }

protected:
    void runTactic(int m, int n, int k, const Config& tactic, char* workspace, const cudaStream_t& stream) override;

    void computeTmpSize(int maxM, int n, int k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    bitfusion::common::QuantMode mQuantMode;
};

class SmoothQuantGemmPlugin : public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<SmoothQuantGemmPluginProfiler>;

    SmoothQuantGemmPlugin() = delete;

    SmoothQuantGemmPlugin(
        bitfusion::common::QuantMode quantMode, nvinfer1::DataType type, const PluginProfilerPtr& pluginProfiler);

    SmoothQuantGemmPlugin(const void* data, size_t length, const PluginProfilerPtr& pluginProfiler);

    ~SmoothQuantGemmPlugin() override = default;

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
    void init(nvinfer1::DataType type);

    void configGemm();

private:
    const std::string mLayerName;

    SqGemmRunnerPtr m_sqGemmRunner;
    bitfusion::common::QuantMode mQuantMode;
    size_t m_workspaceMaxSize;

    GemmDims mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;

    nvinfer1::DataType mType;
};

class SmoothQuantGemmPluginCreator : public BaseCreator
{
public:
    SmoothQuantGemmPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<SmoothQuantGemmPluginProfiler> gemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}
