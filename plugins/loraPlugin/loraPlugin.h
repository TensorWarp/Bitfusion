#ifndef TRT_LORA_PLUGIN_H
#define TRT_LORA_PLUGIN_H
#include "../../common/cublasMMWrapper.h"
#include "../../plugins/common/gemmPluginProfiler.h"
#include "../../plugins/common/plugin.h"
#include "../../plugins/gemmPlugin/gemmPlugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace bitfusion::plugins
{

using CublasGemmWrapper = bitfusion::common::CublasMMWrapper;
using CublasGemmWrapperPtr = std::shared_ptr<CublasGemmWrapper>;

class LoraPlugin : public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<CublasLtGemmPluginProfiler>;

    LoraPlugin() = delete;

    LoraPlugin(int in_hidden_size, std::vector<int> out_hidden_sizes, int transA, int transB, int num_lora_modules,
        nvinfer1::DataType type, const PluginProfilerPtr& profiler, bool remove_input_padding, int max_context_length,
        int max_low_rank);

    LoraPlugin(const void* data, size_t length, const PluginProfilerPtr& profiler);

    ~LoraPlugin() override = default;

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
    void init();
    void configGemm();
    void setGemmConfig();

    using IndexType = std::int32_t;

    IndexType getInputTensorIdx() const
    {
        return 0;
    }

    IndexType getHostRequestTypesIdx() const
    {
        return 1;
    }

    IndexType getLoraRanksIdx() const
    {
        return 2;
    }

    IndexType getLoraWeightsPtrsIdx() const
    {
        return 2 + mNumLoraModules;
    }

    IndexType getHostContextLengthsIdx() const
    {
        TLLM_CHECK(mRemoveInputPadding);
        return 2 + mNumLoraModules + mNumLoraModules;
    }

    enum class RequestType : int32_t
    {
        kCONTEXT = 0,
        kGENERATION = 1
    };

private:
    const std::string mLayerName;

    int mInHiddenSize;
    std::vector<int> mOutHiddenSizes;
    int mTransA;
    int mTransB;
    nvinfer1::DataType mType;
    bool mRemoveInputPadding;
    int mMaxContextLength;
    int mMaxLowRank;
    int mNumLoraModules;

    CublasGemmWrapperPtr mCublasWrapper;

    GemmDims mDims{};
    GemmIdCublas mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

class LoraPluginCreator : public BaseCreator
{
public:
    LoraPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<CublasLtGemmPluginProfiler> gemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}

#endif
