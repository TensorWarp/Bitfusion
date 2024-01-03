#pragma once

#include "../../common/cublasMMWrapper.h"
#include "../../common/quantization.h"
#include "../../kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "../../kernels/gptKernels.h"
#include "../../plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>
#include "../../common/cublasMMWrapper.h"

namespace bitfusion::plugins
{

class BertAttentionPlugin : public BasePlugin
{
public:
    BertAttentionPlugin() = delete;

    BertAttentionPlugin(int num_heads, int head_size, float q_scaling, bool qk_half_accum,
        bitfusion::kernels::ContextFMHAType context_fmha_type, nvinfer1::DataType type,
        bool do_relative_attention = false, int max_distance = 0, bool remove_padding = false);

    BertAttentionPlugin(const void* data, size_t length);

    ~BertAttentionPlugin() override = default;

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

    template <typename T>
    int enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

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
    const std::string mLayerName;

    int mNumHeads;
    int mHeadSize;
    float mQScaling;
    nvinfer1::DataType mType;
    bool mRelativeAttention = false;
    int mMaxDistance = 0;
    bool mRemovePadding = false;

    bool mQKHalfAccum = false;

    bool mEnableContextFMHA = false;
    bool mFMHAForceFP32Acc = false;
    int mSM = bitfusion::common::getSMVersion();

    UniqPtrWNullCopy<bitfusion::kernels::FusedMHARunnerV2> mFMHARunner;
    UniqPtrWNullCopy<bitfusion::common::CublasMMWrapper> mCublasWrapper;
};

class BertAttentionPluginCreator : public BaseCreator
{
public:
    BertAttentionPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}
