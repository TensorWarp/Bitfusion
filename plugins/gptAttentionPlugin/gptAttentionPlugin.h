#pragma once

#include "../../plugins/common/checkMacrosPlugin.h"
#include "../../common/cublasMMWrapper.h"
#include "../../common/logger.h"
#include "../../common/quantization.h"
#include "../../common/stringUtils.h"
#include "../../kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "../../kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "../../kernels/gptKernels.h"
#include "../../plugins/common/plugin.h"
#include "../../plugins/gptAttentionCommon/gptAttentionCommon.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace bitfusion::plugins
{


class GPTAttentionPlugin : public GPTAttentionPluginCommon
{
public:
    GPTAttentionPlugin(int num_heads, int num_kv_heads, int head_size, int unidirectional, float q_scaling,
        bitfusion::kernels::PositionEmbeddingType position_embedding_type,
        int rotary_embedding_dim,
        float rotary_embedding_base, bitfusion::kernels::RotaryScalingType rotary_embedding_scale_type,
        float rotary_embedding_scale, int rotary_embedding_max_positions, int tp_size, int tp_rank,
        bitfusion::kernels::ContextFMHAType context_fmha_type, bool multi_block_mode, int kv_cache_quant_mode,
        bool remove_input_padding, bitfusion::kernels::AttentionMaskType mask_type, bool paged_kv_cache,
        int tokens_per_block, nvinfer1::DataType type, int32_t max_context_length, bool qkv_bias_enabled,
        bool cross_attention = false, int max_distance = 0, bool use_paged_context_fmha = false, bool use_cache = true);

    GPTAttentionPlugin(const void* data, size_t length);

    ~GPTAttentionPlugin() override = default;

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

    template <typename T, typename KVCacheBuffer>
    int enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    template <typename T>
    int enqueueDispatchKVCacheType(const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream);

    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;

    GPTAttentionPlugin* clone() const noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    enum class RequestType : int32_t
    {
        kCONTEXT = 0,
        kGENERATION = 1
    };

private:
    template <typename T, typename KVCacheBuffer>
    int enqueueSome(int32_t seqIdxBeg, int32_t localNbSeq, int32_t tokenIdxBeg, int32_t localNbTokens,
        const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    using IndexType = std::int32_t;

    std::vector<size_t> mEntryIdx;
    enum class IdxEntry : size_t
    {
        QKV_TENSOR,
        SEQUENCE_LENGTH,
        HOST_PAST_KEY_VALUE_LENGTHS,
        HOST_MAX_ATTENTION_WINDOW,
        CONTEXT_LENGTHS,
        CACHE_INDIR,
        REQUEST_TYPES,
        KV_CACHE_BLOCK_POINTERS,
        HOST_KV_CACHE_BLOCK_POINTERS,
        PAST_KEY_VALUE,
        KV_CACHE_QUANTIZATION_SCALE,
        KV_CACHE_DEQUANTIZATION_SCALE,
        ALIBI_SLOPES,
        RELATIVE_ATTENTION_BIAS,
        CROSS_QKV,
        CROSS_QKV_LENGTH,
        ENCODER_INPUT_LENGTH,
        HOST_CONTEXT_LENGTH,
        QKV_BIAS_TENSOR,
        ENUM_SIZE,
    };

    bool isEntryUsed(const IdxEntry& entry) const;
    void initEntryIdx();
    IndexType getIdx(const IdxEntry& entry) const;
};

class GPTAttentionPluginCreator : public GPTAttentionPluginCreatorCommon
{
public:
    GPTAttentionPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;
};

}
