#pragma once

#include "../../common/cublasMMWrapper.h"
#include "../../common/quantization.h"
#include "../../kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "../../kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "../../kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "../../kernels/gptKernels.h"
#include "../../plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace bitfusion::plugins
{

class GPTAttentionPluginCommon : public BasePlugin
{
public:
    GPTAttentionPluginCommon() = delete;

    GPTAttentionPluginCommon(int num_heads, int num_kv_heads, int head_size, int unidirectional, float q_scaling,
        bitfusion::kernels::PositionEmbeddingType position_embedding_type,
        int rotary_embedding_dim,
        float rotary_embedding_base, bitfusion::kernels::RotaryScalingType rotary_embedding_scale_type,
        float rotary_embedding_scale, int rotary_embedding_max_positions, int tp_size, int tp_rank,
        bitfusion::kernels::ContextFMHAType context_fmha_type, bool multi_block_mode, int kv_cache_quant_mode,
        bool remove_input_padding, bitfusion::kernels::AttentionMaskType mask_type, bool paged_kv_cache,
        int tokens_per_block, nvinfer1::DataType type, int32_t max_context_length, bool qkv_bias_enabled,
        bool cross_attention = false, int max_distance = 0, bool use_paged_context_fmha = false, bool use_cache = true);

    GPTAttentionPluginCommon(const void* data, size_t length);

    ~GPTAttentionPluginCommon() override = default;

    template <typename T>
    int enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    int initialize() noexcept override;
    void terminate() noexcept override;

    template <typename T>
    T* cloneImpl() const noexcept;

    void destroy() noexcept override;

    static size_t getCommonSerializationSize() noexcept;
    void serializeCommon(void* buffer) const noexcept;
    const int getHeadSize(bool checkInit = true) const;

protected:
    int getMaxNumSeqLenTile(int batch_beam_size = 1) const;
    size_t getWorkspaceSizeForContext(nvinfer1::DataType type, int32_t nbReq, int32_t max_input_length,
        int32_t max_kv_cache_len, int32_t cross_qkv_length = 0) const noexcept;
    size_t getWorkspaceSizeForGeneration(nvinfer1::DataType type, int32_t total_num_seq) const noexcept;

    template <typename T, typename KVCacheBuffer>
    struct EnqueueContextParams
    {
        T const* attention_input;
        T const* qkv_bias;
        int32_t input_seq_length;
        int32_t max_past_kv_len;
        int32_t max_attention_window;
        int32_t cyclic_attention_window_size;
        int32_t const* q_seq_lengths;
        int32_t const* kv_seq_lengths;
        float const* kv_scale_orig_quant;
        float const* kv_scale_quant_orig;
        T const* alibi_slopes;
        T* context_buf;
        void* key_value_cache;
        void* block_pointers;
        void* host_block_pointers;
        int32_t batch_size;
        int32_t num_tokens;
        int32_t max_blocks_per_sequence;
        void* workspace;
        const T* relative_attention_bias = nullptr;
        int relative_attention_bias_stride = 0;
        T const* cross_qkv = nullptr;
        int32_t cross_qkv_length = 0;
        int32_t const* encoder_input_lengths = nullptr;
        int32_t num_encoder_tokens = 0;
    };

    template <typename T, typename KVCacheBuffer>
    int enqueueContext(const EnqueueContextParams<T, KVCacheBuffer>& params, cudaStream_t stream);

    template <typename T, typename KVCacheBuffer>
    struct EnqueueGenerationParams
    {
        T const* attention_input;
        T const* qkv_bias;
        int32_t const* sequence_lengths;
        int32_t past_kv_length;
        int32_t beam_width;
        int32_t const* context_lengths;
        float const* kv_scale_orig_quant;
        float const* kv_scale_quant_orig;
        T const* alibi_slopes;
        T* context_buf;
        void* key_value_cache;
        void* block_pointers;
        int32_t max_attention_window;
        int32_t cyclic_attention_window_size;
        int32_t num_requests;
        int32_t max_blocks_per_sequence;
        int32_t const* cache_indir;
        void* workspace;
        int32_t const* host_past_key_value_lengths;
        const T* relative_attention_bias = nullptr;
        int relative_attention_bias_stride = 0;
        int32_t const* encoder_input_lengths = nullptr;
        int32_t const* host_context_lengths = nullptr;
    };

    template <typename T, typename KVCacheBuffer>
    int enqueueGeneration(const EnqueueGenerationParams<T, KVCacheBuffer>& params, cudaStream_t stream);

    template <typename T, typename KVCacheBuffer>
    bool convertMMHAParamsToXQAParams(bitfusion::kernels::XQAParams& xqaParams,
        const EnqueueGenerationParams<T, KVCacheBuffer>& generationsParams);

    bool isRelativePosition() const
    {
        return mPositionEmbeddingType == bitfusion::kernels::PositionEmbeddingType::kRELATIVE;
    }

    bool isALiBi() const
    {
        return mPositionEmbeddingType == bitfusion::kernels::PositionEmbeddingType::kALIBI
            || mPositionEmbeddingType == bitfusion::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    bool isAliBiWithScale() const
    {
        return mPositionEmbeddingType == bitfusion::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    bool isRoPE() const
    {
        return mPositionEmbeddingType == bitfusion::kernels::PositionEmbeddingType::kROPE_GPTJ
            || mPositionEmbeddingType == bitfusion::kernels::PositionEmbeddingType::kROPE_GPT_NEOX;
    }

    bool isCrossAttention() const
    {
        return mCrossAttention;
    }

    bool useKVCache() const
    {
        return mUseKVCache;
    }

protected:
    static constexpr int kReservedMaxSeqLenTilePerSeq = 64;

    const std::string mLayerName;

    int mNumHeads;
    int mNumKVHeads;
    int mHeadSize;
    int mUnidirectional;
    float mQScaling;
    int mRotaryEmbeddingDim;
    float mRotaryEmbeddingBase;
    bitfusion::kernels::RotaryScalingType mRotaryEmbeddingScaleType;
    float mRotaryEmbeddingScale;
    int mRotaryEmbeddingMaxPositions;
    bitfusion::kernels::PositionEmbeddingType mPositionEmbeddingType;
    bool mRemovePadding = false;
    bitfusion::kernels::AttentionMaskType mMaskType;
    bool mPagedKVCache = false;
    int mTokensPerBlock;
    bitfusion::common::QuantMode mKVCacheQuantMode;
    int mTpSize = 1;
    int mTpRank = 0;
    nvinfer1::DataType mType;
    int32_t mMaxContextLength;
    bool mQKVBiasEnabled;
    bool mCrossAttention = false;
    int mMaxDistance = 0;
    bool mPagedContextFMHA = false;

    bool mEnableContextFMHA = false;
    bool mFMHAForceFP32Acc = false;
    int mSM = bitfusion::common::getSMVersion();
    int mMultiProcessorCount = bitfusion::common::getMultiProcessorCount();
    int mMaxSharedMemoryPerBlockOptin = bitfusion::common::getMaxSharedMemoryPerBlockOptin();
    UniqPtrWNullCopy<bitfusion::kernels::MHARunner> mFMHARunner;
    UniqPtrWNullCopy<bitfusion::kernels::DecoderXQARunner> mDecoderXQARunner;

    bool mMultiBlockMode;
    int mDeviceId = -1;
    static bool mForceMultiBlockWarned;
    UniqPtrWNullCopy<bitfusion::common::CublasMMWrapper> mCublasWrapper;
    bool mUseKVCache = true;
};

class GPTAttentionPluginCreatorCommon : public BaseCreator
{
public:
    GPTAttentionPluginCreatorCommon();

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    template <typename T>
    T* deserializePluginImpl(const char* name, const void* serialData, size_t serialLength) noexcept;

protected:
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection mFC{};
};

}
