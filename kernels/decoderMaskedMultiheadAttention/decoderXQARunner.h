#pragma once

#include <NvInferRuntime.h>
#include <cuda_fp16.h>

#include "../../common/assert.h"
#include "../../common/cudaUtils.h"
#include "../../common/quantization.h"
#include "../../kernels/gptKernels.h"
#include "../../kernels/kvCacheUtils.h"
#include "../../kernels/multiHeadAttentionCommon.h"

using namespace bitfusion::common;

namespace bitfusion
{
    namespace kernels
    {
        /// <summary>
        /// Template struct to assist in dispatching XQA (XQARunner) based on data type and key-value cache buffer type.
        /// </summary>
        template <typename T, typename KVCacheBuffer>
        struct XQADispatchHelper;

        /// <summary>
        /// XQADispatchHelper specialization for __half data type and KVLinearBuffer.
        /// </summary>
        template <>
        struct XQADispatchHelper<__half, KVLinearBuffer>
        {
            static constexpr bool CanSupport = true;
        };

        /// <summary>
        /// Data type for XQA (XQARunner).
        /// </summary>
        using XQADataType = Data_type;

        /// <summary>
        /// Parameters for the XQA (XQARunner).
        /// </summary>
        struct XQAParams
        {
            /// <summary>
            /// Data type for XQA (XQARunner).
            /// </summary>
            XQADataType data_type = DATA_TYPE_FP16;

            /// <summary>
            /// Data type for the key-value cache.
            /// </summary>
            XQADataType kv_cache_data_type = DATA_TYPE_FP16;

            /// <summary>
            /// Pointer to the output data.
            /// </summary>
            void* output = nullptr;

            /// <summary>
            /// Pointer to the query, key, and value tensors.
            /// </summary>
            const void* qkv = nullptr;

            /// <summary>
            /// Pointer to the cache indices for the key-value cache.
            /// </summary>
            const int32_t* cache_indir = nullptr;

            /// <summary>
            /// Pointer to the original quantized key-value scale factors.
            /// </summary>
            const float* kv_scale_orig_quant = nullptr;

            /// <summary>
            /// Pointer to the quantized original key-value scale factors.
            /// </summary>
            const float* kv_scale_quant_orig = nullptr;

            /// <summary>
            /// Pointer to the lengths of past key-value tensors.
            /// </summary>
            const int32_t* host_past_key_value_lengths = nullptr;

            /// <summary>
            /// Pointer to the lengths of context tensors.
            /// </summary>
            const int32_t* host_context_lengths = nullptr;

            /// <summary>
            /// Pointer to workspace memory.
            /// </summary>
            void* workspaces = nullptr;

            /// <summary>
            /// Batch size for the computation.
            /// </summary>
            uint32_t batch_size = 0;

            /// <summary>
            /// Beam width for multi-beam decoding.
            /// </summary>
            int32_t beam_width = 0;

            /// <summary>
            /// Maximum attention window size.
            /// </summary>
            int32_t max_attention_window_size = 0;

            /// <summary>
            /// Cyclic attention window size.
            /// </summary>
            int32_t cyclic_attention_window_size = 0;

            /// <summary>
            /// Current timestep in the sequence.
            /// </summary>
            int timestep = 0;

            /// <summary>
            /// Pointer to the query, key, and value biases.
            /// </summary>
            const void* qkv_bias;

            /// <summary>
            /// Pointer to the lengths of input sequences.
            /// </summary>
            const int32_t* sequence_lengths;

            /// <summary>
            /// Pointer to the lengths of input contexts.
            /// </summary>
            const int32_t* context_lengths;

            /// <summary>
            /// Pointer to alibi slopes.
            /// </summary>
            const void* alibi_slopes;

            /// <summary>
            /// Number of query heads.
            /// </summary>
            int32_t num_q_heads = 0;

            /// <summary>
            /// Number of key-value heads.
            /// </summary>
            int32_t num_kv_heads = 0;

            /// <summary>
            /// Size of each attention head.
            /// </summary>
            int32_t head_size = 0;

            /// <summary>
            /// Flag indicating unidirectional attention.
            /// </summary>
            int unidirectional;

            /// <summary>
            /// Scaling factor for queries.
            /// </summary>
            float q_scaling = 0;

            /// <summary>
            /// Dimension of rotary embeddings.
            /// </summary>
            int32_t rotary_embedding_dim = 0;

            /// <summary>
            /// Base value for rotary embeddings.
            /// </summary>
            float rotary_embedding_base = 0.0f;

            /// <summary>
            /// Type of scaling used for rotary embeddings.
            /// </summary>
            bitfusion::kernels::RotaryScalingType rotary_embedding_scale_type;

            /// <summary>
            /// Scaling factor for rotary embeddings.
            /// </summary>
            float rotary_embedding_scale;

            /// <summary>
            /// Maximum number of positions for rotary embeddings.
            /// </summary>
            int rotary_embedding_max_positions;

            /// <summary>
            /// Type of position embedding.
            /// </summary>
            bitfusion::kernels::PositionEmbeddingType position_embedding_type;

            /// <summary>
            /// Flag to remove padding from the input.
            /// </summary>
            bool remove_padding = false;

            /// <summary>
            /// Type of attention mask used.
            /// </summary>
            bitfusion::kernels::AttentionMaskType mask_type;

            /// <summary>
            /// Flag indicating paged key-value cache.
            /// </summary>
            bool paged_kv_cache;

            /// <summary>
            /// Number of tokens per block.
            /// </summary>
            int tokens_per_block;

            /// <summary>
            /// Quantization mode for the key-value cache.
            /// </summary>
            bitfusion::common::QuantMode kv_cache_quant_mode;

            /// <summary>
            /// TP size (thread parallelism size).
            /// </summary>
            int tp_size = 1;

            /// <summary>
            /// TP rank (thread parallelism rank).
            /// </summary>
            int tp_rank = 0;

            /// <summary>
            /// Flag indicating the presence of query, key, and value biases.
            /// </summary>
            bool qkv_bias_enabled;

            /// <summary>
            /// Flag indicating cross-attention mode.
            /// </summary>
            bool cross_attention;

            /// <summary>
            /// Maximum distance for attention.
            /// </summary>
            int max_distance = 0;

            /// <summary>
            /// Flag indicating multi-block mode.
            /// </summary>
            bool multi_block_mode;
        };

        /// <summary>
        /// Macro for returning false and providing a reason.
        /// </summary>
#define SUPPORT_RETURN_FALSE(X) { return false; }

        /// <summary>
        /// Class representing the DecoderXQARunner.
        /// </summary>
        class DecoderXQARunner
        {
        public:
            /// <summary>
            /// Constructor for DecoderXQARunner.
            /// </summary>
            /// <param name="data_type">The data type.</param>
            /// <param name="num_heads">The number of heads.</param>
            /// <param name="num_kv_heads">The number of key-value heads.</param>
            /// <param name="head_size">The head size.</param>
            /// <param name="multi_block_mode">Indicates if multi-block mode is used.</param>
            DecoderXQARunner(const XQADataType data_type, int num_heads, int num_kv_heads, int head_size, bool multi_block_mode);

            /// <summary>
            /// Destructor for DecoderXQARunner.
            /// </summary>
            ~DecoderXQARunner();

            /// <summary>
            /// Check if a specific configuration should be used.
            /// </summary>
            /// <typeparam name="T">The data type.</typeparam>
            /// <param name="xqaParams">The XQA parameters.</param>
            /// <returns>True if the configuration should be used, false otherwise.</returns>
            template <typename T>
            bool shouldUse(const XQAParams& xqaParams)
            {
                if (xqaParams.data_type != DATA_TYPE_FP16)
                    SUPPORT_RETURN_FALSE("data type");
                const int nbQHeads = xqaParams.num_q_heads;
                const int nbKVHeads = xqaParams.num_kv_heads;
                const int nbQHeadsPerKV = nbQHeads / nbKVHeads;
                if (nbQHeadsPerKV != 8 || (nbKVHeads != 1 && nbKVHeads != 2 && nbKVHeads != 4 && nbKVHeads != 8))
                    SUPPORT_RETURN_FALSE("nbHeads");
                if (xqaParams.head_size != 128)
                    SUPPORT_RETURN_FALSE("head_size");
                if (xqaParams.unidirectional != 1)
                    SUPPORT_RETURN_FALSE("unidirectional");
                if (xqaParams.q_scaling != 1.0f)
                    SUPPORT_RETURN_FALSE("q_scaling");
                if (xqaParams.rotary_embedding_dim != xqaParams.head_size)
                    SUPPORT_RETURN_FALSE("rotary_embedding_dim");
                if (xqaParams.rotary_embedding_base != 10000.0f)
                    SUPPORT_RETURN_FALSE("rotary_embedding_base");
                if (xqaParams.rotary_embedding_scale_type != bitfusion::kernels::RotaryScalingType::kNONE)
                    SUPPORT_RETURN_FALSE("rotary_embedding_scale_type");
                if (xqaParams.mask_type != bitfusion::kernels::AttentionMaskType::CAUSAL)
                    SUPPORT_RETURN_FALSE("mask_type");
                if (xqaParams.paged_kv_cache)
                    SUPPORT_RETURN_FALSE("paged_kv_cache");
                if (xqaParams.qkv_bias_enabled)
                    SUPPORT_RETURN_FALSE("qkv_bias_enabled");
                if (xqaParams.cross_attention)
                    SUPPORT_RETURN_FALSE("cross_attention");
                if (xqaParams.host_past_key_value_lengths == nullptr)
                    SUPPORT_RETURN_FALSE("host_past_key_value_lengths");
                if (xqaParams.beam_width != 1)
                    SUPPORT_RETURN_FALSE("beam_width");
                if (xqaParams.cyclic_attention_window_size != xqaParams.max_attention_window_size)
                    SUPPORT_RETURN_FALSE("cyclic_attention_window_size != max_attention_window_size");
                return shouldUseImpl(xqaParams);
            }

            /// <summary>
            /// Get the workspace size required.
            /// </summary>
            /// <returns>The workspace size.</returns>
            size_t getWorkspaceSize();

            /// <summary>
            /// Dispatch the XQA (XQARunner) computation.
            /// </summary>
            /// <typeparam name="KVCacheBuffer">The key-value cache buffer type.</typeparam>
            /// <param name="xqa_params">The XQA parameters.</param>
            /// <param name="kv_cache_buffer">The key-value cache buffer.</param>
            /// <param name="stream">The CUDA stream.</param>
            template <typename KVCacheBuffer>
            void dispatch(const XQAParams& xqa_params, KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
            {
                TLLM_CHECK_WITH_INFO((std::is_same<KVCacheBuffer, KVLinearBuffer>::value),
                    "DecoderXQARunner.dispatch supports only KVLinearBuffer now.");
                sync_check_cuda_error();
                this->dispatchCacheBuffer(xqa_params, kv_cache_buffer, stream);
            }

        private:
            /// <summary>
            /// Dispatch the computation for KVLinearBuffer.
            /// </summary>
            /// <param name="xqa_params">The XQA parameters.</param>
            /// <param name="kv_linear_buffer">The key-value linear buffer.</param>
            /// <param name="stream">The CUDA stream.</param>
            void dispatchCacheBuffer(const XQAParams& xqa_params, KVLinearBuffer& kv_linear_buffer, const cudaStream_t& stream);

            /// <summary>
            /// Dispatch the computation for KVBlockArray (not supported).
            /// </summary>
            /// <param name="xqa_params">The XQA parameters.</param>
            /// <param name="kv_block_array">The key-value block array.</param>
            /// <param name="stream">The CUDA stream.</param>
            void dispatchCacheBuffer(const XQAParams& xqa_params, KVBlockArray& kv_block_array, const cudaStream_t& stream);

            /// <summary>
            /// Implementation to determine if the configuration should be used.
            /// </summary>
            /// <param name="xqaParams">The XQA parameters.</param>
            /// <returns>True if the configuration should be used, false otherwise.</returns>
            bool shouldUseImpl(const XQAParams& xqaParams);

            /// <summary>
            /// Run the XQA computation for KVLinearBuffer.
            /// </summary>
            /// <param name="xqa_params">The XQA parameters.</param>
            /// <param name="kv_linear_buffer">The key-value linear buffer.</param>
            /// <param name="stream">The CUDA stream.</param>
            void run(const XQAParams& xqa_params, KVLinearBuffer& kv_linear_buffer, const cudaStream_t& stream);

            /// <summary>
            /// Maximum number of concurrent thread blocks per key-value head factor.
            /// </summary>
            static constexpr int kMaxNbCtaPerKVHeadFactor = 4;

            /// <summary>
            /// Maximum beam width for multi-beam decoding.
            /// </summary>
            static constexpr int kMaxBeamWidth = 4;

            /// <summary>
            /// Implementation class for XQARunner.
            /// </summary>
            class xqaImpl;

            /// <summary>
            /// Pointer to the implementation of XQARunner.
            /// </summary>
            std::unique_ptr<xqaImpl> pimpl;

            /// <summary>
            /// Number of attention heads.
            /// </summary>
            int mNumHeads;

            /// <summary>
            /// Number of key-value attention heads.
            /// </summary>
            int mNumKVHeads;

            /// <summary>
            /// Size of each attention head.
            /// </summary>
            int mHeadSize;

            /// <summary>
            /// Flag indicating whether multi-block mode is enabled.
            /// </summary>
            bool mMultiBlockMode;

            /// <summary>
            /// Number of multi-processors available.
            /// </summary>
            int mMultiProcessorCount;
        };
    }
}
