
#pragma once

#include "../../kernels/kvCacheUtils.h"
#include "tmaDescriptor.h"
#include <limits.h>
#include <stdint.h>

#include "../../kernels/multiHeadAttentionCommon.h"

namespace bitfusion
{
namespace kernels
{

enum class ContextFMHAType
{
    DISABLED,
    ENABLED,
    ENABLED_WITH_FP32_ACC
};

enum class ContextAttentionMaskType
{
    PADDING,
    CAUSAL,
    SLIDING_WINDOW_CAUSAL
};

struct AlibiParams
{
    constexpr static int round_down_to_power_two(int x)
    {
        x = x | (x >> 1);
        x = x | (x >> 2);
        x = x | (x >> 4);
        x = x | (x >> 8);
        x = x | (x >> 16);
        return x - (x >> 1);
    }

    AlibiParams() = default;

    AlibiParams(int h, float scale_after_alibi)
        : scale_after_alibi(scale_after_alibi)
    {
        h_pow_2 = round_down_to_power_two(h);
        alibi_neg4_div_h = -4.0f / h_pow_2;
    }

    AlibiParams(int h, int s, int tp_size, int rank, float scale_after_alibi)
        : AlibiParams(h * tp_size, scale_after_alibi)
    {
        head_idx_offset = h * rank;
        sequence_pos_offset = s * rank;
    }

    int h_pow_2{};
    float alibi_neg4_div_h{};
    float scale_after_alibi{};
    int head_idx_offset = 0;
    int sequence_pos_offset = 0;
};

struct Fused_multihead_attention_params_v2
{
    const void* qkv_ptr;
    const void* packed_mask_ptr;
    void* o_ptr;

    int64_t qkv_stride_in_bytes;
    int64_t packed_mask_stride_in_bytes;
    int64_t o_stride_in_bytes;

    int b, h, s, d;
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    bool enable_i2f_trick;

    const int* cu_seqlens;

    bool interleaved = false;
    bool use_int8_scale_max = false;

    bool has_alibi = false;
    AlibiParams alibi_params{};

    int heads_per_wave;
    int *counters, *max_barriers, *sum_barriers, *locks;
    float *max_scratch_ptr, *sum_scratch_ptr;
    int* o_scratch_ptr;

    int h_kv;

    int sliding_window_size = INT_MAX;

    bool is_s_padded = false;

    cudaTmaDesc tma_desc_q;
    cudaTmaDesc tma_desc_k;
    cudaTmaDesc tma_desc_v;

    void clear()
    {
        qkv_ptr = nullptr;
        packed_mask_ptr = nullptr;
        o_ptr = nullptr;

        qkv_stride_in_bytes = 0;
        packed_mask_stride_in_bytes = 0;
        o_stride_in_bytes = 0;

        b = 0;
        h = 0;
        s = 0;
        d = 0;
        scale_bmm1 = 0;
        scale_softmax = 0;
        scale_bmm2 = 0;

        enable_i2f_trick = false;

        cu_seqlens = nullptr;
        interleaved = false;
        use_int8_scale_max = false;

        h_kv = 0;
        sliding_window_size = INT_MAX;
        is_s_padded = false;

        has_alibi = false;
        alibi_params = AlibiParams{};
    }
};

struct Fused_multihead_attention_paged_kv_params_v2
{
    const void* q_ptr;
    KVBlockArray paged_kv_cache;
    void* o_ptr;
    const void* packed_mask_ptr;

    int64_t q_stride_in_bytes;
    int64_t kv_stride_in_bytes;
    int64_t o_stride_in_bytes;
    int64_t packed_mask_stride_in_bytes;

    int b, h, s, d;
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    bool enable_i2f_trick;

    bool use_int8_scale_max = false;

    bool has_alibi = false;
    AlibiParams alibi_params;

    const int* cu_seqlens;
    const int* cu_q_seqlens;

    cudaTmaDesc tma_desc_q;
    cudaTmaDesc* tma_desc_paged_kv;

    int h_kv = 0;
    int h_q_per_kv = 0;

    int sliding_window_size = INT_MAX;

    bool is_s_padded = false;

    void clear()
    {
        q_ptr = nullptr;
        o_ptr = nullptr;
        packed_mask_ptr = nullptr;

        q_stride_in_bytes = 0;
        kv_stride_in_bytes = 0;
        o_stride_in_bytes = 0;
        packed_mask_stride_in_bytes = 0;

        b = 0;
        h = 0;
        s = 0;
        d = 0;
        scale_bmm1 = 0;
        scale_softmax = 0;
        scale_bmm2 = 0;

        enable_i2f_trick = false;

        cu_seqlens = nullptr;
        cu_q_seqlens = nullptr;
        use_int8_scale_max = false;

        h_kv = 0;
        h_q_per_kv = 0;
        sliding_window_size = INT_MAX;
        is_s_padded = false;

        has_alibi = false;
        alibi_params = AlibiParams{};
    }
};

struct Launch_params
{
    int kernel_s = 0;
    int kernel_kv_s = 0;
    bool ignore_b1opt = false;
    bool force_unroll = false;
    bool force_fp32_acc = false;
    bool interleaved = false;
    bool use_tma = false;
    int* seqlens = nullptr;
    int blocks_per_context_sequence = 0;
    const int64_t* paged_kv_block_ptrs = nullptr;
    bool flash_attention = false;
    bool warp_specialization = false;
    bool granular_tiling = false;
    ContextAttentionMaskType attention_mask_type = ContextAttentionMaskType::PADDING;
    bool useKernelWithoutAlibi = false;
    int multi_processor_count = 0;
    int device_l2_cache_size = 0;

    void set_default_kernel_selection_params()
    {
        kernel_s = 0;
        kernel_kv_s = 0;
        force_unroll = false;
        use_tma = false;
        flash_attention = false;
        warp_specialization = false;
        granular_tiling = false;
        attention_mask_type = (attention_mask_type == ContextAttentionMaskType::PADDING)
            ? ContextAttentionMaskType::PADDING
            : ContextAttentionMaskType::CAUSAL;
        useKernelWithoutAlibi = false;
    }
};

}
}
