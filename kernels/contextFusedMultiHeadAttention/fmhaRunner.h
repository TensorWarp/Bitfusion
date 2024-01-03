
#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "fused_multihead_attention_common.h"
#include "../../common/cudaUtils.h"
#include "tmaDescriptor.h"

namespace bitfusion
{
namespace kernels
{


class MHARunner
{
public:
    MHARunner(const Data_type dataType, const int numHeads, const int headSize, const float qScaling);

    MHARunner() = default;

    virtual ~MHARunner() = default;

    virtual void setup(const int b, const int s, const int sliding_window_size, const int total_seqlen,
        const bool has_alibi = false, const bool scale_alibi = false, const int tp_size = 1, const int tp_rank = 0)
        = 0;

    virtual void setup_paged_kv(const int b, const int s_q, const int s_kv, const int blocks_per_context_sequence,
        const int tokens_per_kv_block, const int sliding_window_size, const int total_seqlen,
        const bool has_alibi = false, const bool scale_alibi = false, const int tp_size = 1, const int tp_rank = 0)
        = 0;

    static bool fmha_supported(const int headSize, const int sm);

    virtual bool fmha_supported() = 0;

    virtual void setup_flags(const bool force_fp32_acc, const bool is_s_padded, const bool causal_mask,
        const int num_kv_heads)
        = 0;

    virtual void run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) = 0;

    virtual void run_paged_kv(const void* q_input, void* paged_kv_tma_desc, const void* paged_kv_block_ptrs_on_host,
        const KVBlockArray paged_kv_cache, const void* cu_q_seqlens, const void* cu_kv_seqlens, void* output,
        cudaStream_t stream)
        = 0;

    virtual bool isValid(int s) const = 0;
};



class FusedMHARunnerV2 : public MHARunner
{
public:
    FusedMHARunnerV2(const Data_type dataType, const int numHeads, const int headSize, const float qScaling);

    ~FusedMHARunnerV2();

    void setup(const int b, const int s, const int sliding_window_size, const int total_seqlen,
        const bool has_alibi = false, const bool scale_alibi = false, const int tp_size = 1,
        const int tp_rank = 0) override;

    void setup_paged_kv(const int b, const int s_q, const int s_kv, const int blocks_per_context_sequence,
        const int tokens_per_kv_block, const int sliding_window_size, const int total_seqlen,
        const bool has_alibi = false, const bool scale_alibi = false, const int tp_size = 1,
        const int tp_rank = 0) override;

    bool fmha_supported() override;

    void run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) override;
    void run_paged_kv(const void* q_input, void* paged_kv_tma_desc, const void* paged_kv_block_ptrs_on_host,
        const KVBlockArray paged_kv_cache, const void* cu_q_seqlens, const void* cu_kv_seqlens, void* output,
        cudaStream_t stream) override;

    void setup_flags(const bool force_fp32_acc, const bool is_s_padded, const bool causal_mask,
        const int num_kv_heads) override;

    bool isValid(int s) const override;

private:
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

}
}
