

#pragma once

#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <iostream>

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/tensor.h"

namespace bitfusion::kernels
{

constexpr size_t WARP_SIZE = 32;
constexpr size_t CUSTOM_AR_SIZE_THRESHOLD = 50331648;
constexpr size_t MAX_ALL_REDUCE_BLOCKS = 24;
constexpr size_t MAX_RANKS_PER_NODE = 8;
constexpr size_t DEFAULT_BLOCK_SIZE = 1024;

// Warning: python definition is in bitfusion/functional.py
// they must be kept in sync
enum class AllReduceStrategyType : int8_t
{
    RING = 0,
    ONESHOT = 1,
    TWOSHOT = 2,
    AUTO = 3,
};

#ifdef ENABLE_BF16
typedef struct bf168
{
    __nv_bfloat162 x;
    __nv_bfloat162 y;
    __nv_bfloat162 z;
    __nv_bfloat162 w;
} bf168;
#endif

struct AllReduceParams
{
    size_t elts_total;
    size_t elts_per_rank;
    size_t elts_per_block;
    size_t rank_offset;
    size_t ranks_per_node, rank, local_rank;
    uint32_t barrier_flag;
    uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
    uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
    void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
    void* local_output_buffer_ptr;

    static AllReduceParams deserialize(const int32_t* buffer, size_t tpSize, size_t tpRank, uint32_t flag_value);
};

template <typename T>
void invokeOneOrTwoShotAllReduceKernel(AllReduceParams& param, AllReduceStrategyType strat, cudaStream_t stream);

void invokeMultiGpuBarrier(AllReduceParams& param, cudaStream_t stream);

template <typename T>
struct CustomARCommTypeConverter
{
    using Type = uint32_t;
};

template <>
struct CustomARCommTypeConverter<half>
{
    using Type = uint16_t;
};

#ifdef ENABLE_BF16
template <>
struct CustomARCommTypeConverter<__nv_bfloat16>
{
    using Type = __nv_bfloat16;
};
#endif

void customAllReduce(kernels::AllReduceParams& params, void* data, size_t elts, size_t size_per_elem,
    common::datatype_enum dataType, AllReduceStrategyType strat, cudaStream_t stream);

} // namespace bitfusion::kernels
