
#pragma once

#include "../../common/cudaTypeUtils.cuh"
#include "../../common/memoryUtils.h"
#include "../../kernels/decoderMaskedMultiheadAttention.h"
#include "../../kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "../../kernels/gptKernels.h"
#include "../../kernels/kvCacheUtils.h"
#include <assert.h>
#include <float.h>
#include <type_traits>

#if (CUDART_VERSION >= 11070)
#define ENABLE_MULTI_BLOCK_OPTION
#endif

#ifdef ENABLE_MULTI_BLOCK_OPTION
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/std/bit>
#endif

namespace bitfusion
{
namespace kernels
{


#ifdef ENABLE_FP8
#define MMHA_FP8_SCALE_Q_INSTEAD_OF_K
#define MMHA_FP8_SCALE_P_INSTEAD_OF_V
#endif


#define MMHA_USE_FP32_ACCUM_FOR_FMA

#define MMHA_USE_FP32_ACCUM_FOR_OUT

#if 0 && defined(MMHA_USE_FP32_ACCUM_FOR_OUT)
#endif

namespace mmha
{




template <typename T, int Dh_MAX>
struct Qk_vec_m_
{
};

template <>
struct Qk_vec_m_<float, 32>
{
    using Type = float;
};

template <>
struct Qk_vec_m_<float, 64>
{
    using Type = float2;
};

template <>
struct Qk_vec_m_<float, 128>
{
    using Type = float4;
};

template <>
struct Qk_vec_m_<float, 256>
{
    using Type = float4;
};

template <>
struct Qk_vec_m_<uint16_t, 32>
{
    using Type = uint32_t;
};

template <>
struct Qk_vec_m_<uint16_t, 64>
{
    using Type = uint32_t;
};

template <>
struct Qk_vec_m_<uint16_t, 128>
{
    using Type = uint2;
};

template <>
struct Qk_vec_m_<uint16_t, 256>
{
    using Type = uint4;
};
#ifdef ENABLE_BF16
template <>
struct Qk_vec_m_<__nv_bfloat16, 32>
{
    using Type = __nv_bfloat162;
};

template <>
struct Qk_vec_m_<__nv_bfloat16, 64>
{
    using Type = __nv_bfloat162;
};

template <>
struct Qk_vec_m_<__nv_bfloat16, 128>
{
    using Type = bf16_4_t;
};

template <>
struct Qk_vec_m_<__nv_bfloat16, 256>
{
    using Type = bf16_8_t;
};
#endif

#ifdef ENABLE_FP8
template <>
struct Qk_vec_m_<__nv_fp8_e4m3, 32>
{
    using Type = fp8_4_t;
};

template <>
struct Qk_vec_m_<__nv_fp8_e4m3, 64>
{
    using Type = fp8_4_t;
};

template <>
struct Qk_vec_m_<__nv_fp8_e4m3, 128>
{
    using Type = fp8_4_t;
};

template <>
struct Qk_vec_m_<__nv_fp8_e4m3, 256>
{
    using Type = fp8_4_t;
};
#endif


template <typename T, int Dh>
struct Qk_vec_k_
{
    using Type = typename Qk_vec_m_<T, Dh>::Type;
};
#ifdef ENABLE_FP8
template <>
struct Qk_vec_k_<__nv_fp8_e4m3, 32>
{
    using Type = float4;
};

template <>
struct Qk_vec_k_<__nv_fp8_e4m3, 64>
{
    using Type = float4;
};

template <>
struct Qk_vec_k_<__nv_fp8_e4m3, 128>
{
    using Type = float4;
};

template <>
struct Qk_vec_k_<__nv_fp8_e4m3, 256>
{
    using Type = float4;
};
#endif


template <typename T, int V_VEC_SIZE>
struct V_vec_m_
{
};

template <>
struct V_vec_m_<float, 1>
{
    using Type = float;
};

template <>
struct V_vec_m_<float, 2>
{
    using Type = float2;
};

template <>
struct V_vec_m_<float, 4>
{
    using Type = float4;
};

template <>
struct V_vec_m_<float, 8>
{
    using Type = Float8_;
};

template <>
struct V_vec_m_<uint16_t, 2>
{
    using Type = uint32_t;
};

template <>
struct V_vec_m_<uint16_t, 4>
{
    using Type = uint2;
};

template <>
struct V_vec_m_<uint16_t, 8>
{
    using Type = uint4;
};
#ifdef ENABLE_BF16
template <>
struct V_vec_m_<__nv_bfloat16, 2>
{
    using Type = __nv_bfloat162;
};

template <>
struct V_vec_m_<__nv_bfloat16, 4>
{
    using Type = bf16_4_t;
};

template <>
struct V_vec_m_<__nv_bfloat16, 8>
{
    using Type = bf16_8_t;
};
#endif


template <typename T, int V_VEC_SIZE>
struct V_vec_k_
{
    using Type = typename V_vec_m_<T, V_VEC_SIZE>::Type;
};
#ifdef ENABLE_FP8
template <>
struct V_vec_k_<__nv_fp8_e4m3, 4>
{
    using Type = float4;
};

template <>
struct V_vec_k_<__nv_fp8_e4m3, 8>
{
    using Type = float4;
};

template <>
struct V_vec_k_<__nv_fp8_e4m3, 16>
{
    using Type = float4;
};
#endif


template <typename T, int K_VEC_SIZE>
struct K_vec_m_
{
    using Type = typename V_vec_m_<T, K_VEC_SIZE>::Type;
};


template <typename T, int K_VEC_SIZE>
struct K_vec_k_
{
    using Type = typename K_vec_m_<T, K_VEC_SIZE>::Type;
};


#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
template <typename T>
struct Qk_vec_accum_fp32_
{
};

template <>
struct Qk_vec_accum_fp32_<float>
{
    using Type = float;
};

template <>
struct Qk_vec_accum_fp32_<float2>
{
    using Type = float2;
};

template <>
struct Qk_vec_accum_fp32_<float4>
{
    using Type = float4;
};

template <>
struct Qk_vec_accum_fp32_<uint32_t>
{
    using Type = float2;
};

template <>
struct Qk_vec_accum_fp32_<uint2>
{
    using Type = Float4_;
};

template <>
struct Qk_vec_accum_fp32_<uint4>
{
    using Type = Float8_;
};

template <>
struct Qk_vec_accum_fp32_<__nv_bfloat16>
{
    using Type = float;
};

template <>
struct Qk_vec_accum_fp32_<__nv_bfloat162>
{
    using Type = float2;
};

template <>
struct Qk_vec_accum_fp32_<bf16_4_t>
{
    using Type = Float4_;
};

template <>
struct Qk_vec_accum_fp32_<bf16_8_t>
{
    using Type = Float8_;
};

#ifdef ENABLE_FP8
template <>
struct Qk_vec_accum_fp32_<fp8_4_t>
{
    using Type = Float4_;
};

#endif


template <typename T>
struct K_vec_accum_fp32_
{
};

template <>
struct K_vec_accum_fp32_<float>
{
    using Type = float;
};

template <>
struct K_vec_accum_fp32_<float2>
{
    using Type = float2;
};

template <>
struct K_vec_accum_fp32_<float4>
{
    using Type = float4;
};

template <>
struct K_vec_accum_fp32_<Float8_>
{
    using Type = Float8_;
};

template <>
struct K_vec_accum_fp32_<uint32_t>
{
    using Type = float2;
};

template <>
struct K_vec_accum_fp32_<uint2>
{
    using Type = Float4_;
};

template <>
struct K_vec_accum_fp32_<uint4>
{
    using Type = Float8_;
};

template <>
struct K_vec_accum_fp32_<__nv_bfloat16>
{
    using Type = float;
};

template <>
struct K_vec_accum_fp32_<__nv_bfloat162>
{
    using Type = float2;
};

template <>
struct K_vec_accum_fp32_<bf16_4_t>
{
    using Type = Float4_;
};

template <>
struct K_vec_accum_fp32_<bf16_8_t>
{
    using Type = Float8_;
};
#ifdef ENABLE_FP8
template <>
struct K_vec_accum_fp32_<__nv_fp8_e4m3>
{
    using Type = float;
};

template <>
struct K_vec_accum_fp32_<fp8_2_t>
{
    using Type = float2;
};

template <>
struct K_vec_accum_fp32_<fp8_4_t>
{
    using Type = Float4_;
};

template <>
struct K_vec_accum_fp32_<fp8_8_t>
{
    using Type = Float8_;
};
#endif

template <>
struct K_vec_accum_fp32_<int8_t>
{
    using Type = float;
};

template <>
struct K_vec_accum_fp32_<int16_t>
{
    using Type = float2;
};

template <>
struct K_vec_accum_fp32_<int32_t>
{
    using Type = Float4_;
};

template <>
struct K_vec_accum_fp32_<int64_t>
{
    using Type = Float8_;
};

#endif


#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
template <typename T>
struct V_vec_accum_fp32_
{
};

template <>
struct V_vec_accum_fp32_<float>
{
    using Type = float;
};

template <>
struct V_vec_accum_fp32_<float2>
{
    using Type = float2;
};

template <>
struct V_vec_accum_fp32_<float4>
{
    using Type = float4;
};

template <>
struct V_vec_accum_fp32_<uint32_t>
{
    using Type = float2;
};

template <>
struct V_vec_accum_fp32_<uint2>
{
    using Type = Float4_;
};

template <>
struct V_vec_accum_fp32_<uint4>
{
    using Type = Float8_;
};
#ifdef ENABLE_BF16
template <>
struct V_vec_accum_fp32_<__nv_bfloat162>
{
    using Type = float2;
};

template <>
struct V_vec_accum_fp32_<bf16_4_t>
{
    using Type = Float4_;
};

template <>
struct V_vec_accum_fp32_<bf16_8_t>
{
    using Type = Float8_;
};
#endif
#ifdef ENABLE_FP8
template <>
struct V_vec_accum_fp32_<fp8_4_t>
{
    using Type = Float4_;
};

#endif
#endif


template <typename Tout, typename Tin>
__inline__ __device__ constexpr Tout vec_conversion(const Tin& x)
{
    static_assert(std::is_same<Tout, Tin>::value, "Type mismatch");
    return x;
}

template <>
__inline__ __device__ Float8_ vec_conversion<Float8_, uint4>(const uint4& a)
{
    Float8_ fc;
    fc.x = half2_to_float2(a.x);
    fc.y = half2_to_float2(a.y);
    fc.z = half2_to_float2(a.z);
    fc.w = half2_to_float2(a.w);
    return fc;
}

#ifdef ENABLE_BF16
template <>
__inline__ __device__ Float8_ vec_conversion<Float8_, bf16_8_t>(const bf16_8_t& a)
{
    Float8_ fc;
    fc.x = bf1622float2(a.x);
    fc.y = bf1622float2(a.y);
    fc.z = bf1622float2(a.z);
    fc.w = bf1622float2(a.w);
    return fc;
}
#endif

#ifdef ENABLE_FP8
template <>
__inline__ __device__ float vec_conversion<float, __nv_fp8_e4m3>(const __nv_fp8_e4m3& a)
{
    return float(a);
}

template <>
__inline__ __device__ __nv_fp8_e4m3 vec_conversion<__nv_fp8_e4m3, float>(const float& a)
{
    return __nv_fp8_e4m3(a);
}

template <>
__inline__ __device__ float2 vec_conversion<float2, fp8_2_t>(const fp8_2_t& a)
{
    return float2(a);
}

template <>
__inline__ __device__ fp8_2_t vec_conversion<fp8_2_t, float2>(const float2& a)
{
    return fp8_2_t(a);
}

template <>
__inline__ __device__ float4 vec_conversion<float4, fp8_4_t>(const fp8_4_t& a)
{
    return float4(a);
}

template <>
__inline__ __device__ fp8_4_t vec_conversion<fp8_4_t, float4>(const float4& a)
{
    return fp8_4_t(a);
}
#endif


template <int THREADS_PER_KEY, typename Q_vec, typename K_vec, int N>
inline __device__ float qk_dot_(const Q_vec (&q)[N], const K_vec (&k)[N])
{
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using K_vec_accum = typename K_vec_accum_fp32_<K_vec>::Type;
#else
    using K_vec_accum = K_vec;
#endif
    K_vec_accum qk_vec = mul<K_vec_accum, Q_vec, K_vec>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii)
    {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }

    float qk = sum(qk_vec);
#pragma unroll
    for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2)
    {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

template <int THREADS_PER_KEY, typename Q_vec, typename K_vec, int N>
inline __device__ float qk_scale_dot_(const Q_vec (&q)[N], const K_vec (&k)[N], const float k_scale)
{
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using K_vec_accum = typename K_vec_accum_fp32_<K_vec>::Type;
#else
    using K_vec_accum = K_vec;
#endif
    K_vec_accum k_vec = mul<K_vec_accum, float, K_vec>(k_scale, k[0]);
    K_vec_accum qk_vec = mul<K_vec_accum, Q_vec, K_vec_accum>(q[0], k_vec);
#pragma unroll
    for (int ii = 1; ii < N; ++ii)
    {
        K_vec_accum k_vec = mul<K_vec_accum, float, K_vec>(k_scale, k[ii]);
        qk_vec = fma(q[ii], k_vec, qk_vec);
    }

    float qk = sum(qk_vec);
#pragma unroll
    for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2)
    {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}


template <typename T, int THREADS_PER_KEY>
struct Qk_dot
{
    template <typename Q_vec, typename K_vec, int N>
    static inline __device__ float dot(const Q_vec (&q)[N], const K_vec (&k)[N])
    {
        return qk_dot_<THREADS_PER_KEY>(q, k);
    }

    template <typename Q_vec, typename K_vec, int N>
    static inline __device__ float scale_dot(const Q_vec (&q)[N], const K_vec (&k)[N], const float k_scale)
    {
#ifdef MMHA_USE_HMMA
        static_assert("HMMA doesn't support k scales");
#endif
        return qk_scale_dot_<THREADS_PER_KEY>(q, k, k_scale);
    }

    template <int WARP_SIZE = 32>
    static inline __device__ bool is_leader(const int tidx)
    {
        return (tidx % THREADS_PER_KEY) == 0;
    }
};


template <typename K_vec>
inline __device__ void hmma_fp32(float4& c, const K_vec& a, K_vec b)
{
    assert(false);
}

template <>
inline __device__ void hmma_fp32(float4& c, const uint32_t& a, uint32_t b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
        "    {%0, %1, %2, %3}, \n"
        "    {%4, %5}, \n"
        "    {%6}, \n"
        "    {%0, %1, %2, %3}; \n"
        : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w)
        : "r"(a), "r"(a), "r"(b));
}

template <>
inline __device__ void hmma_fp32(float4& c, const uint2& a, uint2 b)
{
    hmma_fp32(c, a.x, b.x);
    hmma_fp32(c, a.y, b.y);
}

template <>
inline __device__ void hmma_fp32(float4& c, const uint4& a, uint4 b)
{
    hmma_fp32(c, a.x, b.x);
    hmma_fp32(c, a.y, b.y);
    hmma_fp32(c, a.z, b.z);
    hmma_fp32(c, a.w, b.w);
}


template <typename K_vec, int THREADS_PER_KEY, int N>
inline __device__ float qk_hmma_dot_(const K_vec (&q)[N], const K_vec (&k)[N])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        hmma_fp32(acc, q[ii], k[ii]);
    }

    int lane = threadIdx.x % 32;

    int row = lane / 4;
    int col = lane % 4 * 2;

    float result = (row == col) ? acc.x : acc.y;

    if (THREADS_PER_KEY > 4)
    {
        result += __shfl_xor_sync(unsigned(-1), result, 4);
    }
    if (THREADS_PER_KEY > 8)
    {
        result += __shfl_xor_sync(unsigned(-1), result, 9);
    }
    if (THREADS_PER_KEY > 16)
    {
        result += __shfl_xor_sync(unsigned(-1), result, 18);
    }

    return result;

#else
    return 0.f;
#endif
}


template <int THREADS_PER_KEY>
struct Qk_dot<uint16_t, THREADS_PER_KEY>
{
    template <typename Q_vec, typename K_vec, int N>
    static inline __device__ float dot(const Q_vec (&q)[N], const K_vec (&k)[N])
    {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA)
        return qk_hmma_dot_<K_vec, THREADS_PER_KEY, N>(q, k);
#else
        return qk_dot_<THREADS_PER_KEY>(q, k);
#endif
    }

    template <typename Q_vec, typename K_vec, int N>
    static inline __device__ float scale_dot(const Q_vec (&q)[N], const K_vec (&k)[N], const float k_scale)
    {
#ifdef MMHA_USE_HMMA
        static_assert("HMMA doesn't support k scales");
#endif
        return qk_scale_dot_<THREADS_PER_KEY>(q, k, k_scale);
    }

    template <int WARP_SIZE = 32>
    static inline __device__ bool is_leader(const int tidx)
    {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA)
        int leader = 0;
        int lane = tidx % WARP_SIZE;
        if (THREADS_PER_KEY == 4)
        {
            leader = int(lane / 8);
        }
        else
        {
            leader = int(lane / THREADS_PER_KEY) * int(THREADS_PER_KEY / 8);
        }
#else
        const bool leader = 0;
#endif
        return (tidx % THREADS_PER_KEY) == leader;
    }
};


template <typename Tk, typename V_vec_accum, typename V_vec_m, bool INT8_KV_CACHE, bool FP8_KV_CACHE>
inline __device__ void Logit_value_fma(
    V_vec_accum& out, const Tk* logits_smem, const V_vec_m& v_vec, const float v_scale, const bool is_mask)
{
#if defined(MMHA_USE_FP32_ACCUM_FOR_LOGITS)
    float logit = is_mask ? 0.f : reinterpret_cast<float*>(logits_smem)[0];
    if constexpr (INT8_KV_CACHE)
    {
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out = fma(logit, cast_to_float(v_vec_), out);
    }
    else if constexpr (FP8_KV_CACHE)
    {
#ifdef MMHA_FP8_SCALE_P_INSTEAD_OF_V
        out = fma(logit, cast_to_float(v_vec), out);
#else
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out = fma(logit, cast_to_float(v_vec_), out);
#endif
    }
    else
    {
        out = fma(logit, cast_to_float(v_vec), out);
    }
#else
    Tk logit = is_mask ? Tk(0.f) : logits_smem[0];
    if constexpr (INT8_KV_CACHE)
    {
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out = fma(logit, v_vec_, out);
    }
    else if constexpr (FP8_KV_CACHE)
    {
#ifdef MMHA_FP8_SCALE_P_INSTEAD_OF_V
        out = fma(logit, v_vec, out);
#else
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out = fma(logit, v_vec_, out);
#endif
    }
    else
    {
        out = fma(logit, v_vec, out);
    }
#endif
};


template <int WARPS_PER_BLOCK, int WARP_SIZE = 32>
inline __device__ float block_sum(float* red_smem, float sum)
{

    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
    {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    if (lane == 0)
    {
        red_smem[warp] = sum;
    }

    __syncthreads();

    if (lane < WARPS_PER_BLOCK)
    {
        sum = red_smem[lane];
    }

#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2)
    {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    return __shfl_sync(uint32_t(-1), sum, 0);
}

#if defined(MMHA_USE_FP32_ACCUM_FOR_LOGITS)


inline __device__ float cast_to_float(float u)
{
    return u;
}


inline __device__ float2 cast_to_float(float2 u)
{
    return u;
}


inline __device__ float4 cast_to_float(float4 u)
{
    return u;
}


inline __device__ Float4_ cast_to_float(Float4_ u)
{
    return u;
}


inline __device__ Float8_ cast_to_float(Float8_ u)
{
    return u;
}


inline __device__ float2 cast_to_float(uint32_t u)
{
    return half2_to_float2(u);
}


inline __device__ Float4_ cast_to_float(uint2 u)
{
    Float4_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    return tmp;
}


inline __device__ Float8_ cast_to_float(uint4 u)
{
    Float8_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    tmp.z = half2_to_float2(u.z);
    tmp.w = half2_to_float2(u.w);
    return tmp;
}


inline __device__ float2 cast_to_float(__nv_bfloat162 u)
{
    float2 tmp;
    tmp = __bfloat1622float2(u);
    return tmp;
}


inline __device__ Float4_ cast_to_float(bf16_4_t u)
{
    Float4_ tmp;
    tmp.x = __bfloat1622float2(u.x);
    tmp.y = __bfloat1622float2(u.y);
    return tmp;
}


inline __device__ Float8_ cast_to_float(bf16_8_t u)
{
    Float8_ tmp;
    tmp.x = __bfloat1622float2(u.x);
    tmp.y = __bfloat1622float2(u.y);
    tmp.z = __bfloat1622float2(u.z);
    tmp.w = __bfloat1622float2(u.w);
    return tmp;
}

#endif


template <typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}


template <typename T>
inline __device__ __host__ T div(T m, T n)
{
    return m / n;
}


template <typename T>
struct kernel_type_t
{
    using Type = T;
};


inline __device__ __host__ constexpr unsigned dh_max(unsigned dh)
{
    return next_power_of_two(mmha::const_max(dh, 32u));
}


template <typename T>
inline __device__ __host__ constexpr unsigned threads_per_value(unsigned dh_max)
{
    return dh_max * sizeof(T) / 16;
}


template <typename T, unsigned Dh_MAX>
inline __device__ __host__ constexpr unsigned threads_per_key()
{
    constexpr unsigned threads = (unsigned) (Dh_MAX * sizeof(T) / 16u);
    if ((threads & (threads - 1)) != 0)
    {
        assert(false);
    }
    return std::min(32u, threads);
}


inline __device__ constexpr uint32_t shfl_mask(int threads)
{
    assert(threads <= 32);
    return threads == 32 ? -1u : (1u << threads) - 1u;
}


template <typename T, typename T_VEC, unsigned VECS_PER_CHUNK>
__device__ inline constexpr uint2 chunk_index(unsigned tidx)
{
    auto const idx_chunk = tidx / VECS_PER_CHUNK;

    static_assert(sizeof(T_VEC) % sizeof(T) == 0);
    unsigned constexpr kVecSize{sizeof(T_VEC) / sizeof(T)};
    auto const idx_vec = (tidx % VECS_PER_CHUNK) * kVecSize;

    return uint2{idx_chunk, idx_vec};
}


template <
    typename T,
    typename Tcache,
    typename KVCacheBuffer,
    unsigned Dh,
    unsigned THREADS_PER_BLOCK,
    bool DO_CROSS_ATTENTION,
    bool HAS_BEAMS,
    bool DO_MULTI_BLOCK = false,
    unsigned THREADS_PER_KEY = threads_per_key<T, dh_max(Dh)>(),
    unsigned THREADS_PER_VALUE = threads_per_value<T>(dh_max(Dh)),
    unsigned K_LOOP_UNROLL = 4,
    unsigned V_LOOP_UNROLL = 8>
__global__ void masked_multihead_attention_kernel(
    Multihead_attention_params<T, DO_CROSS_ATTENTION> params, KVCacheBuffer kvCacheBuffer)
{

    using Tk = typename kernel_type_t<T>::Type;
    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;
    static constexpr bool FP8_KV_CACHE = std::is_same<Tcache, __nv_fp8_e4m3>::value;
    static constexpr bool INT8_KV_CACHE = std::is_same<Tcache, int8_t>::value;

    constexpr unsigned WARP_SIZE{32};
    constexpr unsigned WARPS_PER_BLOCK{THREADS_PER_BLOCK / WARP_SIZE};

    constexpr auto Dh_MAX = dh_max(Dh);
    constexpr bool IS_Dh_MAX = Dh == Dh_MAX;
    static_assert(Dh_MAX >= WARP_SIZE);
    static_assert(Dh_MAX >= Dh);

    const auto cyclic_kv_cache_len = static_cast<unsigned>(params.cyclic_attention_window_size);
    const auto timestep = static_cast<unsigned>(DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep);

#ifdef ENABLE_MULTI_BLOCK_OPTION
    constexpr bool MULTI_BLOCK_FLAG = DO_MULTI_BLOCK;
#else
    constexpr bool MULTI_BLOCK_FLAG = false;
#endif

    extern __shared__ char smem_[];

    auto qk_smem = reinterpret_cast<float*>(smem_);

    __shared__ float qk_current_smem[1];

    char* logits_smem_ = smem_;
#ifndef MMHA_USE_FP32_ACCUM_FOR_LOGITS
    if (sizeof(Tk) != 4)
    {
        const auto max_timesteps = DO_CROSS_ATTENTION ? cyclic_kv_cache_len : min(timestep, cyclic_kv_cache_len);
        logits_smem_ += divUp(max_timesteps + 1, 4u) * 16;
    }
    Tk* logits_smem = reinterpret_cast<Tk*>(logits_smem_);
#else
    float* logits_smem = reinterpret_cast<float*>(logits_smem_);
#endif

    __shared__ Tk logits_current_smem[1];

    Tk* out_smem = reinterpret_cast<Tk*>(smem_);

    __shared__ float red_smem[WARPS_PER_BLOCK * 2];

    using Qk_vec_m = typename Qk_vec_m_<T, Dh_MAX>::Type;
    using Qk_vec_k = typename Qk_vec_k_<T, Dh_MAX>::Type;
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using Qk_vec_accum = typename Qk_vec_accum_fp32_<Qk_vec_k>::Type;
#else
    using Qk_vec_accum = Qk_vec_k;
#endif

    static_assert(Dh_MAX % THREADS_PER_KEY == 0);

    constexpr int K_VEC_SIZE = 16u / sizeof(T);
    static_assert(Dh_MAX % K_VEC_SIZE == 0);
    using K_vec_k = typename K_vec_k_<T, K_VEC_SIZE>::Type;
    using K_vec_m = typename packed_type<Tcache, num_elems<K_vec_k>::value>::type;
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using K_vec_accum = typename Qk_vec_accum_fp32_<K_vec_k>::Type;
#else
    using K_vec_accum = K_vec_k;
#endif

    __shared__ __align__(mmha::const_max(sizeof(Qk_vec_k), sizeof(K_vec_k))) Tk q_smem[Dh_MAX];
    __shared__ __align__(mmha::const_max(sizeof(Qk_vec_k), sizeof(K_vec_k))) Tk k_smem[Dh_MAX];

    static_assert(Dh_MAX % THREADS_PER_VALUE == 0);

    constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
    using V_vec_k = typename V_vec_k_<T, V_VEC_SIZE>::Type;
    using V_vec_m = typename packed_type<Tcache, num_elems<V_vec_k>::value>::type;
    static_assert(V_VEC_SIZE == sizeof(V_vec_k) / sizeof(T));

    constexpr auto bias_smem_size = DO_CROSS_ATTENTION ? Dh_MAX : 1u;
    __shared__ __align__(mmha::const_max(mmha::const_max(sizeof(Qk_vec_k), sizeof(K_vec_k)), sizeof(V_vec_k)))
        Tk bias_smem[bias_smem_size];

    constexpr unsigned QK_VEC_SIZE{sizeof(Qk_vec_m) / sizeof(T)};
    static_assert(Dh_MAX % QK_VEC_SIZE == 0);
    constexpr unsigned QK_VECS_PER_Dh_MAX{Dh_MAX / QK_VEC_SIZE};
    static_assert(THREADS_PER_BLOCK >= QK_VECS_PER_Dh_MAX);

    const auto batch_beam_idx = blockIdx.y;
    if (params.finished != nullptr && params.finished[batch_beam_idx])
    {
        return;
    }

    const unsigned hi{blockIdx.x};
    const int qhead_per_kv{params.num_heads / params.num_kv_heads};
    const unsigned hi_kv{hi / qhead_per_kv};
    const auto num_heads = static_cast<unsigned>(params.num_heads);
    const auto num_heads_kv = static_cast<unsigned>(params.num_kv_heads);

    const unsigned tidx{threadIdx.x};

    const unsigned c_tile{MULTI_BLOCK_FLAG ? blockIdx.z : 0};

    static constexpr bool HANDLE_KV{!DO_CROSS_ATTENTION};

    float qk_max = -FLT_MAX;

    float qk = 0.0F;

    bool has_relative_attention_bias = params.relative_attention_bias != nullptr;
    const bool implicit_rel_attn_bias = params.max_distance != 0 && has_relative_attention_bias;
    int relative_attention_bias_stride
        = params.relative_attention_bias_stride;
    int max_distance = params.max_distance;

    const int tlength = DO_CROSS_ATTENTION
        ? params.memory_length_per_sample[batch_beam_idx] - 1
        : (params.length_per_sample ? (params.length_per_sample[batch_beam_idx] - 1) : static_cast<int>(timestep));
    const int cyclic_tlength = tlength % cyclic_kv_cache_len;
    const int kv_loop_length = min(tlength, cyclic_kv_cache_len);
    const int beam0_context_length
        = HAS_BEAMS && tlength > cyclic_kv_cache_len ? 0 : params.input_lengths[batch_beam_idx];

    const auto qk_vec_idx = tidx * QK_VEC_SIZE;
    const auto is_valid_qk_vec = qk_vec_idx < Dh;

    const bool load_qkv_quant = params.qkv_scale_quant_orig != nullptr;
    const bool write_attention_quant = params.attention_out_scale_orig_quant != nullptr;

    using T_scale = typename kv_cache_scale_type_t<T, Tcache>::Type;
    T_scale kv_scale_orig_quant, kv_scale_quant_orig;
    const float kv_scale_quant_orig_f = (ENABLE_8BITS_CACHE ? params.kv_scale_quant_orig[0] : 1.0f);
    convert_from_float(&kv_scale_quant_orig, kv_scale_quant_orig_f);
    convert_from_float(&kv_scale_orig_quant, (ENABLE_8BITS_CACHE ? params.kv_scale_orig_quant[0] : 1.0f));

    Qk_vec_k q, k, q_bias, k_bias;
    zero(q);
    zero(k);
    zero(q_bias);
    zero(k_bias);
    float rotary_embedding_base = params.rotary_embedding_base;
    float rotary_embedding_scale = params.rotary_embedding_scale;
    if (is_valid_qk_vec)
    {
        mmha::update_rotary_base_n_scale(rotary_embedding_base, rotary_embedding_scale,
            params.rotary_embedding_scale_type, params.rotary_embedding_dim, params.rotary_embedding_max_positions,
            tlength);
        uint32_t q_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads * Dh);
        const auto q_offset = bitfusion::common::flat_index_strided3(batch_beam_idx, hi, qk_vec_idx, q_stride, Dh);

        if (load_qkv_quant)
        {
            using Packed_Int8_t = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
            using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
            const auto q_scaling = params.qkv_scale_quant_orig[0];
            const auto q_quant
                = *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.q)[q_offset]);
            convert_from_float(&q, mul<Packed_Float_t, float>(q_scaling, float_from_int8(q_quant)));
        }
        else
        {
            q = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.q[q_offset]));
        }

        if constexpr (DO_CROSS_ATTENTION)
        {
            const auto k_idx = QK_VEC_SIZE * tidx;
            const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(cyclic_tlength, hi, Dh, k_idx);
            Tcache* k_cache = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(batch_beam_idx, cyclic_tlength));

            k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&k_cache[inBlockIdx]));
        }
        else
        {
            uint32_t k_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
            const auto k_offset
                = bitfusion::common::flat_index_strided3(batch_beam_idx, hi_kv, qk_vec_idx, k_stride, Dh);

            if (load_qkv_quant)
            {
                using Packed_Int8_t = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
                const auto k_scaling = params.qkv_scale_quant_orig[1];
                const auto k_quant
                    = *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.k)[k_offset]);

                convert_from_float(&k, mul<Packed_Float_t, float>(k_scaling, float_from_int8(k_quant)));
            }
            else
            {
                k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.k[k_offset]));
            }
        }

        if (params.q_bias != nullptr)
        {
            const auto q_bias_offset = bitfusion::common::flat_index2(hi, qk_vec_idx, Dh);
            q_bias
                = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.q_bias[q_bias_offset]));
        }
        if (HANDLE_KV && params.k_bias != nullptr)
        {
            const auto k_bias_offset = bitfusion::common::flat_index2(hi_kv, qk_vec_idx, Dh);
            k_bias
                = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.k_bias[k_bias_offset]));
        }
    }

    q = add(q, q_bias);
    if (HANDLE_KV)
    {
        k = add(k, k_bias);
    }

    const auto beam_width = static_cast<unsigned>(params.beam_width);
    const int batch_idx = batch_beam_idx / beam_width;
    const bool do_ia3 = HANDLE_KV && params.ia3_tasks != nullptr;
    const auto ia3_ti_hi = do_ia3
        ? bitfusion::common::flat_index2(static_cast<unsigned>(params.ia3_tasks[batch_idx]), hi, num_heads)
        : 0;

    if (do_ia3 && is_valid_qk_vec)
    {
        k = mul<Qk_vec_k, Qk_vec_k, Qk_vec_k>(k,
            vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(
                &params.ia3_key_weights[bitfusion::common::flat_index2(ia3_ti_hi, qk_vec_idx, Dh)])));
    }

    switch (params.position_embedding_type)
    {
    case PositionEmbeddingType::kLEARNED_ABSOLUTE:
    case PositionEmbeddingType::kRELATIVE:
    case PositionEmbeddingType::kALIBI:
    case PositionEmbeddingType::kALIBI_WITH_SCALE:
    {
        break;
    }
    case PositionEmbeddingType::kROPE_GPTJ:
    {
        if (HANDLE_KV)
        {
            apply_rotary_embedding(q, k, tidx, params.rotary_embedding_dim, params.rotary_embedding_base,
                params.rotary_embedding_scale, tlength);
        }
        else
        {
            apply_rotary_embedding(q, tidx, params.rotary_embedding_dim, params.rotary_embedding_base,
                params.rotary_embedding_scale, tlength);
        }
        break;
    }
    case PositionEmbeddingType::kROPE_GPT_NEOX:
    {
        const bool do_rotary = is_valid_qk_vec && QK_VEC_SIZE * tidx < params.rotary_embedding_dim;

        T* q_smem_ = reinterpret_cast<T*>(smem_);
        T* k_smem_ = q_smem_ + params.rotary_embedding_dim;

        const int half_rotary_dim = params.rotary_embedding_dim / 2;
        const int half_idx = qk_vec_idx / half_rotary_dim;
        const int intra_half_idx = qk_vec_idx % half_rotary_dim;
        const int smem_pitch = half_rotary_dim;

        assert(half_rotary_dim % QK_VEC_SIZE == 0);

        if (do_rotary)
        {
            *reinterpret_cast<Qk_vec_k*>(q_smem_ + half_idx * smem_pitch + intra_half_idx) = q;
            if (HANDLE_KV)
            {
                *reinterpret_cast<Qk_vec_k*>(k_smem_ + half_idx * smem_pitch + intra_half_idx) = k;
            }
        }

        __syncthreads();

        const int transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
        constexpr int tidx_factor = (QK_VEC_SIZE > 1) ? QK_VEC_SIZE / 2 : 1;
        if (do_rotary)
        {
            mmha::vec_from_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
            if (HANDLE_KV)
            {
                mmha::vec_from_smem_transpose(k, k_smem_, transpose_idx, smem_pitch);

                mmha::apply_rotary_embedding(q, k, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                    rotary_embedding_base, rotary_embedding_scale, tlength);

                mmha::write_smem_transpose(k, k_smem_, transpose_idx, smem_pitch);
            }
            else
            {
                mmha::apply_rotary_embedding(q, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                    rotary_embedding_base, rotary_embedding_scale, tlength);
            }
            mmha::write_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
        }

        __syncthreads();

        if (do_rotary)
        {
            q = *reinterpret_cast<Qk_vec_k*>(q_smem_ + half_idx * smem_pitch + intra_half_idx);
            if (HANDLE_KV)
            {
                k = *reinterpret_cast<Qk_vec_k*>(k_smem_ + half_idx * smem_pitch + intra_half_idx);
            }
        }

        __syncthreads();
        break;
    }
    }

    if (qk_vec_idx < Dh_MAX)
    {

#ifdef MMHA_FP8_SCALE_Q_INSTEAD_OF_K
        if constexpr (FP8_KV_CACHE)
        {
            Qk_vec_k scaled_q;
            zero(scaled_q);
            if (is_valid_qk_vec)
            {
                scaled_q = mul<Qk_vec_k, Tk, Qk_vec_k>(kv_scale_quant_orig, q);
            }
            reinterpret_cast<Qk_vec_k*>(&q_smem[qk_vec_idx])[0] = scaled_q;
        }
        else
#endif
        {
            Qk_vec_k zero_q;
            zero(zero_q);
            reinterpret_cast<Qk_vec_k*>(&q_smem[qk_vec_idx])[0] = is_valid_qk_vec ? q : zero_q;
        }

        reinterpret_cast<Qk_vec_k*>(&k_smem[qk_vec_idx])[0] = k;

        qk = dot<Qk_vec_accum, Qk_vec_k>(q, k);
        if (QK_VECS_PER_Dh_MAX <= WARP_SIZE)
        {
#pragma unroll
            for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2)
            {
                qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), qk, mask);
            }
        }
    }

    if (QK_VECS_PER_Dh_MAX > WARP_SIZE)
    {
        constexpr int WARPS_PER_RED = (QK_VECS_PER_Dh_MAX + WARP_SIZE - 1) / WARP_SIZE;
        qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
    }

    const T* relative_attention_bias_ptr = nullptr;
    const T* relative_attention_bias_ptr_fixed = nullptr;
    if (has_relative_attention_bias)
    {
        int64_t offset = implicit_rel_attn_bias
            ? ((int64_t) hi * relative_attention_bias_stride - tlength)
            : ((int64_t) hi * relative_attention_bias_stride + tlength) * relative_attention_bias_stride;
        relative_attention_bias_ptr = &params.relative_attention_bias[offset];
        relative_attention_bias_ptr_fixed = &params.relative_attention_bias[offset];
    }

    float relative_attention_bias = 0.f;
    if (has_relative_attention_bias && tidx == 0)
    {
        relative_attention_bias = add(relative_attention_bias, relative_attention_bias_ptr[tlength]);
    }

    if (tidx == 0)
    {
        qk = qk * params.inv_sqrt_dh + relative_attention_bias;

        qk_max = qk;

        if (MULTI_BLOCK_FLAG)
        {
            qk_current_smem[0] = qk;
        }
        else
        {
            qk_smem[kv_loop_length] = qk;
        }
    }

    __syncthreads();

    constexpr unsigned K_ELTS_PER_CHUNK{THREADS_PER_KEY * K_VEC_SIZE};

    const auto k_idx = chunk_index<T, K_vec_k, THREADS_PER_KEY>(tidx);

    constexpr unsigned K_VECS_PER_THREAD{Dh_MAX / K_ELTS_PER_CHUNK};
    static_assert(Dh_MAX == K_ELTS_PER_CHUNK * K_VECS_PER_THREAD);

    K_vec_accum q_vec[K_VECS_PER_THREAD];
#pragma unroll
    for (unsigned ii = 0; ii < K_VECS_PER_THREAD; ++ii)
    {
        q_vec[ii] = vec_conversion<K_vec_accum, K_vec_k>(*reinterpret_cast<const K_vec_k*>(
            &q_smem[bitfusion::common::flat_index2(ii, k_idx.y, K_ELTS_PER_CHUNK)]));
    }

    constexpr unsigned K_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_KEY};
    constexpr unsigned K_PER_WARP{WARP_SIZE / THREADS_PER_KEY};
    constexpr unsigned UNROLLED_K_PER_WARP = K_PER_WARP * K_LOOP_UNROLL;
    constexpr unsigned UNROLLED_K_PER_ITER = K_PER_ITER * K_LOOP_UNROLL;

    void** k_cache_base_row_ptr = reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::K_IDX, batch_beam_idx));

    const auto timesteps_per_block = static_cast<unsigned>(params.timesteps_per_block);

    const int context_length
        = DO_CROSS_ATTENTION ? kv_loop_length : (HAS_BEAMS ? beam0_context_length : kv_loop_length);

    const auto context_ti_end = MULTI_BLOCK_FLAG
        ? divUp(timesteps_per_block, UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP
        : divUp(static_cast<unsigned>(context_length), UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP;

    const auto generation_ti_end = MULTI_BLOCK_FLAG
        ? divUp(timesteps_per_block, K_PER_WARP) * K_PER_WARP
        : divUp(static_cast<unsigned>(kv_loop_length), K_PER_WARP) * K_PER_WARP;

    const auto bi_seq_len_offset = static_cast<std::size_t>(batch_beam_idx) * params.max_attention_window_size;
    const int* beam_indices = HAS_BEAMS ? &params.cache_indir[bi_seq_len_offset] : nullptr;

    const auto c_tile_times_timesteps_per_block = c_tile * timesteps_per_block;


    const bool is_leader = Qk_dot<T, THREADS_PER_KEY>::is_leader(tidx);

    float linear_bias_slope = 0.f;
    if (params.linear_bias_slopes != nullptr)
    {
        linear_bias_slope = mul<float>(params.linear_bias_slopes[hi], 1.f);
    }

    for (int ti = k_idx.x; ti < context_ti_end; ti += UNROLLED_K_PER_ITER)
    {
        const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        K_vec_m k_vec_cache[K_LOOP_UNROLL][K_VECS_PER_THREAD];

#pragma unroll
        for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
        {
#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
            {
                auto const jj = min(k_idx.y + k_vec_i * K_ELTS_PER_CHUNK, Dh - K_VEC_SIZE);
                const int valid_time_now = min(time_now + k_loop * K_PER_ITER, context_length - 1);
                const int seqIdx = batch_idx * beam_width;

                Tcache* k_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(seqIdx, valid_time_now));

                int inBlockIdx = kvCacheBuffer.getKVLocalIdx(valid_time_now, hi_kv, Dh, jj);
                k_vec_cache[k_loop][k_vec_i] = *reinterpret_cast<const K_vec_m*>(&k_cache_batch[inBlockIdx]);
            }
        }

#pragma unroll
        for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
        {
            const int local_time_now = time_now + k_loop * K_PER_ITER;
            const int local_ti = ti + k_loop * K_PER_ITER;

            K_vec_m k_vec[K_VECS_PER_THREAD];
#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
            {
                k_vec[k_vec_i] = *reinterpret_cast<K_vec_m*>(&k_vec_cache[k_loop][k_vec_i]);
            }

            const bool is_active = local_time_now < context_length;

            if (implicit_rel_attn_bias)
            {
                int relative_buckets = 0;
                int relative_position = local_time_now - tlength;
                int num_buckets = relative_attention_bias_stride;
                num_buckets /= 2;
                relative_buckets += relative_position > 0 ? num_buckets : 0;
                relative_position = abs(relative_position);
                int max_exact = num_buckets / 2;
                bool is_small = relative_position < max_exact;
                int relative_position_if_large = max_exact
                    + (int) (logf(relative_position * 1.0f / max_exact) / logf((float) max_distance / max_exact)
                        * (num_buckets - max_exact));
                relative_position_if_large = min(relative_position_if_large, num_buckets - 1);
                relative_buckets += is_small ? relative_position : relative_position_if_large;
                relative_attention_bias_ptr
                    = relative_attention_bias_ptr_fixed + (tlength - local_time_now) + relative_buckets;
            }

            float relative_attention_bias = 0.f;
            if (is_active && has_relative_attention_bias)
            {
                relative_attention_bias = add(relative_attention_bias, relative_attention_bias_ptr[local_time_now]);
            }

            float qk_ = 0.f;
#ifdef MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            if constexpr (FP8_KV_CACHE)
            {
                qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
            }
            else
#endif
            {
                if constexpr (ENABLE_8BITS_CACHE)
                {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::scale_dot(q_vec, k_vec, kv_scale_quant_orig_f)
                        * params.inv_sqrt_dh;
                }
                else
                {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
                }
            }

            if (MULTI_BLOCK_FLAG && local_ti >= timesteps_per_block)
            {
                continue;
            }

            qk_ += linear_bias_slope * (local_time_now - tlength) + relative_attention_bias;

            if (is_active && is_leader)
            {
                qk_max = fmaxf(qk_max, qk_);
                qk_smem[local_ti] = qk_;
            }
        }
    }

    if (HAS_BEAMS && !DO_CROSS_ATTENTION
        && (!MULTI_BLOCK_FLAG || (c_tile + 1) * timesteps_per_block > beam0_context_length))
    {
        const int input_length_ = MULTI_BLOCK_FLAG ? beam0_context_length % timesteps_per_block : beam0_context_length;
        const int generation_start_ti = k_idx.x + input_length_ / K_PER_WARP * K_PER_WARP;

        for (int ti = generation_start_ti; ti < generation_ti_end; ti += K_PER_ITER)
        {
            const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

            K_vec_m k_vec[K_VECS_PER_THREAD];

#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
            {
                const int jj = min(k_idx.y + k_vec_i * K_ELTS_PER_CHUNK, Dh - K_VEC_SIZE);
                const int valid_time_now = min(time_now, kv_loop_length - 1);
                int beam_offset = beam_indices[valid_time_now];
                const int seqIdx = batch_idx * beam_width + beam_offset;
                Tcache* k_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(seqIdx, valid_time_now));

                int inBlockIdx = kvCacheBuffer.getKVLocalIdx(valid_time_now, hi_kv, Dh, jj);
                k_vec[k_vec_i] = (*reinterpret_cast<const K_vec_m*>(&k_cache_batch[inBlockIdx]));
            }

            const bool is_active = time_now >= context_length && time_now < kv_loop_length;

            if (implicit_rel_attn_bias)
            {
                int relative_buckets = 0;
                int relative_position = time_now - tlength;
                int num_buckets = relative_attention_bias_stride;
                num_buckets /= 2;
                relative_buckets += relative_position > 0 ? num_buckets : 0;
                relative_position = abs(relative_position);
                int max_exact = num_buckets / 2;
                bool is_small = relative_position < max_exact;
                int relative_position_if_large = max_exact
                    + (int) (logf(relative_position * 1.0f / max_exact) / logf((float) max_distance / max_exact)
                        * (num_buckets - max_exact));
                relative_position_if_large = min(relative_position_if_large, num_buckets - 1);
                relative_buckets += is_small ? relative_position : relative_position_if_large;
                relative_attention_bias_ptr
                    = relative_attention_bias_ptr_fixed + (tlength - time_now) + relative_buckets;
            }

            float relative_attention_bias = 0.f;
            if (is_active && has_relative_attention_bias)
            {
                relative_attention_bias = add(relative_attention_bias, relative_attention_bias_ptr[time_now]);
            }

            float qk_ = 0.f;
#ifdef MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            if constexpr (FP8_KV_CACHE)
            {
                qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
            }
            else
#endif
            {
                if constexpr (ENABLE_8BITS_CACHE)
                {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::scale_dot(q_vec, k_vec, kv_scale_quant_orig_f)
                        * params.inv_sqrt_dh;
                }
                else
                {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
                }
            }
            qk_ += linear_bias_slope * (time_now - tlength) + relative_attention_bias;

            if (is_active && is_leader)
            {
                qk_max = fmaxf(qk_max, qk_);
                qk_smem[ti] = qk_;
            }
        }
    }



#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA)
    if (THREADS_PER_KEY <= 4)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 4));
    }
    if (THREADS_PER_KEY <= 8)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 9));
    }
    if (THREADS_PER_KEY <= 16)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 18));
    }
#else
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }
#endif

    const auto warp = tidx / WARP_SIZE;
    const auto lane = tidx % WARP_SIZE;

    if (lane == 0)
    {
        red_smem[warp] = qk_max;
    }

    __syncthreads();


    if (HANDLE_KV && hi == (hi_kv * qhead_per_kv) && qk_vec_idx < Dh)
    {
        Qk_vec_k k_vec = *reinterpret_cast<Qk_vec_k*>(&k_smem[qk_vec_idx]);
        const auto k_idx = QK_VEC_SIZE * tidx;
        const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(cyclic_tlength, hi_kv, Dh, k_idx);
        Tcache* k_cache = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(batch_beam_idx, cyclic_tlength));

        if constexpr (ENABLE_8BITS_CACHE)
        {
            store_8bits_kv_cache_vec(reinterpret_cast<Tcache*>(k_cache), k_vec, inBlockIdx, kv_scale_orig_quant);
        }
        else
        {
            *reinterpret_cast<Qk_vec_m*>(&k_cache[inBlockIdx]) = vec_conversion<Qk_vec_m, Qk_vec_k>(k_vec);
        }
    }

    qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }

    qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

    float sum = 0.f;

    const int logit_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
    for (int ti = tidx; ti <= logit_loop_end; ti += THREADS_PER_BLOCK)
    {

        const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        if (!MULTI_BLOCK_FLAG)
        {
            float logit = __expf(qk_smem[time_now] - qk_max);
            sum += logit;
            qk_smem[time_now] = logit;
        }
        else
        {
            if (time_now < kv_loop_length && ti != timesteps_per_block)
            {
                float logit = __expf(qk_smem[ti] - qk_max);
                sum += logit;
                qk_smem[ti] = logit;
            }
            else if (time_now == kv_loop_length)
            {
                float logit = __expf(qk_current_smem[0] - qk_max);
                sum += logit;
                qk_current_smem[0] = logit;
            }
        }
    }

    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

#ifdef MMHA_FP8_SCALE_P_INSTEAD_OF_V
    float logit_scale = (FP8_KV_CACHE ? kv_scale_quant_orig_f : 1.0f);
#else
    float logit_scale = 1.f;
#endif
    float inv_sum = __fdividef(logit_scale, sum + 1.e-6f);

    const int normlization_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
    for (int ti = tidx; ti <= normlization_loop_end; ti += THREADS_PER_BLOCK)
    {
        const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        if (!MULTI_BLOCK_FLAG)
        {
            convert_from_float(&logits_smem[ti], qk_smem[ti] * inv_sum);
        }
        else
        {
            if (time_now < kv_loop_length && ti != timesteps_per_block)
            {
                convert_from_float(&logits_smem[ti], qk_smem[ti]);
            }
            else if (time_now == kv_loop_length)
            {
                convert_from_float(&logits_current_smem[0], qk_current_smem[0]);
            }
        }
    }


    const auto v_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);
    const auto vo = v_idx.x;
    const auto vi = v_idx.y;
    void** v_cache_base_row_ptr = reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::V_IDX, batch_beam_idx));
    void** v_cache_batch_row_ptr
        = reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::V_IDX, batch_idx * beam_width));

    constexpr unsigned V_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_VALUE};
    constexpr unsigned UNROLLED_V_PER_ITER = V_PER_ITER * V_LOOP_UNROLL;

    bool const is_valid_vi = IS_Dh_MAX || vi < Dh;

    V_vec_k v_bias;
    zero(v_bias);
    if (is_valid_vi && HANDLE_KV && vo == kv_loop_length % V_PER_ITER)
    {
        if (params.v_bias != nullptr)
        {
            const auto v_bias_offset = bitfusion::common::flat_index2(hi_kv, vi, Dh);
            v_bias = *reinterpret_cast<const V_vec_k*>(&params.v_bias[v_bias_offset]);
        }

        if (DO_CROSS_ATTENTION)
        {
            *reinterpret_cast<V_vec_k*>(&bias_smem[vi]) = v_bias;
        }
    }

    __syncthreads();


#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
    using V_vec_accum = typename V_vec_accum_fp32_<V_vec_k>::Type;
#else
    using V_vec_accum = V_vec_k;
#endif
    V_vec_accum out;
    zero(out);

    if (is_valid_vi)
    {
        const int context_length
            = DO_CROSS_ATTENTION ? kv_loop_length : (HAS_BEAMS ? beam0_context_length : kv_loop_length);
        int context_v_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : context_length;
        int generation_v_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
        for (int ti = vo; ti < context_v_loop_end; ti += UNROLLED_V_PER_ITER)
        {
            V_vec_m v_vec_cache[V_LOOP_UNROLL];
#pragma unroll
            for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
            {
                int time_idx = ti + v_loop * V_PER_ITER + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                time_idx = min(time_idx, kv_loop_length - 1);
                int rowIdx = batch_idx * beam_width;

                const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh, vi);
                Tcache* v_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));

                v_vec_cache[v_loop] = *reinterpret_cast<const V_vec_m*>(&v_cache_batch[inBlockIdx]);
            }

#pragma unroll
            for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
            {
                V_vec_m v_vec = reinterpret_cast<V_vec_m*>(&v_vec_cache[v_loop])[0];

                int local_time_idx = ti + v_loop * V_PER_ITER;
                int time_idx = local_time_idx + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);

                const bool is_mask
                    = (MULTI_BLOCK_FLAG && local_time_idx >= timesteps_per_block) || (time_idx >= context_length);

                Logit_value_fma<Tk, V_vec_accum, V_vec_m, INT8_KV_CACHE, FP8_KV_CACHE>(
                    out, reinterpret_cast<Tk*>(logits_smem + local_time_idx), v_vec, kv_scale_quant_orig_f, is_mask);
            }
        }

        if (HAS_BEAMS && !DO_CROSS_ATTENTION)
        {
            const auto generation_start_ti
                = MULTI_BLOCK_FLAG ? vo : (vo + (beam0_context_length / V_PER_ITER) * V_PER_ITER);
            if (!MULTI_BLOCK_FLAG || (c_tile + 1) * timesteps_per_block > beam0_context_length)
            {
                for (int ti = generation_start_ti; ti < generation_v_loop_end; ti += V_PER_ITER)
                {
                    int time_idx = ti + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                    int local_time_idx = ti;
                    if (time_idx < beam0_context_length || (MULTI_BLOCK_FLAG && time_idx >= kv_loop_length))
                    {
                        continue;
                    }
                    int rowIdx = batch_idx * beam_width + beam_indices[time_idx];

                    const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh, vi);
                    Tcache* v_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));
                    V_vec_m v_vec = reinterpret_cast<const V_vec_m*>(&v_cache_batch[inBlockIdx])[0];

                    Logit_value_fma<Tk, V_vec_accum, V_vec_m, INT8_KV_CACHE, FP8_KV_CACHE>(
                        out, reinterpret_cast<Tk*>(logits_smem + local_time_idx), v_vec, kv_scale_quant_orig_f, false);
                }
            }
        }
    }

    __syncthreads();

    const int ctile_idx = tlength / timesteps_per_block;

    if (vo == kv_loop_length % V_PER_ITER && is_valid_vi && (!MULTI_BLOCK_FLAG || (c_tile == ctile_idx)))
    {
        const int tokenIdx = cyclic_tlength;
        const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(tokenIdx, hi_kv, Dh, vi);
        Tcache* v_cache_base = reinterpret_cast<Tcache*>(kvCacheBuffer.getBlockPtr(v_cache_base_row_ptr, tokenIdx));

        V_vec_k v;
        if (DO_CROSS_ATTENTION)
        {
            v = vec_conversion<V_vec_k, V_vec_k>(*reinterpret_cast<const V_vec_k*>(&v_cache_base[inBlockIdx]));
        }
        else
        {
            uint32_t v_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
            const auto v_offset = bitfusion::common::flat_index_strided3(batch_beam_idx, hi_kv, vi, v_stride, Dh);

            if (load_qkv_quant)
            {
                using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_k>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<V_vec_k>::value>::type;
                const auto v_scaling = params.qkv_scale_quant_orig[2];
                const auto v_quant
                    = *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.v)[v_offset]);

                convert_from_float(&v, mul<Packed_Float_t, float>(v_scaling, float_from_int8(v_quant)));
            }
            else
            {
                v = *reinterpret_cast<const V_vec_k*>(&params.v[v_offset]);
            }
        }

        if (HANDLE_KV)
        {
            v = add(v, v_bias);

            if (do_ia3)
            {
                v = mul<V_vec_k, V_vec_k, V_vec_k>(v,
                    *reinterpret_cast<const V_vec_k*>(
                        &params.ia3_value_weights[bitfusion::common::flat_index2(ia3_ti_hi, vi, Dh)]));
            }
        }

        if (hi == (hi_kv * qhead_per_kv))
        {
            if (ENABLE_8BITS_CACHE)
            {
                store_8bits_kv_cache_vec(v_cache_base, v, inBlockIdx, kv_scale_orig_quant);
            }
            else
            {
                *reinterpret_cast<V_vec_k*>(&v_cache_base[inBlockIdx]) = v;
            }
        }

#if defined(MMHA_USE_FP32_ACCUM_FOR_LOGITS)
        if (!MULTI_BLOCK_FLAG)
        {
            out = fma(logits_smem[kv_loop_length], cast_to_float(v), out);
        }
        else
        {
            out = fma(logits_current_smem[0], cast_to_float(v), out);
        }
#else
        if (!MULTI_BLOCK_FLAG)
        {
            out = fma(logits_smem[kv_loop_length], v, out);
        }
        else
        {
            out = fma(logits_current_smem[0], v, out);
        }
#endif
    }
    __syncthreads();

#pragma unroll
    for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2)
    {

        int midpoint = active_groups / 2;

        if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh))
        {
#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
            convert_from_float(reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]), out);
#else
            *reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
        }
        __syncthreads();

        if (vo < midpoint && (Dh == Dh_MAX || vi < Dh))
        {
            out = add(*reinterpret_cast<const V_vec_k*>(&out_smem[vo * Dh + vi]), out);
        }
        __syncthreads();
    }

    const auto bhi = bitfusion::common::flat_index2(batch_beam_idx, hi, num_heads);
    const auto bhi_seq_len_tile = bhi * params.seq_len_tile;
    if (vo == 0 && (Dh == Dh_MAX || vi < Dh))
    {
        const auto bhvi = bitfusion::common::flat_index2(bhi, vi, Dh);
#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
        if (write_attention_quant)
        {
            using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_accum>::value>::type;
            out = mul<V_vec_accum, float>(*params.attention_out_scale_orig_quant, out);
            *reinterpret_cast<Packed_Int8_t*>(&(reinterpret_cast<int8_t*>(params.out)[bhvi])) = cast_to_int8(out);
        }
        else
        {
            if (!MULTI_BLOCK_FLAG)
            {
                V_vec_k final_out;
                convert_from_float(&final_out, out);
                *reinterpret_cast<V_vec_k*>(&params.out[bhvi]) = final_out;
            }
            else
            {
                int partial_out_offset = c_tile * params.batch_size * num_heads * params.hidden_size_per_head;
                int partial_stats_offset = bhi_seq_len_tile + c_tile;

                V_vec_k partial_out;
                convert_from_float(&partial_out, out);
                *reinterpret_cast<V_vec_k*>(&params.partial_out[partial_out_offset + bhvi]) = partial_out;
                convert_from_float(reinterpret_cast<float*>(&params.partial_max[partial_stats_offset]), qk_max);
                convert_from_float(reinterpret_cast<float*>(&params.partial_sum[partial_stats_offset]), sum);
            }
        }
#else
        *reinterpret_cast<V_vec_accum*>(&params.out[bhvi]) = out;
#endif
    }

#ifdef ENABLE_MULTI_BLOCK_OPTION
    if (MULTI_BLOCK_FLAG)
    {

        cuda::atomic_ref<int, cuda::thread_scope_device> count_ref{params.block_counter[bhi]};
        bool last_block{false};
        if (tidx == 0)
        {
            if (count_ref.fetch_add(1, cuda::memory_order_acq_rel) == (gridDim.z - 1))
            {
                last_block = true;
            }
        }

        if (__syncthreads_or(last_block))
        {


            float final_max = -FLT_MAX;
            float thread_partial_max = -FLT_MAX;
            thread_partial_max = params.partial_max[bhi_seq_len_tile + min(tidx, gridDim.z - 1)];

            __syncthreads();

            typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            final_max = BlockReduce(temp_storage).Reduce(thread_partial_max, cub::Max(), gridDim.z);

            __shared__ float final_max_smem;
            if (tidx == 0)
            {
                final_max_smem = final_max;
            }
            __syncthreads();

            final_max = final_max_smem;


            float final_sum = 0.f;
            if (tidx < gridDim.z)
            {
                thread_partial_max = params.partial_max[bhi_seq_len_tile + tidx];
                const auto thread_partial_sum = params.partial_sum[bhi_seq_len_tile + tidx];
                final_sum += __expf(thread_partial_max - final_max) * thread_partial_sum;
            }

            final_sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], final_sum);


            T* out_oi_smem = reinterpret_cast<T*>(smem_);

            const auto o_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);

            V_vec_k zero_k;
            zero(zero_k);
            V_vec_k thread_accumulated_out = zero_k;

            const auto oi = o_idx.y;

            const auto oo = o_idx.x;

            for (int tile_idx = o_idx.x; tile_idx < gridDim.z; tile_idx += V_PER_ITER)
            {
                int thread_partial_out_offset = tile_idx * params.batch_size * num_heads * params.hidden_size_per_head;
                float thread_partial_max_for_out = params.partial_max[bhi_seq_len_tile + tile_idx];
                V_vec_k thread_partial_out
                    = *reinterpret_cast<const V_vec_k*>(&params.partial_out[thread_partial_out_offset + bhi * Dh + oi]);
                Tk factor_compute;
                convert_from_float(&factor_compute, __expf(thread_partial_max_for_out - final_max));
                thread_partial_out = mul<V_vec_k, Tk, V_vec_k>(factor_compute, thread_partial_out);
                thread_accumulated_out = add(thread_partial_out, thread_accumulated_out);
            }

#pragma unroll
            for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2)
            {

                int midpoint = active_groups / 2;

                if (oo >= midpoint && oo < active_groups && (Dh == Dh_MAX || oi < Dh))
                {
                    *reinterpret_cast<V_vec_k*>(&out_oi_smem[(oo - midpoint) * Dh + oi]) = thread_accumulated_out;
                }
                __syncthreads();

                if (oo < midpoint && (Dh == Dh_MAX || oi < Dh))
                {
                    thread_accumulated_out
                        = add(thread_accumulated_out, *reinterpret_cast<const V_vec_k*>(&out_oi_smem[oo * Dh + oi]));
                }
                __syncthreads();
            }


            if (oo == 0 && (Dh == Dh_MAX || oi < Dh))
            {
                const auto inv_sum = __fdividef(1.f, final_sum + 1.e-6f);

                Tk inv_sum_compute;
                convert_from_float(&inv_sum_compute, inv_sum);

                thread_accumulated_out = mul<V_vec_k, Tk, V_vec_k>(inv_sum_compute, thread_accumulated_out);
                *reinterpret_cast<V_vec_k*>(&params.out[bhi * Dh + oi]) = thread_accumulated_out;
            }

            if (tidx == 0)
            {
                params.block_counter[bhi] = 0;
            }
        }
    }
#endif
}

}

}
}
