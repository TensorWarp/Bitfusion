#pragma once

#include "common.h"
#include "utility.h"

namespace bitfusion
{
    namespace kernels
    {
        template <typename ActType>
        struct ActTypeDetails;

        template <>
        struct ActTypeDetails<half>
        {
            using CutlassType = cutlass::half_t;
            using Vec2 = half2;

            __device__ __forceinline__ static Vec2 to_vec2(half v)
            {
                return __half2half2(v);
            }
        };
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
        template <>
        struct ActTypeDetails<__nv_bfloat16>
        {
            using CutlassType = cutlass::bfloat16_t;
            using Vec2 = __nv_bfloat162;

            __device__ __forceinline__ static Vec2 to_vec2(__nv_bfloat16 v)
            {
                return __bfloat162bfloat162(v);
            }
        };
#endif

        template <typename ActType, WeightOnlyQuantType QType>
        struct ConverterSelector
        {
            static_assert(QType == WeightOnlyQuantType::Int4b || QType == WeightOnlyQuantType::Int8b);

            using WeiType = std::conditional_t<QType == WeightOnlyQuantType::Int4b, cutlass::uint4b_t, uint8_t>;
            static constexpr int kConvertCount = QType == WeightOnlyQuantType::Int4b ? 8 : 4;
            using Converter
                = cutlass::FastInterleavedAndBiasedNumericArrayConverter<typename ActTypeDetails<ActType>::CutlassType, WeiType,
                kConvertCount>;
        };

        template <typename ActType, WeightOnlyQuantType QType>
        struct WeightOnlyDetails;

        template <typename ActType>
        struct WeightOnlyDetails<ActType, WeightOnlyQuantType::Int4b>
        {
            static constexpr int kElemBits = 4;
            static constexpr int kInterleave = 4;
            static constexpr int kStride = 64;

            static constexpr int kShuffleSize = 32;
            static constexpr int kShuffleBasicTile = 2;
            static constexpr int kShuffleContinous = 4;
            static constexpr int kShuffleStrided = 4;

            template <int Num, int WarpSize>
            __device__ __forceinline__ static void sync(float* res, float(*sm)[Num * kInterleave])
            {
#pragma unroll
                for (int i = 0; i < Num; ++i)
                {
                    res[i] += __shfl_xor_sync(~0, res[i], 16);
                    res[i] += __shfl_xor_sync(~0, res[i], 8);
                    res[i] += __shfl_xor_sync(~0, res[i], 1);
                }
                __syncthreads();
                int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
                if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
                {
#pragma unroll
                    for (int i = 0; i < Num; ++i)
                    {
                        sm[warp][i * kInterleave + lane / 2] = res[i];
                    }
                }
                __syncthreads();
            }
        };

        template <typename ActType>
        struct WeightOnlyDetails<ActType, WeightOnlyQuantType::Int8b>
        {
            static constexpr int kElemBits = 8;
            static constexpr int kInterleave = 2;
            static constexpr int kStride = 64;

            static constexpr int kShuffleSize = 16;
            static constexpr int kShuffleBasicTile = 2;
            static constexpr int kShuffleContinous = 2;
            static constexpr int kShuffleStrided = 4;

            template <int Num, int WarpSize>
            __device__ __forceinline__ static void sync(float* res, float(*sm)[Num * kInterleave])
            {
#pragma unroll
                for (int i = 0; i < Num; ++i)
                {
                    res[i] += __shfl_xor_sync(~0, res[i], 16);
                    res[i] += __shfl_xor_sync(~0, res[i], 8);
                    res[i] += __shfl_xor_sync(~0, res[i], 2);
                    res[i] += __shfl_xor_sync(~0, res[i], 1);
                }
                __syncthreads();
                int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
                if (lane == 0 || lane == 4)
                {
#pragma unroll
                    for (int i = 0; i < Num; ++i)
                    {
                        sm[warp][i * kInterleave + lane / 4] = res[i];
                    }
                }
                __syncthreads();
            }
        };

        template <typename ActType, WeightOnlyQuantType QType>
        struct WeightOnlyKernelDetails
        {
            using Layout = WeightOnlyDetails<ActType, QType>;

            static constexpr int kElemBits = Layout::kElemBits;
            static constexpr int kInterleave = Layout::kInterleave;
            static constexpr int kStride = Layout::kStride;

            static constexpr int kShuffleSize = Layout::kShuffleSize;
            static constexpr int kShuffleBasicTile = Layout::kShuffleBasicTile;
            static constexpr int kShuffleContinous = Layout::kShuffleContinous;
            static constexpr int kShuffleStrided = Layout::kShuffleStrided;


            static constexpr int kConvertCount = ConverterSelector<ActType, QType>::kConvertCount;
            using Converter = typename ConverterSelector<ActType, QType>::Converter;

            static constexpr int kAccessSize = 128;
            using AccessType = uint4;

            static constexpr int kElemsPerByte = 8 / kElemBits;
            static constexpr int kElemsPerThread = kAccessSize / kElemBits;
            static constexpr int kBytePerThread = kElemsPerThread / kElemsPerByte;
            static constexpr int kThreadsNumPerTile = kStride / kElemsPerThread;
            static constexpr int kThreadsNumPerInterleave = kThreadsNumPerTile * kInterleave;

            static constexpr int kConvertIters = kElemsPerThread / kConvertCount;

            static constexpr int kActivationElemNumPerAccess = kAccessSize / (sizeof(ActType) * 8);
            static constexpr int kActivationAccessNum = kElemsPerThread / kActivationElemNumPerAccess;
        };

        template <typename WeightOnlyFlag>
        struct WeightOnlyProperties;

        template <>
        struct WeightOnlyProperties<WeightOnlyPerChannel>
        {
            static constexpr bool kIsFineGrained = false;
            static constexpr int kGroupSize = 0;
        };

        template <int GS>
        struct WeightOnlyProperties<WeightOnlyGroupWise<GS>>
        {
            static constexpr bool kIsFineGrained = true;
            static constexpr int kGroupSize = GS;
        };

        template <typename ActType, WeightOnlyQuantType QType, typename WeightOnlyFlag, bool Zero, int BlockSize>
        struct WeightOnlyScaleLoader
        {
            using ElemType = ActType;
            using Details = WeightOnlyKernelDetails<ActType, QType>;
            static constexpr bool kIsFineGrained = WeightOnlyProperties<WeightOnlyFlag>::kIsFineGrained;
            static constexpr int kGroupSize = WeightOnlyProperties<WeightOnlyFlag>::kGroupSize;

        private:
            const ElemType* _scales;
            const ElemType* _zeros;
            int _stride;
            int _offset;

        public:
            __device__ __forceinline__ WeightOnlyScaleLoader(
                const ElemType* scales, const ElemType* zeros, int initial_offset, int stride)
                : _scales(scales)
                , _zeros(zeros)
                , _stride(stride)
            {
                _scales += initial_offset;
                if constexpr (Zero)
                {
                    _zeros += initial_offset;
                }
                _offset = threadIdx.x / Details::kThreadsNumPerInterleave * Details::kStride
                    + (threadIdx.x % Details::kThreadsNumPerTile) * Details::kElemsPerThread;
            }

            __device__ __forceinline__ void load(ElemType& scale, ElemType& zero, int nid)
            {
                int offset = nid * Details::kInterleave;
                if constexpr (kIsFineGrained)
                {
                    offset += _offset / kGroupSize * _stride;
                }
                scale = _scales[offset];
                if constexpr (Zero)
                {
                    zero = _zeros[offset];
                }
                else
                {
                    zero = static_cast<ElemType>(0.f);
                }
            }

            __device__ __forceinline__ void advance()
            {
                _offset += BlockSize * Details::kElemsPerThread / Details::kInterleave;
            }

            __device__ __forceinline__ int offset()
            {
                return _offset;
            }
        };

        template <typename ActType, WeightOnlyQuantType QType, typename WeightOnlyFlag, template <typename T> class ActOp,
            bool Zero, bool Bias, bool ActScale, int NPerBlock, int Batch, int BlockSize>
        __device__ void weight_only_batched_gemv(const uint8_t* qweight, const ActType* scales, const ActType* zeros,
            const ActType* in, const ActType* act_scale, const ActType* bias, ActType* out, const int n, const int k)
        {
            static_assert(NPerBlock == 1 || (NPerBlock % 2 == 0));
            using ActType2 = typename ActTypeDetails<ActType>::Vec2;
            using Details = WeightOnlyKernelDetails<ActType, QType>;

            using Converter = typename Details::Converter;
            using AccType = typename Details::AccessType;
            using CvtSrcType = typename Converter::source_type;
            using CvtResType = typename Converter::result_type;
            using ScaleLoader = WeightOnlyScaleLoader<ActType, QType, WeightOnlyFlag, Zero, BlockSize>;
            extern __shared__ uint8_t shmem[];
            constexpr int Interleave = Details::kInterleave;
            constexpr int WarpSize = 32;
            constexpr int Num = Batch * NPerBlock;
            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int n_start_id = bid * NPerBlock * Interleave;
            const int interleave_n_id = (tid / Details::kThreadsNumPerTile) % Interleave;

            qweight += n_start_id * k / Details::kElemsPerByte;
            ScaleLoader scale_loader(scales, zeros, n_start_id + interleave_n_id, n);

            float(*sm)[Num * Interleave] = reinterpret_cast<float(*)[Num * Interleave]>(shmem);

            ActType accumulator[Num];
            for (int i = 0; i < Num; ++i)
            {
                accumulator[i] = static_cast<ActType>(0.f);
            }

            for (int local_k = tid * Details::kElemsPerThread; local_k < k * Interleave;
                local_k += BlockSize * Details::kElemsPerThread)
            {
                ActType weights_f16[Details::kElemsPerThread * NPerBlock];
                ActType scale[NPerBlock], zero[NPerBlock];
#pragma unroll
                for (int idx = 0; idx < NPerBlock; ++idx)
                {
                    uint8_t weights_quantized[Details::kBytePerThread];
                    load<AccType>(weights_quantized,
                        qweight + idx * Interleave * k / Details::kElemsPerByte + local_k / Details::kElemsPerByte);
                    scale_loader.load(scale[idx], zero[idx], idx);
                    ActType weights_vec[Details::kElemsPerThread];
#pragma unroll
                    for (int i = 0; i < Details::kConvertIters; ++i)
                    {
                        assign<CvtResType>(weights_vec + i * Details::kConvertCount,
                            Converter::convert(*reinterpret_cast<CvtSrcType*>(
                                weights_quantized + i * Details::kConvertCount / Details::kElemsPerByte)));
                    }
#pragma unroll
                    for (int i = 0; i < Details::kShuffleContinous; ++i)
                    {
#pragma unroll
                        for (int j = 0; j < Details::kShuffleStrided; ++j)
                        {
                            ActType2 v = *reinterpret_cast<ActType2*>(weights_vec + i * Details::kShuffleBasicTile
                                + j * Details::kShuffleContinous * Details::kShuffleBasicTile);
                            v = __hfma2(
                                v, ActTypeDetails<ActType>::to_vec2(scale[idx]), ActTypeDetails<ActType>::to_vec2(zero[idx]));
                            weights_f16[(i * Details::kShuffleStrided * Details::kShuffleBasicTile
                                + j * Details::kShuffleBasicTile + 0)
                                * NPerBlock
                                + idx]
                                = v.x;
                            weights_f16[(i * Details::kShuffleStrided * Details::kShuffleBasicTile
                                + j * Details::kShuffleBasicTile + 1)
                                * NPerBlock
                                + idx]
                                = v.y;
                        }
                    }
                }
                ActType act_scale_v[Details::kElemsPerThread];
                if constexpr (ActScale)
                {
#pragma unroll
                    for (int idx = 0; idx < Details::kActivationAccessNum; ++idx)
                    {
                        load<AccType>(act_scale_v + idx * Details::kActivationElemNumPerAccess,
                            act_scale + scale_loader.offset() + idx * Details::kActivationElemNumPerAccess);
                    }
                }
#pragma unroll
                for (int b = 0; b < Batch; ++b)
                {
                    ActType in_v[Details::kElemsPerThread];
#pragma unroll
                    for (int idx = 0; idx < Details::kActivationAccessNum; ++idx)
                    {
                        load<AccType>(in_v + idx * Details::kActivationElemNumPerAccess,
                            in + b * k + scale_loader.offset() + idx * Details::kActivationElemNumPerAccess);
                        if constexpr (ActScale)
                        {
#pragma unroll
                            for (int i = 0; i < Details::kActivationElemNumPerAccess; i += 2)
                            {
                                *reinterpret_cast<ActType2*>(in_v + idx * Details::kActivationElemNumPerAccess + i) = __hmul2(
                                    *reinterpret_cast<ActType2*>(in_v + idx * Details::kActivationElemNumPerAccess + i),
                                    *reinterpret_cast<ActType2*>(act_scale_v + idx * Details::kActivationElemNumPerAccess + i));
                            }
                        }
                    }
                    if constexpr (NPerBlock == 1)
                    {
                        ActType2 v = ActTypeDetails<ActType>::to_vec2(static_cast<ActType>(0.f));
#pragma unroll
                        for (int y = 0; y < Details::kElemsPerThread; y += 2)
                        {
                            v = __hfma2(
                                *reinterpret_cast<ActType2*>(weights_f16 + y), *reinterpret_cast<ActType2*>(in_v + y), v);
                        }
                        accumulator[b] += __hadd(v.x, v.y);
                    }
                    else
                    {
#pragma unroll
                        for (int x = 0; x < NPerBlock / 2; ++x)
                        {
#pragma unroll
                            for (int y = 0; y < Details::kElemsPerThread; ++y)
                            {
                                *reinterpret_cast<ActType2*>(accumulator + b * NPerBlock + x * 2)
                                    = __hfma2(*reinterpret_cast<ActType2*>(weights_f16 + y * NPerBlock + x * 2),
                                        ActTypeDetails<ActType>::to_vec2(in_v[y]),
                                        *reinterpret_cast<ActType2*>(accumulator + b * NPerBlock + x * 2));
                            }
                        }
                    }
                }
                scale_loader.advance();
            }
            float reses[Num];
#pragma unroll
            for (int i = 0; i < Num; ++i)
            {
                reses[i] = static_cast<float>(accumulator[i]);
            }

            Details::Layout::sync<Num, WarpSize>(reses, sm);

            for (int i = tid; i < Num * Interleave; i += BlockSize)
            {
                int nid = i % (NPerBlock * Interleave);
                float v = 0.f;
                for (int j = 0; j < BlockSize / WarpSize; ++j)
                {
                    v += sm[j][i];
                }
                float bias_v = 0.f;
                if constexpr (Bias)
                {
                    bias_v = static_cast<float>(bias[n_start_id + nid]);
                }
                int b = i / NPerBlock / Interleave;
                out[b * n + n_start_id + nid] = static_cast<ActType>(ActOp<float>::apply(v + bias_v));
            }
        }

        template <typename ActType, WeightOnlyQuantType QType, typename WeightOnlyFlag, template <typename T> class ActOp,
            bool Zero, bool Bias, bool ActScale, int NPerBlock, int Batch, int BlockSize>
        __global__ void weight_only_batched_gemv_wrapper(const uint8_t* qweight, const ActType* scales, const ActType* zeros,
            const ActType* in, const ActType* act_scale, const ActType* bias, ActType* out, const int n, const int k)
        {
            if constexpr (std::is_same_v<ActType, half>)
            {
                weight_only_batched_gemv<ActType, QType, WeightOnlyFlag, ActOp, Zero, Bias, ActScale, NPerBlock, Batch,
                    BlockSize>(qweight, scales, zeros, in, act_scale, bias, out, n, k);
            }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
            else if (std::is_same_v<ActType, nv_bfloat16>)
            {
                weight_only_batched_gemv<ActType, QType, WeightOnlyFlag, ActOp, Zero, Bias, ActScale, NPerBlock, Batch,
                    BlockSize>(qweight, scales, zeros, in, act_scale, bias, out, n, k);
            }
#endif
        }

        template <WeightOnlyQuantType QType, typename WeightOnlyFlag, template <typename T> class ActOp, bool Zero, bool Bias,
            int NPerBlock, int Batch, int BlockSize>
        struct WeightOnlyBatchedGemvKernelLauncher
        {
            static void run(const WeightOnlyParams& params, cudaStream_t stream)
            {
                if (params.act_type == WeightOnlyActivationType::FP16)
                {
                    constexpr int kInterleave = WeightOnlyDetails<half, QType>::kInterleave;
                    dim3 grid(params.n / NPerBlock / kInterleave);
                    dim3 block(BlockSize);
                    int size = sizeof(float) * BlockSize / 32 * Batch * NPerBlock * kInterleave;
                    if (params.act_scale != nullptr)
                    {
                        weight_only_batched_gemv_wrapper<half, QType, WeightOnlyFlag, ActOp, Zero, Bias, true, NPerBlock, Batch,
                            BlockSize> << <grid, block, size, stream >> > (params.qweight,
                                reinterpret_cast<const half*>(params.scales), reinterpret_cast<const half*>(params.zeros),
                                reinterpret_cast<const half*>(params.in), reinterpret_cast<const half*>(params.act_scale),
                                reinterpret_cast<const half*>(params.bias), reinterpret_cast<half*>(params.out), params.n,
                                params.k);
                    }
                    else
                    {
                        weight_only_batched_gemv_wrapper<half, QType, WeightOnlyFlag, ActOp, Zero, Bias, false, NPerBlock,
                            Batch, BlockSize> << <grid, block, size, stream >> > (params.qweight,
                                reinterpret_cast<const half*>(params.scales), reinterpret_cast<const half*>(params.zeros),
                                reinterpret_cast<const half*>(params.in), reinterpret_cast<const half*>(params.act_scale),
                                reinterpret_cast<const half*>(params.bias), reinterpret_cast<half*>(params.out), params.n,
                                params.k);
                    }
                }
#if defined(ENABLE_BF16)
                else if (params.act_type == WeightOnlyActivationType::BF16)
                {
                    constexpr int kInterleave = WeightOnlyDetails<nv_bfloat16, QType>::kInterleave;
                    dim3 grid(params.n / NPerBlock / kInterleave);
                    dim3 block(BlockSize);
                    int size = sizeof(float) * BlockSize / 32 * Batch * NPerBlock * kInterleave;
                    if (params.act_scale != nullptr)
                    {
                        weight_only_batched_gemv_wrapper<__nv_bfloat16, QType, WeightOnlyFlag, ActOp, Zero, Bias, true,
                            NPerBlock, Batch, BlockSize> << <grid, block, size, stream >> > (params.qweight,
                                reinterpret_cast<const __nv_bfloat16*>(params.scales),
                                reinterpret_cast<const __nv_bfloat16*>(params.zeros),
                                reinterpret_cast<const __nv_bfloat16*>(params.in),
                                reinterpret_cast<const __nv_bfloat16*>(params.act_scale),
                                reinterpret_cast<const __nv_bfloat16*>(params.bias), reinterpret_cast<__nv_bfloat16*>(params.out),
                                params.n, params.k);
                    }
                    else
                    {
                        weight_only_batched_gemv_wrapper<__nv_bfloat16, QType, WeightOnlyFlag, ActOp, Zero, Bias, false,
                            NPerBlock, Batch, BlockSize> << <grid, block, size, stream >> > (params.qweight,
                                reinterpret_cast<const __nv_bfloat16*>(params.scales),
                                reinterpret_cast<const __nv_bfloat16*>(params.zeros),
                                reinterpret_cast<const __nv_bfloat16*>(params.in),
                                reinterpret_cast<const __nv_bfloat16*>(params.act_scale),
                                reinterpret_cast<const __nv_bfloat16*>(params.bias), reinterpret_cast<__nv_bfloat16*>(params.out),
                                params.n, params.k);
                    }
                }
#endif
            }
        };
    }
}