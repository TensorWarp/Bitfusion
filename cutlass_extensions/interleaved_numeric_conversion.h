#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"

namespace cutlass
{

    template <typename T, typename S, int N>
    struct FastInterleavedAndBiasedNumericArrayConverter
    {
    };

    template <>
    struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint8_t, 4>
    {
        using result_type = Array<half_t, 4>;
        using source_type = Array<uint8_t, 4>;

        CUTLASS_DEVICE
            static result_type convert(source_type const& source)
        {
            result_type result;

            uint32_t* h = reinterpret_cast<uint32_t*>(&result);
            uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

            static constexpr uint32_t mask_for_elt_01 = 0x5250;
            static constexpr uint32_t mask_for_elt_23 = 0x5351;
            static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
            asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
            asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

            static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
            asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
            asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));

            return result;
        }

        CUTLASS_DEVICE
            result_type operator()(source_type const& s)
        {
            return convert(s);
        }
    };

    template <int N>
    struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint8_t, N>
    {
        static constexpr int VEC_WIDTH = 4;
        static_assert(!(N% VEC_WIDTH), "N must be multiple of 4.");

        using result_type = Array<half_t, N>;
        using source_type = Array<uint8_t, N>;

        CUTLASS_DEVICE
            static result_type convert(source_type const& source)
        {
            using scalar_result_type = typename result_type::Element;
            using scalar_source_type = typename source_type::Element;
            FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
                convert_vector_;

            result_type result;
            using vec_result = Array<scalar_result_type, VEC_WIDTH>;
            using vec_source = Array<scalar_source_type, VEC_WIDTH>;

            vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
            vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

            CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < N / VEC_WIDTH; ++i)
                {
                    result_ptr[i] = convert_vector_(source_ptr[i]);
                }

            return result;
        }

        CUTLASS_DEVICE
            result_type operator()(source_type const& s)
        {
            return convert(s);
        }
    };

    template <>
    struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint8_t, 4>
    {
        using result_type = Array<bfloat16_t, 4>;
        using source_type = Array<uint8_t, 4>;

        CUTLASS_DEVICE
            static result_type convert(source_type const& source)
        {
            result_type result;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

            uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&result);
            uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

            static constexpr uint32_t fp32_base = 0x4B000000;
            float fp32_intermediates[4];

            uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);
            fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
            fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7652);
            fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7651);
            fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

            CUTLASS_PRAGMA_UNROLL
                for (int ii = 0; ii < 4; ++ii)
                {
                    fp32_intermediates[ii] -= 8388736.f;
                }

            CUTLASS_PRAGMA_UNROLL
                for (int ii = 0; ii < 2; ++ii)
                {
                    bf16_result_ptr[ii]
                        = __byte_perm(fp32_intermediates_casted[2 * ii + 0], fp32_intermediates_casted[2 * ii + 1], 0x7632);
                }
#else
            result.clear();
            arch::device_breakpoint();
#endif
            return result;
        }

        CUTLASS_DEVICE
            result_type operator()(source_type const& s)
        {
            return convert(s);
        }
    };

    template <int N>
    struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint8_t, N>
    {
        static constexpr int VEC_WIDTH = 4;
        static_assert(!(N% VEC_WIDTH), "N must be multiple of 4.");

        using result_type = Array<bfloat16_t, N>;
        using source_type = Array<uint8_t, N>;

        CUTLASS_DEVICE
            static result_type convert(source_type const& source)
        {
            using scalar_result_type = typename result_type::Element;
            using scalar_source_type = typename source_type::Element;
            FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
                convert_vector_;

            result_type result;
            using vec_result = Array<scalar_result_type, VEC_WIDTH>;
            using vec_source = Array<scalar_source_type, VEC_WIDTH>;

            vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
            vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

            CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < N / VEC_WIDTH; ++i)
                {
                    result_ptr[i] = convert_vector_(source_ptr[i]);
                }

            return result;
        }

        CUTLASS_DEVICE
            result_type operator()(source_type const& s)
        {
            return convert(s);
        }
    };

    template <>
    struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint4b_t, 8>
    {
        using result_type = Array<half_t, 8>;
        using source_type = Array<uint4b_t, 8>;

        CUTLASS_DEVICE
            static result_type convert(source_type const& source)
        {
            result_type result;

            uint32_t* h = reinterpret_cast<uint32_t*>(&result);
            uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

            static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
            static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
            static constexpr uint32_t TOP_MASK = 0x00f000f0;
            static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;


            const uint32_t top_i4s = i4s >> 8;
            asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[0])
                : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
            asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[1])
                : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
            asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[2])
                : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
            asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[3])
                : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));


            static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
            static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
            static constexpr uint32_t NEG_72 = 0xd480d480;

            asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
            asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));

            return result;
        }

        CUTLASS_DEVICE
            result_type operator()(source_type const& s)
        {
            return convert(s);
        }
    };

    template <int N>
    struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint4b_t, N>
    {
        static constexpr int VEC_WIDTH = 8;
        static_assert(!(N% VEC_WIDTH), "N must be multiple of 8.");

        using result_type = Array<half_t, N>;
        using source_type = Array<uint4b_t, N>;

        CUTLASS_DEVICE
            static result_type convert(source_type const& source)
        {
            using scalar_result_type = typename result_type::Element;
            using scalar_source_type = typename source_type::Element;
            FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
                convert_vector_;

            result_type result;
            using vec_result = Array<scalar_result_type, VEC_WIDTH>;
            using vec_source = Array<scalar_source_type, VEC_WIDTH>;

            vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
            vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

            CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < N / VEC_WIDTH; ++i)
                {
                    result_ptr[i] = convert_vector_(source_ptr[i]);
                }

            return result;
        }

        CUTLASS_DEVICE
            result_type operator()(source_type const& s)
        {
            return convert(s);
        }
    };

    template <>
    struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint4b_t, 8>
    {
        using result_type = Array<bfloat16_t, 8>;
        using source_type = Array<uint4b_t, 8>;

        CUTLASS_DEVICE
            static result_type convert(source_type const& source)
        {
            result_type result;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

            uint32_t* h = reinterpret_cast<uint32_t*>(&result);
            uint32_t const source_i4s = reinterpret_cast<uint32_t const&>(source);

            static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
            static constexpr uint32_t MASK = 0x000f000f;
            static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

            uint32_t i4s = source_i4s;
            asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[0])
                : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
            CUTLASS_PRAGMA_UNROLL
                for (int ii = 1; ii < result_type::kElements / 2; ++ii)
                {
                    i4s >>= sizeof_bits<typename source_type::Element>::value;
                    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                        : "=r"(h[ii])
                        : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
                }

            static constexpr uint32_t BF16_BIAS = 0xC308C308;
            static constexpr uint32_t BF16_ONE = 0x3F803F80;

            CUTLASS_PRAGMA_UNROLL
                for (int ii = 0; ii < result_type::kElements / 2; ++ii)
                {
                    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[ii]) : "r"(h[ii]), "r"(BF16_ONE), "r"(BF16_BIAS));
                }
#else
            arch::device_breakpoint();
            result.clear();
#endif
            return result;
        }

        CUTLASS_DEVICE
            result_type operator()(source_type const& s)
        {
            return convert(s);
        }
    };

    template <int N>
    struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint4b_t, N>
    {
        static constexpr int VEC_WIDTH = 8;
        static_assert(!(N% VEC_WIDTH), "N must be multiple of 8.");

        using result_type = Array<bfloat16_t, N>;
        using source_type = Array<uint4b_t, N>;

        CUTLASS_DEVICE
            static result_type convert(source_type const& source)
        {
            using scalar_result_type = typename result_type::Element;
            using scalar_source_type = typename source_type::Element;
            FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
                convert_vector_;

            result_type result;
            using vec_result = Array<scalar_result_type, VEC_WIDTH>;
            using vec_source = Array<scalar_source_type, VEC_WIDTH>;

            vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
            vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

            CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < N / VEC_WIDTH; ++i)
                {
                    result_ptr[i] = convert_vector_(source_ptr[i]);
                }

            return result;
        }

        CUTLASS_DEVICE
            result_type operator()(source_type const& s)
        {
            return convert(s);
        }
    };


}