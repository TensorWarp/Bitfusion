#pragma once

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/platform/platform.h"

#include "../../../../cutlass_extensions/arch/mma.h"
#include "../../../../cutlass_extensions/tile_interleaved_layout.h"

namespace cutlass
{
    namespace gemm
    {
        namespace kernel
        {

            template <typename TypeB, typename Arch, typename Enable = void>
            struct LayoutDetailsB
            {
            };

            template <typename TypeB>
            struct LayoutDetailsB<TypeB, arch::Sm70>
            {
                static constexpr int ThreadblockK = 64;
                using Layout = layout::ColumnMajor;
                static constexpr int ElementsPerAccess = 8;
                using Operator = cutlass::arch::OpMultiplyAdd;
            };

            template <typename Arch>
            struct LayoutDetailsB<half_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type>
            {
                static constexpr int ThreadblockK = 64;
                using Layout = layout::ColumnMajor;
                static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<half_t>::value;
                using Operator = cutlass::arch::OpMultiplyAdd;
            };

            template <typename Arch>
            struct LayoutDetailsB<bfloat16_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type>
            {
                static constexpr int ThreadblockK = 64;
                using Layout = layout::ColumnMajor;
                static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<bfloat16_t>::value;
                using Operator = cutlass::arch::OpMultiplyAdd;
            };

            template <typename Arch>
            struct LayoutDetailsB<uint8_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type>
            {
                static constexpr int ThreadblockK = 64;

            private:
                static constexpr int ElementsPerCacheLine = 128 * 8 / sizeof_bits<uint8_t>::value;
                static constexpr int ColumnsInterleaved = ElementsPerCacheLine / ThreadblockK;

            public:
                using Layout = layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;
                static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<uint8_t>::value;
                using Operator = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;
            };

            template <typename Arch>
            struct LayoutDetailsB<uint4b_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type>
            {
                static constexpr int ThreadblockK = 64;

            private:
                static constexpr int ElementsPerCacheLine = 128 * 8 / sizeof_bits<uint4b_t>::value;
                static constexpr int ColumnsInterleaved = ElementsPerCacheLine / ThreadblockK;

            public:
                using Layout = layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;
                static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<uint4b_t>::value;
                using Operator = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;
            };

        }
    }
}