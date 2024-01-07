#pragma once

#include "../../common/cudaUtils.h"
#include "../../common/assert.h"
#include "../../cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"
#include "../../cutlass_extensions/tile_interleaved_layout.h"
#include "common.h"

namespace bitfusion
{
    namespace kernels
    {
        template <typename TypeB, typename Layout>
        struct SupportedLayout
        {
            static constexpr bool value = false;
        };

        template <>
        struct SupportedLayout<uint8_t, cutlass::layout::ColumnMajorTileInterleave<64, 2>>
        {
            static constexpr bool value = true;
        };

        template <>
        struct SupportedLayout<cutlass::uint4b_t, cutlass::layout::ColumnMajorTileInterleave<64, 4>>
        {
            static constexpr bool value = true;
        };

        template <typename TypeB, typename Arch>
        bool isEnabled()
        {
            using Layout = typename cutlass::gemm::kernel::LayoutDetailsB<TypeB, Arch>::Layout;
            return SupportedLayout<TypeB, Layout>::value;
        }

        template <typename TypeB>
        bool isEnabledForArch(int arch)
        {
            if (arch >= 70 && arch < 75)
            {
                return isEnabled<TypeB, cutlass::arch::Sm70>();
            }
            else if (arch >= 75 && arch < 80)
            {
                return isEnabled<TypeB, cutlass::arch::Sm75>();
            }
            else if (arch >= 80 && arch <= 90)
            {
                return isEnabled<TypeB, cutlass::arch::Sm80>();
            }
            else
            {
                CHECK_WITH_INFO(false, "Unsupported Arch");
                return false;
            }
        }

        inline bool isWeightOnlyBatchedGemvEnabled(WeightOnlyQuantType qtype)
        {
            const int arch = bitfusion::common::getSMVersion();
            if (qtype == WeightOnlyQuantType::Int4b)
            {
                return isEnabledForArch<cutlass::uint4b_t>(arch);
            }
            else if (qtype == WeightOnlyQuantType::Int8b)
            {
                return isEnabledForArch<uint8_t>(arch);
            }
            else
            {
                CHECK_WITH_INFO(false, "Unsupported WeightOnlyQuantType");
                return false;
            }
        }
    }
}