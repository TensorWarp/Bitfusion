#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"

#include "../../../../cutlass_extensions/arch/mma.h"
#include "../../../../cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h"

namespace cutlass
{
    namespace gemm
    {
        namespace warp
        {


            template <
                typename WarpShape_,
                typename InstructionShape_,
                typename ElementA,
                typename LayoutA,
                typename ElementB,
                typename LayoutB,
                typename ElementC,
                typename LayoutC,
                int PartitionsK,
                bool AccumulatorsInRowMajor>
            struct DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                arch::OpMultiplyAddDequantizeInterleavedBToA, PartitionsK, AccumulatorsInRowMajor>
            {

            private:
                using ComputeInstructionShape = InstructionShape_;

                static constexpr int LoadInstructionK = 8 * sizeof_bits<ElementA>::value / sizeof_bits<ElementB>::value;

                using LoadInstructionShape = GemmShape<InstructionShape_::kM, InstructionShape_::kN, LoadInstructionK>;

            public:
                using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
                    cutlass::arch::Mma<InstructionShape_, 32, ElementA, cutlass::layout::RowMajor, ElementA,
                    cutlass::layout::ColumnMajor, ElementC, cutlass::layout::RowMajor, arch::OpMultiplyAdd>,
                    cutlass::MatrixShape<1, 1>>;

                using Type = cutlass::gemm::warp::MmaTensorOpComputeBWithF16<WarpShape_, ElementA, LayoutA, ElementB, LayoutB,
                    ElementC, LayoutC, Policy, LoadInstructionShape, PartitionsK, AccumulatorsInRowMajor>;
            };


        }
    }
}