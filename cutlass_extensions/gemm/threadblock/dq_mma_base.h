#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "../../../../cutlass_extensions/weight_only_quant_op.h"


namespace cutlass
{
    namespace gemm
    {
        namespace threadblock
        {

            template <typename WarpMma, int kExpansionFactor = 1>
            CUTLASS_DEVICE void run_warp_mma(WarpMma& warp_mma, typename WarpMma::FragmentC& D,
                typename WarpMma::FragmentA const& A, typename WarpMma::FragmentB const& B, typename WarpMma::FragmentC const& C,
                const int warp_tileB_k_offset)
            {
                warp_mma(D, A, B, C);
            }

            template <typename WarpMma, int kExpansionFactor = WarpMma::kExpansionFactor>
            CUTLASS_DEVICE void run_warp_mma(WarpMma& warp_mma, typename WarpMma::FragmentC& D,
                typename WarpMma::TransformedFragmentA const& A, typename WarpMma::TransformedFragmentB const& B,
                typename WarpMma::FragmentC const& C, const int warp_tileB_k_offset)
            {
                warp_mma(D, A, B, C, warp_tileB_k_offset);
            }


            template <
                typename Shape_,
                typename Policy_,
                typename ElementScale_,
                int Stages,
                WeightOnlyQuantOp DequantOp,
                typename Enable = bool>
            class DqMmaBase
            {
            public:
                using Shape = Shape_;

                using Policy = Policy_;

                using ElementScale = ElementScale_;

                static_assert(DequantOp != WeightOnlyQuantOp::UNDEFINED, "");

                static constexpr int ScalebiasStages = isFinegrained(DequantOp) ? Stages : 1;
                static constexpr int ScaleElementsPerStage = Shape::kN;
                static constexpr int BiasElementsPerStage = hasZero(DequantOp) ? Shape::kN : 0;


                using Operator = typename Policy::Operator;

                using WarpGemm = typename Policy::Operator::Shape;

                using WarpCount = GemmShape<Shape::kM / WarpGemm::kM, Shape::kN / WarpGemm::kN, Shape::kK / WarpGemm::kK>;

                static int const kWarpGemmIterations = (WarpGemm::kK / Operator::Policy::MmaShape::kK);

                static constexpr int kNumKIterationsPerWarpBLoad
                    = Operator::IteratorB::InstructionShape::kRow / Operator::InstructionShape::kK;

                static_assert(!(kWarpGemmIterations% kNumKIterationsPerWarpBLoad), "");
                static constexpr int kWarpGemmIterationsForB = kWarpGemmIterations / kNumKIterationsPerWarpBLoad;

                static int const kStages = Stages;

                using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

                using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;


                class SharedStorage
                {
                public:

                    using ShapeA
                        = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow, Shape::kK* kStages + Policy::SmemPaddingA::kColumn>;

                    using ShapeB
                        = MatrixShape<Shape::kK* kStages + Policy::SmemPaddingB::kRow, Shape::kN + Policy::SmemPaddingB::kColumn>;

                    using ShapeScale = MatrixShape<ScalebiasStages, ScaleElementsPerStage>;
                    using ShapeZero = MatrixShape<ScalebiasStages, BiasElementsPerStage>;

                public:

                    AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

                    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

                    AlignedBuffer<ElementScale, ShapeScale::kCount> operand_scale;

                    AlignedBuffer<ElementScale, ShapeZero::kCount> operand_zero;

                public:

                    CUTLASS_DEVICE
                        static typename Operator::LayoutA LayoutA()
                    {
                        return Operator::LayoutA::packed({ ShapeA::kRow, ShapeA::kColumn });
                    }

                    CUTLASS_HOST_DEVICE
                        static typename Operator::LayoutB LayoutB()
                    {
                        return Operator::LayoutB::packed({ ShapeB::kRow, ShapeB::kColumn });
                    }

                    CUTLASS_HOST_DEVICE
                        TensorRefA operand_A_ref()
                    {
                        return TensorRefA{ operand_A.data(), LayoutA() };
                    }

                    CUTLASS_HOST_DEVICE
                        TensorRefB operand_B_ref()
                    {
                        return TensorRefB{ operand_B.data(), LayoutB() };
                    }
                };

            protected:

                typename Operator::IteratorA warp_tile_iterator_A_;

                typename Operator::IteratorB warp_tile_iterator_B_;

            public:
                CUTLASS_DEVICE
                    DqMmaBase(
                        SharedStorage& shared_storage,
                        int thread_idx,
                        int warp_idx,
                        int lane_idx)
                    : warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx)
                    , warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx)
                {
                }
            };


        }
    }
}