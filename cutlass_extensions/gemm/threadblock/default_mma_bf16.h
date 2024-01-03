#pragma once

#include "cutlass/gemm/threadblock/default_mma.h"
#include "../../../../cutlass_extensions/gemm/threadblock/default_dq_mma_multistage.h"
#include "../../../../cutlass_extensions/gemm/threadblock/default_dq_mma_pipelined.h"

namespace cutlass
{
    namespace gemm
    {
        namespace threadblock
        {


            template <
                typename LayoutA,
                int kAlignmentA,
                typename LayoutB,
                int kAlignmentB,
                typename ElementAccumulator,
                typename ArchTag,
                typename ThreadblockShape,
                typename WarpShape,
                typename InstructionShape,
                typename Operator,
                SharedMemoryClearOption SharedMemoryClear,
                bool GatherA,
                bool GatherB>
            struct DefaultMma<bfloat16_t, LayoutA, kAlignmentA, bfloat16_t, LayoutB, kAlignmentB, ElementAccumulator,
                layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2, Operator, false,
                SharedMemoryClear, GatherA, GatherB>
            {

            private:
                static constexpr bool arch_has_bf16_mma = ArchTag::kMinComputeCapability >= 80;
                using MmaElementA = typename platform::conditional<arch_has_bf16_mma, bfloat16_t, half_t>::type;
                using MmaElementB = typename platform::conditional<arch_has_bf16_mma, bfloat16_t, half_t>::type;

            public:
                using MmaCore =
                    typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape, MmaElementA,
                    LayoutA, MmaElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, 2, Operator>;

                using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
                    cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, bfloat16_t, LayoutA, 1,
                    typename MmaCore::IteratorThreadMapA, kAlignmentA, GatherA>;

                using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
                    cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, bfloat16_t, LayoutB, 0,
                    typename MmaCore::IteratorThreadMapB, kAlignmentB, GatherB>;

                using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<typename MmaCore::Shape, IteratorA,
                    typename MmaCore::SmemIteratorA, IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
                    layout::RowMajor, typename MmaCore::MmaPolicy>;
            };

            template <
                typename LayoutA,
                int kAlignmentA,
                typename LayoutB,
                int kAlignmentB,
                typename ElementAccumulator,
                typename ThreadblockShape,
                typename WarpShape,
                typename InstructionShape,
                typename Operator,
                SharedMemoryClearOption SharedMemoryClear,
                bool GatherA,
                bool GatherB>
            struct DefaultMma<bfloat16_t, LayoutA, kAlignmentA, bfloat16_t, LayoutB, kAlignmentB, ElementAccumulator,
                layout::RowMajor, arch::OpClassTensorOp, arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, 2, Operator,
                false, SharedMemoryClear, GatherA, GatherB>
            {

                using MmaCore =
                    typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape, bfloat16_t,
                    LayoutA, bfloat16_t, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, 3, Operator>;

                using ThreadMapA = typename MmaCore::IteratorThreadMapA;
                using AccessTypeA = cutlass::Array<bfloat16_t, kAlignmentA>;
                using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>, bfloat16_t, LayoutA, 1, ThreadMapA,
                    AccessTypeA, GatherA>;

                using ThreadMapB = typename MmaCore::IteratorThreadMapB;
                using AccessTypeB = cutlass::Array<bfloat16_t, kAlignmentB>;
                using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>, bfloat16_t, LayoutB, 0, ThreadMapB,
                    AccessTypeB, GatherB>;

                using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<typename MmaCore::Shape, IteratorA,
                    typename MmaCore::SmemIteratorA, MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
                    MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor, typename MmaCore::MmaPolicy, 2>;
            };


            template <
                typename LayoutA,
                int kAlignmentA,
                typename LayoutB,
                int kAlignmentB,
                typename ElementAccumulator,
                typename ArchTag,
                typename ThreadblockShape,
                typename WarpShape,
                typename InstructionShape,
                typename Operator>
            struct DefaultMma<cutlass::bfloat16_t, LayoutA, kAlignmentA, uint8_t, LayoutB, kAlignmentB, ElementAccumulator,
                layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2, Operator>
            {

            private:
                static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;

                using Mma = DqMma<bfloat16_t, LayoutA, kAlignmentA, uint8_t, LayoutB, kAlignmentB, bfloat16_t, layout::RowMajor,
                    kAlignmentScale, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape,
                    WarpShape, InstructionShape, 2, Operator>;

            public:
                using MmaCore = typename Mma::MmaCore;

                using IteratorA = typename Mma::IteratorA;

                using IteratorB = typename Mma::IteratorB;

                using ThreadblockMma = typename Mma::ThreadblockMma;
            };

            template <
                typename LayoutA,
                int kAlignmentA,
                typename LayoutB,
                int kAlignmentB,
                typename ElementAccumulator,
                typename ArchTag,
                typename ThreadblockShape,
                typename WarpShape,
                typename InstructionShape,
                typename Operator>
            struct DefaultMma<cutlass::bfloat16_t, LayoutA, kAlignmentA, uint4b_t, LayoutB, kAlignmentB, ElementAccumulator,
                layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2, Operator>
            {

            private:
                static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;

                using Mma = DqMma<bfloat16_t, LayoutA, kAlignmentA, uint4b_t, LayoutB, kAlignmentB, bfloat16_t, layout::RowMajor,
                    kAlignmentScale, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape,
                    WarpShape, InstructionShape, 2, Operator>;

            public:
                using MmaCore = typename Mma::MmaCore;

                using IteratorA = typename Mma::IteratorA;

                using IteratorB = typename Mma::IteratorB;

                using ThreadblockMma = typename Mma::ThreadblockMma;
            };

            template <
                typename LayoutA,
                int kAlignmentA,
                typename LayoutB,
                int kAlignmentB,
                typename ElementAccumulator,
                typename ArchTag,
                typename ThreadblockShape,
                typename WarpShape,
                typename InstructionShape,
                typename Operator,
                int kStages,
                SharedMemoryClearOption SharedMemoryClear>
            struct DefaultMma<cutlass::bfloat16_t, LayoutA, kAlignmentA, uint8_t, LayoutB, kAlignmentB, ElementAccumulator,
                layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, kStages, Operator,
                false, SharedMemoryClear>
            {

            private:
                static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;

                using Mma = DqMma<bfloat16_t, LayoutA, kAlignmentA, uint8_t, LayoutB, kAlignmentB, bfloat16_t, layout::RowMajor,
                    kAlignmentScale, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape,
                    WarpShape, InstructionShape, kStages, Operator, SharedMemoryClear>;

            public:
                using MmaCore = typename Mma::MmaCore;

                using IteratorA = typename Mma::IteratorA;

                using IteratorB = typename Mma::IteratorB;

                using ThreadblockMma = typename Mma::ThreadblockMma;
            };

            template <
                typename LayoutA,
                int kAlignmentA,
                typename LayoutB,
                int kAlignmentB,
                typename ElementAccumulator,
                typename ArchTag,
                typename ThreadblockShape,
                typename WarpShape,
                typename InstructionShape,
                typename Operator,
                int kStages,
                SharedMemoryClearOption SharedMemoryClear>
            struct DefaultMma<cutlass::bfloat16_t, LayoutA, kAlignmentA, uint4b_t, LayoutB, kAlignmentB, ElementAccumulator,
                layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, kStages, Operator,
                false, SharedMemoryClear>
            {

            private:
                static constexpr int kAlignmentScale = 128 / sizeof_bits<bfloat16_t>::value;

                using Mma = DqMma<bfloat16_t, LayoutA, kAlignmentA, uint4b_t, LayoutB, kAlignmentB, bfloat16_t, layout::RowMajor,
                    kAlignmentScale, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape,
                    WarpShape, InstructionShape, kStages, Operator, SharedMemoryClear>;

            public:
                using MmaCore = typename Mma::MmaCore;

                using IteratorA = typename Mma::IteratorA;

                using IteratorB = typename Mma::IteratorB;

                using ThreadblockMma = typename Mma::ThreadblockMma;
            };

        }
    }
}