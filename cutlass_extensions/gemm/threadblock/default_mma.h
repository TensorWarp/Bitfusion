#pragma once

#include "../../../../cutlass_extensions/gemm/threadblock/default_dq_mma_multistage.h"
#include "../../../../cutlass_extensions/gemm/threadblock/default_dq_mma_pipelined.h"
#include "../../../../cutlass_extensions/gemm/threadblock/default_mma_bf16.h"

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
                typename Operator>
            struct DefaultMma<cutlass::half_t, LayoutA, kAlignmentA, uint8_t, LayoutB, kAlignmentB, ElementAccumulator,
                layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2, Operator>
            {

            private:
                static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;

                using Mma = DqMma<half_t, LayoutA, kAlignmentA, uint8_t, LayoutB, kAlignmentB, half_t, layout::RowMajor,
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
            struct DefaultMma<cutlass::half_t, LayoutA, kAlignmentA, uint4b_t, LayoutB, kAlignmentB, ElementAccumulator,
                layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2, Operator>
            {

            private:
                static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;

                using Mma = DqMma<half_t, LayoutA, kAlignmentA, uint4b_t, LayoutB, kAlignmentB, half_t, layout::RowMajor,
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
            struct DefaultMma<cutlass::half_t, LayoutA, kAlignmentA, uint8_t, LayoutB, kAlignmentB, ElementAccumulator,
                layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, kStages, Operator,
                false, SharedMemoryClear>
            {

            private:
                static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;

                using Mma = DqMma<half_t, LayoutA, kAlignmentA, uint8_t, LayoutB, kAlignmentB, half_t, layout::RowMajor,
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
            struct DefaultMma<cutlass::half_t, LayoutA, kAlignmentA, uint4b_t, LayoutB, kAlignmentB, ElementAccumulator,
                layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, kStages, Operator,
                false, SharedMemoryClear>
            {

            private:
                static constexpr int kAlignmentScale = 128 / sizeof_bits<half_t>::value;

                using Mma = DqMma<half_t, LayoutA, kAlignmentA, uint4b_t, LayoutB, kAlignmentB, half_t, layout::RowMajor,
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
                typename ThreadblockShape,
                typename WarpShape,
                typename InstructionShape,
                typename Operator,
                SharedMemoryClearOption SharedMemoryClear,
                bool GatherA,
                bool GatherB>
            struct DefaultMma<half_t, LayoutA, kAlignmentA, half_t, LayoutB, kAlignmentB, ElementAccumulator, layout::RowMajor,
                arch::OpClassTensorOp, arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, 2, Operator, false,
                SharedMemoryClear, GatherA, GatherB>
            {

                using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
                    half_t, LayoutA, half_t, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, 3, Operator>;

                using ThreadMapA = typename MmaCore::IteratorThreadMapA;
                using AccessTypeA = cutlass::Array<half_t, kAlignmentA>;
                using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>, half_t, LayoutA, 1, ThreadMapA, AccessTypeA,
                    GatherA>;

                using ThreadMapB = typename MmaCore::IteratorThreadMapB;
                using AccessTypeB = cutlass::Array<half_t, kAlignmentB>;
                using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>, half_t, LayoutB, 0, ThreadMapB, AccessTypeB,
                    GatherB>;

                using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<typename MmaCore::Shape, IteratorA,
                    typename MmaCore::SmemIteratorA, MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
                    MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor, typename MmaCore::MmaPolicy, 2>;
            };

        }
    }
}