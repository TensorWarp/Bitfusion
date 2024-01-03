#pragma once

#include "cutlass/gemm/threadblock/default_mma.h"
#include "../../../../cutlass_extensions/arch/mma.h"

#include "../../../../cutlass_extensions/gemm/threadblock/dq_mma_multistage.h"
#include "../../../../cutlass_extensions/gemm/warp/default_mma_tensor_op.h"
#include "../../../../cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h"
#include "../../../../cutlass_extensions/tile_interleaved_layout.h"

#include "../../../../cutlass_extensions/gemm/threadblock/default_dq_mma.h"
#include "../../../../cutlass_extensions/transform/threadblock/fine_grained_scale_zero_iterator.h"

namespace cutlass
{
    namespace gemm
    {
        namespace threadblock
        {


            template <typename MmaShape, typename Element, typename Layout, WeightOnlyQuantOp QuantOp, int Alignment,
                typename Enable = void>
            struct DefaultScaleIterators;

            template <typename MmaShape, typename Element, typename Layout, WeightOnlyQuantOp QuantOp, int Alignment>
            struct DefaultScaleIterators<MmaShape, Element, Layout, QuantOp, Alignment, std::enable_if_t<isFinegrained(QuantOp)>>
            {
                using IteratorScale
                    = cutlass::transform::threadblock::FineGrainedScaleZeroIterator<cutlass::MatrixShape<1, MmaShape::kN>, Element,
                    Layout, 0, Alignment>;

                using SmemIteratorScale = IteratorScale;
            };

            template <typename MmaShape, typename Element, typename Layout, WeightOnlyQuantOp QuantOp, int Alignment>
            struct DefaultScaleIterators<MmaShape, Element, Layout, QuantOp, Alignment, std::enable_if_t<!isFinegrained(QuantOp)>>
            {
                static_assert((MmaShape::kN% Alignment) == 0, "");

            private:
                using IteratorScaleThreadMap = transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaShape::kN, 1>,
                    MmaShape::kN / Alignment, Alignment>;

            public:
                using IteratorScale = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaShape::kN>,
                    Element, Layout, 0, IteratorScaleThreadMap, Alignment>;

                using SmemIteratorScale = IteratorScale;
            };


            template <
                typename ElementA,
                typename LayoutA,
                int kAlignmentA,
                typename ElementB,
                typename LayoutB,
                int kAlignmentB,
                typename ElementScale,
                typename LayoutScale,
                int kAlignmentScale,
                typename ElementAccumulator,
                typename OperatorClass,
                typename ArchTag,
                typename ThreadblockShape,
                typename WarpShape,
                typename InstructionShape,
                int kStages,
                typename Operator_,
                SharedMemoryClearOption SharedMemoryClear>
            struct DqMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementScale, LayoutScale, kAlignmentScale,
                ElementAccumulator, layout::RowMajor, OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
                kStages, Operator_, SharedMemoryClear,
                typename platform::enable_if<(
                    ArchTag::kMinComputeCapability >= 80 && !layout::IsColumnMajorTileInterleave<LayoutB>::value)>::type>
            {

                static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
                    "Element A must be fp16 or bf16");

                using OperatorInfo = arch::DetagOperator<Operator_>;
                using Operator = typename OperatorInfo::Operator;
                static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
                    "Mma multistage must dequantize after ldsm");

                static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
                    "Element B must be uint8 or uint4");

                static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
                    ? cutlass::arch::CacheOperation::Global
                    : cutlass::arch::CacheOperation::Always;

                static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
                    ? cutlass::arch::CacheOperation::Global
                    : cutlass::arch::CacheOperation::Always;

                using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
                    ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator, layout::RowMajor, OperatorClass, std::max(kStages, 3),
                    Operator, false, CacheOpA, CacheOpB>;

                using ThreadMapA = typename MmaCore::IteratorThreadMapA;
                using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
                using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>, ElementA, LayoutA, 1, ThreadMapA,
                    AccessTypeA>;

                using ThreadMapB = typename MmaCore::IteratorThreadMapB;
                using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
                using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>, ElementB, LayoutB, 0, ThreadMapB,
                    AccessTypeB>;

                using ScaleIterators = DefaultScaleIterators<typename MmaCore::Shape, ElementScale, LayoutScale,
                    OperatorInfo::QuantOp, kAlignmentScale>;

                using IteratorScale = typename ScaleIterators::IteratorScale;

                using SmemIteratorScale = typename ScaleIterators::SmemIteratorScale;

                using Converter = FastInterleavedAndBiasedNumericArrayConverter<ElementA, ElementB,
                    MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

                using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<typename MmaCore::Shape, IteratorA,
                    typename MmaCore::SmemIteratorA, MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
                    MmaCore::kCacheOpB, IteratorScale, SmemIteratorScale, ElementAccumulator, layout::RowMajor,
                    typename MmaCore::MmaPolicy, kStages, Converter, OperatorInfo::QuantOp, SharedMemoryClear>;
            };

            template <
                typename ElementA,
                typename LayoutA,
                int kAlignmentA,
                typename ElementB,
                typename LayoutB,
                int kAlignmentB,
                typename ElementScale,
                typename LayoutScale,
                int kAlignmentScale,
                typename ElementAccumulator,
                typename OperatorClass,
                typename ArchTag,
                typename ThreadblockShape,
                typename WarpShape,
                typename InstructionShape,
                int kStages,
                typename Operator_,
                SharedMemoryClearOption SharedMemoryClear>
            struct DqMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementScale, LayoutScale, kAlignmentScale,
                ElementAccumulator, layout::RowMajor, OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
                kStages, Operator_, SharedMemoryClear,
                typename platform::enable_if<(
                    ArchTag::kMinComputeCapability >= 80 && layout::IsColumnMajorTileInterleave<LayoutB>::value)>::type>
            {

                static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
                    "Element A must be fp16 or bf16");

                using OperatorInfo = arch::DetagOperator<Operator_>;
                using Operator = typename OperatorInfo::Operator;
                static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
                    "Mma multistage must dequantize after ldsm");

                static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
                    "Element B must be uint8 or uint4");

                static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
                    ? cutlass::arch::CacheOperation::Global
                    : cutlass::arch::CacheOperation::Always;

                static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
                    ? cutlass::arch::CacheOperation::Global
                    : cutlass::arch::CacheOperation::Always;

                using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
                    ElementA, LayoutA, ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
                    std::max(kStages, 3), Operator, false, CacheOpA, CacheOpB>;

                using ThreadMapA = typename MmaCore::IteratorThreadMapA;
                using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
                using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>, ElementA, LayoutA, 1, ThreadMapA,
                    AccessTypeA>;

            private:
                static constexpr int ColumnsInterleaved = LayoutB::kColumnsInterleaved;
                static constexpr int RowsPerTile = LayoutB::kRowsPerTile;
                static_assert(!(MmaCore::Shape::kN% ColumnsInterleaved), "");
                static_assert(RowsPerTile == MmaCore::Shape::kK, "");

                using OriginalThreadMap = typename MmaCore::IteratorThreadMapB;
                using OriginalWarpArrangement = typename OriginalThreadMap::Detail::WarpThreadArrangement;
                static_assert(!(OriginalWarpArrangement::kStrided% ColumnsInterleaved), "");

                using GmemIteratorShape
                    = MatrixShape<MmaCore::Shape::kK* ColumnsInterleaved, MmaCore::Shape::kN / ColumnsInterleaved>;
                using GmemThreadMapB = transform::PitchLinearWarpRakedThreadMap<
                    layout::PitchLinearShape<GmemIteratorShape::kRow, GmemIteratorShape::kColumn>, OriginalThreadMap::kThreads,
                    layout::PitchLinearShape<OriginalWarpArrangement::kContiguous* ColumnsInterleaved,
                    OriginalWarpArrangement::kStrided / ColumnsInterleaved>,
                    MmaCore::kAccessSizeInBits / sizeof_bits<ElementB>::value>;

            public:
                using ThreadMapB = typename MmaCore::IteratorThreadMapB;
                using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
                using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<GmemIteratorShape, ElementB,
                    layout::ColumnMajor, 0, GmemThreadMapB, AccessTypeB>;

                using ScaleIterators = DefaultScaleIterators<typename MmaCore::Shape, ElementScale, LayoutScale,
                    OperatorInfo::QuantOp, kAlignmentScale>;

                using IteratorScale = typename ScaleIterators::IteratorScale;

                using SmemIteratorScale = typename ScaleIterators::SmemIteratorScale;

                using Converter = FastInterleavedAndBiasedNumericArrayConverter<ElementA, ElementB,
                    MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

                using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<typename MmaCore::Shape, IteratorA,
                    typename MmaCore::SmemIteratorA, MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
                    MmaCore::kCacheOpB, IteratorScale, SmemIteratorScale, ElementAccumulator, layout::RowMajor,
                    typename MmaCore::MmaPolicy, kStages, Converter, OperatorInfo::QuantOp, SharedMemoryClear>;
            };

        }
    }
}