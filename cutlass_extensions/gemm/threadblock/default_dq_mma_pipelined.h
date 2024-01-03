#pragma once

#include "cutlass/gemm/threadblock/default_mma.h"
#include "../../../../cutlass_extensions/arch/mma.h"

#include "../../../../cutlass_extensions/gemm/threadblock/dq_mma_pipelined.h"
#include "../../../../cutlass_extensions/gemm/warp/default_mma_tensor_op.h"
#include "../../../../cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h"
#include "../../../../cutlass_extensions/tile_interleaved_layout.h"

#include "../../../../cutlass_extensions/gemm/threadblock/default_dq_mma.h"

namespace cutlass
{
    namespace gemm
    {
        namespace threadblock
        {


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
                typename Operator_>
            struct DqMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementScale, LayoutScale, kAlignmentScale,
                ElementAccumulator, layout::RowMajor, OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2,
                Operator_, SharedMemoryClearOption::kNone,
                typename platform::enable_if<(
                    ArchTag::kMinComputeCapability < 80 && !layout::IsColumnMajorTileInterleave<LayoutB>::value)>::type>
            {

                static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
                    "Element A must be fp16 or bf16");

                static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
                    "Element B must be uint8 or uint4");

                using OperatorInfo = arch::DetagOperator<Operator_>;
                using Operator = typename OperatorInfo::Operator;
                static_assert(OperatorInfo::QuantOp == WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, "");

                static constexpr bool DqAfterLDG = platform::is_same<arch::OpMultiplyAdd, Operator>::value;
                static constexpr bool arch_has_bf16_mma = ArchTag::kMinComputeCapability >= 80;
                using MmaCoreElementA = typename platform::conditional<arch_has_bf16_mma, ElementA, half_t>::type;
                using MmaCoreElementB = typename platform::conditional<DqAfterLDG, MmaCoreElementA, ElementB>::type;

                using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
                    MmaCoreElementA, LayoutA, MmaCoreElementB, LayoutB, ElementAccumulator, layout::RowMajor, OperatorClass, 2,
                    Operator>;

                using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
                    cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, ElementA, LayoutA, 1,
                    typename MmaCore::IteratorThreadMapA, kAlignmentA>;

                using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
                    cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, ElementB, LayoutB, 0,
                    typename MmaCore::IteratorThreadMapB, kAlignmentB>;

                static_assert((MmaCore::Shape::kN% kAlignmentScale) == 0, "");
                using IteratorScaleThreadMap
                    = transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
                    MmaCore::Shape::kN / kAlignmentScale, kAlignmentScale>;

                using IteratorScale
                    = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                    ElementScale, LayoutScale, 0, IteratorScaleThreadMap, kAlignmentScale>;

                using SmemScaleType = typename platform::conditional<arch_has_bf16_mma, ElementScale, half_t>::type;
                using SmemIteratorScale
                    = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                    SmemScaleType, LayoutScale, 0, IteratorScaleThreadMap, kAlignmentScale>;

                using Converters = SetConverters<IteratorB, typename MmaCore::MmaPolicy::Operator, Operator>;

                using ThreadblockMma = cutlass::gemm::threadblock::DqMmaPipelined<typename MmaCore::Shape, IteratorA,
                    typename MmaCore::SmemIteratorA, IteratorB, typename MmaCore::SmemIteratorB, IteratorScale, SmemIteratorScale,
                    ElementAccumulator, layout::RowMajor, typename MmaCore::MmaPolicy, typename Converters::TransformAfterLDG,
                    typename Converters::TransformAfterLDS, OperatorInfo::QuantOp>;
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
                typename Operator_>
            struct DqMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementScale, LayoutScale, kAlignmentScale,
                ElementAccumulator, layout::RowMajor, OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2,
                Operator_, SharedMemoryClearOption::kNone,
                typename platform::enable_if<(
                    ArchTag::kMinComputeCapability < 80 && layout::IsColumnMajorTileInterleave<LayoutB>::value)>::type>
            {

                static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
                    "Element A must be fp16 or bf16");

                static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
                    "Element B must be uint8 or uint4");

                using OperatorInfo = arch::DetagOperator<Operator_>;
                using Operator = typename OperatorInfo::Operator;
                static_assert(OperatorInfo::QuantOp == WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, "");

                static constexpr bool DqAfterLDG = platform::is_same<arch::OpMultiplyAdd, Operator>::value;
                static constexpr bool arch_has_bf16_mma = ArchTag::kMinComputeCapability >= 80;
                using MmaCoreElementA = typename platform::conditional<arch_has_bf16_mma, ElementA, half_t>::type;
                using MmaCoreElementB = typename platform::conditional<DqAfterLDG, MmaCoreElementA, ElementB>::type;

                using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
                    MmaCoreElementA, LayoutA, MmaCoreElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor,
                    OperatorClass, 2, Operator>;

                using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
                    cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, ElementA, LayoutA, 1,
                    typename MmaCore::IteratorThreadMapA, kAlignmentA>;

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
                using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<GmemIteratorShape, ElementB,
                    layout::ColumnMajor, 0, GmemThreadMapB, kAlignmentB>;

                static_assert((MmaCore::Shape::kN% kAlignmentScale) == 0, "");
                using IteratorScaleThreadMap
                    = transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
                    MmaCore::Shape::kN / kAlignmentScale, kAlignmentScale>;

                using IteratorScale
                    = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                    ElementScale, LayoutScale, 0, IteratorScaleThreadMap, kAlignmentScale>;

                using SmemScaleType = typename platform::conditional<arch_has_bf16_mma, ElementScale, half_t>::type;
                using SmemIteratorScale
                    = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaCore::Shape::kN>,
                    SmemScaleType, LayoutScale, 0, IteratorScaleThreadMap, kAlignmentScale>;

                using Converters = SetConverters<IteratorB, typename MmaCore::MmaPolicy::Operator, Operator>;

                using ThreadblockMma = cutlass::gemm::threadblock::DqMmaPipelined<typename MmaCore::Shape, IteratorA,
                    typename MmaCore::SmemIteratorA, IteratorB, typename MmaCore::SmemIteratorB, IteratorScale, SmemIteratorScale,
                    ElementAccumulator, layout::RowMajor, typename MmaCore::MmaPolicy, typename Converters::TransformAfterLDG,
                    typename Converters::TransformAfterLDS, OperatorInfo::QuantOp>;
            };

        }
    }
}