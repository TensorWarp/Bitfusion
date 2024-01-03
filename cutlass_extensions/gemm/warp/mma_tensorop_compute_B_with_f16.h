#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/platform/platform.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/mma_sm80.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"


namespace cutlass
{
    namespace gemm
    {
        namespace warp
        {

            template <
                typename Shape_,
                typename ElementA_,
                typename LayoutA_,
                typename ElementB_,
                typename LayoutB_,
                typename ElementC_,
                typename LayoutC_,
                typename Policy_,
                typename SharedMemoryInstructionShape_,
                int PartitionsK_ = 1,
                bool AccumulatorsInRowMajor = false,
                typename Enable = bool>
            class MmaTensorOpComputeBWithF16
            {
            public:
                using Shape = Shape_;

                using ElementA = ElementA_;

                using LayoutA = LayoutA_;

                using ElementB = ElementB_;

                using LayoutB = LayoutB_;

                using ElementC = ElementC_;

                using LayoutC = LayoutC_;

                using Policy = Policy_;

                using ArchMmaOperator = typename Policy::Operator;

                using MathOperator = typename ArchMmaOperator::Operator;

                using ArchTag = typename ArchMmaOperator::ArchTag;
                static_assert((platform::is_same<typename ArchMmaOperator::ElementA, half_t>::value
                    && platform::is_same<typename ArchMmaOperator::ElementB, half_t>::value)
                    || (platform::is_same<typename ArchMmaOperator::ElementA, bfloat16_t>::value
                        && platform::is_same<typename ArchMmaOperator::ElementB, bfloat16_t>::value
                        && ArchTag::kMinComputeCapability >= 80),
                    "MmaTensorOpCvtBToA only supports underlying HMMA");

                static_assert(platform::is_same<ElementA, half_t>::value
                    || (platform::is_same<ElementA, bfloat16_t>::value && ArchTag::kMinComputeCapability >= 80),
                    "MmaTensorOpCvtBToA only supports Fp16 A or Bf16 A on Ampere+");

                using OperatorClass = arch::OpClassTensorOp;

                using InstructionShape = typename ArchMmaOperator::Shape;

                using SharedMemoryInstructionShape = SharedMemoryInstructionShape_;

                static_assert(
                    SharedMemoryInstructionShape::kM == InstructionShape::kM, "M dimension of compute instruction must match load");
                static_assert(
                    SharedMemoryInstructionShape::kN == InstructionShape::kN, "N dimension of compute instruction must match load");

                static constexpr int kExpansionFactor = SharedMemoryInstructionShape::kK / InstructionShape::kK;

                static_assert(!(Shape::kK% SharedMemoryInstructionShape::kK), "");

                static ComplexTransform const kTransformA = ComplexTransform::kNone;

                static ComplexTransform const kTransformB = ComplexTransform::kNone;

                static int const kThreadCount = 32;

                static int const kPartitionsK = PartitionsK_;

            public:
                using IteratorA
                    = MmaTensorOpMultiplicandTileIterator<MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
                    MatrixShape<InstructionShape::kM, InstructionShape::kK>, Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

                using FragmentA = typename IteratorA::Fragment;

                using TransformedFragmentA = Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;

                using IteratorB = MmaTensorOpMultiplicandTileIterator<MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB,
                    LayoutB, MatrixShape<SharedMemoryInstructionShape::kK, InstructionShape::kN>, Policy::OpDelta::kRow,
                    kThreadCount, kPartitionsK>;

                using FragmentB = typename IteratorB::Fragment;

                using TransformedFragmentB = Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

                using IteratorC = MmaTensorOpAccumulatorTileIterator<MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
                    typename ArchMmaOperator::Shape, typename Policy::OpDelta>;

                using FragmentC = typename IteratorC::Fragment;

                using MmaIterations = MatrixShape<(Shape::kM + ArchMmaOperator::Shape::kM - 1) / ArchMmaOperator::Shape::kM,
                    (Shape::kN + ArchMmaOperator::Shape::kN - 1) / ArchMmaOperator::Shape::kN>;

            public:
                ArchMmaOperator mma;

            public:

                CUTLASS_DEVICE
                    MmaTensorOpComputeBWithF16() {}

                CUTLASS_DEVICE
                    void operator()(FragmentC& D, TransformedFragmentA const& A, TransformedFragmentB const& B, FragmentC const& C,
                        const int warp_tileB_k_offset) const
                {

                    using MmaOperandA = typename ArchMmaOperator::FragmentA;
                    using MmaOperandB = typename ArchMmaOperator::FragmentB;
                    using MmaOperandC = typename ArchMmaOperator::FragmentC;

                    static_assert(
                        TransformedFragmentB::kElements == MmaOperandB::kElements * kExpansionFactor * MmaIterations::kColumn,
                        "Each thread should have a pack of mma registers for each column iteration AND for the expanded K dim of "
                        "B");

                    D = C;

                    MmaOperandA const* ptr_A = reinterpret_cast<MmaOperandA const*>(&A);
                    MmaOperandB const* ptr_B = reinterpret_cast<MmaOperandB const*>(&B);
                    MmaOperandC* ptr_D = reinterpret_cast<MmaOperandC*>(&D);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
                    CUTLASS_PRAGMA_UNROLL
                        for (int n = 0; n < MmaIterations::kColumn; ++n)
                        {

                            CUTLASS_PRAGMA_UNROLL
                                for (int m = 0; m < MmaIterations::kRow; ++m)
                                {

                                    int m_serpentine = ((n % 2) ? (MmaIterations::kRow - 1 - m) : m);

                                    int n_offsetB = warp_tileB_k_offset + kExpansionFactor * n;
                                    if (AccumulatorsInRowMajor)
                                    {
                                        mma(ptr_D[n + m_serpentine * MmaIterations::kColumn], ptr_A[m_serpentine], ptr_B[n_offsetB],
                                            ptr_D[n + m_serpentine * MmaIterations::kColumn]);
                                    }
                                    else
                                    {
                                        mma(ptr_D[m_serpentine + n * MmaIterations::kRow], ptr_A[m_serpentine], ptr_B[n_offsetB],
                                            ptr_D[m_serpentine + n * MmaIterations::kRow]);
                                    }
                                }
                        }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                    CUTLASS_PRAGMA_UNROLL
                        for (int m = 0; m < MmaIterations::kRow; ++m)
                        {

                            CUTLASS_PRAGMA_UNROLL
                                for (int n = 0; n < MmaIterations::kColumn; ++n)
                                {

                                    int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);

                                    int n_serpentine_offsetB = warp_tileB_k_offset + kExpansionFactor * n_serpentine;
                                    if (AccumulatorsInRowMajor)
                                    {
                                        mma(ptr_D[n_serpentine + m * MmaIterations::kColumn], ptr_A[m], ptr_B[n_serpentine_offsetB],
                                            ptr_D[n_serpentine + m * MmaIterations::kColumn]);
                                    }
                                    else
                                    {
                                        mma(ptr_D[m + n_serpentine * MmaIterations::kRow], ptr_A[m], ptr_B[n_serpentine_offsetB],
                                            ptr_D[m + n_serpentine * MmaIterations::kRow]);
                                    }
                                }
                        }
#else
                    assert(0);
#endif
                }
            };


        }
    }
}