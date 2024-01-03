#pragma once

#include "../../../../cutlass_extensions/arch/mma.h"
#include "../../../../cutlass_extensions/interleaved_numeric_conversion.h"

namespace cutlass
{
namespace gemm
{
namespace threadblock
{

template <
    typename IteratorB,
    typename MmaOperator,
    typename MathOperator>
struct SetConverters
{
};

template <
    typename IteratorB,
    typename MmaOperator>
struct SetConverters<IteratorB, MmaOperator, arch::OpMultiplyAdd>
{
    using TransformAfterLDG
        = FastInterleavedAndBiasedNumericArrayConverter<typename MmaOperator::ArchMmaOperator::ElementB,
            typename IteratorB::Element, IteratorB::Fragment::kElements>;

    using TransformAfterLDS = NumericArrayConverter<typename MmaOperator::ArchMmaOperator::ElementB,
        typename MmaOperator::ArchMmaOperator::ElementB, MmaOperator::FragmentB::kElements>;
};


template <
    typename IteratorB,
    typename MmaOperator>
struct SetConverters<IteratorB, MmaOperator, arch::OpMultiplyAddDequantizeInterleavedBToA>
{
    using TransformAfterLDG = NumericArrayConverter<typename IteratorB::Element, typename IteratorB::Element,
        IteratorB::Fragment::kElements>;

    using TransformAfterLDS
        = FastInterleavedAndBiasedNumericArrayConverter<typename MmaOperator::ArchMmaOperator::ElementB,
            typename TransformAfterLDG::result_type::Element, MmaOperator::FragmentB::kElements>;
};


template <
    typename ElementA_,
    typename LayoutA_,
    int kAlignmentA,
    typename ElementB_,
    typename LayoutB_,
    int kAlignmentB,
    typename ElementScale_,
    typename LayoutScale_,
    int kAlignmentScale,
    typename ElementAccumulator_,
    typename LayoutC_,
    typename OperatorClass_,
    typename ArchTag_,
    typename ThreadblockShape_,
    typename WarpShape_,
    typename InstructionShape_,
    int Stages,
    typename Operator_,
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    typename Enable = void>
struct DqMma;

}
}
}
