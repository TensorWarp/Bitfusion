#pragma once
#include "../../../../cutlass_extensions/weight_only_quant_op.h"


namespace cutlass
{
    namespace arch
    {

        struct OpMultiplyAddDequantizeInterleavedBToA;

        struct OpMultiplyAddDequantizeInterleavedBToA_percol_scale;
        struct OpMultiplyAddDequantizeInterleavedBToA_fine_scale;
        struct OpMultiplyAddDequantizeInterleavedBToA_fine_scalebias;

        template <typename MmaOp, WeightOnlyQuantOp QuantOp_>
        struct TagOperator
        {
            using TaggedOperator = MmaOp;
        };

        template <>
        struct TagOperator<OpMultiplyAddDequantizeInterleavedBToA, WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>
        {
            using TaggedOperator = OpMultiplyAddDequantizeInterleavedBToA_percol_scale;
        };

        template <>
        struct TagOperator<OpMultiplyAddDequantizeInterleavedBToA, WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>
        {
            using TaggedOperator = OpMultiplyAddDequantizeInterleavedBToA_fine_scale;
        };

        template <>
        struct TagOperator<OpMultiplyAddDequantizeInterleavedBToA, WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>
        {
            using TaggedOperator = OpMultiplyAddDequantizeInterleavedBToA_fine_scalebias;
        };

        template <typename TaggedMmaOp>
        struct DetagOperator
        {
            using Operator = TaggedMmaOp;
            static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;
        };

        template <>
        struct DetagOperator<OpMultiplyAddDequantizeInterleavedBToA_percol_scale>
        {
            using Operator = OpMultiplyAddDequantizeInterleavedBToA;
            static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;
        };

        template <>
        struct DetagOperator<OpMultiplyAddDequantizeInterleavedBToA_fine_scale>
        {
            using Operator = OpMultiplyAddDequantizeInterleavedBToA;
            static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;
        };

        template <>
        struct DetagOperator<OpMultiplyAddDequantizeInterleavedBToA_fine_scalebias>
        {
            using Operator = OpMultiplyAddDequantizeInterleavedBToA;
            static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS;
        };

    }
}