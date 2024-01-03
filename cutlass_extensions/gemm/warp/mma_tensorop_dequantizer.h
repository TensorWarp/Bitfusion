#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/functional.h"
#include "cutlass/platform/platform.h"

#include "../../../../cutlass_extensions/weight_only_quant_op.h"
#include "../../../../common/cudaBf16Wrapper.h"


namespace cutlass
{
    namespace gemm
    {
        namespace warp
        {


            template <
                typename MmaOperator_,
                typename Shape_,
                Operand Operand,
                typename Element_,
                typename Layout_,
                int Threads,
                WeightOnlyQuantOp QuantOp_,
                typename Enable = void>
            class MmaTensorOpDequantizer;

            template <
                typename MmaOperator_,
                typename Shape_,
                WeightOnlyQuantOp QuantOp_>
            class MmaTensorOpDequantizer<MmaOperator_, Shape_, Operand::kB, bfloat16_t, layout::RowMajor, 32, QuantOp_,
                typename platform::enable_if<MmaOperator_::ArchTag::kMinComputeCapability >= 80
                && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type>
            {

            public:
                using MmaOperator = MmaOperator_;

                using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

                using InstructionShape = typename ArchMmaOperator::Shape;

                static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

                using ElementScale = bfloat16_t;

                using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

                static constexpr int kColsPerMmaPerThread = 1;
                using FragmentScale = Array<ElementScale, kColsPerMmaPerThread* MmaOperator::MmaIterations::kColumn>;
                using FragmentZero = Array<ElementScale, kColsPerMmaPerThread* MmaOperator::MmaIterations::kColumn>;

                using Shape = Shape_;

                using Layout = layout::RowMajor;

                using TensorRef = TensorRef<ElementScale, Layout>;

                static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;

                CUTLASS_DEVICE
                    MmaTensorOpDequantizer(TensorRef smem_scales, TensorRef smem_zeros, const int warp_idx_n, const int lane_idx)
                {
                    const int warp_offset = warp_idx_n * Shape::kN;
                    const int quad = lane_idx / 4;
                    const int thread_offset = warp_offset + quad;
                    pointer_scale_ = smem_scales.data() + thread_offset;
                    if constexpr (hasZero(QuantOp))
                    {
                        pointer_zero_ = smem_zeros.data() + thread_offset;
                    }
                }

                CUTLASS_DEVICE
                    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
                    : MmaTensorOpDequantizer(smem_scales, TensorRef(), warp_idx_n, lane_idx)
                {
                }

                CUTLASS_DEVICE
                    void load(FragmentScale& scale_frag)
                {
                    CUTLASS_PRAGMA_UNROLL
                        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
                        {
                            scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
                        }
                }

                CUTLASS_DEVICE
                    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
                {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
                    using _MmaOperandB = typename ArchMmaOperator::FragmentB;
                    using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor* _MmaOperandB::kElements>;
                    static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                        == FragmentDequantizedOperand::kElements,
                        "");

                    const __nv_bfloat16* scale_ptr = reinterpret_cast<const __nv_bfloat16*>(&scale_frag);
                    ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
                    CUTLASS_PRAGMA_UNROLL
                        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
                        {
                            static_assert(ExpandedMmaOperandB::kElements % 2 == 0, "");

                            __nv_bfloat162 scalex2 = __bfloat162bfloat162(scale_ptr[mma_n_iter]);
                            __nv_bfloat162* operand_bf16x2_ptr = reinterpret_cast<__nv_bfloat162*>(&operand_frag_ptr[mma_n_iter]);

                            CUTLASS_PRAGMA_UNROLL
                                for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii)
                                {
                                    operand_bf16x2_ptr[ii] = __hmul2(operand_bf16x2_ptr[ii], scalex2);
                                }
                        }
#else
                    arch::device_breakpoint();
#endif
                }

                CUTLASS_DEVICE
                    void load(FragmentScale& scale_frag, FragmentScale& zero_frag)
                {
                    if constexpr (hasZero(QuantOp))
                    {
                        CUTLASS_PRAGMA_UNROLL
                            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
                            {
                                scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
                                zero_frag[mma_n_iter] = pointer_zero_[mma_n_iter * InstructionShape::kN];
                            }
                    }
                    else
                    {
                        CUTLASS_PRAGMA_UNROLL
                            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
                            {
                                scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
                            }
                    }
                }

                CUTLASS_DEVICE
                    void dequantize(
                        FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag, const FragmentScale& zero_frag)
                {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
                    using _MmaOperandB = typename ArchMmaOperator::FragmentB;
                    using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor* _MmaOperandB::kElements>;
                    static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                        == FragmentDequantizedOperand::kElements,
                        "");

                    const __nv_bfloat16* scale_ptr = reinterpret_cast<const __nv_bfloat16*>(&scale_frag);
                    const __nv_bfloat16* zero_ptr = reinterpret_cast<const __nv_bfloat16*>(&zero_frag);

                    ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
                    CUTLASS_PRAGMA_UNROLL
                        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
                        {
                            static_assert(ExpandedMmaOperandB::kElements % 2 == 0, "");

                            __nv_bfloat162 scalex2 = __bfloat162bfloat162(scale_ptr[mma_n_iter]);
                            __nv_bfloat162 zerox2 = __bfloat162bfloat162(zero_ptr[mma_n_iter]);
                            __nv_bfloat162* operand_bf16x2_ptr = reinterpret_cast<__nv_bfloat162*>(&operand_frag_ptr[mma_n_iter]);

                            if constexpr (hasZero(QuantOp))
                            {
                                CUTLASS_PRAGMA_UNROLL
                                    for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii)
                                    {
                                        operand_bf16x2_ptr[ii] = __hfma2(operand_bf16x2_ptr[ii], scalex2, zerox2);
                                    }
                            }
                            else
                            {
                                CUTLASS_PRAGMA_UNROLL
                                    for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii)
                                    {
                                        operand_bf16x2_ptr[ii] = __hmul2(operand_bf16x2_ptr[ii], scalex2);
                                    }
                            }
                        }
#else
                    arch::device_breakpoint();
#endif
                }

                CUTLASS_DEVICE
                    void add_pointer_offset(int64_t const& offset)
                {
                    static_assert(sizeof(ElementScale) > 1, "");
                    pointer_scale_ += offset;
                    pointer_zero_ += offset;
                }

            private:
                ElementScale const* pointer_scale_;
                ElementScale const* pointer_zero_;
            };


            template <
                typename MmaOperator_,
                typename Shape_,
                WeightOnlyQuantOp QuantOp_>
            class MmaTensorOpDequantizer<MmaOperator_, Shape_, Operand::kB, half_t, layout::RowMajor, 32, QuantOp_,
                typename platform::enable_if<MmaOperator_::ArchTag::kMinComputeCapability >= 75
                && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type>
            {

            public:
                using MmaOperator = MmaOperator_;

                using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

                using InstructionShape = typename ArchMmaOperator::Shape;

                static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

                using ElementScale = half_t;

                using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

                static constexpr int kColsPerMmaPerThread = 1;
                using FragmentScale = Array<ElementScale, kColsPerMmaPerThread* MmaOperator::MmaIterations::kColumn>;
                using FragmentZero = Array<ElementScale, kColsPerMmaPerThread* MmaOperator::MmaIterations::kColumn>;

                using Shape = Shape_;

                using Layout = layout::RowMajor;

                using TensorRef = TensorRef<ElementScale, Layout>;

                static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;

                CUTLASS_DEVICE
                    MmaTensorOpDequantizer(TensorRef smem_scales, TensorRef smem_zeros, const int warp_idx_n, const int lane_idx)
                {
                    const int warp_offset = warp_idx_n * Shape::kN;
                    const int quad = lane_idx / 4;
                    const int thread_offset = warp_offset + quad;
                    pointer_scale_ = smem_scales.data() + thread_offset;
                    if constexpr (hasZero(QuantOp))
                    {
                        pointer_zero_ = smem_zeros.data() + thread_offset;
                    }
                }

                CUTLASS_DEVICE
                    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
                    : MmaTensorOpDequantizer(smem_scales, TensorRef(), warp_idx_n, lane_idx)
                {
                }

                CUTLASS_DEVICE
                    void load(FragmentScale& scale_frag)
                {
                    CUTLASS_PRAGMA_UNROLL
                        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
                        {
                            scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
                        }
                }

                CUTLASS_DEVICE
                    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
                {
                    using _MmaOperandB = typename ArchMmaOperator::FragmentB;
                    using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor* _MmaOperandB::kElements>;
                    static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                        == FragmentDequantizedOperand::kElements,
                        "");

                    multiplies<ExpandedMmaOperandB> mul_op;

                    ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
                    CUTLASS_PRAGMA_UNROLL
                        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
                        {
                            operand_frag_ptr[mma_n_iter] = mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
                        }
                }

                CUTLASS_DEVICE
                    void load(FragmentScale& scale_frag, FragmentScale& zero_frag)
                {
                    if constexpr (hasZero(QuantOp))
                    {
                        CUTLASS_PRAGMA_UNROLL
                            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
                            {
                                scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
                                zero_frag[mma_n_iter] = pointer_zero_[mma_n_iter * InstructionShape::kN];
                            }
                    }
                    else
                    {
                        CUTLASS_PRAGMA_UNROLL
                            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
                            {
                                scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
                            }
                    }
                }

                CUTLASS_DEVICE
                    void dequantize(
                        FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag, const FragmentScale& zero_frag)
                {
                    using _MmaOperandB = typename ArchMmaOperator::FragmentB;
                    using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor* _MmaOperandB::kElements>;
                    static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                        == FragmentDequantizedOperand::kElements,
                        "");

                    multiplies<ExpandedMmaOperandB> mul_op;
                    ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);

                    if constexpr (hasZero(QuantOp))
                    {
                        plus<ExpandedMmaOperandB> plus_op;

                        CUTLASS_PRAGMA_UNROLL
                            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
                            {
                                operand_frag_ptr[mma_n_iter]
                                    = plus_op(mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]), zero_frag[mma_n_iter]);
                            }
                    }
                    else
                    {
                        CUTLASS_PRAGMA_UNROLL
                            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
                            {
                                operand_frag_ptr[mma_n_iter] = mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
                            }
                    }
                }

                CUTLASS_DEVICE
                    void add_pointer_offset(int64_t const& offset)
                {
                    static_assert(sizeof(ElementScale) > 1, "");
                    pointer_scale_ += offset;
                    pointer_zero_ += offset;
                }

            private:
                ElementScale const* pointer_scale_;
                ElementScale const* pointer_zero_;
            };


            template <
                typename MmaOperator_,
                typename Shape_,
                WeightOnlyQuantOp QuantOp_>
            class MmaTensorOpDequantizer<MmaOperator_, Shape_, Operand::kB, half_t, layout::RowMajor, 32, QuantOp_,
                typename platform::enable_if<platform::is_same<typename MmaOperator_::ArchTag, arch::Sm70>::value
                && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::RowMajor>::value>::type>
            {

            public:
                static_assert(platform::is_same<typename MmaOperator_::InterleavedTileShape, GemmShape<32, 32, 4>>::value, "");

                using MmaOperator = MmaOperator_;

                using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

                using InstructionShape = typename ArchMmaOperator::Shape;

                using ElementScale = half_t;

                using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

                using Shape = Shape_;

                static constexpr int ColsPerMmaTile = 32;
                static constexpr int TileNIterations = Shape::kN / ColsPerMmaTile;
                using FragmentScale = Array<ElementScale, TileNIterations * 8>;
                using AccessType = Array<ElementScale, 8>;

                using Layout = layout::RowMajor;

                using TensorRef = TensorRef<ElementScale, Layout>;

                static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;
                static_assert(QuantOp == WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, "");

                CUTLASS_DEVICE
                    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
                {
                    const int warp_offset = warp_idx_n * Shape::kN;
                    const int base_col = lane_idx & 0xF8;
                    const int thread_offset = warp_offset + base_col;
                    pointer_ = smem_scales.data() + thread_offset;
                }

                CUTLASS_DEVICE
                    void load(FragmentScale& scale_frag)
                {
                    AccessType* scale_frag_ptr = reinterpret_cast<AccessType*>(&scale_frag);

                    CUTLASS_PRAGMA_UNROLL
                        for (int tile_iter = 0; tile_iter < TileNIterations; ++tile_iter)
                        {
                            scale_frag_ptr[tile_iter] = *reinterpret_cast<AccessType const*>(pointer_ + ColsPerMmaTile * tile_iter);
                        }
                }

                CUTLASS_DEVICE
                    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
                {
                    static_assert(FragmentScale::kElements == FragmentDequantizedOperand::kElements, "");

                    multiplies<FragmentDequantizedOperand> mul_op;
                    operand_frag = mul_op(operand_frag, scale_frag);
                }

            private:
                ElementScale const* pointer_;
            };


            template <
                typename MmaOperator_,
                typename Shape_,
                WeightOnlyQuantOp QuantOp_>
            class MmaTensorOpDequantizer<MmaOperator_, Shape_, Operand::kB, half_t, layout::RowMajor, 32, QuantOp_,
                typename platform::enable_if<platform::is_same<typename MmaOperator_::ArchTag, arch::Sm70>::value
                && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type>
            {

            public:
                static_assert(platform::is_same<typename MmaOperator_::InterleavedTileShape, GemmShape<32, 32, 4>>::value, "");

                using MmaOperator = MmaOperator_;

                using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

                using InstructionShape = typename ArchMmaOperator::Shape;

                using ElementScale = half_t;

                using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

                using Shape = Shape_;

                static constexpr int ColsPerMmaTile = 32;
                static constexpr int TileNIterations = Shape::kN / ColsPerMmaTile;
                using FragmentScale = Array<ElementScale, TileNIterations * 2>;

                using Layout = layout::RowMajor;

                using TensorRef = TensorRef<ElementScale, Layout>;

                static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;
                static_assert(QuantOp == WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, "");

                CUTLASS_DEVICE
                    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
                {
                    const int warp_offset = warp_idx_n * Shape::kN;
                    const int base_col = lane_idx & 0xF8 + lane_idx % 4;
                    const int thread_offset = warp_offset + base_col;
                    pointer_ = smem_scales.data() + thread_offset;
                }

                CUTLASS_DEVICE
                    void load(FragmentScale& scale_frag)
                {
                    CUTLASS_PRAGMA_UNROLL
                        for (int tile_iter = 0; tile_iter < TileNIterations; ++tile_iter)
                        {
                            CUTLASS_PRAGMA_UNROLL
                                for (int mma_iter = 0; mma_iter < 2; ++mma_iter)
                                {
                                    scale_frag[tile_iter * 2 + mma_iter] = pointer_[ColsPerMmaTile * tile_iter + 4 * mma_iter];
                                }
                        }
                }

                CUTLASS_DEVICE
                    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
                {
                    using MmaOperandB = typename ArchMmaOperator::FragmentB;
                    static constexpr int total_n_mmas = 2 * TileNIterations;
                    static_assert(MmaOperandB::kElements * total_n_mmas == FragmentDequantizedOperand::kElements, "");

                    multiplies<MmaOperandB> mul_op;

                    MmaOperandB* operand_frag_ptr = reinterpret_cast<MmaOperandB*>(&operand_frag);
                    CUTLASS_PRAGMA_UNROLL
                        for (int mma_n_iter = 0; mma_n_iter < total_n_mmas; ++mma_n_iter)
                        {
                            operand_frag_ptr[mma_n_iter] = mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
                        }
                }

            private:
                ElementScale const* pointer_;
            };


        }
    }
}