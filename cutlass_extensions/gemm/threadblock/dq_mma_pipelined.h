#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"

#include "../../../../cutlass_extensions/gemm/threadblock/dq_mma_base.h"
#include "../../../../cutlass_extensions/gemm/warp/mma_tensorop_dequantizer.h"
#include "../../../../cutlass_extensions/interleaved_numeric_conversion.h"

#include "../../../../cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"
#include "../../../../cutlass_extensions/gemm_configs.h"


namespace cutlass
{
    namespace gemm
    {
        namespace threadblock
        {


            template <
                typename Shape_,
                typename IteratorA_,
                typename SmemIteratorA_,
                typename IteratorB_,
                typename SmemIteratorB_,
                typename IteratorScale_,
                typename SmemIteratorScale_,
                typename ElementC_,
                typename LayoutC_,
                typename Policy_,
                typename TransformBAfterLDG_,
                typename TransformBAfterLDS_,
                WeightOnlyQuantOp QuantOp_,
                typename Enable = bool>
            class DqMmaPipelined : public DqMmaBase<Shape_, Policy_, typename SmemIteratorScale_::Element, 2, QuantOp_>
            {
            public:
                using Base = DqMmaBase<Shape_, Policy_, typename SmemIteratorScale_::Element, 2, QuantOp_>;

                using Shape = Shape_;
                using IteratorA = IteratorA_;
                using IteratorB = IteratorB_;
                using ElementC = ElementC_;
                using LayoutC = LayoutC_;
                using Policy = Policy_;

                using IteratorScale = IteratorScale_;
                using ElementScale = typename IteratorScale::Element;
                using LayoutScale = typename IteratorScale::Layout;

                using SmemIteratorA = SmemIteratorA_;
                using SmemIteratorB = SmemIteratorB_;
                using SmemIteratorScale = SmemIteratorScale_;

                using TransformBAfterLDG = TransformBAfterLDG_;
                using TransformBAfterLDS = TransformBAfterLDS_;

                static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;


                using FragmentA = typename IteratorA::Fragment;

                using FragmentB = typename IteratorB::Fragment;

                using FragmentScale = typename IteratorScale::Fragment;

                using FragmentC = typename Policy::Operator::FragmentC;

                using Operator = typename Policy::Operator;

                using ArchTag = typename Policy::Operator::ArchTag;

                using Dequantizer = warp::MmaTensorOpDequantizer<Operator, typename Base::WarpGemm, Operand::kB,
                    typename SmemIteratorScale::Fragment::Element, LayoutScale, 32, QuantOp>;

                static ComplexTransform const kTransformA = Operator::kTransformA;

                static ComplexTransform const kTransformB = Operator::kTransformB;

                static_assert((Base::kStages == 2), "DqMmaPipelined requires kStages set to value 2");

            private:
                using WarpFragmentA = typename Operator::FragmentA;
                using WarpFragmentB = typename Operator::FragmentB;
                Dequantizer warp_dequantizer_;

                using ElementB = typename IteratorB::Element;
                using LayoutDetailsForB = kernel::LayoutDetailsB<ElementB, ArchTag>;

                static constexpr bool RequiresTileInterleave
                    = layout::IsColumnMajorTileInterleave<typename LayoutDetailsForB::Layout>::value;
                static_assert(!RequiresTileInterleave || (RequiresTileInterleave && (Shape::kK == LayoutDetailsForB::ThreadblockK)),
                    "Layout K must match threadblockK");

            protected:
                SmemIteratorA smem_iterator_A_;

                SmemIteratorB smem_iterator_B_;

                SmemIteratorScale smem_iterator_scale_;

            public:
                CUTLASS_DEVICE
                    DqMmaPipelined(typename Base::SharedStorage&
                        shared_storage,
                        const int group_size,
                        int thread_idx,
                        int warp_idx,
                        int lane_idx
                    )
                    : Base(shared_storage, thread_idx, warp_idx, lane_idx)
                    , warp_dequantizer_({ shared_storage.operand_scale.data(), LayoutScale(Shape::kN) },
                        (warp_idx % (Base::WarpCount::kM* Base::WarpCount::kN)) / Base::WarpCount::kM, lane_idx)
                    , smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx)
                    , smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
                    , smem_iterator_scale_(LayoutScale(Shape::kN), shared_storage.operand_scale.data(), { 1, Shape::kN }, thread_idx)
                {


                    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
                    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

                    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
                    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

                    this->warp_tile_iterator_A_.add_tile_offset({ warp_idx_m, Base::kWarpGemmIterations * warp_idx_k });
                    this->warp_tile_iterator_B_.add_tile_offset({ Base::kWarpGemmIterationsForB * warp_idx_k, warp_idx_n });
                }

                CUTLASS_DEVICE
                    void operator()(int gemm_k_iterations,
                        FragmentC& accum,
                        IteratorA iterator_A,
                        IteratorB iterator_B,
                        IteratorScale iterator_scale,
                        FragmentC const& src_accum)
                {

                    TransformBAfterLDG ldg_converter;
                    TransformBAfterLDS lds_converter;

                    using TransformA
                        = NumericArrayConverter<typename WarpFragmentA::Element, typename FragmentA::Element, FragmentA::kElements>;

                    using TransformScale = NumericArrayConverter<typename SmemIteratorScale::Fragment::Element,
                        typename FragmentScale::Element, FragmentScale::kElements>;

                    TransformA transformA;
                    TransformScale transformScale;

                    accum = src_accum;

                    FragmentA tb_frag_A;
                    FragmentB tb_frag_B;
                    FragmentScale tb_frag_scales;

                    using WarpFragmentScale = typename Dequantizer::FragmentScale;
                    WarpFragmentScale warp_frag_scales;

                    tb_frag_A.clear();
                    tb_frag_B.clear();
                    tb_frag_scales.clear();

                    iterator_A.load(tb_frag_A);
                    iterator_B.load(tb_frag_B);
                    iterator_scale.load(tb_frag_scales);

                    ++iterator_A;
                    ++iterator_B;

                    this->smem_iterator_A_.store(transformA(tb_frag_A));
                    this->smem_iterator_B_.store(ldg_converter(tb_frag_B));
                    this->smem_iterator_scale_.store(transformScale(tb_frag_scales));

                    ++this->smem_iterator_A_;
                    ++this->smem_iterator_B_;

                    __syncthreads();

                    warp_dequantizer_.load(warp_frag_scales);

                    WarpFragmentA warp_frag_A[2];
                    WarpFragmentB warp_frag_B[2];

                    this->warp_tile_iterator_A_.set_kgroup_index(0);
                    this->warp_tile_iterator_B_.set_kgroup_index(0);

                    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
                    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

                    ++this->warp_tile_iterator_A_;
                    ++this->warp_tile_iterator_B_;

                    Operator warp_mma;

                    int smem_write_stage_idx = 1;

                    iterator_A.clear_mask(gemm_k_iterations <= 1);
                    iterator_B.clear_mask(gemm_k_iterations <= 1);



                    CUTLASS_GEMM_LOOP
                        for (; gemm_k_iterations > 0; --gemm_k_iterations)
                        {

                            CUTLASS_PRAGMA_UNROLL
                                for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
                                {


                                    if (warp_mma_k == Base::kWarpGemmIterations - 1)
                                    {

                                        this->smem_iterator_A_.store(transformA(tb_frag_A));

                                        this->smem_iterator_B_.store(ldg_converter(tb_frag_B));

                                        __syncthreads();

                                        ++this->smem_iterator_A_;
                                        ++this->smem_iterator_B_;

                                        if (smem_write_stage_idx == 1)
                                        {
                                            this->smem_iterator_A_.add_tile_offset({ 0, -Base::kStages });
                                            this->smem_iterator_B_.add_tile_offset({ -Base::kStages, 0 });
                                        }
                                        else
                                        {
                                            this->warp_tile_iterator_A_.add_tile_offset(
                                                { 0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations });
                                            this->warp_tile_iterator_B_.add_tile_offset(
                                                { -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterationsForB, 0 });
                                        }

                                        smem_write_stage_idx ^= 1;
                                    }

                                    this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
                                    this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                                    ++this->warp_tile_iterator_A_;

                                    const int warp_tileB_k_compute_offset = warp_mma_k % Base::kNumKIterationsPerWarpBLoad;
                                    const int warp_tileB_k_load_offset = warp_mma_k / Base::kNumKIterationsPerWarpBLoad;
                                    if (warp_tileB_k_compute_offset == Base::kNumKIterationsPerWarpBLoad - 1)
                                    {
                                        this->warp_tile_iterator_B_.set_kgroup_index(
                                            (warp_tileB_k_load_offset + 1) % Base::kWarpGemmIterationsForB);
                                        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_tileB_k_load_offset + 1) % 2]);
                                        ++this->warp_tile_iterator_B_;
                                    }

                                    if (warp_mma_k == 0)
                                    {

                                        iterator_A.load(tb_frag_A);
                                        iterator_B.load(tb_frag_B);

                                        ++iterator_A;
                                        ++iterator_B;

                                        iterator_A.clear_mask(gemm_k_iterations <= 2);
                                        iterator_B.clear_mask(gemm_k_iterations <= 2);
                                    }

                                    typename TransformBAfterLDS::result_type converted_frag_B
                                        = lds_converter(warp_frag_B[warp_tileB_k_load_offset % 2]);
                                    warp_dequantizer_.dequantize(converted_frag_B, warp_frag_scales);
                                    run_warp_mma(
                                        warp_mma, accum, warp_frag_A[warp_mma_k % 2], converted_frag_B, accum, warp_tileB_k_compute_offset);
                                }
                        }
                }
            };


        }
    }
}