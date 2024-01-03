#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "../../../../cutlass_extensions/gemm/threadblock/dq_mma_base.h"
#include "../../../../cutlass_extensions/gemm/warp/mma_tensorop_dequantizer.h"
#include "../../../../cutlass_extensions/interleaved_numeric_conversion.h"


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
                cutlass::arch::CacheOperation::Kind CacheOpA,
                typename IteratorB_,
                typename SmemIteratorB_,
                cutlass::arch::CacheOperation::Kind CacheOpB,
                typename IteratorScale_,
                typename SmemIteratorScale_,
                typename ElementC_,
                typename LayoutC_,
                typename Policy_,
                int Stages,
                typename TransformBAfterLDS_,
                WeightOnlyQuantOp QuantOp_,
                SharedMemoryClearOption SharedMemoryClear>
            class DqMmaMultistage<Shape_, IteratorA_, SmemIteratorA_, CacheOpA, IteratorB_, SmemIteratorB_, CacheOpB,
                IteratorScale_, SmemIteratorScale_, ElementC_, LayoutC_, Policy_, Stages, TransformBAfterLDS_, QuantOp_,
                SharedMemoryClear, std::enable_if_t<!isFinegrained(QuantOp_)>>
                : public DqMmaBase<Shape_, Policy_, typename IteratorScale_::Element, Stages, QuantOp_>
            {
            public:
                using Base = DqMmaBase<Shape_, Policy_, typename IteratorScale_::Element, Stages, QuantOp_>;
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

                static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
                static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

                using TransformBAfterLDS = TransformBAfterLDS_;

                static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;


                using FragmentScale = typename IteratorScale::Fragment;

                using FragmentC = typename Policy::Operator::FragmentC;

                using Operator = typename Policy::Operator;

                using ArchTag = arch::Sm80;

                using Dequantizer = warp::MmaTensorOpDequantizer<Operator, typename Base::WarpGemm, Operand::kB, ElementScale,
                    LayoutScale, 32, QuantOp>;

                static ComplexTransform const kTransformA = Operator::kTransformA;

                static ComplexTransform const kTransformB = Operator::kTransformB;

                struct Detail
                {

                    static_assert(Base::kWarpGemmIterations > 1,
                        "The pipelined structure requires at least two warp-level "
                        "GEMM operations.");

                    static int const AsyncCopyIterationsPerStageA = IteratorA::ThreadMap::Iterations::kCount;

                    static int const AsyncCopyIterationsPerStageB = IteratorB::ThreadMap::Iterations::kCount;

                    static int const kStages = Stages;

                    static int const kAccessesPerGroupA
                        = (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

                    static int const kAccessesPerGroupB
                        = (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
                };

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

            private:

                SmemIteratorA smem_iterator_A_;

                SmemIteratorB smem_iterator_B_;

                SmemIteratorScale smem_iterator_scale_;

            public:
                CUTLASS_DEVICE
                    DqMmaMultistage(
                        typename Base::SharedStorage& shared_storage,
                        const int group_size,
                        int thread_idx,
                        int warp_idx,
                        int lane_idx)
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
                    void copy_tiles_and_advance(
                        IteratorA& iterator_A, IteratorB& iterator_B, int group_start_A = 0, int group_start_B = 0)
                {
                    iterator_A.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
                    this->smem_iterator_A_.set_iteration_index(group_start_A);

                    CUTLASS_PRAGMA_UNROLL
                        for (int j = 0; j < Detail::kAccessesPerGroupA; ++j)
                        {
                            if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA)
                            {
                                typename IteratorA::AccessType* dst_ptr
                                    = reinterpret_cast<typename IteratorA::AccessType*>(this->smem_iterator_A_.get());

                                int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value
                                    * IteratorA::ThreadMap::kElementsPerAccess / IteratorA::kAccessesPerVector / 8;

                                CUTLASS_PRAGMA_UNROLL
                                    for (int v = 0; v < IteratorA::kAccessesPerVector; ++v)
                                    {
                                        auto gmem_ptr = iterator_A.get();

                                        if (SharedMemoryClear == SharedMemoryClearOption::kZfill)
                                        {
                                            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(dst_ptr + v, gmem_ptr, iterator_A.valid());
                                        }
                                        else
                                        {
                                            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(dst_ptr + v, gmem_ptr, iterator_A.valid());
                                        }

                                        ++iterator_A;
                                    }

                                ++this->smem_iterator_A_;
                            }
                        }

                    iterator_B.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);
                    this->smem_iterator_B_.set_iteration_index(group_start_B);

                    CUTLASS_PRAGMA_UNROLL
                        for (int j = 0; j < Detail::kAccessesPerGroupB; ++j)
                        {
                            if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB)
                            {
                                typename IteratorB::AccessType* dst_ptr
                                    = reinterpret_cast<typename IteratorB::AccessType*>(this->smem_iterator_B_.get());

                                int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value
                                    * IteratorB::ThreadMap::kElementsPerAccess / IteratorB::kAccessesPerVector / 8;

                                CUTLASS_PRAGMA_UNROLL
                                    for (int v = 0; v < IteratorB::kAccessesPerVector; ++v)
                                    {
                                        auto gmem_ptr = iterator_B.get();

                                        if (SharedMemoryClear == SharedMemoryClearOption::kZfill)
                                        {
                                            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(dst_ptr + v, gmem_ptr, iterator_B.valid());
                                        }
                                        else
                                        {
                                            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(dst_ptr + v, gmem_ptr, iterator_B.valid());
                                        }

                                        ++iterator_B;
                                    }
                                ++this->smem_iterator_B_;
                            }
                        }
                }

                CUTLASS_DEVICE
                    void operator()(
                        int gemm_k_iterations,
                        FragmentC& accum,
                        IteratorA iterator_A,
                        IteratorB iterator_B,
                        IteratorScale iterator_scale,
                        FragmentC const& src_accum)
                {


                    TransformBAfterLDS lds_converter;

                    FragmentScale tb_frag_scales;
                    tb_frag_scales.clear();
                    iterator_scale.load(tb_frag_scales);
                    this->smem_iterator_scale_.store(tb_frag_scales);

                    CUTLASS_PRAGMA_UNROLL
                        for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations)
                        {

                            iterator_A.clear_mask(gemm_k_iterations == 0);
                            iterator_B.clear_mask(gemm_k_iterations == 0);

                            iterator_A.set_iteration_index(0);
                            this->smem_iterator_A_.set_iteration_index(0);

                            CUTLASS_PRAGMA_UNROLL
                                for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j)
                                {
                                    typename IteratorA::AccessType* dst_ptr
                                        = reinterpret_cast<typename IteratorA::AccessType*>(this->smem_iterator_A_.get());

                                    CUTLASS_PRAGMA_UNROLL
                                        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v)
                                        {
                                            int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value
                                                * IteratorA::ThreadMap::kElementsPerAccess / IteratorA::kAccessesPerVector / 8;

                                            int src_bytes = (iterator_A.valid() ? kSrcBytes : 0);

                                            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                                                dst_ptr + v, iterator_A.get(), iterator_A.valid());

                                            ++iterator_A;
                                        }

                                    ++this->smem_iterator_A_;
                                }

                            iterator_B.set_iteration_index(0);
                            this->smem_iterator_B_.set_iteration_index(0);

                            CUTLASS_PRAGMA_UNROLL
                                for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j)
                                {
                                    typename IteratorB::AccessType* dst_ptr
                                        = reinterpret_cast<typename IteratorB::AccessType*>(this->smem_iterator_B_.get());

                                    CUTLASS_PRAGMA_UNROLL
                                        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v)
                                        {
                                            int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value
                                                * IteratorB::ThreadMap::kElementsPerAccess / IteratorB::kAccessesPerVector / 8;

                                            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                                                dst_ptr + v, iterator_B.get(), iterator_B.valid());

                                            ++iterator_B;
                                        }

                                    ++this->smem_iterator_B_;
                                }

                            iterator_A.add_tile_offset({ 0, 1 });
                            iterator_B.add_tile_offset({ 1, 0 });

                            this->smem_iterator_A_.add_tile_offset({ 0, 1 });
                            this->smem_iterator_B_.add_tile_offset({ 1, 0 });

                            cutlass::arch::cp_async_fence();
                        }

                    accum = src_accum;


                    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage)
                    {

                        SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

                        typename IteratorA::AccessType zero_A;
                        zero_A.clear();

                        last_smem_iterator_A.set_iteration_index(0);

                        CUTLASS_PRAGMA_UNROLL
                            for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j)
                            {

                                typename IteratorA::AccessType* dst_ptr
                                    = reinterpret_cast<typename IteratorA::AccessType*>(last_smem_iterator_A.get());

                                *dst_ptr = zero_A;

                                ++last_smem_iterator_A;
                            }

                        SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
                        typename IteratorB::AccessType zero_B;

                        zero_B.clear();
                        last_smem_iterator_B.set_iteration_index(0);

                        CUTLASS_PRAGMA_UNROLL
                            for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j)
                            {

                                typename IteratorB::AccessType* dst_ptr
                                    = reinterpret_cast<typename IteratorB::AccessType*>(last_smem_iterator_B.get());

                                *dst_ptr = zero_B;

                                ++last_smem_iterator_B;
                            }
                    }

                    cutlass::arch::cp_async_wait<Base::kStages - 2>();
                    __syncthreads();

                    WarpFragmentA warp_frag_A[2];
                    WarpFragmentB warp_frag_B[2];
                    typename Dequantizer::FragmentScale warp_frag_scales;

                    Operator warp_mma;

                    this->warp_tile_iterator_A_.set_kgroup_index(0);
                    this->warp_tile_iterator_B_.set_kgroup_index(0);

                    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
                    this->warp_tile_iterator_B_.load(warp_frag_B[0]);
                    warp_dequantizer_.load(warp_frag_scales);

                    ++this->warp_tile_iterator_A_;
                    ++this->warp_tile_iterator_B_;

                    iterator_A.clear_mask(gemm_k_iterations == 0);
                    iterator_B.clear_mask(gemm_k_iterations == 0);

                    int smem_write_stage_idx = Base::kStages - 1;
                    int smem_read_stage_idx = 0;


                    CUTLASS_GEMM_LOOP
                        for (; gemm_k_iterations > (-Base::kStages + 1);)
                        {

                            CUTLASS_PRAGMA_UNROLL
                                for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
                                {


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

                                    typename TransformBAfterLDS::result_type converted_frag_B
                                        = lds_converter(warp_frag_B[warp_tileB_k_load_offset % 2]);
                                    warp_dequantizer_.dequantize(converted_frag_B, warp_frag_scales);

                                    run_warp_mma(
                                        warp_mma, accum, warp_frag_A[warp_mma_k % 2], converted_frag_B, accum, warp_tileB_k_compute_offset);

                                    if (warp_mma_k < Base::kWarpGemmIterations - 1)
                                    {
                                        int group_start_iteration_A, group_start_iteration_B;

                                        group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
                                        group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

                                        copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A, group_start_iteration_B);
                                    }

                                    if (warp_mma_k + 2 == Base::kWarpGemmIterations)
                                    {
                                        int group_start_iteration_A, group_start_iteration_B;
                                        group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
                                        group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

                                        copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A, group_start_iteration_B);

                                        cutlass::arch::cp_async_fence();

                                        arch::cp_async_wait<Base::kStages - 2>();
                                        __syncthreads();

                                        iterator_A.add_tile_offset({ 0, 1 });
                                        iterator_B.add_tile_offset({ 1, 0 });

                                        this->smem_iterator_A_.add_tile_offset({ 0, 1 });
                                        this->smem_iterator_B_.add_tile_offset({ 1, 0 });

                                        if (smem_write_stage_idx == (Base::kStages - 1))
                                        {
                                            this->smem_iterator_A_.add_tile_offset({ 0, -Base::kStages });
                                            this->smem_iterator_B_.add_tile_offset({ -Base::kStages, 0 });
                                            smem_write_stage_idx = 0;
                                        }
                                        else
                                        {
                                            ++smem_write_stage_idx;
                                        }

                                        if (smem_read_stage_idx == (Base::kStages - 1))
                                        {
                                            this->warp_tile_iterator_A_.add_tile_offset(
                                                { 0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations });
                                            this->warp_tile_iterator_B_.add_tile_offset(
                                                { -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterationsForB, 0 });
                                            smem_read_stage_idx = 0;
                                        }
                                        else
                                        {
                                            ++smem_read_stage_idx;
                                        }

                                        --gemm_k_iterations;
                                        iterator_A.clear_mask(gemm_k_iterations == 0);
                                        iterator_B.clear_mask(gemm_k_iterations == 0);
                                    }
                                }
                        }

                    if (SharedMemoryClear == SharedMemoryClearOption::kZfill)
                    {
                        cutlass::arch::cp_async_fence();
                        cutlass::arch::cp_async_wait<0>();
                        __syncthreads();
                    }
                }
            };


        }
    }
}