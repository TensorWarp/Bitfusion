#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"
#include "cutlass/matrix_coord.h"

#include "../../../../cutlass_extensions/gemm/kernel/gemm_moe_problem_visitor.h"
#include "../../../../cutlass_extensions/gemm/kernel/moe_problem_visitor.h"


namespace cutlass
{
    namespace gemm
    {
        namespace kernel
        {

            template <typename ThreadblockShape, GroupScheduleMode GroupScheduleMode_, int PrefetchTileCount, int ThreadCount,
                bool Transposed = false>
            struct GemmMoeProblemVisitor
                : public MoeProblemVisitor<detail::GemmGroupedProblemSizeHelper<ThreadblockShape, Transposed>, ThreadblockShape,
                GroupScheduleMode_, PrefetchTileCount, ThreadCount>
            {

                static bool const kTransposed = Transposed;

                using ProblemSizeHelper = detail::GemmGroupedProblemSizeHelper<ThreadblockShape, Transposed>;
                using Base
                    = MoeProblemVisitor<ProblemSizeHelper, ThreadblockShape, GroupScheduleMode_, PrefetchTileCount, ThreadCount>;
                using Params = typename Base::Params;
                using SharedStorage = typename Base::SharedStorage;

                CUTLASS_DEVICE
                    GemmMoeProblemVisitor(Params const& params_, SharedStorage& shared_storage_, int32_t block_idx)
                    : Base(params_, shared_storage_, block_idx)
                {
                }
            };


        }
    }
}