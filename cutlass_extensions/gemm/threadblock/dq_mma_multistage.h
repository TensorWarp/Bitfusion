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
                SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
                typename Enable = void>
            class DqMmaMultistage;

        }
    }
}

#include "../../../../cutlass_extensions/gemm/threadblock/dq_mma_multistage_finegrained.h"
#include "../../../../cutlass_extensions/gemm/threadblock/dq_mma_multistage_percol.h"
