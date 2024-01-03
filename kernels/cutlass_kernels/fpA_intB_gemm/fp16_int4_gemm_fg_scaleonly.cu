

#include "../../../kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace bitfusion
{
namespace kernels
{
namespace cutlass_kernels
{
template class CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>;
} // namespace cutlass_kernels
} // namespace kernels
} // namespace bitfusion
