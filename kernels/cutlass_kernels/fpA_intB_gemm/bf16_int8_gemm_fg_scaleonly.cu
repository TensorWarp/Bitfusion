

#include "../../../kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace bitfusion
{
namespace kernels
{
namespace cutlass_kernels
{
#ifdef ENABLE_BF16
template class CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>;
#endif
} // namespace cutlass_kernels
} // namespace kernels
} // namespace bitfusion
