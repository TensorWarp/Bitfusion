

#include "../../../kernels/cutlass_kernels/int8_gemm/int8_gemm_template.h"

namespace bitfusion
{
namespace kernels
{
namespace cutlass_kernels
{

template class CutlassInt8GemmRunner<half>;

} // namespace cutlass_kernels
} // namespace kernels
} // namespace bitfusion
