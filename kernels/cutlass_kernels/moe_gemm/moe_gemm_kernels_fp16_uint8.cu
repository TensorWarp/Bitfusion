

#include "../../cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h"

namespace bitfusion
{
template class MoeGemmRunner<half, uint8_t>;
}
