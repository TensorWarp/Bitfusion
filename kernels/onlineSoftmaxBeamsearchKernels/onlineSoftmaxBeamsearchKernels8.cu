

#include "onlineSoftmaxBeamsearchKernelsTemplate.h"

namespace bitfusion
{
namespace kernels
{

INSTANTIATE_BEAMSEARCH_K(float, 8);
INSTANTIATE_BEAMSEARCH_K(half, 8);

} // namespace kernels
} // namespace bitfusion
