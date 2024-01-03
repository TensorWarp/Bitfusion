

#include "onlineSoftmaxBeamsearchKernelsTemplate.h"

namespace bitfusion
{
namespace kernels
{

INSTANTIATE_BEAMSEARCH_K(float, 4);
INSTANTIATE_BEAMSEARCH_K(half, 4);

} // namespace kernels
} // namespace bitfusion
