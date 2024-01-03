

#include "onlineSoftmaxBeamsearchKernelsTemplate.h"

namespace bitfusion
{
namespace kernels
{
#ifndef FAST_BUILD // skip beam_width between [?, 16] for fast build
INSTANTIATE_BEAMSEARCH_K(float, 16);
INSTANTIATE_BEAMSEARCH_K(half, 16);
#endif // FAST_BUILD
} // namespace kernels
} // namespace bitfusion
