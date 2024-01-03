

#include "onlineSoftmaxBeamsearchKernelsTemplate.h"

namespace bitfusion
{
namespace kernels
{

#ifndef FAST_BUILD // skip beam_width between [?, 32] for fast build
INSTANTIATE_BEAMSEARCH_K(float, 32);
INSTANTIATE_BEAMSEARCH_K(half, 32);
#endif // FAST_BUILD

} // namespace kernels
} // namespace bitfusion
