

#include "onlineSoftmaxBeamsearchKernelsTemplate.h"

namespace bitfusion
{
namespace kernels
{

#ifndef FAST_BUILD // skip beam_width between [?, 64] for fast build
INSTANTIATE_BEAMSEARCH_K(float, 64);
INSTANTIATE_BEAMSEARCH_K(half, 64);
#endif // FAST_BUILD

} // namespace kernels
} // namespace bitfusion
