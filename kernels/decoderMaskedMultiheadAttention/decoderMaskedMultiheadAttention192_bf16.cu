

#include "decoderMaskedMultiheadAttentionLaunch.h"

namespace bitfusion
{
namespace kernels
{

namespace
{
auto constexpr kSizePerHead = 192;
} // namespace

namespace mmha
{

#ifndef FAST_BUILD // skip mmha_192 for fast build
#ifdef ENABLE_BF16
INSTANTIATE_MMHA_LAUNCHERS(__nv_bfloat16, kSizePerHead)
#endif // ENABLE_BF16
#endif // FAST_BUILD

} // namespace mmha

} // namespace kernels
} // namespace bitfusion
