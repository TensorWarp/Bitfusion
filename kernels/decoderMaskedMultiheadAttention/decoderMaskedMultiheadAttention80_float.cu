

#include "decoderMaskedMultiheadAttentionLaunch.h"

namespace bitfusion
{
namespace kernels
{

namespace
{
auto constexpr kSizePerHead = 80;
} // namespace

namespace mmha
{

#ifndef FAST_BUILD // skip mmha_80 for fast build
INSTANTIATE_MMHA_LAUNCHERS(float, kSizePerHead)
#endif             // FAST_BUILD

} // namespace mmha

} // namespace kernels
} // namespace bitfusion
