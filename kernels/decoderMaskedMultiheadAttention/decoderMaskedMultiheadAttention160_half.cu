

#include "decoderMaskedMultiheadAttentionLaunch.h"

namespace bitfusion
{
namespace kernels
{

namespace
{
auto constexpr kSizePerHead = 160;
} // namespace

namespace mmha
{

#ifndef FAST_BUILD // skip mmha_160 for fast build
INSTANTIATE_MMHA_LAUNCHERS(uint16_t, kSizePerHead)
#endif             // FAST_BUILD

} // namespace mmha

} // namespace kernels
} // namespace bitfusion
