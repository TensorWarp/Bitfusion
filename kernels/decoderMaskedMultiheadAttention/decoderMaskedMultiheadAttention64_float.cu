

#include "decoderMaskedMultiheadAttentionLaunch.h"

namespace bitfusion
{
namespace kernels
{

namespace
{
auto constexpr kSizePerHead = 64;
} // namespace

namespace mmha
{

INSTANTIATE_MMHA_LAUNCHERS(float, kSizePerHead)

} // namespace mmha

} // namespace kernels
} // namespace bitfusion
