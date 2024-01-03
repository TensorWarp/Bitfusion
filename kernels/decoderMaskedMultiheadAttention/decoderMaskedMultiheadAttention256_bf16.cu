

#include "decoderMaskedMultiheadAttentionLaunch.h"

namespace bitfusion
{
namespace kernels
{

namespace
{
auto constexpr kSizePerHead = 256;
} // namespace

namespace mmha
{

#ifdef ENABLE_BF16
INSTANTIATE_MMHA_LAUNCHERS(__nv_bfloat16, kSizePerHead)
#endif

} // namespace mmha

} // namespace kernels
} // namespace bitfusion
