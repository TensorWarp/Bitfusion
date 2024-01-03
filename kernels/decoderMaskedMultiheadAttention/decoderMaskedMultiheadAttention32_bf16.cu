#include "decoderMaskedMultiheadAttentionLaunch.h"

namespace bitfusion
{
	namespace kernels
	{
		namespace
		{
			/// <summary>
			/// The size per head for multi-head attention.
			/// </summary>
			auto constexpr kSizePerHead = 32;
		}

		namespace mmha
		{

#ifdef ENABLE_BF16
			/// <summary>
			/// Instantiate the launchers for Masked Multihead Attention with __nv_bfloat16 precision.
			/// </summary>
			INSTANTIATE_MMHA_LAUNCHERS(__nv_bfloat16, kSizePerHead)
#endif
		}
	}
}
