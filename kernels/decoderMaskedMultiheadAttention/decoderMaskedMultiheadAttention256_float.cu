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
			auto constexpr kSizePerHead = 256;
		}

		namespace mmha
		{

			/// <summary>
			/// Instantiate the launchers for Masked Multihead Attention with float precision.
			/// </summary>
			INSTANTIATE_MMHA_LAUNCHERS(float, kSizePerHead)

		}
	}
}
