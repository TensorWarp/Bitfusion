#pragma once

#include "common.h"

namespace bitfusion
{
	namespace kernels
	{
		void weight_only_batched_gemv_launcher(const WeightOnlyParams& params, cudaStream_t stream);
	}
}