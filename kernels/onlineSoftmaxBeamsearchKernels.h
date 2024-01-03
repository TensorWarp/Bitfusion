
#pragma once

#include "beamSearchTopkKernels.h"
#include "decodingCommon.h"

namespace bitfusion
{
namespace kernels
{

template <typename T>
void invokeTopkSoftMax(const T* log_probs, const T* bias, const FinishedState* finished, const int* sequence_lengths,
    float* cum_log_probs, float* output_log_probs, int** output_ids_ptr, void* tmp_storage, const int temp_storage_size,
    BeamHypotheses* beam_hyps, const int batch_size, const int beam_width, const int vocab_size, const int* end_ids,
    const float* diversity_rates, const float* length_penalties, cudaStream_t stream);

} // namespace kernels
} // namespace bitfusion
