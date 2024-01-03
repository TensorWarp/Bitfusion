
#pragma once

#include <cuda_fp16.h>

#include "penaltyTypes.h"

namespace bitfusion
{
namespace kernels
{

template <typename T>
void invokeAddBiasApplyPenalties(T* logits, const int** output_ids_ptr, const int** parent_ids_ptr,
    const int* input_lengths, const int* sequence_lengths, const T* bias, const int ite, const int local_batch_size,
    const int batch_size, const int beam_width, const int vocab_size, const int vocab_size_padded, const int* end_ids,
    const float* temperatures, const std::vector<float>& h_temperatures, const float* repetition_penalties,
    const std::vector<float>& h_repetition_penalties, const RepetitionPenaltyType repetition_penalty_type,
    const int* min_lengths, int max_seq_len, cudaStream_t stream);

} // namespace kernels
} // namespace bitfusion
