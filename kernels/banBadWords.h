#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bitfusion
{
    namespace kernels
    {

        template <typename T>
        void invokeBanBadWords(T* logits, const int** output_ids_ptr, const int** parent_ids_ptr, int batch_size,
            int local_batch_size, int beam_width, const int* bad_words, bool share_words, size_t bad_words_len,
            int vocab_size_padded, const int* sequence_lengths, int max_seq_len, cudaStream_t stream);

    } // namespace kernels
} // namespace bitfusion