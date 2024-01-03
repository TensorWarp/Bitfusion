#pragma once

#include "../kernels/decodingCommon.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bitfusion
{
    namespace kernels
    {

        template <typename T>
        void invokeBanRepeatNgram(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,
            const int* parent_ids_buf, int batch_size, int local_batch_size, int beam_width,
            const int* no_repeat_ngram_size_buf, int id_offset, int vocab_size_padded, size_t step, cudaStream_t stream);

    } // namespace kernels
} // namespace bitfusion