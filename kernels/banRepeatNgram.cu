#include "../common/cudaUtils.h"
#include "banRepeatNgram.h"

using namespace bitfusion::common;

namespace bitfusion
{
    namespace kernels
    {

        template <typename T>
        __global__ void ban_repeat_ngram(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,
            const int* parent_ids_buf, int batch_size, int beam_width, const int* no_repeat_ngram_size_buf, int id_offset,
            int vocab_size_padded, size_t step)
        {

            const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int local_batch_idx = blockIdx.y / beam_width;
            const int beam_idx = blockIdx.y % beam_width;
            const bool beam_search = beam_width > 1;
            const int no_repeat_ngram_size = no_repeat_ngram_size_buf[local_batch_idx];

            if (no_repeat_ngram_size == 0 || step < no_repeat_ngram_size)
            {
                return;
            }

            if ((finished_buf != nullptr) && (finished_buf[id_offset + local_batch_idx * beam_width + beam_idx].isFinished()))
            {
                return;
            }

            extern __shared__ int shared_tokens[];
            int shared_tokens_length = blockDim.x + no_repeat_ngram_size - 1;
            int* last_tokens = &shared_tokens[shared_tokens_length];
            int last_tokens_length = no_repeat_ngram_size - 1;

            if (threadIdx.x == 0)
            {
                int parent_id = beam_idx;
                int start_record_idx = min(output_idx + shared_tokens_length, (int)step);
                int shared_token_idx = start_record_idx == step ? step - output_idx - 1 : shared_tokens_length - 1;
                int last_token_idx = last_tokens_length - 1;

                for (int curr_idx = step - 1; curr_idx >= output_idx; curr_idx--)
                {
                    if (last_token_idx >= 0)
                    {
                        last_tokens[last_token_idx--] = output_ids_buf[blockIdx.y][curr_idx * batch_size * beam_width
                            + id_offset + local_batch_idx * beam_width + parent_id];
                    }

                    if (curr_idx < start_record_idx)
                    {
                        shared_tokens[shared_token_idx--] = output_ids_buf[blockIdx.y][curr_idx * batch_size * beam_width
                            + id_offset + local_batch_idx * beam_width + parent_id];
                    }

                    if (beam_search)
                    {
                        parent_id = parent_ids_buf[curr_idx * batch_size * beam_width + id_offset + local_batch_idx * beam_width
                            + parent_id];
                    }
                }
            }

            __syncthreads();

            if (output_idx > step - no_repeat_ngram_size)
            {
                return;
            }

            bool ban_ngram = true;

            for (int ngram_idx = 0; ngram_idx < no_repeat_ngram_size - 1; ngram_idx++)
            {
                if (shared_tokens[threadIdx.x + ngram_idx] != last_tokens[ngram_idx])
                {
                    ban_ngram = false;
                    break;
                }
            }

            if (ban_ngram)
            {
                int banned_token = shared_tokens[threadIdx.x + no_repeat_ngram_size - 1];
                logits[local_batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token]
                    = static_cast<T>(-INFINITY);
            }
        }

        template <typename T>
        void invokeBanRepeatNgram(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,
            const int* parent_ids_buf, int batch_size, int local_batch_size, int beam_width,
            const int* no_repeat_ngram_size_buf, int id_offset, int vocab_size_padded, size_t step, cudaStream_t stream)
        {
            int max_no_repeat_ngram_size = 32;

            dim3 block, grid;
            constexpr size_t max_blocks{ 256 };
            block.x = min(((step + 32 - 1) / 32) * 32, max_blocks);
            grid.x = (step + block.x - 1) / block.x;
            grid.y = local_batch_size * beam_width;

            ban_repeat_ngram << <grid, block, (block.x + 2 * (max_no_repeat_ngram_size - 1)) * sizeof(int), stream >> > (logits,
                output_ids_buf, finished_buf, parent_ids_buf, batch_size, beam_width, no_repeat_ngram_size_buf, id_offset,
                vocab_size_padded, step);
            sync_check_cuda_error();
        }

#define INVOKE_BAN_REPEAT_NGRAM(T)                                                                                     \
    template void invokeBanRepeatNgram(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,       \
        const int* parent_ids_buf, int batch_size, int local_batch_size, int beam_width,                               \
        const int* no_repeat_ngram_size_buf, int id_offset, int vocab_size_padded, size_t step, cudaStream_t stream);

        INVOKE_BAN_REPEAT_NGRAM(float)
            INVOKE_BAN_REPEAT_NGRAM(half)
#ifdef ENABLE_BF16
            INVOKE_BAN_REPEAT_NGRAM(__nv_bfloat16)
#endif
#undef INVOKE_BAN_REPEAT_NGRAM

    }

}