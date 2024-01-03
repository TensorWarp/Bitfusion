#pragma once

#include "decodingCommon.h"
#include <cuda_runtime.h>

namespace bitfusion
{
    namespace kernels
    {

        struct BeamHypotheses
        {
            int* output_ids_tgt = nullptr;
            int* sequence_lengths_tgt = nullptr;
            float* cum_log_probs = nullptr;
            float* normed_scores = nullptr;
            float* log_probs = nullptr;
            float* min_normed_scores = nullptr;
            int* num_beams = nullptr;
            bool* is_done = nullptr;

            const int* output_ids_src;
            const int** output_ids_src_ptr;
            const int* parent_ids_src;
            const int** parent_ids_src_ptr;
            const int* sequence_lengths_src;
            const int* end_ids;
            const float* log_probs_src;
            const int* input_lengths;

            int step;
            int ite;
            int batch_size;
            int local_batch_size;
            int max_seq_len;
            float* length_penalties;

            bool early_stopping = true;
            bool is_return_normed_score = true;
        };

        template <typename T>
        void invokeTopkBeamSearch(void* workspace, size_t& workspace_size, T* log_probs, int* ids, BeamHypotheses* beam_hyps,
            const bool* finished, const int* sequence_lengths, const int batch_size, const int beam_width,
            const int vocab_size_padded_, const T diversity_rate, const float length_penalty, const int* end_ids,
            cudaStream_t stream);

        template <typename T>
        void invokeTileEncoderResults(T* tiled_encoder_output, int* tiled_encoder_sequence_length, const T* encoder_output,
            const int* encoder_sequence_length, const size_t batch_size, const size_t beam_width, const size_t mem_max_seq_len,
            const size_t d_model, cudaStream_t stream);

        void invokeInsertUnfinishedPath(BeamHypotheses beam_hyps, const FinishedState* finished, const float* cum_log_probs,
            const int batch_size, const int beam_width, cudaStream_t stream);

        void invokeCopyBatchMajorToGeneralPtr(
            void* output_ids_ptr, int* output_ids, int batch_size, int beam_width, int max_seq_len, cudaStream_t stream);

        void invokeCopyGeneralPtrToBatchMajor(
            int* output_ids, void* output_ids_ptr, int batch_size, int beam_width, int max_seq_len, cudaStream_t stream);

        void invokeSeqlenMajorToBatchMajor(
            int* batchMajoredIds, int* seqlenMajorIds, int batch_size, int beam_width, int max_seq_len, cudaStream_t stream);

    }
}