
#pragma once

#include "../common/tensor.h"
#include "../kernels/beamSearchTopkKernels.h"
#include "baseLayer.h"
#include "onlineBeamSearchLayer.h"
#include "topKSamplingLayer.h"
#include "topPSamplingLayer.h"
#include "../runtime/iTensor.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace tc = bitfusion::common;

namespace bitfusion
{
namespace kernels
{
struct BeamHypotheses;
}

namespace layers
{
template <typename T>
class DynamicDecodeLayer : public BaseLayer
{
public:
    DynamicDecodeLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream, tc::IAllocator* allocator,
        bool is_free_buffer_after_forward, cudaDeviceProp* cuda_device_prop);

    ~DynamicDecodeLayer() override;
    DynamicDecodeLayer(DynamicDecodeLayer const& dynamic_decode_layer);

    class SetupParams
    {
    public:
        std::optional<std::vector<float>> temperature;
        std::optional<std::vector<std::int32_t>> min_length;
        std::optional<std::vector<float>> repetition_penalty;
        std::optional<std::vector<float>> presence_penalty;

        std::optional<std::vector<std::uint32_t>> runtime_top_k;
        std::optional<std::vector<float>> runtime_top_p;
        std::optional<std::vector<unsigned long long>> random_seed;

        std::optional<std::vector<float>> top_p_decay;
        std::optional<std::vector<float>> top_p_min;
        std::optional<std::vector<std::int32_t>> top_p_reset_ids;

        std::optional<std::vector<float>> beam_search_diversity_rate;
        std::optional<std::vector<float>> length_penalty;
    };

    void setup(size_t batch_size, size_t beam_width, SetupParams const& setupParams);

    class ForwardParams
    {
    public:
        ForwardParams(int step, int ite, int maxInputLength, int maxAttentionWindow, int localBatchSize,
            tc::Tensor logits, tc::Tensor endIds)
            : step{step}
            , ite{ite}
            , max_input_length{maxInputLength}
            , max_attention_window{maxAttentionWindow}
            , local_batch_size{localBatchSize}
            , logits{std::move(logits)}
            , end_ids{std::move(endIds)}
        {
        }

        int step;
        int ite;
        int max_input_length;
        int max_attention_window;
        int local_batch_size;
        tc::Tensor logits;
        tc::Tensor end_ids;

        std::optional<tc::Tensor> finished;
        std::optional<tc::Tensor> src_cache_indirection;
        std::optional<tc::Tensor> sequence_limit_length;
        std::optional<tc::Tensor> embedding_bias;
        std::optional<tc::Tensor> input_lengths;
        std::optional<tc::Tensor> bad_words_list;
        std::optional<tc::Tensor> stop_words_list;
        std::optional<tc::Tensor> no_repeat_ngram_size;
    };

    class OutputParams
    {
    public:
        explicit OutputParams(tc::Tensor outputIds)
            : output_ids{std::move(outputIds)}
        {
        }

        tc::Tensor output_ids;
        tc::Tensor newTokens;
        std::optional<tc::Tensor> finished;
        std::optional<tc::Tensor> finished_sum;
        std::optional<tc::Tensor> cum_log_probs;
        std::optional<tc::Tensor> parent_ids;
        std::optional<tc::Tensor> sequence_length;
        std::optional<tc::Tensor>
            output_log_probs_tiled;
        std::optional<tc::Tensor>
            output_log_probs;
        std::optional<tc::Tensor>
            tgt_cache_indirection;
        std::shared_ptr<kernels::BeamHypotheses>
            beamHypotheses;

        tc::Tensor output_ids_ptr;
        tc::Tensor parent_ids_ptr;
    };

    void forward(OutputParams& outputs, ForwardParams const& params);
    void allocateBuffer(size_t batch_size, size_t beam_width, size_t max_seq_len);
    void freeBuffer();

private:
    void initialize();

    std::unique_ptr<OnlineBeamSearchLayer<T>> mOnlineBeamsearchDecode;
    std::unique_ptr<TopKSamplingLayer<T>> mTopKDecode;
    std::unique_ptr<TopPSamplingLayer<T>> mTopPDecode;

    size_t vocab_size_;
    size_t vocab_size_padded_;
    cudaDeviceProp* cuda_device_prop_;
    int* zero_parent_ids = nullptr;
    runtime::IBuffer::SharedPtr mIdsPtrHost;

    bool has_diff_runtime_args_ = false;
    int* h_pinned_finished_sum_ = nullptr;
};

}
}
