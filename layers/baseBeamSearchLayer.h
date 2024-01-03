
#pragma once

#include "../common/tensor.h"
#include "../kernels/beamSearchTopkKernels.h"
#include "../kernels/decodingCommon.h"
#include "../kernels/penaltyTypes.h"
#include "baseLayer.h"
#include "decodingParams.h"

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
class BaseBeamSearchLayer : public BaseLayer
{
public:
    using SetupParams = DecodingSetupParams;

    BaseBeamSearchLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream, tc::IAllocator* allocator,
        bool is_free_buffer_after_forward);

    BaseBeamSearchLayer(BaseBeamSearchLayer<T> const& beam_search_layer);

    ~BaseBeamSearchLayer() override;

    using SoftmaxParams = DecodingParams;

    class ForwardParams : public SoftmaxParams
    {
    public:
        ForwardParams(int step, int ite, tc::Tensor logits, tc::Tensor endIds, tc::Tensor src_cache_indirection,
            int max_attention_window, int max_seq_len)
            : SoftmaxParams(step, ite, std::move(logits), std::move(endIds))
            , src_cache_indirection{std::move(src_cache_indirection)}
            , max_attention_window{max_attention_window}
            , max_seq_len{max_seq_len}
        {
        }

        int max_attention_window;
        int max_seq_len;
        tc::Tensor src_cache_indirection;

        std::optional<tc::Tensor> embedding_bias;
        std::optional<tc::Tensor> input_lengths;
    };

    class BeamSearchOutputParams : public DecodingOutputParams
    {
    public:
        explicit BeamSearchOutputParams(tc::Tensor outputIds, tc::Tensor parentIds, tc::Tensor tgt_cache_indirection)
            : DecodingOutputParams{std::move(outputIds)}
            , parent_ids{std::move(parentIds)}
            , tgt_cache_indirection{std::move(tgt_cache_indirection)}
        {
        }

        tc::Tensor parent_ids;
        tc::Tensor
            tgt_cache_indirection;
        std::shared_ptr<kernels::BeamHypotheses>
            beamHypotheses;

        tc::Tensor
            parent_ids_ptr;
    };

    void forward(BeamSearchOutputParams& outputs, ForwardParams const& params);

protected:
    size_t vocab_size_;
    size_t vocab_size_padded_;

    size_t topk_softmax_workspace_size_;
    void* topk_softmax_workspace_ = nullptr;

    std::vector<float> mTemperature;
    std::vector<int> mMinLength;
    std::vector<float> mRepetitionPenalty;
    float* temperature_buf_;
    int* min_lengths_buf_;
    float* repetition_penalty_buf_;
    bitfusion::kernels::RepetitionPenaltyType mRepetitionPenaltyType;

    virtual void invokeSoftMax(BeamSearchOutputParams& outputs, SoftmaxParams const& params) = 0;

    void setupBase(size_t batch_size, SetupParams const& setupParams);

private:
    void allocateBuffer(size_t batch_size);
    void freeBuffer();
};

void update_indir_cache_kernelLauncher(int* tgt_indir_cache, const int* src_indir_cache, const int* beam_ids,
    const bitfusion::kernels::FinishedState* finished, int batch_dim, int beam_width, int max_seq_len, int ite,
    cudaStream_t stream);

}
}
