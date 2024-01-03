
#pragma once

#include <curand_kernel.h>

#include "../common/tensor.h"
#include "../kernels/penaltyTypes.h"
#include "baseLayer.h"
#include "decodingParams.h"

namespace tc = bitfusion::common;

namespace bitfusion
{
namespace layers
{

template <typename T>
class BaseSamplingLayer : public BaseLayer
{
public:
    BaseSamplingLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
        bitfusion::common::IAllocator* allocator, bool is_free_buffer_after_forward,
        cudaDeviceProp* cuda_device_prop);

    BaseSamplingLayer(BaseSamplingLayer const& sampling_layer);

    ~BaseSamplingLayer() override;

    class SetupParams : public DecodingSetupParams
    {
    public:
        std::optional<std::vector<std::uint32_t>> runtime_top_k;
        std::optional<std::vector<float>> runtime_top_p;
        std::optional<std::vector<unsigned long long>> random_seed;
    };

    class ForwardParams : public DecodingParams
    {
    public:
        ForwardParams(int step, int ite, tc::Tensor logits, tc::Tensor end_ids, int max_seq_len)
            : DecodingParams{step, ite, std::move(logits), std::move(end_ids)}
            , max_seq_len{max_seq_len}
        {
        }

        int max_seq_len;

        std::optional<tc::Tensor> embedding_bias;
        std::optional<tc::Tensor> input_lengths;
    };

    void forward(DecodingOutputParams& outputs, ForwardParams const& params);

protected:
    size_t vocab_size_;
    size_t vocab_size_padded_;

    size_t sampling_workspace_size_;
    void* sampling_workspace_ = nullptr;
    curandState_t* curandstate_buf_ = nullptr;
    unsigned long long* random_seeds_buf_ = nullptr;

    float* temperature_buf_ = nullptr;
    float* repetition_penalty_buf_ = nullptr;
    int32_t* min_lengths_buf_ = nullptr;
    bool* skip_decode_buf_ = nullptr;
    T* runtime_logits_buf_ = nullptr;

    std::vector<float> mTemperature;
    std::vector<float> mRepetitionPenalty;
    std::vector<int32_t> mMinLengths;
    bool* skip_decode_ = nullptr;
    bool skip_any_ = false;

    bitfusion::kernels::RepetitionPenaltyType repetition_penalty_type_
        = bitfusion::kernels::RepetitionPenaltyType::None;

    virtual void runSampling(DecodingOutputParams& outputs, DecodingParams const& params) = 0;

    virtual void freeBuffer();
    void setupBase(size_t batch_size, SetupParams const& setupParams);

private:
    void allocateBuffer(size_t batch_size);
    bool isValidBatchSize(size_t batch_size);
};

}
}
