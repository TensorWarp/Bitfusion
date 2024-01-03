
#pragma once

#include "../kernels/decodingCommon.h"
#include "../kernels/onlineSoftmaxBeamsearchKernels.h"
#include "baseBeamSearchLayer.h"

#include <optional>

namespace tc = bitfusion::common;

namespace bitfusion
{
namespace layers
{

template <typename T>
class OnlineBeamSearchLayer : public BaseBeamSearchLayer<T>
{
public:
    using Base = BaseBeamSearchLayer<T>;

    class SetupParams : public Base::SetupParams
    {
    public:
        std::optional<std::vector<float>> beam_search_diversity_rate;
        std::optional<std::vector<float>> length_penalty;
    };

    OnlineBeamSearchLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream, tc::IAllocator* allocator,
        bool is_free_buffer_after_forward);

    OnlineBeamSearchLayer(OnlineBeamSearchLayer<T> const& beam_search_layer);

    ~OnlineBeamSearchLayer() override;

    void setup(size_t batch_size, SetupParams const& setupParams);

protected:
    using Base::vocab_size_;
    using Base::vocab_size_padded_;

    using Base::topk_softmax_workspace_size_;
    using Base::topk_softmax_workspace_;

    using typename Base::BeamSearchOutputParams;
    using typename Base::SoftmaxParams;

    void invokeSoftMax(BeamSearchOutputParams& outputs, SoftmaxParams const& params) override;

    using Base::stream_;
    using Base::is_allocate_buffer_;
    using Base::allocator_;

    std::vector<float> mDiversityRate;
    std::vector<float> mLengthPenalty;
    float* diversity_rates_buf_;
    float* length_penalties_buf_;

private:
    void allocateBuffer(size_t batch_size);
    void freeBuffer();
};

}
}
