
#pragma once

#include "../common/memoryUtils.h"
#include "../common/tensor.h"
#include "../kernels/decodingCommon.h"
#include "baseSamplingLayer.h"

namespace bitfusion
{
namespace layers
{

template <typename T>
class TopKSamplingLayer : public BaseSamplingLayer<T>
{
public:
    using Base = BaseSamplingLayer<T>;
    using SetupParams = typename Base::SetupParams;

    TopKSamplingLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
        bitfusion::common::IAllocator* allocator, bool is_free_buffer_after_forward);
    TopKSamplingLayer(TopKSamplingLayer<T> const& top_k_sampling_layer);
    ~TopKSamplingLayer();

    void setup(size_t batch_size, SetupParams const& setupParams);

protected:
    void runSampling(DecodingOutputParams& outputs, DecodingParams const& params) override;

    void freeBuffer() override;

    uint32_t runtime_max_top_k_ = 1;
    uint32_t* runtime_top_k_buf_ = nullptr;
    float* runtime_top_p_buf_ = nullptr;
    using Base::vocab_size_;
    using Base::vocab_size_padded_;

    using Base::sampling_workspace_size_;
    using Base::sampling_workspace_;
    using Base::curandstate_buf_;
    using Base::random_seeds_buf_;
    using Base::skip_decode_buf_;
    using Base::skip_decode_;
    using Base::skip_any_;
    using Base::runtime_logits_buf_;

    using Base::stream_;
    using Base::allocator_;
    using Base::is_allocate_buffer_;

private:
    void allocateBuffer(size_t batch_size, std::vector<uint32_t> const& top_k);
};

}
}
