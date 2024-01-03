
#pragma once

#include "../common/tensor.h"
#include "../kernels/decodingCommon.h"
#include "baseSamplingLayer.h"

namespace tc = bitfusion::common;

namespace bitfusion
{
namespace layers
{

template <typename T>
class TopPSamplingLayer : public BaseSamplingLayer<T>
{
public:
    using Base = BaseSamplingLayer<T>;

    class SetupParams : public Base::SetupParams
    {
    public:
        std::optional<std::vector<float>> top_p_decay;
        std::optional<std::vector<float>> top_p_min;
        std::optional<std::vector<std::int32_t>> top_p_reset_ids;
    };

    TopPSamplingLayer(std::size_t vocab_size, std::size_t vocab_size_padded, cudaStream_t stream,
        bitfusion::common::IAllocator* allocator, bool is_free_buffer_after_forward,
        cudaDeviceProp* cuda_device_prop);
    TopPSamplingLayer(TopPSamplingLayer<T> const& top_p_sampling_layer);
    ~TopPSamplingLayer();

    void setup(std::size_t batch_size, SetupParams const& setupParams);

protected:
    void runSampling(DecodingOutputParams& outputs, DecodingParams const& params) override;
    void freeBuffer() override;

    std::uint32_t* runtime_top_k_buf_ = nullptr;
    float* runtime_top_p_buf_ = nullptr;
    float runtime_max_top_p_;
    float* initial_top_p_buf_ = nullptr;
    float* top_p_decay_buf_ = nullptr;
    float* top_p_min_buf_ = nullptr;
    std::int32_t* top_p_reset_ids_buf_ = nullptr;

    std::int32_t* topp_id_vals_buf_ = nullptr;
    std::int32_t* topp_offset_buf_ = nullptr;
    std::int32_t* begin_topp_offset_buf_ = nullptr;
    std::size_t cub_temp_storage_size_;

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
    void allocateBuffer(std::size_t batch_size, std::vector<float> const& top_k);
};

}
}
