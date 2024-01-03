#pragma once

#include "../common/allocator.h"
#include "../common/tensor.h"

namespace bitfusion
{
    namespace layers
    {

        class BaseLayer
        {
        public:
            BaseLayer(cudaStream_t stream, bitfusion::common::IAllocator* allocator, bool is_free_buffer_after_forward,
                cudaDeviceProp* cuda_device_prop = nullptr)
                : stream_(stream)
                , allocator_(allocator)
                , cuda_device_prop_(cuda_device_prop)
                , is_free_buffer_after_forward_(is_free_buffer_after_forward) {};
            virtual ~BaseLayer() = default;

            virtual cudaStream_t getStream()
            {
                return stream_;
            }

            virtual void setStream(cudaStream_t stream)
            {
                stream_ = stream;
            }

        protected:
            cudaStream_t stream_;
            bitfusion::common::IAllocator* allocator_;
            cudaDeviceProp* cuda_device_prop_ = nullptr;

            bool is_free_buffer_after_forward_;
            bool is_allocate_buffer_ = false;
        };

    }
}