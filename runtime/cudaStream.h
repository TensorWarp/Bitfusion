
#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/logger.h"
#include "cudaEvent.h"

#include <cuda_runtime_api.h>

#include <memory>

namespace bitfusion::runtime
{

class CudaStream
{
public:
    explicit CudaStream(unsigned int flags = cudaStreamNonBlocking, int priority = 0)
        : mDevice{bitfusion::common::getDevice()}
    {
        cudaStream_t stream;
        CUDA_CHECK(::cudaStreamCreateWithPriority(&stream, flags, priority));
        LOG_TRACE("Created stream %p", stream);
        bool constexpr ownsStream{true};
        mStream = StreamPtr{stream, Deleter{ownsStream}};
    }

    explicit CudaStream(cudaStream_t stream, int device, bool ownsStream = true)
        : mDevice{device}
    {
        CHECK_WITH_INFO(stream != nullptr, "stream is nullptr");
        mStream = StreamPtr{stream, Deleter{ownsStream}};
    }

    [[nodiscard]] int getDevice() const
    {
        return mDevice;
    }

    [[nodiscard]] cudaStream_t get() const
    {
        return mStream.get();
    }

    void synchronize() const
    {
        CUDA_CHECK(::cudaStreamSynchronize(get()));
    }

    void record(CudaEvent::pointer event) const
    {
        CUDA_CHECK(::cudaEventRecord(event, get()));
    }

    void record(CudaEvent const& event) const
    {
        record(event.get());
    }

    void wait(CudaEvent::pointer event) const
    {
        CUDA_CHECK(::cudaStreamWaitEvent(get(), event));
    }

    void wait(CudaEvent const& event) const
    {
        wait(event.get());
    }

private:
    class Deleter
    {
    public:
        explicit Deleter(bool ownsStream)
            : mOwnsStream{ownsStream}
        {
        }

        explicit Deleter()
            : Deleter{true}
        {
        }

        constexpr void operator()(cudaStream_t stream) const
        {
            if (mOwnsStream && stream != nullptr)
            {
                CUDA_CHECK(::cudaStreamDestroy(stream));
                LOG_TRACE("Destroyed stream %p", stream);
            }
        }

    private:
        bool mOwnsStream;
    };

    using StreamPtr = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, Deleter>;

    StreamPtr mStream;
    int mDevice{-1};
};

}
