
#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/logger.h"

#include <cuda_runtime_api.h>

#include <memory>

namespace bitfusion::runtime
{

class CudaEvent
{
public:
    using pointer = cudaEvent_t;

    explicit CudaEvent(unsigned int flags = cudaEventDisableTiming)
    {
        pointer event;
        CUDA_CHECK(::cudaEventCreate(&event, flags));
        LOG_TRACE("Created event %p", event);
        bool constexpr ownsEvent{true};
        mEvent = EventPtr{event, Deleter{ownsEvent}};
    }

    explicit CudaEvent(pointer event, bool ownsEvent = true)
    {
        CHECK_WITH_INFO(event != nullptr, "event is nullptr");
        mEvent = EventPtr{event, Deleter{ownsEvent}};
    }

    [[nodiscard]] pointer get() const
    {
        return mEvent.get();
    }

    void synchronize() const
    {
        CUDA_CHECK(::cudaEventSynchronize(get()));
    }

private:
    class Deleter
    {
    public:
        explicit Deleter(bool ownsEvent)
            : mOwnsEvent{ownsEvent}
        {
        }

        explicit Deleter()
            : Deleter{true}
        {
        }

        constexpr void operator()(pointer event) const
        {
            if (mOwnsEvent && event != nullptr)
            {
                CUDA_CHECK(::cudaEventDestroy(event));
                LOG_TRACE("Destroyed event %p", event);
            }
        }

    private:
        bool mOwnsEvent;
    };

    using element_type = std::remove_pointer_t<pointer>;
    using EventPtr = std::unique_ptr<element_type, Deleter>;

    EventPtr mEvent;
};

}
