
#pragma once

#include "cudaStream.h"
#include "iBuffer.h"
#include "worldConfig.h"

struct ncclComm;
typedef struct ncclComm* ncclComm_t;

namespace bitfusion::runtime
{

class NcclCommunicator
{
public:
    template <typename T>
    void send(T* sendbuff, size_t count, int peer, CudaStream const& stream) const;

    template <typename T>
    void send(IBuffer const& buf, int peer, CudaStream const& stream) const
    {
        send(bufferCast<T>(buf), buf.getSize(), peer, stream);
    }

    template <typename T>
    void receive(T* sendbuff, size_t count, int peer, CudaStream const& stream) const;

    template <typename T>
    void receive(IBuffer& buf, int peer, CudaStream const& stream) const
    {
        receive(bufferCast<T>(buf), buf.getSize(), peer, stream);
    }

    static std::shared_ptr<NcclCommunicator> createPipelineComm(WorldConfig const& worldConfig);

private:
    ncclComm_t mComm;
};

}
