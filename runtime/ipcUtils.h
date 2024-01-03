

#pragma once

#include "../kernels/customAllReduceKernels.h"
#include "bufferManager.h"
#include "iTensor.h"
#include "worldConfig.h"

namespace bitfusion::runtime
{

void setPeerAccess(WorldConfig worldConfig, bool enable = true);

class IpcMemory
{
public:
    using TensorPtr = ITensor::SharedPtr;

    size_t static constexpr FLAGS_SIZE = kernels::MAX_ALL_REDUCE_BLOCKS * sizeof(uint32_t);

    IpcMemory(WorldConfig worldConfig, std::size_t bufferSize);
    ~IpcMemory();

    [[nodiscard]] const std::vector<void*>& getCommPtrsTensor() const
    {
        return mCommPtrs;
    }

private:
    void allocateIpcMemory();
    void destroyIpcMemory();

    WorldConfig mWorldConfig;
    std::vector<void*> mCommPtrs;
    std::size_t mBufferSize;
    void* mBufferPtr;
};

}
