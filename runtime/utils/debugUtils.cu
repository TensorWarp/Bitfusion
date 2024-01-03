#include "debugUtils.h"

#include "../../common/cudaUtils.h"
#include "../../common/memoryUtils.h"

namespace
{

__global__ void checkTensorNanKernel(const float* data, std::size_t size, int* foundNan)
{
    auto tidx = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t found = 0;

    for (auto idx = tidx; idx < size; idx += blockDim.x * gridDim.x)
    {
        auto value = data[idx];
        if (isnan(value))
        {
            found = 1;
            break;
        }
    }
    atomicCAS(foundNan, 0, found);
}
}

using namespace bitfusion::runtime;
namespace tc = bitfusion::common;

namespace bitfusion::runtime::utils
{

void invokeCheckTensorNanKernel(const float* data, std::size_t size, int* foundNan, cudaStream_t stream)
{
    constexpr uint32_t kThreadsPerCta = 256;
    checkTensorNanKernel<<<tc::ceilDiv(size, kThreadsPerCta), kThreadsPerCta, 0, stream>>>(data, size, foundNan);
}

bool tensorHasNan(const IBuffer& tensor, BufferManager& manager)
{
    auto foundNan = manager.pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    auto foundNanPtr = bufferCast<int32_t>(*foundNan);
    foundNanPtr[0] = 0;
    const auto size = tensor.getSize();
    invokeCheckTensorNanKernel(bufferCast<float>(tensor), size, foundNanPtr, manager.getStream().get());
    manager.getStream().synchronize();
    return static_cast<bool>(foundNanPtr[0]);
}
}
