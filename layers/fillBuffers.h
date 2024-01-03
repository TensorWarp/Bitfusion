
#pragma once

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/assert.h"
#include "../common/memoryUtils.h"

namespace bitfusion
{
namespace layers
{

struct FillBuffers
{

    template <typename T>
    void operator()(std::optional<std::vector<T>> const& optParam, T const defaultValue, std::vector<T>& hostBuffer,
        T*& deviceBuffer) const
    {
        using bitfusion::common::cudaAutoCpy;

        hostBuffer.resize(batch_size);
        if (!optParam)
        {
            std::fill(std::begin(hostBuffer), std::end(hostBuffer), defaultValue);
        }
        else if (optParam->size() == 1)
        {
            std::fill(std::begin(hostBuffer), std::end(hostBuffer), optParam->front());
        }
        else
        {
            TLLM_CHECK_WITH_INFO(optParam->size() == batch_size, "Argument vector size mismatch.");
            std::copy(optParam->begin(), optParam->end(), std::begin(hostBuffer));
        }
        cudaAutoCpy(deviceBuffer, hostBuffer.data(), batch_size, stream);
    }

    size_t batch_size;
    cudaStream_t stream;
};

}

}
