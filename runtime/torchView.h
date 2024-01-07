
#pragma once

#include "../common/assert.h"
#include "iTensor.h"
#include "torchUtils.h"

#include <ATen/ATen.h>
#include <torch/types.h>

#include <memory>

namespace bitfusion::runtime
{
class TorchView : virtual public ITensor
{
public:
    static ITensor::UniquePtr of(at::Tensor&& tensor)
    {
        return ITensor::UniquePtr{new TorchView{std::move(tensor)}};
    }

    static ITensor::UniquePtr of(at::Tensor tensor)
    {
        return ITensor::UniquePtr{new TorchView{std::move(tensor)}};
    }

    void* data() override
    {
        return LIKELY(getSize() > 0) ? mTensor.data_ptr() : nullptr;
    }

    [[nodiscard]] void const* data() const override
    {
        return LIKELY(getSize() > 0) ? mTensor.data_ptr() : nullptr;
    }

    [[nodiscard]] size_t getSize() const override
    {
        return mTensor.numel();
    }

    [[nodiscard]] std::size_t getCapacity() const override
    {
        return mCapacity;
    }

    [[nodiscard]] DataType getDataType() const override
    {
        return TorchUtils::dataType(mTensor.scalar_type());
    }

    [[nodiscard]] MemoryType getMemoryType() const override
    {
        return mTensor.is_cuda() ? MemoryType::kGPU : mTensor.is_pinned() ? MemoryType::kPINNED : MemoryType::kCPU;
    }

    void release() override
    {
        resize(0);
    }

    [[nodiscard]] Shape const& getShape() const override
    {
        return mDims;
    }

    void reshape(Shape const& dims) override
    {
        try
        {
            mTensor.resize_(TorchUtils::shape(dims));
        }
        catch (c10::Error const& e)
        {
            THROW("%s", e.what_without_backtrace());
        }
        mDims = dims;
        if (auto const newSize = volumeNonNegative(dims); mCapacity < newSize)
        {
            mCapacity = newSize;
        }
    }

private:
    explicit TorchView(at::Tensor&& tensor)
        : mTensor(tensor)
        , mDims{TorchUtils::shape(mTensor.sizes())}
        , mCapacity{static_cast<std::size_t>(mTensor.numel())}
    {
        CHECK(mTensor.is_contiguous());
    };

    at::Tensor mTensor;
    Shape mDims;
    std::size_t mCapacity;
};
}
