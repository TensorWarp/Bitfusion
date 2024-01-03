
#pragma once

#include "bufferView.h"

#include <stdexcept>

namespace bitfusion::runtime
{
class TensorView : virtual public ITensor, public BufferView
{
public:
    using Base = BufferView;

    TensorView(ITensor::SharedPtr const& buffer, size_t offset, size_t size)
        : BufferView{buffer, offset * sizeDim0(*buffer), size * sizeDim0(*buffer)}
        , mDims{buffer->getShape()}
    {
        auto const dim0 = static_cast<size_t>((mDims.nbDims >= 0 && mDims.d[0] >= 0) ? mDims.d[0] : 0);
        if (offset > dim0)
        {
            throw std::out_of_range("offset exceeds dimension 0");
        }

        if (offset + size > dim0)
        {
            throw std::out_of_range("slice exceeds dimension 0");
        }
        mDims.d[0] = size;
    }

    TensorView(IBuffer::SharedPtr const& buffer, size_t offset, size_t size, nvinfer1::Dims const& dims)
        : BufferView{buffer, offset, size}
        , mDims{dims}
    {
        Base::resize(nonNegative(volume(dims)));
    }

    [[nodiscard]] nvinfer1::Dims const& getShape() const override
    {
        return mDims;
    }

    void reshape(nvinfer1::Dims const& dims) override
    {
        Base::resize(nonNegative(volume(dims)));
        mDims = dims;
    }

    void resize(std::size_t newSize) override
    {
        ITensor::resize(newSize);
    }

    void release() override
    {
        Base::release();
        mDims.nbDims = 0;
    }

private:
    static std::size_t sizeDim0(ITensor const& tensor)
    {
        auto& shape = tensor.getShape();
        return shape.nbDims >= 0 ? ITensor::volume(shape) / shape.d[0] : 0;
    }

    nvinfer1::Dims mDims{};
};
}
