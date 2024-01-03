
#pragma once

#include "../common/assert.h"
#include "iBuffer.h"
#include "tllmBuffers.h"

#include <memory>
#include <string>

namespace bitfusion::runtime
{
class BufferView : virtual public IBuffer
{
public:
    explicit BufferView(IBuffer::SharedPtr buffer, std::size_t offset, std::size_t size)
        : mBuffer(std::move(buffer))
        , mOffset{offset}
        , mSize{size}
    {
        if (offset > mBuffer->getSize())
        {
            throw std::out_of_range(std::string("offset ") + std::to_string(offset) + " exceeds buffer size "
                + std::to_string(mBuffer->getSize()));
        }

        if (offset + size > mBuffer->getSize())
        {
            throw std::out_of_range(std::string("slice ") + std::to_string(offset + size) + " exceeds buffer size "
                + std::to_string(mBuffer->getSize()));
        }
    }

    void* data() override
    {
        if (getSize() == 0)
            return nullptr;
        return mBuffer->data(mOffset);
    }

    [[nodiscard]] void const* data() const override
    {
        if (getSize() == 0)
            return nullptr;
        return mBuffer->data(mOffset);
    }

    [[nodiscard]] size_t getSize() const override
    {
        return mSize;
    }

    [[nodiscard]] size_t getCapacity() const override
    {
        return mBuffer->getCapacity() - mOffset;
    }

    [[nodiscard]] nvinfer1::DataType getDataType() const override
    {
        return mBuffer->getDataType();
    }

    [[nodiscard]] MemoryType getMemoryType() const override
    {
        return mBuffer->getMemoryType();
    }

    void resize(std::size_t newSize) override
    {
        TLLM_CHECK(newSize <= getCapacity());
        mSize = newSize;
    }

    void release() override
    {
        mSize = 0;
    }

    ~BufferView() override = default;

private:
    IBuffer::SharedPtr mBuffer;
    std::size_t mOffset, mSize;
};

}
