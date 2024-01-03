
#pragma once

#include "assert.h"
#include "../runtime/common.h"
#include "../runtime/iBuffer.h"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "../common/assert.h"

namespace nvinfer1
{
class IExecutionContext;
}

namespace bitfusion::runtime
{

class ITensor : virtual public IBuffer
{
public:
    using UniquePtr = std::unique_ptr<ITensor>;
    using SharedPtr = std::shared_ptr<ITensor>;
    using UniqueConstPtr = std::unique_ptr<ITensor const>;
    using SharedConstPtr = std::shared_ptr<ITensor const>;
    using Shape = nvinfer1::Dims;
    using DimType = std::remove_reference_t<decltype(Shape::d[0])>;

    ~ITensor() override = default;

    [[nodiscard]] virtual Shape const& getShape() const = 0;

    virtual void reshape(Shape const& dims) = 0;

    void resize(std::size_t newSize) override
    {
        if (newSize == getSize())
            return;

        reshape(makeShape({castSize(newSize)}));
    }

    ITensor(ITensor const&) = delete;

    ITensor& operator=(ITensor const&) = delete;

    static std::int64_t volume(Shape const& dims)
    {
        {
            return dims.nbDims < 0 ? -1
                : dims.nbDims == 0
                ? 0
                : std::accumulate(dims.d, dims.d + dims.nbDims, std::int64_t{1}, std::multiplies<>{});
        }
    }

    static std::size_t volumeNonNegative(Shape const& shape)
    {
        auto const vol = volume(shape);
        TLLM_CHECK_WITH_INFO(0 <= vol, "Invalid tensor shape");
        return static_cast<std::size_t>(vol);
    }

    static Shape squeeze(Shape const& shape, SizeType dim);

    static Shape unsqueeze(Shape const& shape, SizeType dim);

    void squeeze(SizeType dim)
    {
        reshape(squeeze(getShape(), dim));
    }

    void unsqueeze(SizeType dim)
    {
        reshape(unsqueeze(getShape(), dim));
    }

    static UniquePtr slice(SharedPtr tensor, std::size_t offset, std::size_t size);

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, std::size_t offset, std::size_t size)
    {
        return ITensor::slice(constPointerCast(std::forward<TConstPtr>(tensor)), offset, size);
    }

    static UniquePtr slice(SharedPtr tensor, std::size_t offset)
    {
        auto const dims = tensor->getShape();
        auto const size = (dims.nbDims > 0 ? dims.d[0] : 0) - offset;
        return ITensor::slice(std::move(tensor), offset, size);
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, std::size_t offset)
    {
        return ITensor::slice(constPointerCast(std::forward<TConstPtr>(tensor)), offset);
    }

    static UniquePtr view(IBuffer::SharedPtr buffer, Shape const& dims);

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr view(TConstPtr&& tensor, Shape const& dims)
    {
        return ITensor::view(constPointerCast(std::forward<TConstPtr>(tensor)), dims);
    }

    static UniquePtr view(SharedPtr tensor)
    {
        auto shapes = tensor->getShape();
        return ITensor::view(std::move(tensor), shapes);
    }

    static UniquePtr wrap(void* data, nvinfer1::DataType type, Shape const& shape, std::size_t capacity);

    static UniquePtr wrap(void* data, nvinfer1::DataType type, Shape const& shape)
    {
        return wrap(data, type, shape, volumeNonNegative(shape));
    }

    template <typename T>
    static UniquePtr wrap(T* data, Shape const& shape, std::size_t capacity)
    {
        return wrap(data, TRTDataType<T>::value, shape, capacity);
    }

    template <typename T>
    static UniquePtr wrap(T* data, Shape const& shape)
    {
        return wrap<T>(data, shape, volumeNonNegative(shape));
    }

    template <typename T>
    static UniquePtr wrap(std::vector<T>& v, Shape const& shape)
    {
        return wrap<T>(v.data(), shape, v.capacity());
    }

    static Shape makeShape(std::initializer_list<SizeType> const& dims);

    static std::string toString(Shape const& dims);

    static bool shapeEquals(Shape const& lhs, Shape const& rhs)
    {
        return shapeEquals(lhs, rhs.d, rhs.nbDims);
    }

    template <typename T>
    static bool shapeEquals(Shape const& lhs, T const* dims, SizeType count)
    {
        return lhs.nbDims == count && std::equal(lhs.d, lhs.d + lhs.nbDims, dims);
    }

    bool shapeEquals(Shape const& other) const
    {
        return shapeEquals(getShape(), other);
    }

    bool shapeEquals(std::initializer_list<SizeType> const& other) const
    {
        return shapeEquals(getShape(), other.begin(), other.size());
    }

    template <typename T>
    bool shapeEquals(T const* dims, SizeType count) const
    {
        return shapeEquals(getShape(), dims, count);
    }

protected:
    ITensor() = default;

    static DimType castSize(size_t newSize)
    {
        TLLM_CHECK_WITH_INFO(
            newSize <= std::numeric_limits<DimType>::max(), "New size is too large. Use reshape() instead.");
        return static_cast<DimType>(newSize);
    }
};

inline std::ostream& operator<<(std::ostream& output, ITensor::Shape const& dims)
{
    return output << ITensor::toString(dims);
}

std::ostream& operator<<(std::ostream& output, ITensor const& tensor);

}
