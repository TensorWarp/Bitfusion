#include "iTensor.h"
#include "../common/memoryUtils.h"
#include "../common/stringUtils.h"
#include "bufferManager.h"
#include "tensorView.h"

#include <memory>
#include <iostream>
#include <vector>
#include <cstdint>
#include <ranges>
#include <format>
#include "Buffers.h"

namespace bitfusion::runtime {

    namespace tc = bitfusion::common;

    /// <summary>
    /// Create a new ITensor by slicing an existing tensor.
    /// </summary>
    /// <param name="tensor">The source tensor to be sliced.</param>
    /// <param name="offset">The starting offset of the slice.</param>
    /// <param name="size">The size of the slice.</param>
    /// <returns>A unique pointer to the sliced ITensor.</returns>
    ITensor::UniquePtr ITensor::slice(SharedPtr tensor, std::size_t offset, std::size_t size) {
        return std::make_unique<TensorView>(std::move(tensor), offset, size);
    }

    /// <summary>
    /// Create a new ITensor as a view of an existing buffer with specified dimensions.
    /// </summary>
    /// <param name="buffer">The source buffer to be viewed.</param>
    /// <param name="dims">The dimensions of the view.</param>
    /// <returns>A unique pointer to the viewed ITensor.</returns>
    ITensor::UniquePtr ITensor::view(IBuffer::SharedPtr buffer, nvinfer1::Dims const& dims) {
        auto const size = buffer->getSize();
        return std::make_unique<TensorView>(std::move(buffer), 0, size, dims);
    }

    /// <summary>
    /// Create an nvinfer1::Dims object from a list of dimensions.
    /// </summary>
    /// <param name="dims">The list of dimensions.</param>
    /// <returns>An nvinfer1::Dims object representing the dimensions.</returns>
    nvinfer1::Dims ITensor::makeShape(std::initializer_list<SizeType> const& dims) {
        constexpr int MAX_DIMS = nvinfer1::Dims::MAX_DIMS;
        CHECK_WITH_INFO(dims.size() <= MAX_DIMS, "Number of dimensions is too large");

        nvinfer1::Dims shape{};
        shape.nbDims = static_cast<decltype(shape.nbDims)>(dims.size());
        std::copy(dims.begin(), dims.end(), shape.d);
        return shape;
    }

    /// <summary>
    /// Convert an nvinfer1::Dims object to a string representation.
    /// </summary>
    /// <param name="dims">The nvinfer1::Dims object to be converted.</param>
    /// <returns>A string representation of the dimensions.</returns>
    std::string ITensor::toString(nvinfer1::Dims const& dims) {
        if (dims.nbDims < 0) {
            return "invalid";
        }
        else if (dims.nbDims == 0) {
            return "()";
        }
        else {
            return tc::arr2str(dims.d, dims.nbDims);
        }
    }

    /// <summary>
    /// Create a new ITensor wrapping an existing memory buffer with a specified shape and capacity.
    /// </summary>
    /// <param name="data">The pointer to the memory buffer.</param>
    /// <param name="type">The data type of the elements in the buffer.</param>
    /// <param name="shape">The shape of the ITensor.</param>
    /// <param name="capacity">The capacity of the buffer.</param>
    /// <returns>A unique pointer to the wrapped ITensor.</returns>
    ITensor::UniquePtr ITensor::wrap(void* data, nvinfer1::DataType type, nvinfer1::Dims const& shape, std::size_t capacity) {
        auto const size = volumeNonNegative(shape);
        CHECK_WITH_INFO(size <= capacity, "Requested size is larger than capacity");
        auto memoryType = IBuffer::memoryType(data);

        ITensor::UniquePtr result;
        auto const capacityInBytes = capacity * BufferDataType(type).getSize();

        switch (memoryType) {
        case MemoryType::kPINNED:
            result = std::make_unique<GenericTensor<PinnedBorrowingAllocator>>(
                shape, capacity, type, PinnedBorrowingAllocator(data, capacityInBytes));
            break;
        case MemoryType::kCPU:
            result = std::make_unique<GenericTensor<CpuBorrowingAllocator>>(
                shape, capacity, type, CpuBorrowingAllocator(data, capacityInBytes));
            break;
        case MemoryType::kGPU:
            result = std::make_unique<GenericTensor<GpuBorrowingAllocator>>(
                shape, capacity, type, GpuBorrowingAllocator(data, capacityInBytes));
            break;
        default:
            THROW("Unknown memory type");
        }
        return result;
    }

    /// <summary>
    /// Squeeze a dimension from the given shape.
    /// </summary>
    /// <param name="shape">The input shape to be squeezed.</param>
    /// <param name="dim">The dimension to be squeezed.</param>
    /// <returns>The squeezed shape.</returns>
    ITensor::Shape ITensor::squeeze(Shape const& shape, SizeType dim) {
        CHECK_WITH_INFO(0 < shape.nbDims, "Cannot squeeze 1-dimensional tensor");
        CHECK_WITH_INFO(dim < shape.nbDims, tc::fmtstr("Invalid index %d, tensor has %d dimensions", dim, shape.nbDims));
        CHECK_WITH_INFO(shape.d[dim] == 1, "Can only squeeze dimension of size 1");

        Shape newDims{ shape.nbDims - 1 };
        std::copy(shape.d, shape.d + dim, newDims.d);
        std::copy(shape.d + dim + 1, shape.d + shape.nbDims, newDims.d + dim);
        return newDims;
    }

    /// <summary>
    /// Unsqueeze a dimension in the given shape.
    /// </summary>
    /// <param name="shape">The input shape to be unsqueezed.</param>
    /// <param name="dim">The dimension to be unsqueezed.</param>
    /// <returns>The unsqueezed shape.</returns>
    ITensor::Shape ITensor::unsqueeze(Shape const& shape, SizeType dim) {
        constexpr int MAX_DIMS = Shape::MAX_DIMS;
        CHECK_WITH_INFO(shape.nbDims < MAX_DIMS, "Too many dimensions to unsqueeze");
        CHECK_WITH_INFO(0 <= dim && dim <= shape.nbDims, tc::fmtstr("Invalid dim %d, tensor has %d dimensions", dim, shape.nbDims));

        Shape newDims{ shape.nbDims + 1 };
        std::copy(shape.d, shape.d + dim, newDims.d);
        newDims.d[dim] = 1;
        std::copy(shape.d + dim, shape.d + shape.nbDims, newDims.d + dim + 1);
        return newDims;
    }

    /// <summary>
    /// Print the elements of the ITensor to the given output stream.
    /// </summary>
    /// <typeparam name="T">The data type of the elements in the ITensor.</typeparam>
    /// <param name="tensor">The ITensor to be printed.</param>
    /// <param name="out">The output stream to print to.</param>
    template <typename T>
    void printTensor(ITensor const& tensor, std::ostream& out) {
        CHECK_WITH_INFO(tensor.getDataType() == TRTDataType<std::remove_cvref_t<T>>::value,
            tc::fmtstr("Data type mismatch: %d vs %d", static_cast<std::int32_t>(tensor.getDataType()),
                static_cast<std::int32_t>(TRTDataType<std::remove_cvref_t<T>>::value)));
        auto const& shape = tensor.getShape();
        out << "shape: " << shape << std::endl;
        out << "vals: " << std::endl;

        BufferManager::ITensorPtr host{};
        T const* hostData;
        if (tensor.getMemoryType() == MemoryType::kGPU) {
            auto streamPtr = std::make_shared<CudaStream>();
            BufferManager manager{ streamPtr };
            host = manager.copyFrom(tensor, MemoryType::kCPU);
            streamPtr->synchronize();
            hostData = bufferCast<T>(*host);
        }
        else {
            hostData = bufferCast<T>(tensor);
        }

        using TOutput = std::conditional_t<std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>, std::int32_t, T>;

        if (shape.nbDims > 3) {
            out << "Not printing elements for more than 3 dims\n";
        }
        else if (shape.nbDims == 3 && shape.d[2] > 1) {
            for (int i = 0; i < shape.d[0]; ++i) {
                for (int j = 0; j < shape.d[1]; ++j) {
                    out << "i=" << i << " j=" << j << ": ";
                    tc::arr2outCasted<TOutput>(out, hostData + tc::flat_index(shape.d, i, j, 0), shape.d[2]) << "\n";
                }
            }
        }
        else if (shape.nbDims >= 2 && shape.d[1] > 1) {
            for (int i = 0; i < shape.d[0]; ++i) {
                out << "i=" << i << ": ";
                tc::arr2outCasted<TOutput>(out, hostData + tc::flat_index(shape.d, i, 0), shape.d[1]) << "\n";
            }
        }
        else {
            tc::arr2outCasted<TOutput>(out, hostData, shape.d[0]) << "\n";
        }
        out << std::flush;
    }

    /// <summary>
    /// Overload the << operator to print the ITensor to the given output stream.
    /// </summary>
    /// <param name="out">The output stream to print to.</param>
    /// <param name="tensor">The ITensor to be printed.</param>
    /// <returns>The output stream after printing the ITensor.</returns>
    std::ostream& operator<<(std::ostream& out, ITensor const& tensor) {
        switch (tensor.getDataType()) {
        case nvinfer1::DataType::kFLOAT: printTensor<float>(tensor, out); break;
        case nvinfer1::DataType::kHALF: printTensor<half>(tensor, out); break;
        case nvinfer1::DataType::kBOOL: printTensor<bool>(tensor, out); break;
        case nvinfer1::DataType::kINT8: printTensor<std::int8_t>(tensor, out); break;
        case nvinfer1::DataType::kINT32: printTensor<std::int32_t>(tensor, out); break;
        case nvinfer1::DataType::kINT64: printTensor<std::int64_t>(tensor, out); break;
        case nvinfer1::DataType::kUINT8: printTensor<std::uint8_t>(tensor, out); break;
#ifdef ENABLE_BF16
        case nvinfer1::DataType::kBF16: printTensor<__nv_bfloat16>(tensor, out); break;
#endif
        default: THROW("Unsupported data type");
        }

        return out;
    }
}