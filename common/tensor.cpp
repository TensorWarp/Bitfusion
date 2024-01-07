#include "tensor.h"
#include "cudaBf16Wrapper.h"
#include "cudaUtils.h"
#include "memoryUtils.h"
#include "stringUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <dirent.h>
#endif

namespace bitfusion::common {

    /// <summary>
    /// Default constructor for the Tensor class.
    /// </summary>
    Tensor::Tensor() = default;

    /// <summary>
    /// Constructor for the Tensor class.
    /// </summary>
    /// <param name="_where">Memory type where the tensor is located.</param>
    /// <param name="_type">Data type of the tensor.</param>
    /// <param name="_shape">Shape of the tensor.</param>
    /// <param name="_data">Pointer to the data of the tensor.</param>
    Tensor::Tensor(MemoryType _where, DataType _type, const std::vector<size_t>& _shape, const void* _data)
        : where(_where),
        type(_type),
        shape(_shape),
        data(_data) {
    }

    /// <summary>
    /// Get the size (number of elements) of the tensor.
    /// </summary>
    /// <returns>The size of the tensor.</returns>
    auto Tensor::size() const -> size_t {
        return data ? std::reduce(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>()) : 0;
    }

    /// <summary>
    /// Get the size of the tensor in bytes.
    /// </summary>
    /// <returns>The size of the tensor in bytes.</returns>
    auto Tensor::sizeBytes() const -> size_t {
        return size() * getTypeSize(type);
    }

    /// <summary>
    /// Convert the MemoryType enum to a string representation.
    /// </summary>
    /// <returns>The string representation of the MemoryType.</returns>
    auto Tensor::whereToString() const -> std::string {
        static const std::unordered_map<MemoryType, std::string> memToString{
            { MEMORY_CPU, "CPU" },{ MEMORY_CPU_PINNED, "CPU_PINNED" },{ MEMORY_GPU, "GPU" }
        };
        return memToString.at(where);
    }

    /// <summary>
    /// Convert the Tensor object to a string representation.
    /// </summary>
    /// <returns>The string representation of the Tensor.</returns>
    auto Tensor::toString() const -> std::string {
        const auto memTypeStr = whereToString();

        static const std::unordered_map<DataType, std::string> typeToString{
            { TYPE_BOOL, "BOOL" },{ TYPE_UINT8, "UINT8" },{ TYPE_UINT16, "UINT16" },{ TYPE_UINT32, "UINT32" },
            { TYPE_UINT64, "UINT64" },{ TYPE_INT8, "INT8" },{ TYPE_INT16, "INT16" },{ TYPE_INT32, "INT32" },
            { TYPE_INT64, "INT64" },{ TYPE_BF16, "BF16" },{ TYPE_FP16, "FP16" },{ TYPE_FP32, "FP32" },
            { TYPE_FP64, "FP64" },{ TYPE_BYTES, "BYTES" },{ TYPE_INVALID, "INVALID" },{ TYPE_FP8_E4M3, "E4M3" },
            { TYPE_VOID, "VOID" }
        };

        return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]", memTypeStr.c_str(), typeToString.at(type).c_str(),
            vec2str(shape).c_str(), data);
    }

    /// <summary>
    /// Get the size in bytes of a specific data type.
    /// </summary>
    /// <param name="type">The data type.</param>
    /// <returns>The size in bytes of the data type.</returns>
    auto Tensor::getTypeSize(DataType type) -> size_t {
        static const std::unordered_map<DataType, size_t> typeMap{
            { TYPE_BOOL, sizeof(bool) },{ TYPE_BYTES, sizeof(char) },
            { TYPE_UINT8, sizeof(uint8_t) },{ TYPE_UINT16, sizeof(uint16_t) },{ TYPE_UINT32, sizeof(uint32_t) },
            { TYPE_UINT64, sizeof(uint64_t) },{ TYPE_INT8, sizeof(int8_t) },{ TYPE_INT16, sizeof(int16_t) },
            { TYPE_INT32, sizeof(int32_t) },{ TYPE_INT64, sizeof(int64_t) },
    #ifdef ENABLE_BF16
            { TYPE_BF16, sizeof(__nv_bfloat16) },
    #endif
    #ifdef ENABLE_FP8
            { TYPE_FP8_E4M3, sizeof(__nv_fp8_e4m3) },
    #endif
            { TYPE_FP16, sizeof(half) },{ TYPE_FP32, sizeof(float) },{ TYPE_FP64, sizeof(double) }
        };

        return typeMap.at(type);
    }

    /// <summary>
    /// Get the Numpy type description for a specific data type.
    /// </summary>
    /// <param name="type">The data type.</param>
    /// <returns>The Numpy type description string.</returns>
    auto Tensor::getNumpyTypeDesc(DataType type) const -> std::string {
        static const std::unordered_map<DataType, std::string> typeMap{
            { TYPE_INVALID, "x" },{ TYPE_BOOL, "?" },{ TYPE_BYTES, "b" },{ TYPE_UINT8, "u1" },
            { TYPE_UINT16, "u2" },{ TYPE_UINT32, "u4" },{ TYPE_UINT64, "u8" },{ TYPE_INT8, "i1" },
            { TYPE_INT16, "i2" },{ TYPE_INT32, "i4" },{ TYPE_INT64, "i8" },{ TYPE_FP16, "f2" },
            { TYPE_FP32, "f4" },{ TYPE_FP64, "f8" }
        };

        if (type == TYPE_BF16) {
            LOG_WARNING(
                "getNumpyTypeDesc(TYPE_BF16) returns an invalid type 'x' since Numpy doesn't "
                "support bfloat16 as of now, it will be properly extended if numpy supports. "
                "Please refer for the discussions https://github.com/numpy/numpy/issues/19808.");
        }

        return typeMap.contains(type) ? typeMap.at(type) : "x";
    }

    /// <summary>
    /// Slice the tensor into a new tensor with a specified shape and offset.
    /// </summary>
    /// <param name="shape">The shape of the sliced tensor.</param>
    /// <param name="offset">The offset in the original tensor.</param>
    /// <returns>The sliced tensor.</returns>
    auto Tensor::slice(const std::vector<size_t>& shape, size_t offset) const -> Tensor {
        if (data != nullptr) {
            size_t nElts = size();
            size_t nSlicedElts = std::reduce(shape.begin(), shape.end(), size_t{ 1 }, std::multiplies<size_t>());
            CHECK_WITH_INFO(nSlicedElts + offset <= nElts,
                fmtstr("The number (%ld) of elements of sliced tensor exceeds that (%ld) of the original tensor",
                    nSlicedElts + offset, nElts));
        }
        return Tensor(where, type, shape, getPtrWithOffset(offset));
    }

    /// <summary>
    /// Constructor for the TensorMap class from an unordered map of string keys and Tensor values.
    /// </summary>
    /// <param name="tensorMap">The unordered map of string keys and Tensor values.</param>
    TensorMap::TensorMap(const std::unordered_map<std::string, Tensor>& tensorMap) {
        for (const auto& [key, value] : tensorMap) {
            if (value.isValid()) {
                insert(key, value);
            }
            else {
                LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", key.c_str()));
            }
        }
    }

    /// <summary>
    /// Constructor for the TensorMap class from a vector of Tensor values.
    /// </summary>
    /// <param name="tensorMap">The vector of Tensor values.</param>
    TensorMap::TensorMap(const std::vector<Tensor>& tensorMap) {
        for (size_t i = 0; i < tensorMap.size(); i++) {
            insert(std::to_string(i), tensorMap[i]);
        }
    }

    /// <summary>
    /// Constructor for the TensorMap class from an initializer list of pairs containing string keys and Tensor values.
    /// </summary>
    /// <param name="tensorMap">The initializer list of pairs containing string keys and Tensor values.</param>
    TensorMap::TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensorMap) {
        for (const auto& [key, value] : tensorMap) {
            if (value.isValid()) {
                insert(key, value);
            }
            else {
                LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", key.c_str()));
            }
        }
    }

    /// <summary>
    /// Destructor for the TensorMap class.
    /// </summary>
    TensorMap::~TensorMap() = default;

    /// <summary>
    /// Get a vector of string keys for the TensorMap.
    /// </summary>
    /// <returns>A vector of string keys.</returns>
    auto TensorMap::keys() const -> std::vector<std::string> {
        std::vector<std::string> key_names;
        for (const auto& [key, value] : tensor_map_) {
            key_names.push_back(key);
        }
        return key_names;
    }

    /// <summary>
    /// Convert the TensorMap to a string representation.
    /// </summary>
    /// <returns>The string representation of the TensorMap.</returns>
    auto TensorMap::toString() -> std::string {
        std::string result = "{";
        const auto& key_names = keys();
        for (size_t i = 0; i < key_names.size(); ++i) {
            result += key_names[i] + ": " + at(key_names[i]).toString();
            if (i < key_names.size() - 1) {
                result += ", ";
            }
        }
        return result + "}";
    }
}