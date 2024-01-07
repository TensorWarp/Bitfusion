#include "tensor.h"
#include "cudaBf16Wrapper.h"
#include "cudaUtils.h"
#include "memoryUtils.h"
#include "stringUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <dirent.h>
#endif

namespace bitfusion::common {

    Tensor::Tensor()
        : where(MEMORY_CPU),
        type(TYPE_INVALID),
        shape({}),
        data(nullptr) {
    }

    Tensor::Tensor(MemoryType _where, DataType _type, const std::vector<size_t>& _shape, const void* _data)
        : where(_where),
        type(_type),
        shape(_shape),
        data(_data) {
    }

    size_t Tensor::size() const {
        if (data == nullptr || shape.empty()) {
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    }

    size_t Tensor::sizeBytes() const {
        return size() * getTypeSize(type);
    }

    std::string Tensor::whereToString() const {
        static const std::unordered_map<MemoryType, std::string> memToString{
            { MEMORY_CPU, "CPU" },{ MEMORY_CPU_PINNED, "CPU_PINNED" },{ MEMORY_GPU, "GPU" }
        };
        return memToString.at(where);
    }

    std::string Tensor::toString() const {
        std::string memTypeStr = whereToString();

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

    size_t Tensor::getTypeSize(DataType type) {
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

    std::string Tensor::getNumpyTypeDesc(DataType type) const {
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

        return typeMap.count(type) > 0 ? typeMap.at(type) : "x";
    }

    Tensor Tensor::slice(const std::vector<size_t>& shape, size_t offset) const {
        if (data != nullptr) {
            size_t nElts = size();
            size_t nSlicedElts = std::accumulate(shape.begin(), shape.end(), size_t{ 1 }, std::multiplies<size_t>());
            CHECK_WITH_INFO(nSlicedElts + offset <= nElts,
                fmtstr("The number (%ld) of elements of sliced tensor exceeds that (%ld) of the original tensor",
                    nSlicedElts + offset, nElts));
        }
        return Tensor(where, type, shape, getPtrWithOffset(offset));
    }

    TensorMap::TensorMap(const std::unordered_map<std::string, Tensor>& tensorMap) {
        for (auto& kv : tensorMap) {
            if (kv.second.isValid()) {
                insert(kv.first, kv.second);
            }
            else {
                LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", kv.first.c_str()));
            }
        }
    }

    TensorMap::TensorMap(const std::vector<Tensor>& tensorMap) {
        for (size_t i = 0; i < tensorMap.size(); i++) {
            insert(std::to_string(i), tensorMap[i]);
        }
    }

    TensorMap::TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensorMap) {
        for (auto& pair : tensorMap) {
            if (pair.second.isValid()) {
                insert(pair.first, pair.second);
            }
            else {
                LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str()));
            }
        }
    }

        TensorMap::~TensorMap()
        {
            tensor_map_.clear();
    }

        std::vector<std::string> TensorMap::keys() const
        {
            std::vector<std::string> key_names;
            for (auto& kv : tensor_map_)
            {
                key_names.push_back(kv.first);
        }
            return key_names;
    }

        std::string TensorMap::toString()
        {
        std::stringstream ss;
        ss << "{";
            std::vector<std::string> key_names = keys();
            for (size_t i = 0; i < tensor_map_.size(); ++i)
            {
                ss << key_names[i] << ": " << at(key_names[i]).toString();
                if (i < tensor_map_.size() - 1)
                {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }
}