#pragma once
#include <stdexcept>
#include <type_traits>

namespace DataSetEnums
{
    enum Attributes
    {
        Sparse = 1,
        Boolean = 2,
        Compressed = 4,
        Recurrent = 8,
        Mutable = 16,
        SparseIgnoreZero = 32,
        Indexed = 64,
        Weighted = 128
    };

    enum Kind
    {
        Numeric = 0,
        Image = 1,
        Audio = 2,
        Text = 3
    };

    enum Sharding
    {
        None = 0,
        Model = 1,
        Data = 2,
    };

    enum DataType
    {
        UInt = 0,
        Int = 1,
        LLInt = 2,
        ULLInt = 3,
        Float = 4,
        Double = 5,
        RGB8 = 6,
        RGB16 = 7,
        UChar = 8,
        Char = 9
    };

    enum class RegularizationType {
        L1,
        L2
    };

    enum class DatasetType {
        Indicator,
        Analog
    };

    enum class AttentionMaskType {
        PADDING = 0,
        CAUSAL = 1,
        BIDIRECTIONAL = 2,
        BIDIRECTIONALGLM = 3
    };

    enum class PositionEmbeddingType : int8_t {
        kLEARNED_ABSOLUTE = 0,
        kROPE_GPTJ = 1,
        kROPE_GPT_NEOX = 2,
        kALIBI = 3,
        kALIBI_WITH_SCALE = 4,
        kRELATIVE = 5
    };

    enum class RotaryScalingType : int8_t {
        kNONE = 0,
        kLINEAR = 1,
        kDYNAMIC = 2,
    };

    enum class KvCacheDataType {
        BASE = 0,
        INT8,
        FP8
    };

    template <typename T>
    concept ValidDataType = std::is_same_v<T, uint32_t> ||
        std::is_same_v<T, int32_t> ||
        std::is_same_v<T, int64_t> ||
        std::is_same_v<T, uint64_t> ||
        std::is_same_v<T, float> ||
        std::is_same_v<T, double> ||
        std::is_same_v<T, char> ||
        std::is_same_v<T, unsigned char>;

    template <ValidDataType T>
    inline DataType getDataType()
    {
        if constexpr (std::is_same_v<T, uint32_t>)
        {
            return DataType::UInt;
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            return DataType::Int;
        }
        else if constexpr (std::is_same_v<T, int64_t>)
        {
            return DataType::LLInt;
        }
        else if constexpr (std::is_same_v<T, uint64_t>)
        {
            return DataType::ULLInt;
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return DataType::Float;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return DataType::Double;
        }
        else if constexpr (std::is_same_v<T, char>)
        {
            return DataType::Char;
        }
        else if constexpr (std::is_same_v<T, unsigned char>)
        {
            return DataType::UChar;
        }
        else
        {
            throw std::runtime_error("Default data type not defined");
        }
    }

}
