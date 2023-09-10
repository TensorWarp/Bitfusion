#pragma once

#include <stdexcept>

namespace DataSetEnums
{
    /// <summary>
    /// Enumeration of attributes that can be associated with a data set.
    /// </summary>
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

    /// <summary>
    /// Enumeration of data set kinds.
    /// </summary>
    enum Kind
    {
        Numeric = 0,
        Image = 1,
        Audio = 2
    };

    /// <summary>
    /// Enumeration of data sharding options.
    /// </summary>
    enum Sharding
    {
        None = 0,
        Model = 1,
        Data = 2,
    };

    /// <summary>
    /// Enumeration of data types.
    /// </summary>
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

    enum ErrorCode {
        Success,
        JSONParseError,
        ConfigurationError,
        ThreadPoolError,
        PluginError
    };

    enum LogLevel {
        INFO,
        ERROR,
        DEBUG
    };

    /// <summary>
    /// Template function to get the DataType associated with a specific data type.
    /// </summary>
    /// <typeparam name="T">The data type.</typeparam>
    /// <returns>The DataType enum value corresponding to the data type.</returns>
    template <typename T>
    inline DataType getDataType()
    {
        throw std::runtime_error("Default data type not defined");
    }

    template <>
    inline DataType getDataType<uint32_t>()
    {
        return DataType::UInt;
    }

    template <>
    inline DataType getDataType<int32_t>()
    {
        return DataType::Int;
    }

    template <>
    inline DataType getDataType<int64_t>()
    {
        return DataType::LLInt;
    }

    template <>
    inline DataType getDataType<uint64_t>()
    {
        return DataType::ULLInt;
    }

    template <>
    inline DataType getDataType<float>()
    {
        return DataType::Float;
    }

    template <>
    inline DataType getDataType<double>()
    {
        return DataType::Double;
    }

    template <>
    inline DataType getDataType<char>()
    {
        return DataType::Char;
    }

    template <>
    inline DataType getDataType<unsigned char>()
    {
        return DataType::UChar;
    }
}
