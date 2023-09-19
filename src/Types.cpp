#include <stdexcept>
#include <sstream>

#include "GpuTypes.h"
#include "Types.h"
#include "Kernels.cuh"
#include <span>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>
#include <set>
#include <unordered_map>

/// Template class instantiations for various data types.
template class DataSet<float>;
template class DataSet<double>;
template class DataSet<unsigned char>;
template class DataSet<char>;
template class DataSet<uint32_t>;
template class DataSet<uint64_t>;
template class DataSet<int32_t>;
template class DataSet<int64_t>;

/// A map that associates training modes with their corresponding string representations.
static std::map<TrainingMode, std::string> sTrainingModeMap = {
    {TrainingMode::SGD,         "SGD"},
    {TrainingMode::Momentum,    "Momentum"},
    {TrainingMode::AdaGrad,     "AdaGrad"},
    {TrainingMode::Nesterov,    "Nesterov"},
    {TrainingMode::RMSProp,     "RMSProp"},
    {TrainingMode::AdaDelta,    "AdaDelta"},
    {TrainingMode::Adam,        "Adam"},
    {TrainingMode::LAMB,        "LAMB"},
    {TrainingMode::RAdam,       "RAdam"},
    {TrainingMode::Lookahead,   "Lookahead"},
    {TrainingMode::SWA,         "SWA"},
    {TrainingMode::Ranger,      "Ranger"},
    {TrainingMode::Nadam,       "Nadam"},
    {TrainingMode::Adabelief,   "Adabelief"},
    {TrainingMode::SAM,         "SAM"},
    {TrainingMode::NovoGrad,    "NovoGrad"}
};

/// Overload of the output stream operator to print a TrainingMode enum as a string.
std::ostream& operator<< (std::ostream& out, const TrainingMode& e)
{
    // Output the string representation of the TrainingMode enum using the sTrainingModeMap.
    out << sTrainingModeMap[e];
    return out;
}

/// A map that associates error functions with their corresponding string representations.
static std::map<ErrorFunction, std::string> sErrorFunctionMap = {
    {ErrorFunction::L1,                             "L1"},
    {ErrorFunction::L2,                             "L2"},
    {ErrorFunction::CrossEntropy,                   "CrossEntropy"},
    {ErrorFunction::ScaledMarginalCrossEntropy,     "ScaledMarginalCrossEntropy"},
    {ErrorFunction::Hinge,                          "Hinge"},
    {ErrorFunction::L2Hinge,                        "L2Hinge"},
    {ErrorFunction::MeanAbsoluteError,              "MeanAbsoluteError"},
    {ErrorFunction::MeanSquaredError,               "MeanSquaredError"},
    {ErrorFunction::RootMeanSquaredError,           "RootMeanSquaredError"},
    {ErrorFunction::KullbackLeiblerDivergence,      "KullbackLeiblerDivergence"},
    {ErrorFunction::JaccardIndex,                   "JaccardIndex"},
    {ErrorFunction::DiceCoefficient,                "DiceCoefficient"},
    {ErrorFunction::LogCosh,                        "LogCosh"},
    {ErrorFunction::CosineSimilarity,               "CosineSimilarity"},
    {ErrorFunction::CategoricalCrossEntropy,        "CategoricalCrossEntropy"},
    {ErrorFunction::WassersteinDistance,            "WassersteinDistance"},
    {ErrorFunction::TripletMarginLoss,              "TripletMarginLoss"},
    {ErrorFunction::EarthMoversDistance,            "EarthMoversDistance"},
    {ErrorFunction::FocalLoss,                      "FocalLoss"},
    {ErrorFunction::SparseCategoricalCrossEntropy,  "SparseCategoricalCrossEntropy"},
    {ErrorFunction::LogLoss,                        "LogLoss"},
    {ErrorFunction::HuberLoss,                      "HuberLoss"},
    {ErrorFunction::ExponentialLoss,                "ExponentialLoss"},
    {ErrorFunction::HuberizedHingeLoss,             "HuberizedHingeLoss"},
    {ErrorFunction::WeightedHuberLoss,              "WeightedHuberLoss"},
    {ErrorFunction::RankingLoss,                    "RankingLoss"},
    {ErrorFunction::ContrastiveLoss,                "ContrastiveLoss"},
    {ErrorFunction::TripletLoss,                    "TripletLoss"},
    {ErrorFunction::CenterLoss,                     "CenterLoss"},
    {ErrorFunction::GaussianKLDivergence,           "GaussianKLDivergence"},
    {ErrorFunction::LogitMarginLoss,                "LogitMarginLoss"},
};

/// Overload of the output stream operator to print an ErrorFunction enum as a string.
std::ostream& operator<< (std::ostream& out, const ErrorFunction& e)
{
    // Output the string representation of the ErrorFunction enum using the sErrorFunctionMap.
    out << sErrorFunctionMap[e];
    return out;
}

/// A map that associates activation functions with their corresponding string representations.
static std::map<Activation, std::string> sActivationMap = {
    {Activation::Sigmoid,                              "Sigmoid"},
    {Activation::Tanh,                                 "Tanh"},
    {Activation::Linear,                               "Linear"},
    {Activation::ParametricRectifiedLinear,            "ParametricRectifiedLinear"},
    {Activation::SoftSign,                             "SoftSign"},
    {Activation::SoftPlus,                             "SoftPlus"},
    {Activation::SoftMax,                              "SoftMax"},
    {Activation::RELUMax,                              "RELUMax"},
    {Activation::LinearMax,                            "LinearMax"},
    {Activation::RectifiedLinear,                      "RectifiedLinear"},
    {Activation::LeakyRectifiedLinear,                 "LeakyRectifiedLinear"},
    {Activation::ExponentialLinear,                    "ExponentialLinear"},
    {Activation::ScaledExponentialLinear,              "ScaledExponentialLinear"}
};

/// Overload of the output stream operator to print an Activation enum as a string.
std::ostream& operator<< (std::ostream& out, const Activation& a)
{
    // Output the string representation of the Activation enum using the sActivationMap.
    out << sActivationMap[a];
    return out;
}

/// A map that associates weight initialization methods with their corresponding string representations.
static std::map<WeightInitialization, std::string> sWeightInitializationMap = {
    {WeightInitialization::Xavier,           "Xavier"},
    {WeightInitialization::CaffeXavier,      "CaffeXavier"},
    {WeightInitialization::Gaussian,         "Gaussian"},
    {WeightInitialization::Uniform,          "Uniform"},
    {WeightInitialization::UnitBall,         "UnitBall"},
    {WeightInitialization::Constant,         "Constant"},
    {WeightInitialization::SELU,             "SELU"}
};

/// Overload of the output stream operator to print a WeightInitialization enum as a string.
std::ostream& operator<< (std::ostream& out, const WeightInitialization& w)
{
    // Output the string representation of the WeightInitialization enum using the sWeightInitializationMap.
    out << sWeightInitializationMap[w];
    return out;
}

/// A map that associates pooling functions with their corresponding string representations.
static std::map<PoolingFunction, std::string> sPoolingFunctionMap = {
    {PoolingFunction::None,                       "None"},
    {PoolingFunction::Max,                        "Max"},
    {PoolingFunction::Average,                    "Average"},
    {PoolingFunction::Maxout,                     "Maxout"},
    {PoolingFunction::DotProduct,                 "DotProduct"},
    {PoolingFunction::Cosine,                     "Cosine"},
    {PoolingFunction::Stochastic,                 "Stochastic"},
    {PoolingFunction::LCN,                        "LocalContrastNormalization"},
    {PoolingFunction::LRN,                        "LocalResponseNormalization"},
    {PoolingFunction::GlobalTemporal,             "GlobalTemporal"}
};

/// Overload of the output stream operator to print a PoolingFunction enum as a string.
std::ostream& operator<< (std::ostream& out, const PoolingFunction& a)
{
    // Output the string representation of the PoolingFunction enum using the sPoolingFunctionMap.
    out << sPoolingFunctionMap[a];
    return out;
}

/// A map that associates dataset kinds with their corresponding string representations.
static std::map<DataSetEnums::Kind, std::string> sKindMap = {
    {DataSetEnums::Numeric, "Numeric"},
    {DataSetEnums::Image,   "Image"},
    {DataSetEnums::Audio,   "Audio"},
    {DataSetEnums::Text,    "Text"}
};

/// Overload of the output stream operator to print a DataSetEnums::Kind enum as a string.
std::ostream& operator<< (std::ostream& out, DataSetEnums::Kind& k)
{
    // Output the string representation of the DataSetEnums::Kind enum using the sKindMap.
    out << sKindMap[k];
    return out;
}


/// A map that associates custom attributes with their corresponding string representations.
static std::map<DataSetEnums::Attributes, std::string> sAttributesMap = {
    {DataSetEnums::Sparse,                       "Sparse"},
    {DataSetEnums::Boolean,                      "Boolean"},
    {DataSetEnums::Compressed,                   "Compressed"},
    {DataSetEnums::Recurrent,                    "Recurrent"},
    {DataSetEnums::Mutable,                      "Mutable"},
    {DataSetEnums::Attributes::SparseIgnoreZero, "SparseIgnoreZero"},
    {DataSetEnums::Attributes::Indexed,          "Indexed"},
    {DataSetEnums::Attributes::Weighted,         "Weighted"},
};

/// Overload of the output stream operator to print a custom attributes type as a string.
/// <param name="out">The output stream.</param>
/// <param name="a">The custom attributes type to print.</param>
/// <returns>The output stream with the custom attributes type as a string representation.</returns>
std::ostream& operator<< (std::ostream& out, DataSetEnums::Attributes& a)
{
    // Output the string representation of the custom attributes type using the sAttributesMap.
    out << sAttributesMap[a];
    return out;
}

/// A map that associates custom sharding types with their corresponding string representations.
static std::map<DataSetEnums::Sharding, std::string> sShardingMap = {
    {DataSetEnums::None,  "None"},
    {DataSetEnums::Model, "Model"},
    {DataSetEnums::Data,  "Data"}
};

/// Overload of the output stream operator to print a custom sharding type as a string.
/// <param name="out">The output stream.</param>
/// <param name="s">The custom sharding type to print.</param>
/// <returns>The output stream with the custom sharding type as a string representation.</returns>
std::ostream& operator<< (std::ostream& out, DataSetEnums::Sharding& s)
{
    // Output the string representation of the custom sharding type using the sShardingMap.
    out << sShardingMap[s];
    return out;
}


/// A map that associates custom data types with their corresponding string representations.
static std::map<DataSetEnums::DataType, std::string> sDataTypeMap = {
    {DataSetEnums::UInt,   "UInt"},
    {DataSetEnums::Int,    "Int"},
    {DataSetEnums::LLInt,  "LLInt"},
    {DataSetEnums::ULLInt, "ULLInt"},
    {DataSetEnums::Float,  "Float"},
    {DataSetEnums::Double, "Double"},
    {DataSetEnums::RGB8,   "RGB8"},
    {DataSetEnums::RGB16,  "RGB16"},
    {DataSetEnums::UChar,  "UChar"},
    {DataSetEnums::Char,   "Char"}
};

/// Overload of the output stream operator to print a custom data type as a string.
/// <param name="out">The output stream.</param>
/// <param name="t">The custom data type to print.</param>
/// <returns>The output stream with the custom data type as a string representation.</returns>
std::ostream& operator<< (std::ostream& out, DataSetEnums::DataType& t)
{
    // Output the string representation of the custom data type using the sDataTypeMap.
    out << sDataTypeMap[t];
    return out;
}

/// Maps a custom data type to its corresponding MPI data type.
/// <param name="datatype">The custom data type to map.</param>
/// <returns>The MPI data type corresponding to the input custom data type.</returns>
static MPI_Datatype getMPIDataType(DataSetEnums::DataType datatype)
{
    MPI_Datatype mpiType;

    // Switch statement to map the custom data type to its corresponding MPI data type.
    switch (datatype)
    {
        // If the custom data type is UInt, set mpiType to MPI_UINT32_T.
    case DataSetEnums::UInt:
        mpiType = MPI_UINT32_T;
        break;

        // If the custom data type is Int, set mpiType to MPI_INT32_T.
    case DataSetEnums::Int:
        mpiType = MPI_INT32_T;
        break;

        // If the custom data type is ULLInt, set mpiType to MPI_UINT64_T.
    case DataSetEnums::ULLInt:
        mpiType = MPI_UINT64_T;
        break;

        // If the custom data type is LLInt, set mpiType to MPI_INT64_T.
    case DataSetEnums::LLInt:
        mpiType = MPI_INT64_T;
        break;

        // If the custom data type is Float, set mpiType to MPI_FLOAT.
    case DataSetEnums::Float:
        mpiType = MPI_FLOAT;
        break;

        // If the custom data type is Double, set mpiType to MPI_DOUBLE.
    case DataSetEnums::Double:
        mpiType = MPI_DOUBLE;
        break;
    }

    // Return the MPI data type corresponding to the input custom data type.
    return mpiType;
}

/// Maps a custom data type to its corresponding netCDF data type.
/// <param name="datatype">The custom data type to map.</param>
/// <returns>The netCDF data type corresponding to the input custom data type.</returns>
static netCDF::NcType getNetCDFDataType(DataSetEnums::DataType datatype)
{
    switch (datatype)
    {
        // Map the custom data type UInt to the netCDF data type ncUint.
    case DataSetEnums::UInt:
        return netCDF::ncUint;

        // Map the custom data type Int to the netCDF data type ncInt.
    case DataSetEnums::Int:
        return netCDF::ncInt;

        // Map the custom data type ULLInt to the netCDF data type ncUint64.
    case DataSetEnums::ULLInt:
        return netCDF::ncUint64;

        // Map the custom data type LLInt to the netCDF data type ncInt64.
    case DataSetEnums::LLInt:
        return netCDF::ncInt64;

        // Map the custom data type Float to the netCDF data type ncFloat.
    case DataSetEnums::Float:
        return netCDF::ncFloat;

        // Map the custom data type Double to the netCDF data type ncDouble.
    case DataSetEnums::Double:
        return netCDF::ncDouble;
    }
}

/// Checks if a string has a specified suffix.
/// <param name="str">The string to check.</param>
/// <param name="suffix">The suffix to check for.</param>
/// <returns>True if the string has the specified suffix, otherwise false.</returns>
static inline bool has_suffix(const std::string& str, const std::string& suffix)
{
    // Check if the length of the string is greater than or equal to the length of the suffix.
    // Also, compare the characters starting from the end of the string with the suffix.
    return str.size() >= suffix.size() &&
        str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Broadcasts a string using MPI (Message Passing Interface).
/// <param name="s">The string to broadcast.</param>
/// <returns>MPI_SUCCESS if the broadcast operation is successful.</returns>
int MPI_Bcast_string(std::string& s)
{
    // Get the length of the string.
    int length = s.size();

    // Broadcast the length of the string to all processes.
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Create a buffer to hold the string data.
    std::vector<char> buff(length + 1);

    // Broadcast the string data to all processes.
    if (MPI_Bcast(buff.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        // Handle the case where the broadcast operation fails.
        // (You may want to add error handling code here.)
    }

    // Null-terminate the buffer to make it a valid C-string.
    buff[length] = '\0';

    // Assign the received C-string to the input string 's'.
    s = buff.data();

    return MPI_SUCCESS;
}

/// <summary>
/// Default constructor for DataSetDimensions.
/// Initializes dimensions to 1x1x1.
/// </summary>
DataSetDimensions::DataSetDimensions() :
    
    // Initializes dimensions to 1x1x1.
    DataSetDimensions(1, 1, 1)
{}

/// Overloaded constructor for DataSetDimensions.
/// Initializes dimensions with the provided width, height, and length values.
/// It also calculates the number of dimensions based on non-zero values.
/// <param name="width">The width dimension of the data set.</param>
/// <param name="height">The height dimension of the data set.</param>
/// <param name="length">The length dimension of the data set.</param>
DataSetDimensions::DataSetDimensions(uint32_t width, uint32_t height, uint32_t length) :
    // Initialize the width dimension.
    _width(width),
    // Initialize the height dimension.
    _height(height),
    // Initialize the length dimension.
    _length(length),
    // Initialize the number of dimensions to 0.
    _dimensions(0)
{
    // Check if width is greater than 1 and increment dimensions accordingly.
    if (width > 1)
    {
        ++_dimensions;
    }

    // Check if height is greater than 1 and increment dimensions accordingly.
    if (height > 1)
    {
        ++_dimensions;
    }

    // Check if length is greater than 1 and increment dimensions accordingly.
    if (length > 1)
    {
        ++_dimensions;
    }
}

template<typename T> DataSetBase* createDataSet(const DataSetDescriptor &descriptor)
{
    using DataSetEnums::Attributes;

    uint32_t attributes = descriptor._attributes;
    if (!DataSetDescriptor::isSupported(attributes))
    {
        std::stringstream msg;
        msg << "Unsupported attributes " << attributes << " for dataset " << descriptor._name;
       std::runtime_error(msg.str());
    }

    DataSetBase *dataset;
    if (attributes & Attributes::Sparse)
    {
        dataset = new DataSet<T>(descriptor._examples, descriptor._sparseDensity, descriptor._dim, false,
                                   descriptor._name);
    } else
    {
        dataset = new DataSet<T>(descriptor._examples, descriptor._dim, descriptor._name);
    }

    // Return a pointer to the created DataSetBase instance.
    return dataset;
}

// Factory function to create a DataSetBase instance based on the provided DataSetDescriptor.
// This function determines the data type and other attributes of the dataset and creates it accordingly.
// If the provided data type is unsupported, it throws a runtime error.
DataSetBase* createDataSet(const DataSetDescriptor& descriptor)
{
    // Declare a pointer to a DataSetBase instance to be created.
    DataSetBase* dataset;

    // Import the DataType enum from DataSetEnums for easier use.
    using DataSetEnums::DataType;

    // Use a switch statement to handle different data types.
    switch (descriptor._dataType) {
    case DataType::UInt:
        // Create a dataset of type uint32_t.
        dataset = createDataSet<uint32_t>(descriptor);
        break;
    case DataType::Int:
        // Create a dataset of type int.
        dataset = createDataSet<int>(descriptor);
        break;
    case DataType::Float:
        // Create a dataset of type float.
        dataset = createDataSet<float>(descriptor);
        break;
    case DataType::Double:
        // Create a dataset of type double.
        dataset = createDataSet<double>(descriptor);
        break;
    case DataType::Char:
        // Create a dataset of type char.
        dataset = createDataSet<char>(descriptor);
        break;
    case DataType::UChar:
    case DataType::RGB8:
        // Create a dataset of type uint8_t (for DataType::UChar and DataType::RGB8).
        dataset = createDataSet<uint8_t>(descriptor);
        break;
    default:
        // If the provided data type is unsupported, throw a runtime error with a message.
        std::stringstream msg;
        msg << "Unsupported data type: " << descriptor._dataType
            << ". DataType must be one of: UInt, Int, Float, Double, Char, UChar, RGB8";
        throw std::runtime_error(msg.str());
    }

    // Return a pointer to the created DataSetBase instance.
    return dataset;
}


/// <summary>
/// Default constructor for the DataSetBase class.
/// </summary>
/// <remarks>
/// This constructor initializes a DataSetBase object with default values for its member variables. It is typically used when
/// creating an uninitialized instance of DataSetBase and is followed by setting specific values for its attributes.
/// </remarks>
DataSetBase::DataSetBase() :
    
    /// <summary>
    /// The name of the dataset.
    /// </summary>
    _name(""),
    
    /// <summary>
    /// The attributes of the dataset, such as Sparse, Indexed, Weighted, or None.
    /// </summary>
    _attributes(DataSetEnums::None),
    
    /// <summary>
    /// The total number of examples in the dataset.
    /// </summary>
    _examples(0),
    
    /// <summary>
    /// The number of unique examples in the dataset.
    /// </summary>
    _uniqueExamples(0),
    
    /// <summary>
    /// The number of dimensions in the dataset.
    /// </summary>
    _dimensions(0),
    
    /// <summary>
    /// The width of the dataset.
    /// </summary>
    _width(0),
    
    /// <summary>
    /// The height of the dataset.
    /// </summary>
    _height(0),
    
    /// <summary>
    /// The length of the dataset.
    /// </summary>
    _length(0),
    
    /// <summary>
    /// The stride of the dataset.
    /// </summary>
    _stride(0),
    
    /// <summary>
    /// The sharding type of the dataset, if any.
    /// </summary>
    _sharding(DataSetEnums::Sharding::None),
    
    /// <summary>
    /// The minimum value of X in the dataset.
    /// </summary>
    _minX(0),
    
    /// <summary>
    /// The maximum value of X in the dataset.
    /// </summary>
    _maxX(0),
    
    /// <summary>
    /// The total size of sparse data in the dataset.
    /// </summary>
    _sparseDataSize(0),
    
    /// <summary>
    /// The number of transposed sparse indices in the dataset.
    /// </summary>
    _sparseTransposedIndices(0),
    
    /// <summary>
    /// The density of the sparse dataset.
    /// </summary>
    _sparseDensity(0),
    
    /// <summary>
    /// Indicates whether denoising is enabled for the dataset.
    /// </summary>
    _bDenoising(false),
    
    /// <summary>
    /// A vector for sparse start data.
    /// </summary>
    _pbSparseStart(),
    
    /// <summary>
    /// A vector for sparse end data.
    /// </summary>
    _pbSparseEnd(),
    
    /// <summary>
    /// A vector for sparse index data.
    /// </summary>
    _pbSparseIndex(),
    
    /// <summary>
    /// A vector for indexed data.
    /// </summary>
    _pbIndex(),
    
    /// <summary>
    /// A vector for transposed sparse start data.
    /// </summary>
    _pbSparseTransposedStart(),
    
    /// <summary>
    /// A vector for transposed sparse end data.
    /// </summary>
    _pbSparseTransposedEnd(),
    
    /// <summary>
    /// A vector for transposed sparse index data.
    /// </summary>
    _pbSparseTransposedIndex(),
    
    /// <summary>
    /// A vector for transposed sparse data.
    /// </summary>
    _pbSparseTransposedData(),
    
    /// <summary>
    /// The batch size for the dataset.
    /// </summary>
    _batch(0),
    
    /// <summary>
    /// A vector for denoising random data.
    /// </summary>
    _pbDenoisingRandom(),
    
    /// <summary>
    /// Indicates whether the dataset is used for streaming.
    /// </summary>
    _bStreaming(false),
    
    /// <summary>
    /// Indicates whether the dataset is indexed.
    /// </summary>
    _bIndexed(false),
    
    /// <summary>
    /// Indicates whether the dataset has been modified.
    /// </summary>
    _bDirty(true)
{
}

/// <summary>
/// Constructor for the DataSetBase class.
/// </summary>
/// <param name="name">The name of the dataset.</param>
/// <param name="dataType">The data type of the dataset.</param>
/// <param name="examples">The total number of examples in the dataset.</param>
/// <param name="uniqueExamples">The number of unique examples in the dataset.</param>
/// <param name="datasetDim">The dimensions (width, height, length) of the dataset.</param>
/// <remarks>
/// This constructor initializes a DataSetBase object with the provided name, data type, number of examples, number of unique
/// examples, and dataset dimensions. It sets various internal attributes and data members to their initial values.
/// </remarks>
DataSetBase::DataSetBase(const std::string& name, DataSetEnums::DataType dataType, uint32_t examples,
    uint32_t uniqueExamples, const DataSetDimensions& datasetDim) :

    /// <summary>
    /// Initializes the _name member variable.
    /// </summary>
    _name(name),

    /// <summary>
    /// Initializes the _dataType member variable.
    /// </summary>
    _dataType(dataType),

    /// <summary>
    /// Initializes the _attributes member variable with DataSetEnums::None.
    /// </summary>
    _attributes(DataSetEnums::None),

    /// <summary>
    /// Initializes the _examples member variable.
    /// </summary>
    _examples(examples),

    /// <summary>
    /// Initializes the _uniqueExamples member variable.
    /// </summary>
    _uniqueExamples(uniqueExamples),

    /// <summary>
    /// Initializes the _localExamples member variable with the value of _examples.
    /// </summary>
    _localExamples(examples),

    /// <summary>
    /// Initializes the _dimensions member variable with datasetDim._dimensions.
    /// </summary>
    _dimensions(datasetDim._dimensions),

    /// <summary>
    /// Initializes the _width member variable with datasetDim._width.
    /// </summary>
    _width(datasetDim._width),

    /// <summary>
    /// Initializes the _height member variable with datasetDim._height.
    /// </summary>
    _height(datasetDim._height),

    /// <summary>
    /// Initializes the _length member variable with datasetDim._length.
    /// </summary>
    _length(datasetDim._length),

    /// <summary>
    /// Initializes the _stride member variable with 0.
    /// </summary>
    _stride(0),

    /// <summary>
    /// Initializes the _sharding member variable with DataSetEnums::Sharding::None.
    /// </summary>
    _sharding(DataSetEnums::Sharding::None),

    /// <summary>
    /// Initializes the _minX member variable with 0.
    /// </summary>
    _minX(0),

    /// <summary>
    /// Initializes the _maxX member variable with 0.
    /// </summary>
    _maxX(0),

    /// <summary>
    /// Initializes the _sparseDataSize member variable with 0.
    /// </summary>
    _sparseDataSize(0),

    /// <summary>
    /// Initializes the _sparseTransposedIndices member variable with 0.
    /// </summary>
    _sparseTransposedIndices(0),

    /// <summary>
    /// Initializes the _sparseDensity member variable with 0.
    /// </summary>
    _sparseDensity(0),

    /// <summary>
    /// Initializes the _bDenoising member variable with false.
    /// </summary>
    _bDenoising(false),

    /// <summary>
    /// Initializes the _pbSparseStart member variable.
    /// </summary>
    _pbSparseStart(),

    /// <summary>
    /// Initializes the _pbSparseEnd member variable.
    /// </summary>
    _pbSparseEnd(),

    /// <summary>
    /// Initializes the _pbSparseIndex member variable.
    /// </summary>
    _pbSparseIndex(),

    /// <summary>
    /// Initializes the _pbIndex member variable.
    /// </summary>
    _pbIndex(),

    /// <summary>
    /// Initializes the _pbSparseTransposedStart member variable.
    /// </summary>
    _pbSparseTransposedStart(),

    /// <summary>
    /// Initializes the _pbSparseTransposedEnd member variable.
    /// </summary>
    _pbSparseTransposedEnd(),

    /// <summary>
    /// Initializes the _pbSparseTransposedIndex member variable.
    /// </summary>
    _pbSparseTransposedIndex(),

    /// <summary>
    /// Initializes the _pbSparseTransposedData member variable.
    /// </summary>
    _pbSparseTransposedData(),

    /// <summary>
    /// Initializes the _batch member variable with 0.
    /// </summary>
    _batch(0),

    /// <summary>
    /// Initializes the _pbDenoisingRandom member variable.
    /// </summary>
    _pbDenoisingRandom(),

    /// <summary>
    /// Initializes the _bStreaming member variable with false.
    /// </summary>
    _bStreaming(false),

    /// <summary>
    /// Initializes the _bIndexed member variable with false.
    /// </summary>
    _bIndexed(false),

    /// <summary>
    /// Initializes the _bDirty member variable with true.
    /// </summary>
    _bDirty(true)
{
}

/// <summary>
/// Destructor for the DataSetBase class.
/// </summary>
/// <remarks>
/// This destructor is empty as there is no specific cleanup required for the DataSetBase class. It serves as a placeholder
/// for potential future modifications.
/// </remarks>
DataSetBase::~DataSetBase() {}

/// <summary>
/// Get the dimensions of the dataset.
/// </summary>
/// <returns>The dimensions (width, height, length) of the dataset.</returns>
/// <remarks>
/// This function returns a DataSetDimensions object containing the width, height, and length of the dataset.
/// </remarks>
DataSetDimensions DataSetBase::GetDimensions()
{
    return DataSetDimensions(_width, _height, _length);
}

/// <summary>
/// Gets the memory usage of the DataSet.
/// </summary>
/// <returns>A vector of tuples representing memory usage for each process.</returns>
template<typename T> auto DataSet<T>::getMemoryUsage() -> std::vector<std::tuple<uint64_t, uint64_t>>
{
    // Initialize CPU and GPU memory counters
    uint64_t cpuMemory = 0;
    uint64_t gpuMemory = 0;

    // Check if the dataset is sparse
    if (_attributes & DataSetEnums::Sparse)
    {
        // Calculate memory usage for sparse data
        cpuMemory += _uniqueExamples * 2 * sizeof(uint64_t);
        gpuMemory += _uniqueExamples * 2 * sizeof(uint64_t);
        cpuMemory += _vSparseIndex.size() * sizeof(uint32_t);
        gpuMemory += _vSparseIndex.size() * sizeof(uint32_t);

        // Check if the dataset is not boolean
        if (!(_attributes & DataSetEnums::Boolean))
        {
            // Include memory usage for sparse data values
            cpuMemory += _vSparseData.size() * sizeof(T);
            gpuMemory += _vSparseData.size() * sizeof(T);
        }
    }
    else
    {
        // Calculate memory usage for dense data
        cpuMemory += _vData.size() * sizeof(T);
        gpuMemory += _vData.size() * sizeof(T);
    }

    // Check if the dataset is indexed
    if (_bIndexed)
    {
        // Include memory usage for indexing data
        cpuMemory += _examples * sizeof(uint32_t);
        gpuMemory += _examples * sizeof(uint32_t);
    }

    // Create a vector to store memory usage information for each process
    std::vector<std::tuple<uint64_t, uint64_t>> vResult(getGpu()._numprocs);

    // Store memory usage information for the current process
    vResult[getGpu()._id] = std::make_tuple(cpuMemory, gpuMemory);

    // Use std::span for safer array access and gather memory usage information from all processes
    auto resultSpan = std::span(vResult.data(), vResult.size());
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, resultSpan.data(), sizeof(std::tuple<uint64_t, uint64_t>), MPI_BYTE, MPI_COMM_WORLD);

    // Return the memory usage information for all processes
    return vResult;
}

/// <summary>
/// Constructor for the DataSet class.
/// </summary>
/// <param name="examples">The total number of examples in the dataset.</param>
/// <param name="dim">The dimensions (width, height, length, stride) of the dataset.</param>
/// <param name="name">The name of the dataset.</param>
/// <remarks>
/// This constructor initializes a DataSet object with the given parameters. It sets the sparse density to 1.0, indicating
/// that the dataset is not sparse. It calculates the stride of the dataset based on its dimensions and initializes the
/// dataset's data vector and a GPU buffer for data.
/// </remarks>
/// <param name="T">The data type of the dataset and its components.</typeparam>
template<typename T>
DataSet<T>::DataSet(uint32_t examples, const DataSetDimensions& dim, const std::string& name) :
    DataSetBase(name, DataSetEnums::getDataType<T>(), examples, examples, dim)
{
    // Set the sparse density to 1.0, indicating that the dataset is not sparse
    _sparseDensity = 1.0f;

    // Calculate the stride of the dataset
    _stride = _width * _height * _length;

    // Initialize the data vector with appropriate size
    _vData.resize(_stride * _examples);

    // Create a GPU buffer for data
    _pbData.reset(new GpuBuffer<T>(_vData.size(), false, _bStreaming));
}

/// <summary>
/// Constructor for the DataSet class.
/// </summary>
/// <param name="examples">The total number of examples in the dataset.</param>
/// <param name="uniqueExamples">The number of unique examples in the dataset.</param>
/// <param name="dim">The dimensions (width, height, length, stride) of the dataset.</param>
/// <param name="name">The name of the dataset.</param>
/// <remarks>
/// This constructor initializes a DataSet object with the given parameters. It sets the sparse density to 1.0, indicating
/// that the dataset is not sparse. It sets the dataset's attributes to indicate that it is indexed and initializes the
/// dataset's dimensions and related data structures.
/// </remarks>
/// <param name="T">The data type of the dataset and its components.</typeparam>
template<typename T>
DataSet<T>::DataSet(uint32_t examples, uint32_t uniqueExamples, const DataSetDimensions& dim,
    const std::string& name) :
    DataSetBase(name, DataSetEnums::getDataType<T>(), examples, uniqueExamples, dim)
{
    // Set the sparse density to 1.0, indicating that the dataset is not sparse
    _sparseDensity = 1.0f;

    // Calculate the stride of the dataset
    _stride = _width * _height * _length;

    // Set the attributes to indicate that the dataset is indexed
    _attributes = DataSetEnums::Attributes::Indexed;

    // Set the indexed flag to true
    _bIndexed = true;

    // Initialize the data and index vectors with appropriate sizes
    _vData.resize(_stride * _uniqueExamples);
    _vIndex.resize(_examples, 0);

    // Create GPU buffers for data and index
    _pbData.reset(new GpuBuffer<T>(_vData.size(), false, _bStreaming));
    _pbIndex.reset(new GpuBuffer<uint32_t>(_vIndex.size(), false, _bStreaming));
}

/// <summary>
/// Constructor for the DataSet class.
/// </summary>
/// <param name="examples">The total number of examples in the dataset.</param>
/// <param name="sparseDensity">The density of sparse data in the dataset.</param>
/// <param name="dim">The dimensions (width, height, length, stride) of the dataset.</param>
/// <param name="isWeighted">A flag indicating whether the dataset is weighted.</param>
/// <param name="name">The name of the dataset.</param>
/// <remarks>
/// This constructor initializes a DataSet object with the given parameters. It calculates the size of the sparse data storage
/// based on the dataset's dimensions, the number of examples, and the specified sparse density. It sets the dataset's attributes
/// to indicate that it supports sparse data. If the dataset is weighted, it sets the corresponding attribute.
/// </remarks>
/// <param name="T">The data type of the dataset and its components.</typeparam>
template<typename T>
DataSet<T>::DataSet(uint32_t examples, float sparseDensity, const DataSetDimensions& dim,
    bool isWeighted, const std::string& name) :
    DataSet(examples, examples,
        (size_t)(((double)(dim._width* dim._height* dim._length* examples))* sparseDensity), dim, false,
        isWeighted, name)
{
    // Set the attributes to indicate that the dataset supports sparse data
    _attributes = DataSetEnums::Attributes::Sparse;

    // If the dataset is weighted, set the weighted attribute
    if (isWeighted) {
        _attributes |= DataSetEnums::Attributes::Weighted;
    }
}

/// <summary>
/// Constructor for the DataSet class.
/// </summary>
/// <param name="examples">The total number of examples in the dataset.</param>
/// <param name="uniqueExamples">The number of unique examples in the dataset.</param>
/// <param name="sparseDataSize">The size of the sparse data storage.</param>
/// <param name="dim">The dimensions (width, height, length, stride) of the dataset.</param>
/// <param name="isIndexed">A flag indicating whether the dataset is indexed.</param>
/// <param name="isWeighted">A flag indicating whether the dataset is weighted.</param>
/// <param name="name">The name of the dataset.</param>
/// <remarks>
/// This constructor initializes a DataSet object with the given parameters. It sets the dataset's attributes
/// to indicate that it supports sparse data. It allocates memory for the storage of sparse data, including
/// start, end, data, and index arrays. It also sets up GPU buffers for these sparse data components.
/// If the dataset is indexed, it allocates memory for the index array and sets the corresponding attribute.
/// If the dataset is weighted, it allocates memory for the data weight array and sets the corresponding attribute.
/// </remarks>
/// <param name="T">The data type of the dataset and its components.</typeparam>
template<typename T>
DataSet<T>::DataSet(uint32_t examples, uint32_t uniqueExamples, size_t sparseDataSize,
    const DataSetDimensions& dim, bool isIndexed, bool isWeighted, const std::string& name) :
    DataSetBase(name, DataSetEnums::getDataType<T>(), examples, uniqueExamples, dim)
{
    // Set the attributes to indicate that the dataset supports sparse data
    _attributes = DataSetEnums::Attributes::Sparse;
    _sparseDataSize = sparseDataSize;

    // Initialize vectors for sparse data components
    _vSparseStart.resize(_uniqueExamples, 0);
    _vSparseEnd.resize(_uniqueExamples, 0);
    _vSparseData.resize(_sparseDataSize);
    _vSparseIndex.resize(_sparseDataSize, 0);

    // Calculate sparse stride
    size_t sparseStride = (_sparseDataSize + _uniqueExamples - 1) / _uniqueExamples;

    // Populate sparse start and end arrays
    _vSparseStart[0] = 0;
    _vSparseEnd[0] = sparseStride;
    for (uint32_t i = 1; i < _uniqueExamples; ++i)
    {
        _vSparseStart[i] = _vSparseEnd[i - 1];
        _vSparseEnd[i] = _vSparseStart[i] + sparseStride;
    }

    // Initialize GPU buffers for sparse data components
    _pbSparseStart.reset(new GpuBuffer<uint64_t>(_vSparseStart.size(), false, _bStreaming));
    _pbSparseEnd.reset(new GpuBuffer<uint64_t>(_vSparseEnd.size(), false, _bStreaming));
    _pbSparseData.reset(new GpuBuffer<T>(_vSparseData.size(), false, _bStreaming));
    _pbSparseIndex.reset(new GpuBuffer<uint32_t>(_vSparseIndex.size(), false, _bStreaming));

    // If the dataset is indexed, allocate memory for the index array and set the indexed attribute
    if (isIndexed) {
        _attributes |= DataSetEnums::Attributes::Indexed;
        _bIndexed = true;
        _vIndex.resize(_examples, 0);
        _pbIndex.reset(new GpuBuffer<uint32_t>(_vIndex.size(), false, _bStreaming));
    }

    // If the dataset is weighted, allocate memory for the data weight array and set the weighted attribute
    if (isWeighted)
    {
        _attributes |= DataSetEnums::Attributes::Weighted;
        _vDataWeight.resize(_examples);
        _pbDataWeight.reset(new GpuBuffer<float>(_vDataWeight.size(), false, _bStreaming));
    }
}

/// <summary>
/// Load dense data into the dataset and upload it to the GPU if applicable.
/// </summary>
/// <param name="srcData">A pointer to the source dense data to be loaded.</param>
/// <remarks>
/// This function loads dense data into the dataset if the dataset is configured to support dense data.
/// It checks if the dataset is configured for sparse data, and if so, it throws a runtime error since dense data
/// cannot be set on a sparse dataset.
/// Otherwise, it casts the source dense data to the appropriate data type, copies it into the internal storage,
/// and uploads it to the GPU if applicable.
/// </remarks>
/// <param name="T">The data type of the dataset and source dense data.</typeparam>
template<typename T>
void DataSet<T>::LoadDenseData(const void* srcData) {
    // Cast the source dense data to the appropriate data type
    const T* srcDataTyped = static_cast<const T*>(srcData);

    // Check if the dataset supports sparse data
    if (_attributes & DataSetEnums::Attributes::Sparse) {
        // Throw a runtime error if dense data is being set on a sparse dataset
        throw std::runtime_error("Cannot set dense data on a sparse DataSet");
    }
    else {
        // Copy the source dense data into the internal storage
        std::copy(srcDataTyped, srcDataTyped + _vData.size(), _vData.data());

        // Upload the dense data to the GPU if applicable
        _pbData->Upload(_vData.data());
    }
}

/// <summary>
/// Copy dense data into the dataset.
/// </summary>
/// <param name="srcData">A pointer to the source dense data to be copied.</param>
/// <remarks>
/// This function copies dense data into the dataset if the dataset is configured to support dense data.
/// It checks if the dataset is configured for sparse data, and if so, it throws a runtime error since dense data
/// cannot be set on a sparse dataset.
/// Otherwise, it casts the source dense data to the appropriate data type and copies it into the internal storage.
/// </remarks>
/// <param name="T">The data type of the dataset and source dense data.</typeparam>
template<typename T>
void DataSet<T>::CopyDenseData(const void* srcData) {
    // Cast the source dense data to the appropriate data type
    const T* srcDataTyped = static_cast<const T*>(srcData);

    // Check if the dataset supports sparse data
    if (_attributes & DataSetEnums::Attributes::Sparse) {
        // Throw a runtime error if dense data is being set on a sparse dataset
        throw std::runtime_error("Cannot set dense data on a sparse DataSet");
    }
    else {
        // Copy the source dense data into the internal storage
        std::copy(srcDataTyped, srcDataTyped + _vData.size(), _vData.data());
    }
}

/// <summary>
/// Load sparse data into the dataset.
/// </summary>
/// <param name="srcSparseStart">A pointer to the source sparse start data.</param>
/// <param name="srcSparseEnd">A pointer to the source sparse end data.</param>
/// <param name="srcSparseData">A pointer to the source sparse data to be loaded.</param>
/// <param name="srcSparseIndex">A pointer to the source sparse index data.</param>
/// <remarks>
/// This function loads sparse data into the dataset if the dataset is configured to support sparse data.
/// It checks that the source sparse data starts from index 0 and validates the data length against the internal storage size.
/// Then, it copies the data from the source pointers to the internal storage and uploads it to the GPU if applicable.
/// If the dataset is not configured for sparse data, it throws a runtime error.
/// </remarks>
/// <param name="T">The data type of the dataset.</typeparam>
/// <param name="T">The data type of the source sparse data.</typeparam>
template<typename T>
void DataSet<T>::LoadSparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd,
    const void* srcSparseData, const uint32_t* srcSparseIndex) {
    // Cast the source sparse data to the appropriate data type
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    // Check if the dataset supports sparse data
    if (_attributes & DataSetEnums::Attributes::Sparse) {
        // Check if the source sparse data starts from index 0
        if (srcSparseStart[0] != 0) {
            throw std::runtime_error("Sparse data should be zero-indexed; srcSparseStart[0] != 0");
        }

        // Calculate the expected data length
        uint64_t dataLength = srcSparseEnd[_uniqueExamples - 1];

        // Check if the data length exceeds the internal storage size
        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size()) {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size() << " Required: "
                << dataLength;
            throw std::length_error(msg.str());
        }

        // Copy sparse start and end data
        std::copy(srcSparseStart, srcSparseStart + _uniqueExamples, _vSparseStart.data());
        std::copy(srcSparseEnd, srcSparseEnd + _uniqueExamples, _vSparseEnd.data());

        // Copy sparse data and index
        std::copy(srcSparseDataTyped, srcSparseDataTyped + dataLength, _vSparseData.data());
        std::copy(srcSparseIndex, srcSparseIndex + dataLength, _vSparseIndex.data());

        // Upload sparse data to the GPU if applicable
        _pbSparseStart->Upload(_vSparseStart.data());
        _pbSparseEnd->Upload(_vSparseEnd.data());
        _pbSparseIndex->Upload(_vSparseIndex.data());
        _pbSparseData->Upload(_vSparseData.data());
    }
    else {
        // Throw a runtime error if the dataset does not support sparse data
        throw std::runtime_error("Cannot set sparse data on a non-sparse DataSet");
    }
}

/// <summary>
/// Copy sparse data into the dataset.
/// </summary>
/// <param name="srcSparseStart">A pointer to the source sparse start data.</param>
/// <param name="srcSparseEnd">A pointer to the source sparse end data.</param>
/// <param name="srcSparseData">A pointer to the source sparse data to be copied.</param>
/// <param name="srcSparseIndex">A pointer to the source sparse index data.</param>
/// <remarks>
/// This function copies sparse data into the dataset if the dataset is configured to support sparse data.
/// It checks that the source sparse data starts from index 0 and validates the data length against the internal storage size.
/// Then, it copies the data from the source pointers to the internal storage.
/// If the dataset is not configured for sparse data, it throws a runtime error.
/// </remarks>
/// <param name="T">The data type of the dataset.</typeparam>
/// <param name="T">The data type of the source sparse data.</typeparam>
template<typename T>
void DataSet<T>::CopySparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd,
    const void* srcSparseData, const uint32_t* srcSparseIndex) {
    // Cast the source sparse data to the appropriate data type
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    // Check if the dataset supports sparse data
    if (_attributes & DataSetEnums::Attributes::Sparse) {
        // Check if the source sparse data starts from index 0
        if (srcSparseStart[0] != 0) {
            throw std::runtime_error("Sparse data should be zero-indexed; srcSparseStart[0] != 0");
        }

        // Calculate the expected data length
        uint64_t dataLength = srcSparseEnd[_uniqueExamples - 1];

        // Check if the data length exceeds the internal storage size
        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size()) {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size() << " Required: "
                << dataLength;
            throw std::length_error(msg.str());
        }

        // Copy sparse start and end data
        std::copy(srcSparseStart, srcSparseStart + _uniqueExamples, _vSparseStart.data());
        std::copy(srcSparseEnd, srcSparseEnd + _uniqueExamples, _vSparseEnd.data());

        // Copy sparse data and index
        std::copy(srcSparseDataTyped, srcSparseDataTyped + dataLength, _vSparseData.data());
        std::copy(srcSparseIndex, srcSparseIndex + dataLength, _vSparseIndex.data());
    }
    else {
        // Throw a runtime error if the dataset does not support sparse data
        throw std::runtime_error("Cannot set sparse data on a non-sparse DataSet");
    }
}

/// <summary>
/// Load sparse data into the dataset.
/// </summary>
/// <param name="srcSparseStart">A pointer to the source sparse start data.</param>
/// <param name="srcSparseEnd">A pointer to the source sparse end data.</param>
/// <param name="srcSparseData">A pointer to the source sparse data to be loaded.</param>
/// <param name="srcSparseIndex">A pointer to the source sparse index data.</param>
/// <remarks>
/// This function loads sparse data into the dataset if the dataset is configured to support sparse data.
/// It checks that the source sparse data starts from index 0 and validates the data length against the internal storage size.
/// Then, it copies the data from the source pointers to the internal storage and uploads it to GPU buffers.
/// If the dataset is not configured for sparse data, it throws a runtime error.
/// </remarks>
/// <param name="T">The data type of the dataset.</typeparam>
/// <param name="T">The data type of the source sparse data.</typeparam>
template<typename T>
void DataSet<T>::LoadSparseData(const long* srcSparseStart, const long* srcSparseEnd,
    const void* srcSparseData, const long* srcSparseIndex) {
    // Cast the source sparse data to the appropriate data type
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    // Check if the dataset supports sparse data
    if (_attributes & DataSetEnums::Attributes::Sparse) {
        // Check if the source sparse data starts from index 0
        if (srcSparseStart[0] != 0) {
            throw std::runtime_error("Sparse data should be zero-indexed; srcSparseStart[0] != 0");
        }

        // Calculate the expected data length
        uint64_t dataLength = srcSparseEnd[_uniqueExamples - 1];

        // Check if the data length exceeds the internal storage size
        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size()) {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size() << " Required: "
                << dataLength;
            throw std::length_error(msg.str());
        }

        // Copy sparse start and end data
        for (uint32_t i = 0; i < _uniqueExamples; ++i) {
            _vSparseStart[i] = (uint64_t)srcSparseStart[i];
            _vSparseEnd[i] = (uint64_t)srcSparseEnd[i];
        }

        // Copy sparse data and index
        for (uint64_t i = 0; i < dataLength; ++i) {
            _vSparseData[i] = srcSparseDataTyped[i];
            _vSparseIndex[i] = (uint32_t)srcSparseIndex[i];
        }

        // Upload sparse data to GPU buffers
        _pbSparseStart->Upload(_vSparseStart.data());
        _pbSparseEnd->Upload(_vSparseEnd.data());
        _pbSparseIndex->Upload(_vSparseIndex.data());
        _pbSparseData->Upload(_vSparseData.data());
    }
    else {
        // Throw a runtime error if the dataset does not support sparse data
        throw std::runtime_error("Cannot set sparse data on a non-sparse DataSet");
    }
}

/// <summary>
/// Copy sparse data into the dataset.
/// </summary>
/// <param name="srcSparseStart">A pointer to the source sparse start data.</param>
/// <param name="srcSparseEnd">A pointer to the source sparse end data.</param>
/// <param name="srcSparseData">A pointer to the source sparse data to be copied.</param>
/// <param name="srcSparseIndex">A pointer to the source sparse index data.</param>
/// <remarks>
/// This function copies sparse data into the dataset if the dataset is configured to support sparse data.
/// It checks that the source sparse data starts from index 0 and validates the data length against the internal storage size.
/// Then, it copies the data from the source pointers to the internal storage.
/// If the dataset is not configured for sparse data, it throws a runtime error.
/// </remarks>
/// <param name="T">The data type of the dataset.</typeparam>
/// <param name="T">The data type of the source sparse data.</typeparam>
template<typename T>
void DataSet<T>::CopySparseData(const long* srcSparseStart, const long* srcSparseEnd,
    const void* srcSparseData, const long* srcSparseIndex) {
    // Cast the source sparse data to the appropriate data type
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    // Check if the dataset supports sparse data
    if (_attributes & DataSetEnums::Attributes::Sparse) {
        // Check if the source sparse data starts from index 0
        if (srcSparseStart[0] != 0) {
            throw std::runtime_error("Sparse data should be zero-indexed; srcSparseStart[0] != 0");
        }

        // Calculate the expected data length
        uint64_t dataLength = srcSparseEnd[_uniqueExamples - 1];

        // Check if the data length exceeds the internal storage size
        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size()) {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size() << " Required: "
                << dataLength;
            throw std::length_error(msg.str());
        }

        // Copy sparse start and end data
        for (uint32_t i = 0; i < _uniqueExamples; ++i) {
            _vSparseStart[i] = (uint64_t)srcSparseStart[i];
            _vSparseEnd[i] = (uint64_t)srcSparseEnd[i];
        }

        // Copy sparse data and index
        for (uint64_t i = 0; i < dataLength; ++i) {
            _vSparseData[i] = srcSparseDataTyped[i];
            _vSparseIndex[i] = (uint32_t)srcSparseIndex[i];
        }
    }
    else {
        // Throw a runtime error if the dataset does not support sparse data
        throw std::runtime_error("Cannot set sparse data on a non-sparse DataSet");
    }
}

/// <summary>
/// Load indexed data into the dataset.
/// </summary>
/// <param name="srcIndexedData">A pointer to the source indexed data to be loaded.</param>
/// <remarks>
/// This function loads indexed data into the dataset if the dataset is configured to support indexed data.
/// It copies the data from the source pointer to the internal storage and uploads it to the GPU buffer.
/// If the dataset is not configured for indexed data, it throws a runtime error.
/// </remarks>
/// <param name="T">The data type of the dataset.</typeparam>
/// <param name="T">The data type of the indexed data.</typeparam>
template<typename T>
void DataSet<T>::LoadIndexedData(const uint32_t* srcIndexedData) {
    // Check if the dataset supports indexed data
    if (_attributes & DataSetEnums::Attributes::Indexed) {
        // Copy the indexed data from the source to the internal storage
        std::copy(srcIndexedData, srcIndexedData + _vIndex.size(), _vIndex.data());

        // Upload the indexed data to the GPU buffer
        _pbIndex->Upload(_vIndex.data());
    }
    else {
        // Throw a runtime error if the dataset does not support indexed data
        throw std::runtime_error("Cannot set indexed data on a non-indexed DataSet");
    }
}

/// <summary>
/// Load weighted data into the dataset.
/// </summary>
/// <param name="srcWeightData">A pointer to the source weighted data to be loaded.</param>
/// <remarks>
/// This function loads weighted data into the dataset if the dataset is configured to support weighted data.
/// It copies the data from the source pointer to the internal storage and uploads it to the GPU buffer.
/// If the dataset is not configured for weighted data, it throws a runtime error.
/// </remarks>
/// <param name="T">The data type of the dataset.</typeparam>
/// <param name="T">The data type of the weighted data.</typeparam>
template<typename T>
void DataSet<T>::LoadDataWeight(const float* srcWeightData) {
    // Check if the dataset supports weighted data
    if (_attributes & DataSetEnums::Attributes::Weighted) {
        // Copy the weighted data from the source to the internal storage
        std::copy(srcWeightData, srcWeightData + _vDataWeight.size(), _vDataWeight.data());

        // Upload the weighted data to the GPU buffer
        _pbDataWeight->Upload(_vDataWeight.data());
    }
    else {
        // Throw a runtime error if the dataset does not support weighted data
        throw std::runtime_error("Cannot set weight data on a non-weighted DataSet");
    }
}

/// <summary>
/// Get the value of a non-sparse data point at specific coordinates in a dataset example.
/// </summary>
/// <param name="n">The index of the example.</param>
/// <param name="x">The x-coordinate of the data point.</param>
/// <param name="y">The y-coordinate of the data point.</param>
/// <param name="z">The z-coordinate of the data point.</param>
/// <returns>The value of the specified data point.</returns>
/// <remarks>
/// This function retrieves the value of a non-sparse data point at the specified coordinates within
/// a specific example of the dataset. It performs error checking to ensure the dataset is not sparse,
/// the example index is valid, and the coordinates are within the dataset dimensions.
/// </remarks>
template<typename T>
T DataSet<T>::GetDataPoint(uint32_t n, uint32_t x, uint32_t y, uint32_t z) {
    // Check if the dataset is sparse
    if (_attributes & DataSetEnums::Sparse) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetDataPoint: Attempt to read non-sparse data from a sparse dataset.\n");
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Check if the example index is valid
    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetDataPoint: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // If the dataset is indexed, map the example index to the indexed value
    if (_bIndexed) {
        n = _vIndex[n];
    }

    // Check if the coordinates are within the dataset dimensions
    if ((x >= _width) || (y >= _height) || (z >= _length)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetDataPoint: Invalid data point coordinates (%u, %u, %u) "
                "for dataset dimensions (width: %u, height: %u, length: %u).\n", x, y, z, _width, _height, _length);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Retrieve and return the value of the data point
    return _vData[(n * _stride) + x + _width * (y + z * _height)];
}

/// <summary>
/// Set the value of a non-sparse data point at specific coordinates in a dataset example.
/// </summary>
/// <param name="v">The value to set.</param>
/// <param name="n">The index of the example.</param>
/// <param name="x">The x-coordinate of the data point.</param>
/// <param name="y">The y-coordinate of the data point.</param>
/// <param name="z">The z-coordinate of the data point.</param>
/// <remarks>
/// This function sets the value of a non-sparse data point at specified coordinates within
/// a specific example of the dataset. It performs error checking to ensure the dataset is
/// not sparse, the example index is valid, and the coordinates are within the dataset dimensions.
/// </remarks>
template<typename T>
bool DataSet<T>::SetDataPoint(T v, uint32_t n, uint32_t x, uint32_t y, uint32_t z) {
    // Check if the dataset is sparse
    if (_attributes & DataSetEnums::Sparse) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetDataPoint: Attempt to read non-sparse data from a sparse dataset.\n");
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Check if the example index is valid
    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetDataPoint: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // If the dataset is indexed, map the example index to the indexed value
    if (_bIndexed) {
        n = _vIndex[n];
    }

    // Check if the coordinates are within the dataset dimensions
    if ((x >= _width) || (y >= _height) || (z >= _length)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetDataPoint: Invalid data point coordinates (%u, %u, %u) "
                "for dataset dimensions (width: %u, height: %u, length: %u).\n", x, y, z, _width, _height, _length);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Set the value of the data point
    _vData[(n * _stride) + x + _width * (y + z * _height)] = v;
    return true;
}

/// <summary>
/// Get the number of sparse data points in a specific example of the dataset.
/// </summary>
/// <param name="n">The index of the example.</param>
/// <returns>The number of sparse data points in the specified example.</returns>
/// <remarks>
/// This function retrieves the count of sparse data points within a specific example.
/// It performs error checking to ensure the dataset is sparse, the example index is valid,
/// and the function returns the count of sparse data points for the given example.
/// </remarks>
template<typename T>
uint64_t DataSet<T>::GetSparseDataPoints(uint32_t n) {
    // Check if the dataset is sparse
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseDataPoints: Attempt to read sparse data from a non-sparse dataset.\n");
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Check if the example index is valid
    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseDataPoints: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // If the dataset is indexed, map the example index to the indexed value
    if (_bIndexed) {
        n = _vIndex[n];
    }

    // Return the count of sparse data points in the specified example
    return _vSparseEnd[n] - _vSparseStart[n];
}

/// <summary>
/// Get the index of a specific sparse data point in the dataset.
/// </summary>
/// <param name="n">The index of the example.</param>
/// <param name="i">The index of the sparse data point within the example.</param>
/// <returns>The index value of the specified sparse data point.</returns>
/// <remarks>
/// This function retrieves the index of a sparse data point within an example.
/// It performs error checking to ensure the dataset is sparse, the example index is valid,
/// and the sparse data point index is within the expected range.
/// </remarks>
template<typename T>
uint32_t DataSet<T>::GetSparseIndex(uint32_t n, uint32_t i) {
    // Check if the dataset is sparse
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseIndex: Attempt to read sparse data from a non-sparse dataset.\n");
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Check if the example index is valid
    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseIndex: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // If the dataset is indexed, map the example index to the indexed value
    if (_bIndexed) {
        n = _vIndex[n];
    }

    // Check if the sparse data point index is within a valid range
    if (i >= _vSparseEnd[n] - _vSparseStart[n]) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseIndex: Sparse index %u is out of range [0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Return the index value of the specified sparse data point
    return _vSparseIndex[_vSparseStart[n] + i];
}

/// <summary>
/// Set the index of a specific sparse data point in the dataset.
/// </summary>
/// <param name="n">The index of the example.</param>
/// <param name="i">The index of the sparse data point within the example.</param>
/// <param name="v">The new index value to set.</param>
/// <returns>True if the operation was successful, false otherwise.</returns>
template<typename T>
bool DataSet<T>::SetSparseIndex(uint32_t n, uint32_t i, uint32_t v) {
    // Check if the dataset is sparse
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseIndex: Attempt to set sparse data index on a non-sparse dataset.\n");
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Check if the example index is valid
    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseIndex: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // If the dataset is indexed, map the example index to the indexed value
    if (_bIndexed) {
        n = _vIndex[n];
    }

    // Check if the sparse data point index is within a valid range
    if (i >= _vSparseEnd[n] - _vSparseStart[n]) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseIndex: Sparse index %u is out of range [0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Set the new index value for the specified sparse data point
    _vSparseIndex[_vSparseStart[n] + i] = v;
    _bDirty = true;

    return true;
}

/// <summary>
/// Get the value of a specific sparse data point in the dataset.
/// </summary>
/// <param name="n">The index of the example.</param>
/// <param name="i">The index of the sparse data point within the example.</param>
/// <returns>The value of the specified sparse data point.</returns>
template<typename T>
T DataSet<T>::GetSparseDataPoint(uint32_t n, uint32_t i) {
    // Check if the dataset is sparse
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseDataPoint: Attempt to read sparse data from a non-sparse dataset.\n");
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Check if the example index is valid
    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseDataPoint: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // If the dataset is indexed, map the example index to the indexed value
    if (_bIndexed) {
        n = _vIndex[n];
    }

    // Check if the sparse data point index is within a valid range
    if (i >= _vSparseEnd[n] - _vSparseStart[n]) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseDataPoint: Sparse index %u is out of range [0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Return the value of the specified sparse data point
    return _vSparseData[_vSparseStart[n] + i];
}

/// <summary>
/// Set the value of a specific sparse data point in the dataset.
/// </summary>
/// <param name="n">The index of the example.</param>
/// <param name="i">The index of the sparse data point within the example.</param>
/// <param name="v">The new value for the sparse data point.</param>
/// <returns>True if the update is successful, false otherwise.</returns>
template<typename T>
bool DataSet<T>::SetSparseDataPoint(uint32_t n, uint32_t i, T v) {
    // Check if the dataset is sparse
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseDataPoint: Attempt to modify sparse data in a non-sparse dataset.\n");
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Check if the example index is valid
    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseDataPoint: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // If the dataset is indexed, map the example index to the indexed value
    if (_bIndexed) {
        n = _vIndex[n];
    }

    // Check if the sparse data point index is within a valid range
    if (i >= _vSparseEnd[n] - _vSparseStart[n]) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseDataPoint: Sparse index %u is out of range [0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Update the value of the specified sparse data point
    _vSparseData[_vSparseStart[n] + i] = v;

    // Mark the dataset as dirty, indicating that it has been modified
    _bDirty = true;

    return true;
}

/// <summary>
/// Constructor for the DataSet class.
/// </summary>
/// <typeparam name="T">The data type for the dataset.</typeparam>
/// <param name="fname">The filename of the NetCDF input file.</param>
/// <param name="n">The dataset index.</param>
template<typename T> DataSet<T>::DataSet(const std::string& fname, uint32_t n) :
    _pbData(),
    _pbSparseData()
{
    bool bResult = true;
    if (getGpu()._id == 0)
    {
        bool bOpened = false;
        try
        {
            // Open the NetCDF file 'fname' for reading
            netCDF::NcFile nfc(fname.c_str(), netCDF::NcFile::read);

            // Set the flag 'bOpened' to indicate that the file has been successfully opened
            bOpened = true;

            // Convert the dataset index 'n' to a string for naming purposes
            std::string nstring = std::to_string(n);

            // Create a variable name for the dataset name attribute based on 'name' and the dataset index
            std::string vname = "name" + nstring;

            // Get the NetCDF group attribute named 'vname' from the NetCDF file 'nfc'
            netCDF::NcGroupAtt nameAtt = nfc.getAtt(vname);

            // Check if the dataset name attribute is null (not found in the file)
            if (nameAtt.isNull())
            {
                // Print an error message indicating that no dataset name was supplied in the NetCDF input file
                std::cerr << "NcException: DataSet::DataSet: No dataset name supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            // Get the values of the dataset name attribute and store it in '_name'
            nameAtt.getValues(_name);

            // Print the name of the data set to the console
            std::cout << "DataSet<T>::DataSet: Name of data set: " << _name << std::endl;

            // Create a variable name for the dataset type attribute based on 'dataType' and the dataset index
            vname = "dataType" + nstring;

            // Get the NetCDF group attribute named 'vname' from the NetCDF file 'nfc'
            netCDF::NcGroupAtt dataTypeAtt = nfc.getAtt(vname);

            // Check if the dataset type attribute is null (not found in the file)
            if (dataTypeAtt.isNull())
            {
                // Print an error message indicating that no datatype was supplied in the NetCDF input file
                std::cerr << "NcException: DataSet::DataSet: No datatype supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            int dataType;

            // Get the values of the dataset type attribute and store it in 'dataType'
            dataTypeAtt.getValues(&dataType);

            // Convert the 'dataType' to the corresponding enumeration type and store it in '_dataType'
            _dataType = (DataSetEnums::DataType)dataType;

            // Create a variable name for the dataset attributes attribute based on 'attributes' and the dataset index
            vname = "attributes" + nstring;

            // Get the NetCDF group attribute named 'vname' from the NetCDF file 'nfc'
            netCDF::NcGroupAtt attributesAtt = nfc.getAtt(vname);

            // Check if the dataset attributes attribute is null (not found in the file)
            if (attributesAtt.isNull())
            {
                // Print an error message indicating that no attributes were supplied in the NetCDF input file
                std::cerr << "NcException: DataSet::DataSet: No attributes supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            // Get the values of the dataset attributes attribute and store it in '_attributes'
            attributesAtt.getValues(&_attributes);

            // Check if there are any attributes present, and if so, print them to the console
            if (_attributes != 0)
            {
                int tempAtt = _attributes;
                int position = 0;
                std::cout << "DataSet<T>::DataSet: Attributes:";

                // Iterate through the bits of '_attributes' to determine which attributes are set
                while (tempAtt != 0)
                {
                    if (tempAtt & 1)
                    {
                        DataSetEnums::Attributes a = (DataSetEnums::Attributes)(1 << position);
                        std::cout << " " << a;
                    }
                    tempAtt >>= 1;
                    position++;
                }
                std::cout << std::endl;
            }

            // Create a variable name for the examples count dimension based on 'examplesDim' and the dataset index
            vname = "examplesDim" + nstring;

            // Get the NetCDF dimension named 'vname' from the NetCDF file 'nfc'
            netCDF::NcDim examplesDim = nfc.getDim(vname);

            // Check if the examples count dimension is null (not found in the file)
            if (examplesDim.isNull())
            {
                // Print an error message indicating that no examples count was supplied in the NetCDF input file
                std::cerr << "NcException: DataSet::DataSet: No examples count supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            // Store the size (count) of the examples count dimension in '_examples'
            _examples = examplesDim.getSize();

            // Check if the examples count is zero
            if (_examples == 0)
            {
                // Print an error message indicating a zero-valued examples count in the NetCDF input file
                std::cerr << "NcException: DataSet::DataSet: Zero-valued Examples count in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            // Create a variable name for the unique examples count dimension based on 'uniqueExamplesDim' and the dataset index
            vname = "uniqueExamplesDim" + nstring;

            // Get the NetCDF dimension named 'vname' from the NetCDF file 'nfc'
            netCDF::NcDim uniqueExamplesDim = nfc.getDim(vname);

            // Check if the unique examples count dimension is null (not found in the file)
            if (uniqueExamplesDim.isNull())
            {
                // Set '_uniqueExamples' to be equal to '_examples' (no unique count provided)
                _uniqueExamples = _examples;
            }
            else
            {
                // Store the size (count) of the unique examples count dimension in '_uniqueExamples'
                _uniqueExamples = uniqueExamplesDim.getSize();
            }

            // Create a variable name for the dimensions attribute based on 'dimensions' and the dataset index
            vname = "dimensions" + nstring;

            // Get the NetCDF group attribute named 'vname' from the NetCDF file 'nfc'
            netCDF::NcGroupAtt dimensionsAtt = nfc.getAtt(vname);

            // Check if the dimensions attribute is null (not found in the file)
            if (dimensionsAtt.isNull())
            {
                // Print an error message indicating that no dimension count was supplied in the NetCDF input file
                std::cerr << "NcException: DataSet::DataSet: No dimension count supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            // Get the values of the dimensions attribute and store it in '_dimensions'
            dimensionsAtt.getValues(&_dimensions);

            // Check if the dimension count is less than 1 or greater than 3 (invalid)
            if ((_dimensions < 1) || (_dimensions > 3))
            {
                // Print an error message indicating an invalid dimension count
                std::cerr << "NcException: DataSet::DataSet: Invalid dimension count (" << std::to_string(_dimensions) << ") supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            // Create a variable name for the datapoint width attribute based on 'width' and the dataset index
            vname = "width" + nstring;

            // Get the NetCDF group attribute named 'vname' from the NetCDF file 'nfc'
            netCDF::NcGroupAtt widthAtt = nfc.getAtt(vname);

            // Check if the datapoint width attribute is null (not found in the file)
            if (widthAtt.isNull())
            {
                // Print an error message indicating that no datapoint width was supplied in the NetCDF input file
                std::cerr << "NcException: DataSet::DataSet: No datapoint width supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            // Get the values of the datapoint width attribute and store it in '_width'

            // Check if there are additional dimensions beyond the width
            if (_dimensions > 1)
            {
                // Create a variable name for the datapoint height attribute based on 'height' and the dataset index
                vname = "height" + nstring;

                // Get the NetCDF group attribute named 'vname' from the NetCDF file 'nfc'
                netCDF::NcGroupAtt heightAtt = nfc.getAtt(vname);

                // Check if the datapoint height attribute is null (not found in the file)
                if (heightAtt.isNull())
                {
                    // Print an error message indicating that no datapoint height was supplied in the NetCDF input file
                    std::cerr << "NcException: DataSet::DataSet: No datapoint height supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                // Get the values of the datapoint height attribute and store it in '_height'
            }
            else
            {
                // If there's only one dimension, set '_height' to 1
                _height = 1;
            }

            // Check if there are additional dimensions beyond the height
            if (_dimensions > 2)
            {
                // Create a variable name for the datapoint length attribute based on 'length' and the dataset index
                vname = "length" + nstring;

                // Get the NetCDF group attribute named 'vname' from the NetCDF file 'nfc'
                netCDF::NcGroupAtt lengthAtt = nfc.getAtt(vname);

                // Check if the datapoint length attribute is null (not found in the file)
                if (lengthAtt.isNull())
                {
                    // Print an error message indicating that no datapoint length was supplied in the NetCDF input file
                    std::cerr << "NcException: DataSet::DataSet: No datapoint length supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                // Get the values of the datapoint length attribute and store it in '_length'
            }
            else
            {
                // If there's only one or two dimensions, set '_length' to 1
                _length = 1;
            }

            // Print the dataset dimensions to the console
            std::cerr << "DataSet<T>::DataSet: " << _dimensions << "-dimensional data comprised of (" << _width << ", " << _height << ", " << _length << ") datapoints." << std::endl;

            // Check if any of the dimensions (width, height, or length) is zero
            if ((_width == 0) || (_height == 0) || (_length == 0))
            {
                // Print an error message indicating invalid dataset dimensions
                std::cerr << "NcException: DataSet::DataSet: Invalid dataset dimensions in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            // Check if the dataset attributes include the 'Sparse' flag
            if (_attributes & DataSetEnums::Sparse)
            {
                // Resize the '_vSparseStart' and '_vSparseEnd' vectors to match the number of unique examples
                _vSparseStart.resize(_uniqueExamples);
                _vSparseEnd.resize(_uniqueExamples);

                // Create a variable name for the sparse data dimensions based on 'sparseDataDim' and a string representation of the dataset
                vname = "sparseDataDim" + nstring;

                // Get the NetCDF dimension named 'vname' from the NetCDF file 'nfc'
                netCDF::NcDim sparseDataDim = nfc.getDim(vname);

                // Check if the NetCDF dimension is null (not found in the file)
                if (sparseDataDim.isNull())
                {
                    // Print an error message indicating that no sparse data dimensions were supplied in the NetCDF input file
                    std::cerr << "NcException: DataSet::DataSet: No sparse data dimensions supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                // Store the size of the sparse data (number of elements)
                _sparseDataSize = sparseDataDim.getSize();

                // Check if there is no actual sparse data (size is zero)
                if (_sparseDataSize == 0)
                {
                    // Print an error message indicating a sparse data set with no actual data in the NetCDF input file
                    std::cerr << "NcException: DataSet::DataSet: Sparse data set with no actual data in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                // Resize the '_vSparseIndex' vector to match the sparse data size and print the number of total datapoints
                _vSparseIndex.resize(_sparseDataSize);
                std::cout << "DataSet<T>::DataSet: " << _sparseDataSize << " total datapoints." << std::endl;

                // Create variable names for sparse data offset start, end, and indices
                vname = "sparseStart" + nstring;
                netCDF::NcVar sparseStartVar = nfc.getVar(vname);
                vname = "sparseEnd" + nstring;
                netCDF::NcVar sparseEndVar = nfc.getVar(vname);
                vname = "sparseIndex" + nstring;
                netCDF::NcVar sparseIndexVar = nfc.getVar(vname);

                // Check if the sparse data offset start variable is null (not found in the file)
                if (sparseStartVar.isNull())
                {
                    // Print an error message indicating that no sparse data offset start was supplied in the NetCDF input file
                    std::cerr << "NcException: DataSet::DataSet: No sparse offset start supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                // Check if the sparse data offset end variable is null (not found in the file)
                if (sparseEndVar.isNull())
                {
                    // Print an error message indicating that no sparse data offset end was supplied in the NetCDF input file
                    std::cerr << "NcException: DataSet::DataSet: No sparse data end supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                // Check if the sparse data indices variable is null (not found in the file)
                if (sparseIndexVar.isNull())
                {
                    // Print an error message indicating that no sparse data indices were supplied in the NetCDF input file
                    std::cerr << "NcException: DataSet::DataSet: No sparse data indices supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                // Determine the data type of sparse data offset start variable
                netCDF::NcType vStartType = sparseStartVar.getType();

                // Depending on the data type, read the data into '_vSparseStart'
                if (vStartType == netCDF::ncUint)
                {
                    std::vector<uint32_t> vTempSparseStart(_uniqueExamples);
                    sparseStartVar.getVar((uint32_t*)vTempSparseStart.data());
                    std::copy(vTempSparseStart.begin(), vTempSparseStart.end(), _vSparseStart.begin());
                }
                else
                    sparseStartVar.getVar((uint64_t*)_vSparseStart.data());

                // Determine the data type of sparse data offset end variable
                netCDF::NcType vEndType = sparseEndVar.getType();

                // Depending on the data type, read the data into '_vSparseEnd'
                if (vEndType == netCDF::ncUint)
                {
                    std::vector<uint32_t> vTempSparseEnd(_uniqueExamples);
                    sparseEndVar.getVar((uint32_t*)vTempSparseEnd.data());
                    std::copy(vTempSparseEnd.begin(), vTempSparseEnd.end(), _vSparseEnd.begin());
                }
                else
                    sparseEndVar.getVar((uint64_t*)_vSparseEnd.data());

                // Read the sparse data indices into '_vSparseIndex'
                sparseIndexVar.getVar((uint32_t*)_vSparseIndex.data());

                // Check if the dataset attributes do not include the 'Boolean' flag
                if (!(_attributes & DataSetEnums::Boolean))
                {
                    // Create a variable name for the sparse data based on 'sparseData' and a string representation of the dataset
                    vname = "sparseData" + nstring;

                    // Get the NetCDF variable named 'vname' from the NetCDF file 'nfc'
                    netCDF::NcVar sparseDataVar = nfc.getVar(vname);

                    // Check if the sparse data variable is null (not found in the file)
                    if (sparseDataVar.isNull())
                    {
                        // Print an error message indicating that no sparse data was located in the NetCDF input file
                        std::cerr << "NcException: DataSet::DataSet: No sparse data located in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                    }

                    // Resize the '_vSparseData' vector to match the size of the sparse data dimensions
                    _vSparseData.resize(sparseDataDim.getSize());

                    // Read the sparse data from the NetCDF variable 'sparseDataVar' into the '_vSparseData' vector
                    sparseDataVar.getVar(_vSparseData.data());
                }
            }
            else
            {
                // Calculate the stride based on the width, height, and length dimensions
                _stride = _width * _height * _length;

                // Create a variable name for the data dimensions based on 'dataDim' and a string representation of the dataset
                vname = "dataDim" + nstring;

                // Get the NetCDF dimension named 'vname' from the NetCDF file 'nfc'
                netCDF::NcDim dataDim = nfc.getDim(vname);

                // Check if the NetCDF dimension is null (not found in the file)
                if (dataDim.isNull())
                {
                    // Print an error message indicating that no data dimensions were found in the NetCDF input file
                    std::cerr << "NcException: DataSet::DataSet: No data dimensions located in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                // Create a variable name for the data based on 'data' and a string representation of the dataset
                vname = "data" + nstring;

                // Get the NetCDF variable named 'vname' from the NetCDF file 'nfc'
                netCDF::NcVar dataVar = nfc.getVar(vname);

                // Check if the dataset attributes include the 'Boolean' flag
                if (_attributes & DataSetEnums::Boolean)
                {
                    // Calculate the total size based on width, height, and length
                    uint64_t size = (uint64_t)_width * (uint64_t)_height * (uint64_t)_length;

                    // Resize the '_vData' vector to hold data for all dimensions and elements
                    _vData.resize(dataDim.getSize() * size);

                    // Initialize the '_vData' vector with zeros
                    memset(_vData.data(), 0, _vData.size() * sizeof(T));

                    // Create a temporary vector 'vData' to store data for a single dimension
                    std::vector<T> vData(dataDim.getSize());

                    // Read data from the NetCDF variable 'dataVar' into the 'vData' vector
                    dataVar.getVar(vData.data());

                    // Populate the '_vData' vector with 1.0 values at specific positions based on 'vData'
                    for (int i = 0; i < dataDim.getSize(); i++)
                        _vData[i * size + vData[i]] = (T)1.0;
                }
                else
                {
                    // Resize the '_vData' vector to hold data for all dimensions without any modification
                    _vData.resize(dataDim.getSize());

                    // Read data from the NetCDF variable 'dataVar' into the '_vData' vector
                    dataVar.getVar(_vData.data());
                }
            }

            // Check if the dataset attributes include the 'Weighted' flag
            if (_attributes & DataSetEnums::Weighted)
            {
                // Create a variable name based on 'dataWeight' and a string representation of the dataset
                vname = "dataWeight" + nstring;

                // Get the NetCDF variable named 'vname' from the NetCDF file 'nfc'
                netCDF::NcVar DataWeightVar = nfc.getVar(vname);

                // Check if the NetCDF variable is null (not found in the file)
                if (DataWeightVar.isNull())
                {
                    // Print an error message indicating that no data weights were found in the NetCDF input file
                    std::cerr << "NcException: DataSet::DataSet: No data weights located in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                // Resize the '_vDataWeight' vector to the number of examples
                _vDataWeight.resize(_examples);

                // Read data from the NetCDF variable into the '_vDataWeight' vector
                DataWeightVar.getVar(_vDataWeight.data());
            }

            // Check if the dataset attributes include the 'Indexed' flag
            if (_attributes & DataSetEnums::Indexed)
            {
                // Create a variable name based on 'index' and a string representation of the dataset
                vname = "index" + nstring;

                // Get the NetCDF variable named 'vname' from the NetCDF file 'nfc'
                netCDF::NcVar indexVar = nfc.getVar(vname);

                // Check if the NetCDF variable is null (not found in the file)
                if (indexVar.isNull())
                {
                    // Print an error message indicating that no indexed data was found in the NetCDF input file
                    std::cerr << "NcException: DataSet::DataSet: No indexed data located in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                // Resize the '_vIndex' vector to the number of examples
                _vIndex.resize(_examples);

                // Read data from the NetCDF variable into the '_vIndex' vector
                indexVar.getVar(_vIndex.data());
            }

            std::cout << "DataSet<T>::DataSet: " << _examples << " examples." << std::endl;
            std::cout << "DataSet<T>::DataSet: " << _uniqueExamples << " unique examples." << std::endl;
        }
        // Catch any exceptions of type 'netCDF::exceptions::NcException'
        catch (netCDF::exceptions::NcException& e)
        {
            // Check if the NetCDF file was not successfully opened
            if (!bOpened)
            {
                // Print an error message indicating a failure to open the NetCDF input file
                std::cout << "Exception: DataSet::DataSet: Error opening NetCDF input file " << fname << std::endl;
            }
            else
            {
                // Print the specific exception message provided by 'e'
                std::cout << "Exception: " << e.what() << std::endl;
            }

            // Set the 'bResult' flag to 'false' to indicate an error condition
            bResult = false;
        }
    }

    // Broadcast a single boolean value from process 0 to all other processes
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // If bResult is false on any process, shut down the GPU and exit with an error code
    if (!bResult)
    {
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Broadcast a string (assuming MPI_Bcast_string broadcasts a string) from process 0 to all other processes
    MPI_Bcast_string(_name);

    // Broadcast a single unsigned integer (32-bit) value from process 0 to all other processes
    MPI_Bcast(&_dataType, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_attributes, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_examples, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_uniqueExamples, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_dimensions, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_width, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_height, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_length, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Broadcast a single unsigned 64-bit integer value from process 0 to all other processes
    MPI_Bcast(&_sparseDataSize, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // If the current process is not the GPU process (with ID 0), resize various vectors to zero length
    if (getGpu()._id != 0)
    {
        _vData.resize(0);
        _vSparseStart.resize(_uniqueExamples, 0);
        _vSparseEnd.resize(_uniqueExamples, 0);
        _vSparseIndex.resize(0);
        _vSparseData.resize(0);
    }

    // If the dataset attributes include the 'Indexed' flag, resize the _vIndex vector and broadcast its data
    if (_attributes & DataSetEnums::Indexed)
    {
        _vIndex.resize(_examples);
        MPI_Bcast(_vIndex.data(), _examples, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    }

    // If the dataset attributes include the 'Weighted' flag, resize the _vDataWeight vector and broadcast its data
    if (_attributes & DataSetEnums::Weighted)
    {
        _vDataWeight.resize(_examples);
        MPI_Bcast(_vDataWeight.data(), _examples, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // If the dataset attributes include the 'Sparse' flag, calculate sparse datapoint counts
    if (_attributes & DataSetEnums::Sparse)
    {
        CalculateSparseDatapointCounts();
    }
}

/// <summary>
/// Rename the dataset.
/// </summary>
/// <param name="name">The new name for the dataset.</param>
/// <returns>True if the renaming is successful, false otherwise.</returns>
template<typename T>
bool DataSet<T>::Rename(const std::string& name) {
    // Update the name of the dataset
    _name = name;
    return true;
}

/// <summary>
/// Calculate sparse datapoint counts for the dataset.
/// </summary>
/// <returns>True if the calculation is successful, false otherwise.</returns>
template<typename T>
bool DataSet<T>::CalculateSparseDatapointCounts() {
    // Check if the dataset is sparse
    if (_attributes & DataSetEnums::Sparse) {
        // Calculate the total number of datapoints
        uint64_t N = _width * _height * _length;

        // Initialize vectors to store datapoint counts
        _vSparseDatapointCount.resize(N);
        _vSparseMaxDatapointCount.resize(N);
        _vSparseMultiDatapointCount.resize(N);

        // Initialize counts to zero
        std::fill(_vSparseDatapointCount.begin(), _vSparseDatapointCount.end(), 0);
        std::fill(_vSparseMaxDatapointCount.begin(), _vSparseMaxDatapointCount.end(), 0);
        std::fill(_vSparseMultiDatapointCount.begin(), _vSparseMultiDatapointCount.end(), 0);

        // Create a vector to count occurrences of datapoints
        std::vector<uint32_t> vCount(N, 0);

        // Create a vector to count examples per unique example
        std::vector<uint32_t> vExampleCount(_uniqueExamples, 0);

        // If the dataset is indexed, count examples per index
        if (_attributes & DataSetEnums::Indexed) {
            for (size_t i = 0; i < _examples; i++) {
                vExampleCount[_vIndex[i]]++;
            }
        }
        else {
            // If not indexed, set all example counts to 1
            std::fill(vExampleCount.begin(), vExampleCount.end(), 1);
        }

        // Loop through unique examples
        for (size_t i = 0; i < _uniqueExamples; i++) {
            uint64_t count = _vSparseEnd[i] - _vSparseStart[i];

            // Loop through indices in the example
            for (size_t j = _vSparseStart[i]; j < _vSparseEnd[i]; j++) {
                try {
                    // Increment the count for the current index
                    vCount.at(_vSparseIndex[j])++;
                }
                catch (std::exception& e) {
                    std::cout << "DataSet::CalculateSparseDatapointCounts: vCount address = " << _vSparseIndex[j] << " >= vCount size = " << N << std::endl;
                    std::rethrow_exception(std::current_exception());
                }
            }

            bool bMulti = false;

            // Loop through indices again to calculate datapoint counts
            for (size_t j = _vSparseStart[i]; j < _vSparseEnd[i]; j++) {
                uint32_t x = _vSparseIndex[j];

                if (vCount[x] > 0) {
                    // Update maximum and multi-datapoint counts
                    _vSparseMaxDatapointCount[x] = std::max(_vSparseMaxDatapointCount[x], vCount[x]);
                    if (vCount[x] > 1)
                        _vSparseMultiDatapointCount[x] += vExampleCount[i];
                    _vSparseDatapointCount[x] += vExampleCount[i] * vCount[x];
                    vCount[x] = 0;
                }
            }
        }

        size_t sz = 0;
        size_t batch = 2048;
        size_t active = 0;

        // Loop through indices to calculate sizes and active counts
        for (size_t i = 0; i < N; i++) {
            size_t size1 = _vSparseDatapointCount[i];
            size1 = std::min(batch, size1);
            active += (_vSparseDatapointCount[i] > 0);
            if (_vSparseMaxDatapointCount[i] > 1) {
                size_t size2 = std::min(_vSparseMaxDatapointCount[i] * batch, batch + (_vSparseMaxDatapointCount[i] - 1) * _vSparseMultiDatapointCount[i]);
                size1 = std::max(size1, size2);
            }
            sz += size1;
        }

        // Calculate sparse density
        _sparseDensity = (double_t)_sparseDataSize / (double_t)(_uniqueExamples * N);
        return true;
    }
    else {
        // If the dataset is not sparse, display an error message
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::CalculateSparseDatapointCounts: Attempt to calculate sparse datapoint counts on non-sparse dataset %s.\n", _name.c_str());
        }
        return false;
    }
}

/// <summary>
/// Generate the sparse transposed matrix for a given batch and layer.
/// </summary>
/// <param name="batch">The batch size for generating the matrix.</param>
/// <param name="pLayer">A pointer to the layer for which the matrix is generated.</param>
/// <returns>True if the matrix generation is successful, false otherwise.</returns>
template<typename T>
bool DataSet<T>::GenerateSparseTransposedMatrix(uint32_t batch, Layer* pLayer) {
    // Check if the dataset is marked as dirty
    if (_bDirty) {
        // Calculate sparse datapoint counts if the dataset is dirty
        CalculateSparseDatapointCounts();
        _bDirty = false;
    }

    // Calculate the maximum size for the matrix based on the data and layer dimensions
    uint64_t NData = _width * _height * _length;
    uint32_t Nx, Ny, Nz, Nw;
    std::tie(Nx, Ny, Nz, Nw) = pLayer->GetLocalDimensions();
    uint64_t NLayer = Nx * Ny * Nz * Nw;
    uint64_t N = max(NData, NLayer);

    // Resize the sparse transposed start vector and allocate GPU buffers if needed
    _vSparseTransposedStart.resize(N);
    if (!_pbSparseTransposedStart)
        _pbSparseTransposedStart.reset(new GpuBuffer<uint32_t>(N));
    if (!_pbSparseTransposedEnd)
        _pbSparseTransposedEnd.reset(new GpuBuffer<uint32_t>(N));

    // Store the batch size and initialize an offset
    _batch = batch;
    uint32_t offset = 0;

    // Loop through the sparse datapoint counts
    for (size_t i = 0; i < _vSparseDatapointCount.size(); i++) {
        _vSparseTransposedStart[i] = offset;
        size_t size1 = _vSparseDatapointCount[i];

        // Limit size1 to the batch size or other constraints
        size1 = std::min((size_t)batch, size1);
        if (_vSparseMaxDatapointCount[i] > 1) {
            size_t size2 = std::min(_vSparseMaxDatapointCount[i] * batch, batch + (_vSparseMaxDatapointCount[i] - 1) * _vSparseMultiDatapointCount[i]);
            size1 = std::max(size1, size2);
        }

        // Update the offset
        offset += size1;
        offset = ((offset + 31) >> 5) << 5;
    }

    // Upload the sparse transposed start vector to the GPU
    _pbSparseTransposedStart->Upload(_vSparseTransposedStart.data());

    // Check if the offset requires reallocation of GPU buffers
    if (offset > _sparseTransposedIndices) {
        _sparseTransposedIndices = offset;
        printf("DataSet::GenerateSparseTransposedMatrix: Allocating %lu bytes for sparse transposed weight gradient index matrix %s.\n", _sparseTransposedIndices * sizeof(uint32_t), _name.c_str());

        // Allocate GPU buffers for the index matrix
        _pbSparseTransposedIndex.reset(new GpuBuffer<uint32_t>(_sparseTransposedIndices));

        // Allocate GPU buffers for the value matrix if the dataset is not boolean or weighted
        if (!(_attributes & DataSetEnums::Boolean) || (_attributes & DataSetEnums::Weighted)) {
            std::cout << std::format("DataSet::GenerateSparseTransposedMatrix: Allocating %lu bytes for sparse transposed weight gradient value matrix %s.\n", _sparseTransposedIndices * sizeof(float), _name.c_str());
            _pbSparseTransposedData.reset(new GpuBuffer<float>(_sparseTransposedIndices));
        }
    }

    return true;
}

/// <summary>
/// Set the denoising flag for the dataset.
/// </summary>
/// <param name="flag">The denoising flag to set.</param>
/// <returns>True if the denoising flag was set successfully, false otherwise.</returns>
template<typename T>
bool DataSet<T>::SetDenoising(bool flag) {
    // Check if the dataset is not sparse
    if (!(_attributes & DataSetEnums::Sparse)) {
        // Print an error message if denoising is attempted on a non-sparse dataset
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetDenoising: Attempt to set denoising on non-sparse data set.\n");
        }
        return false;
    }
    else if (!flag && _bDenoising) {
        // Disable denoising and release the denoising random buffer if denoising is currently enabled
        _pbDenoisingRandom.reset();
        _bDenoising = false;
    }
    else if (flag && !_bDenoising) {
        // Enable denoising and create a new denoising random buffer if denoising is not currently enabled
        _pbDenoisingRandom.reset(new GpuBuffer<float>((uint64_t)_vSparseIndex.size()));
    }
    return true;
}

/// <summary>
/// Set the streaming flag for the dataset.
/// </summary>
/// <param name="flag">The streaming flag to set.</param>
/// <returns>True if the streaming flag was set successfully, false otherwise.</returns>
template<typename T>
bool DataSet<T>::SetStreaming(bool flag) {
    // Check if unified memory is not supported
    if (!getGpu()._bUnifiedMemory) {
        std::cout << std::format("DataSet::SetStreaming: Streaming datasets not supported on GPU %d\n", getGpu()._id);
    }

    // Check if the streaming flag is different from the current value
    if (flag != _bStreaming) {
        // Update the streaming flag and mark the dataset as dirty
        _bStreaming = flag && getGpu()._bUnifiedMemory;
        _bDirty = true;
    }

    return true;
}

/// <summary>
/// Get the current streaming flag for the dataset.
/// </summary>
/// <returns>The current streaming flag.</returns>
template<typename T>
bool DataSet<T>::GetStreaming() {
    return _bStreaming;
}

/// <summary>
/// Generate denoising random data for sparse datasets.
/// </summary>
/// <returns>True if denoising randoms were generated successfully, false otherwise.</returns>
template<typename T>
bool DataSet<T>::GenerateDenoisingData() {
    // Check if the dataset is sparse
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            // Display a message for a non-sparse dataset
            std::cout << std::format("DataSet::GenerateDenoisingData: Attempt to generate denoising randoms on non-sparse data set.\n");
        }
        return false;
    }

    // Generate uniform random data using CUDA curand
    curandGenerateUniform(getGpu()._RNG, _pbDenoisingRandom->_pDevData, _vSparseIndex.size());

    return true;
}

/// <summary>
/// Unshards the dataset based on the current sharding mode.
/// </summary>
/// <returns>True if unsharding was successful, false otherwise.</returns>
template<typename T>
bool DataSet<T>::UnShard() {
    if (_sharding == DataSetEnums::Model) {
        if (_attributes & DataSetEnums::Sparse) {
            // Download sparse data from GPU buffers
            _pbSparseStart->Download(_vSparseStart.data());
            _pbSparseEnd->Download(_vSparseEnd.data());
            _pbSparseIndex->Download(_vSparseIndex.data());
            _pbSparseStart.reset();
            _pbSparseEnd.reset();
            _pbSparseIndex.reset();

            // Download sparse data (if non-boolean) and adjust indices
            if (!(_attributes & DataSetEnums::Boolean)) {
                _pbSparseData->Download(_vSparseData.data());
                _pbSparseData.reset();
            }

            int32_t xmin = ((size_t)_width * (size_t)getGpu()._id) / (size_t)getGpu()._numprocs;
            int32_t xmax = ((size_t)_width * ((size_t)getGpu()._id + 1)) / (size_t)getGpu()._numprocs;
            for (auto& index : _vSparseIndex) {
                index -= xmin;
            }

            // Calculate sparse data count for each unique example
            std::vector<uint32_t> vSparseCount(_uniqueExamples);
            for (uint32_t i = 0; i < _uniqueExamples; i++) {
                vSparseCount[i] = _vSparseEnd[i] - _vSparseStart[i];
            }

            uint64_t datapoints = _vSparseIndex.size();
            // Reduce datapoints and sparse counts across GPUs
            MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : &datapoints, &datapoints, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : vSparseCount.data(), vSparseCount.data(), _uniqueExamples, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

            if (getGpu()._id == 0) {
                std::vector<uint64_t> vTempSparseStart(_uniqueExamples);
                std::vector<uint64_t> vTempSparseEnd(_uniqueExamples);
                std::vector<uint32_t> vTempSparseIndex(datapoints);
                std::vector<T> vTempSparseData;
                if (!(_attributes & DataSetEnums::Boolean)) {
                    vTempSparseData.resize(datapoints);
                }
                vTempSparseStart[0] = 0;
                uint64_t start = 0;

                // Reconstruct sparse data for GPU 0
                for (int i = 0; i < _uniqueExamples; i++) {
                    vTempSparseStart[i] = start;
                    vTempSparseEnd[i] = start;
                    for (uint64_t j = _vSparseStart[i]; j < _vSparseEnd[i]; j++) {
                        vTempSparseIndex[vTempSparseEnd[i]] = _vSparseIndex[vTempSparseEnd[i]];
                        if (!(_attributes & DataSetEnums::Boolean)) {
                            vTempSparseData[vTempSparseEnd[i]] = _vSparseData[vTempSparseEnd[i]];
                        }
                        vTempSparseEnd[i]++;
                    }
                    start += vSparseCount[i];
                }

                // Receive and merge sparse data from other GPUs
                for (uint32_t i = 1; i < getGpu()._numprocs; i++) {
                    uint64_t size;
                    MPI_Status status;
                    MPI_Recv(vSparseCount.data(), _uniqueExamples, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    std::vector<uint32_t> vPeerSparseIndex(size);
                    MPI_Recv(&vPeerSparseIndex, size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, &status);
                    std::vector<T> vPeerSparseData;
                    if (!(_attributes & DataSetEnums::Boolean)) {
                        vPeerSparseData.resize(size);
                        MPI_Recv(vPeerSparseData.data(), size, getMPIDataType(_dataType), i, 0, MPI_COMM_WORLD, &status);
                    }

                    for (uint32_t i = 0; i < _uniqueExamples; i++) {
                        uint64_t start = 0;
                        for (int j = 0; j < vSparseCount[i]; j++) {
                            vTempSparseIndex[vTempSparseEnd[i]] = vPeerSparseIndex[start];
                            if (!(_attributes & DataSetEnums::Boolean)) {
                                vTempSparseData[vTempSparseEnd[i]] = vPeerSparseData[start];
                            }
                            vTempSparseEnd[i]++;
                            start++;
                        }
                    }
                }

                // Update dataset with merged sparse data for GPU 0
                _vSparseStart = vTempSparseStart;
                _vSparseEnd = vTempSparseEnd;
                _vSparseIndex = vTempSparseIndex;
                if (!(_attributes & DataSetEnums::Boolean)) {
                    _vSparseData = vTempSparseData;
                }

                // Upload merged sparse data to GPU buffers
                _pbSparseStart.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
                _pbSparseEnd.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
                _pbSparseIndex.reset(new GpuBuffer<uint32_t>((uint64_t)_vSparseIndex.size(), false, _bStreaming));
                _pbSparseStart->Upload(_vSparseStart.data());
                _pbSparseEnd->Upload(_vSparseEnd.data());
                _pbSparseIndex->Upload(_vSparseIndex.data());
                if (!(_attributes & DataSetEnums::Boolean)) {
                    _pbSparseData.reset(new GpuBuffer<T>((uint64_t)_vSparseData.size(), false, _bStreaming));
                    _pbSparseData->Upload(_vSparseData.data());
                }
            }
            else {
                // Send merged sparse data from other GPUs to GPU 0
                uint64_t size = _vSparseIndex.size();
                MPI_Send(vSparseCount.data(), _uniqueExamples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
                MPI_Send(_vSparseIndex.data(), size, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
                if (!(_attributes & DataSetEnums::Boolean)) {
                    MPI_Send(_vSparseData.data(), size, getMPIDataType(_dataType), 0, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            // Download data from GPU buffer
            _pbData->Download(_vData.data());
            _pbData.reset();

            if (getGpu()._id == 0) {
                std::vector<T> vTempData(_vData);
                _vData.resize(_uniqueExamples * _width);

                // Reconstruct data for GPU 0
                uint32_t xmax = _width / getGpu()._numprocs;
                for (uint64_t i = 0; i < _uniqueExamples; i++) {
                    for (uint64_t j = 0; j < xmax; j++) {
                        _vData[i * _width + j] = vTempData[i * xmax + j];
                    }
                }

                // Receive and merge data from other GPUs
                for (int i = 1; i < getGpu()._numprocs; i++) {
                    int xmin = (i * _width) / getGpu()._numprocs;
                    xmax = ((i + 1) * _width) / getGpu()._numprocs;
                    int slice = xmax - xmin;
                    int size = _uniqueExamples * slice;
                    vTempData.resize(size);
                    MPI_Status status;
                    MPI_Recv(vTempData.data(), size, getMPIDataType(_dataType), i, 0, MPI_COMM_WORLD, &status);
                    for (int j = 0; j < _uniqueExamples; j++) {
                        for (int k = 0; k < slice; k++) {
                            _vData[j * _width + xmin + k] = vTempData[j * slice + k];
                        }
                    }
                }

                // Upload merged data to GPU buffer
                _pbData.reset(new GpuBuffer<T>((uint64_t)_vData.size(), false, _bStreaming));
                _pbData->Upload(_vData.data());

            }
            else {
                // Send data from other GPUs to GPU 0
                MPI_Send(_vData.data(), _vData.size(), getMPIDataType(_dataType), 0, 0, MPI_COMM_WORLD);
            }
        }
    }
    else if (_sharding == DataSetEnums::Data) {
        // Handle Data sharding (to be implemented)
    }

    // Reset the sharding mode to None
    _sharding = DataSetEnums::Sharding::None;

    // Upload indexed data (if present) to GPU buffer
    if (_attributes & DataSetEnums::Indexed) {
        _pbIndex.reset(new GpuBuffer<uint32_t>((uint64_t)_vIndex.size(), false, _bStreaming));
        _pbIndex->Upload(_vIndex.data());
    }

    // Upload weighted data (if present) to GPU buffer
    if (_attributes & DataSetEnums::Weighted) {
        _pbDataWeight.reset(new GpuBuffer<float>((uint64_t)_vDataWeight.size(), false, _bStreaming));
        _pbDataWeight->Upload(_vDataWeight.data());
    }

    return true;
}


/// <summary>
/// Shards the dataset based on the specified sharding mode.
/// </summary>
/// <param name="sharding">The sharding mode to apply.</param>
/// <returns>True if sharding was successful, false otherwise.</returns>
template<typename T>
bool DataSet<T>::Shard(DataSetEnums::Sharding sharding) {
    // Check if the specified sharding mode is the same as the current one
    if (sharding == _sharding)
        return true;

    // Unshard the dataset to prepare for new sharding
    UnShard();

    // Sharding based on the 'Model' mode
    if (sharding == DataSetEnums::Model) {
        _sharding = DataSetEnums::Model;

        // Calculate shard boundaries based on GPU ID
        _minX = ((size_t)_width * (size_t)getGpu()._id) / (size_t)getGpu()._numprocs;
        _maxX = ((size_t)_width * (size_t)(getGpu()._id + 1)) / (size_t)getGpu()._numprocs;

        // Handle sparse dataset sharding
        if (_attributes & DataSetEnums::Sparse) {
            if (getGpu()._id == 0) {
                std::cout << std::format("DataSet<T>::Shard: Model Sharding sparse dataset %s across all GPUs.\n", _name.c_str());

                // Iterate over other GPUs to distribute sparse data
                for (size_t i = 1; i < getGpu()._numprocs; i++) {
                    uint32_t xmin = ((size_t)_width * i) / (size_t)getGpu()._numprocs;
                    uint32_t xmax = ((size_t)_width * (i + 1)) / (size_t)getGpu()._numprocs;

                    // Calculate local sparse data for the current GPU
                    std::vector<uint64_t> vLocalSparseStart(_uniqueExamples);
                    std::vector<uint64_t> vLocalSparseEnd(_uniqueExamples);
                    std::vector<uint32_t> vLocalSparseIndex;
                    std::vector<T> vLocalSparseData;

                    // Iterate over unique examples
                    for (int j = 0; j < _uniqueExamples; j++) {
                        vLocalSparseStart[j] = vLocalSparseIndex.size();
                        for (uint64_t k = _vSparseStart[j]; k < _vSparseEnd[j]; k++) {
                            if ((_vSparseIndex[k] >= xmin) && (_vSparseIndex[k] < xmax)) {
                                vLocalSparseIndex.push_back(_vSparseIndex[k] - xmin);
                                if (!(_attributes & DataSetEnums::Boolean)) {
                                    vLocalSparseData.push_back(_vSparseData[k]);
                                }
                            }
                        }
                        vLocalSparseEnd[j] = vLocalSparseIndex.size();
                    }

                    // Send the local sparse data to other GPUs
                    uint64_t size = vLocalSparseIndex.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseStart.data(), _uniqueExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseEnd.data(), _uniqueExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseIndex.data(), size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD);

                    // Send sparse data if it's not a boolean dataset
                    if (!(_attributes & DataSetEnums::Boolean)) {
                        MPI_Datatype mpiType = getMPIDataType(_dataType);
                        MPI_Send(vLocalSparseData.data(), size, mpiType, i, 0, MPI_COMM_WORLD);
                    }
                }

                // Update local sparse data for GPU 0
                std::vector<uint64_t> vTempSparseStart = _vSparseStart;
                std::vector<uint64_t> vTempSparseEnd = _vSparseEnd;
                std::vector<uint32_t> vTempSparseIndex = _vSparseIndex;
                std::vector<T> vTempSparseData = _vSparseData;
                _vSparseIndex.resize(0);
                _vSparseData.resize(0);
                _vSparseStart.resize(_uniqueExamples);
                _vSparseEnd.resize(_uniqueExamples);

                // Reconstruct local sparse data for GPU 0
                for (uint32_t j = 0; j < _uniqueExamples; j++) {
                    _vSparseStart[j] = _vSparseIndex.size();
                    for (uint64_t k = vTempSparseStart[j]; k < vTempSparseEnd[j]; k++) {
                        if ((vTempSparseIndex[k] >= _minX) && (vTempSparseIndex[k] < _maxX)) {
                            _vSparseIndex.push_back(vTempSparseIndex[k]);
                            if (!(_attributes & DataSetEnums::Boolean)) {
                                _vSparseData.push_back(vTempSparseData[k]);
                            }
                        }
                    }
                    _vSparseEnd[j] = _vSparseIndex.size();
                }
            }
            else {
                // Receive sparse data for other GPUs
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                _vSparseStart.resize(_uniqueExamples);
                _vSparseEnd.resize(_uniqueExamples);
                _vSparseIndex.resize(size);
                MPI_Recv(_vSparseStart.data(), _uniqueExamples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(_vSparseEnd.data(), _uniqueExamples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(_vSparseIndex.data(), size, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, &status);

                // Receive sparse data if it's not a boolean dataset
                if (!(_attributes & DataSetEnums::Boolean)) {
                    MPI_Datatype mpiType = getMPIDataType(_dataType);
                    _vSparseData.resize(size);
                    MPI_Recv(_vSparseData.data(), size, mpiType, 0, 0, MPI_COMM_WORLD, &status);
                }

                // Create GPU buffers for sparse data
                _pbSparseStart.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
                _pbSparseEnd.reset(new GpuBuffer<uint64_t>(_uniqueExamples, false, _bStreaming));
                _pbSparseIndex.reset(new GpuBuffer<uint32_t>((uint64_t)_vSparseIndex.size(), false, _bStreaming));
                _pbSparseStart->Upload(_vSparseStart.data());
                _pbSparseEnd->Upload(_vSparseEnd.data());
                _pbSparseIndex->Upload(_vSparseIndex.data());

                // Upload sparse data to GPU buffers if it's not a boolean dataset
                if (!(_attributes & DataSetEnums::Boolean)) {
                    _pbSparseData.reset(new GpuBuffer<T>((uint64_t)_vSparseData.size(), false, _bStreaming));
                    _pbSparseData->Upload(_vSparseData.data());
                }
            }
        }
        else {
            if (getGpu()._id == 0) {
                printf("DataSet<T>::Shard: Model Sharding dataset %s across all GPUs.\n", _name.c_str());

                // Iterate over other GPUs to distribute data
                for (size_t i = 1; i < getGpu()._numprocs; i++) {
                    uint32_t xmin = ((size_t)_width * i) / (size_t)getGpu()._numprocs;
                    uint32_t xmax = ((size_t)_width * (size_t)(i + 1)) / (size_t)getGpu()._numprocs;
                    uint32_t slice = xmax - xmin;
                    std::vector<T> vLocalData(_uniqueExamples * slice);

                    // Copy local data for the current GPU
                    for (size_t j = 0; j < _uniqueExamples; j++) {
                        for (size_t k = 0; k < slice; k++) {
                            vLocalData[j * slice + k] = _vData[j * _width + xmin + k];
                        }
                    }

                    // Send the local data to other GPUs
                    size_t size = vLocalData.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Datatype mpiType = getMPIDataType(_dataType);
                    MPI_Send(vLocalData.data(), _uniqueExamples * slice, mpiType, i, 0, MPI_COMM_WORLD);
                }

                // Update local data for GPU 0
                std::vector<T> vTempData = _vData;
                uint64_t xmax = _width / getGpu()._numprocs;
                _vData.resize(_uniqueExamples * xmax);

                // Reconstruct local data for GPU 0
                for (uint64_t j = 0; j < _uniqueExamples; j++) {
                    for (uint64_t k = 0; k < xmax; k++) {
                        _vData[j * xmax + k] = vTempData[j * _width + k];
                    }
                }
            }
            else {
                // Receive data for other GPUs
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                _vData.resize(size);
                MPI_Datatype mpiType = getMPIDataType(_dataType);
                MPI_Recv(_vData.data(), size, mpiType, 0, 0, MPI_COMM_WORLD, &status);

                // Create GPU buffer for data
                _pbData.reset(new GpuBuffer<T>(_vData.size(), false, _bStreaming));
                _pbData->Upload(_vData.data());
            }
        }
    }

    // Handle indexed sharding
    if (_attributes & DataSetEnums::Indexed) {
        _pbIndex.reset(new GpuBuffer<uint32_t>((uint64_t)_vIndex.size(), false, _bStreaming));
        _pbIndex->Upload(_vIndex.data());
    }
    return true;
}

/// <summary>
/// Saves a DataSet to a NetCDF file.
/// </summary>
/// <param name="fname">The name of the NetCDF file.</param>
/// <returns>True if the dataset was successfully saved, false otherwise.</returns>
template<typename T>
bool DataSet<T>::SaveNetCDF(const std::string& fname) {
    bool bResult = true;

    // Store the old sharding mode and unshard the dataset
    DataSetEnums::Sharding oldSharding = _sharding;
    UnShard();

    // Check if the current process has ID 0
    if (getGpu()._id == 0) {
        try {
            // Create or open the NetCDF file for writing
            netCDF::NcFile nfc(fname, netCDF::NcFile::replace);

            // Set the 'datasets' attribute to 1 (indicating one dataset)
            nfc.putAtt("datasets", netCDF::ncUint, 1);

            // Write the dataset to the NetCDF file and store the result
            bool bResult = WriteNetCDF(nfc, fname, 0);

            // Handle failure to write dataset
            if (!bResult) {
                std::cerr << "SaveNetCDF: Unable to write dataset to NetCDF file " << fname << '\n';
            }
        }
        catch (const netCDF::exceptions::NcException& e) {
            // Handle NetCDF exceptions when creating or writing the file
            std::cerr << "SaveNetCDF: Unable to create or write to NetCDF output file " << fname << '\n';
            std::cerr << e.what() << '\n';
            bResult = false;
        }
    }

    // Broadcast the result flag to all processes
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Handle failure by shutting down GPU and throwing an exception
    if (!bResult) {
        getGpu().Shutdown();
        throw std::runtime_error("SaveNetCDF operation failed.");
    }

    // Restore the old sharding mode
    Shard(oldSharding);

    return bResult;
}


/// <summary>
/// Writes a DataSet to a NetCDF file.
/// </summary>
/// <param name="nfc">An open NetCDF file object.</param>
/// <param name="fname">The name of the NetCDF file.</param>
/// <param name="n">The dataset index.</param>
/// <returns>True if the dataset was successfully written, false otherwise.</returns>
template<typename T>
bool DataSet<T>::WriteNetCDF(netCDF::NcFile& nfc, const std::string& fname, const uint32_t n) {
    bool bResult = true;
    try {
        if (getGpu()._id == 0) {
            std::string nstring = std::to_string(n);

            // Helper function to handle attribute creation and error reporting
            auto createAttribute = [&](const std::string& attrName, const std::string& errMsg) {
                netCDF::NcGroupAtt attribute = nfc.putAtt(attrName, _name);
                if (attribute.isNull()) {
                    std::cerr << "NcException: " << errMsg << " " << fname << " (" << __FILE__ << ", " << __LINE__ << ")" << '\n';
                    bResult = false;
                }
                };

            // Helper function to handle dimension creation and error reporting
            auto createDimension = [&](const std::string& dimName, size_t dimSize, const std::string& errMsg) {
                netCDF::NcDim dimension = nfc.addDim(dimName, dimSize);
                if (dimension.isNull()) {
                    std::cerr << "NcException: " << errMsg << " " << fname << " (" << __FILE__ << ", " << __LINE__ << ")" << '\n';
                    bResult = false;
                }
                };

            // Helper function to handle variable creation and error reporting
            auto createVariable = [&](const std::string& varName, const std::string& varType, const std::string& dimName, const void* data, const std::string& errMsg) {
                netCDF::NcVar variable = nfc.addVar(varName, varType, dimName);
                if (variable.isNull()) {
                    std::cerr << "NcException: " << errMsg << " " << fname << " (" << __FILE__ << ", " << __LINE__ << ")" << '\n';
                    bResult = false;
                }
                else {
                    variable.putVar(data);
                }
                };

            // Create attributes
            createAttribute("name" + nstring, "Failed to write dataset name to NetCDF file");
            createAttribute("attributes" + nstring, "Failed to write dataset attributes to NetCDF file");
            createAttribute("kind" + nstring, "Failed to write dataset kind to NetCDF file");
            createAttribute("datatype" + nstring, "Failed to write dataset type to NetCDF file");
            createAttribute("dimensions" + nstring, "Failed to write dataset dimensions to NetCDF file");
            createAttribute("width" + nstring, "Failed to write dataset width to NetCDF file");

            if (_dimensions > 1) {
                createAttribute("height" + nstring, "Failed to write dataset height to NetCDF file");

                if (_dimensions > 2) {
                    createAttribute("length" + nstring, "Failed to write dataset length to NetCDF file");
                }
            }

            // Create dimensions
            createDimension("uniqueExamplesDim" + nstring, static_cast<size_t>(_uniqueExamples), "Failed to write dataset unique example count to NetCDF file");
            createDimension("examplesDim" + nstring, static_cast<size_t>(_examples), "Failed to write dataset example count to NetCDF file");

            if (_attributes & DataSetEnums::Sparse) {
                createDimension("sparseDataDim" + nstring, _vSparseIndex.size(), "Failed to write dataset sparse datapoint count to NetCDF file");
                createVariable("sparseStart" + nstring, "uint", "uniqueExamplesDim" + nstring, _vSparseStart.data(), "Failed to write dataset sparse start variable to NetCDF file");
                createVariable("sparseEnd" + nstring, "uint", "uniqueExamplesDim" + nstring, _vSparseEnd.data(), "Failed to write dataset sparse end variable to NetCDF file");
                createVariable("sparseIndex" + nstring, "uint64", "sparseDataDim" + nstring, _vSparseIndex.data(), "Failed to write dataset sparse index variable to NetCDF file");

                if (!(_attributes & DataSetEnums::Boolean)) {
                    netCDF::NcType sparseType = getNetCDFDataType(_dataType);
                    createVariable("sparseData" + nstring, sparseType.getName(), "sparseDataDim" + nstring, _vSparseData.data(), "Failed to write dataset sparse data variable to NetCDF file");
                }
            }

            if (_attributes & DataSetEnums::Weighted) {
                createVariable("dataWeight" + nstring, "float", "uniqueExamplesDim" + nstring, _vDataWeight.data(), "Failed to write data weights to NetCDF file");
            }

            if (_attributes & DataSetEnums::Indexed) {
                createVariable("index" + nstring, "uint32", "examplesDim" + nstring, _vIndex.data(), "Failed to create dataset index variable to NetCDF file");
            }
        }
    }
    catch (netCDF::exceptions::NcException& e) {
        std::cout << e.what() << '\n';
        bResult = false;
    }
    return bResult;
}

/// <summary>
/// Destructor for the DataSet class.
/// </summary>
template<typename T>
DataSet<T>::~DataSet() {
}

/// <summary>
/// Saves data to a NetCDF file and returns a boolean indicating success.
/// </summary>
/// <param name="fname">The path to the NetCDF file.</param>
/// <param name="vDataSet">A vector of DataSetBase pointers containing data to be saved.</param>
/// <returns>True if the data was successfully saved, false otherwise.</returns>
bool SaveNetCDF(const std::string& fname, std::vector<DataSetBase*>& vDataSet) {
    // Initialize a vector to store sharding information
    std::vector<DataSetEnums::Sharding> vSharding;
    vSharding.reserve(vDataSet.size());

    // Iterate over data sets and collect sharding information
    for (auto& dataSet : vDataSet) {
        vSharding.push_back(dataSet->_sharding);
        dataSet->UnShard();
    }

    // Get the MPI rank
    int mpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    // Initialize result and opened flag
    bool bResult = true;
    bool bOpened = false;

    try {
        // Only the MPI rank 0 process will perform NetCDF operations
        if (mpiRank == 0) {
            // Create or open the NetCDF file for writing
            netCDF::NcFile nfc(fname, netCDF::NcFile::replace);
            bOpened = true;

            // Create and write the 'datasets' attribute
            netCDF::NcGroupAtt datasetsAtt = nfc.putAtt("datasets", netCDF::ncUint, static_cast<unsigned int>(vDataSet.size()));
            if (datasetsAtt.isNull()) {
                throw std::runtime_error("SaveNetCDF: Unable to write datasets attribute to NetCDF file " + fname);
            }

            // Iterate over data sets and write them to the NetCDF file
            for (uint32_t i = 0; i < vDataSet.size(); i++) {
                if (!vDataSet[i]->WriteNetCDF(nfc, fname, i)) {
                    throw std::runtime_error("SaveNetCDF: Unable to write dataset to NetCDF file " + fname);
                }
            }
        }
    }
    catch (const netCDF::exceptions::NcException& e) {
        // Handle NetCDF exceptions
        if (!bOpened) {
            std::cout << "SaveNetCDF: Unable to create NetCDF output file " << fname << '\n';
        }
        else {
            std::cout << e.what() << '\n';
        }
        bResult = false;
    }

    // Broadcast the result flag to all processes
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Check if broadcast failed
    if (!bResult) {
        // Shutdown GPU and report error
        getGpu().Shutdown();
        std::cerr << "Error: The MPI broadcast failed." << '\n';
    }

    // Restore sharding information to the data sets
    for (uint32_t i = 0; i < vDataSet.size(); i++) {
        vDataSet[i]->Shard(vSharding[i]);
    }

    // Return the result
    return bResult;
}

/// <summary>
/// Loads data from a NetCDF file and returns a vector of DataSetBase pointers.
/// </summary>
/// <param name="fname">The path to the NetCDF file.</param>
/// <returns>A vector of DataSetBase pointers containing loaded data.</returns>
std::vector<DataSetBase*> LoadNetCDF(const std::string& fname) {
    // Initialize vectors and a flag
    std::vector<DataSetBase*> vDataSet;
    std::vector<DataSetEnums::DataType> vDataType;
    bool bResult = true;

    // Check if the current process has ID 0
    if (getGpu()._id == 0) {
        try {
            // Open the NetCDF file for reading
            netCDF::NcFile rnc(fname, netCDF::NcFile::read);

            // Get the 'datasets' attribute
            if (!rnc.getAtt("datasets").isNull()) {
                uint32_t datasets;
                rnc.getAtt("datasets").getValues(&datasets);

                // Iterate over datasets
                for (uint32_t i = 0; i < datasets; i++) {
                    // Create the attribute name
                    std::string vname = "dataType" + std::to_string(i);

                    // Get the 'dataType' attribute
                    if (!rnc.getAtt(vname).isNull()) {
                        uint32_t dataType;
                        rnc.getAtt(vname).getValues(&dataType);

                        // Check and add valid data types to the vector
                        switch (dataType) {
                        case DataSetEnums::UInt:
                        case DataSetEnums::Int:
                        case DataSetEnums::LLInt:
                        case DataSetEnums::ULLInt:
                        case DataSetEnums::Float:
                        case DataSetEnums::Double:
                        case DataSetEnums::RGB8:
                        case DataSetEnums::RGB16:
                        case DataSetEnums::UChar:
                        case DataSetEnums::Char:
                            vDataType.push_back(static_cast<DataSetEnums::DataType>(dataType));
                            break;
                        default:
                            // Handle invalid data type
                            std::cerr << "LoadNetCDF: Invalid data type in binary input file " << fname << '\n';
                        }
                    }
                    else {
                        // Handle missing attribute
                        std::cerr << "NcException: LoadNetCDF: No " << vname << " attribute located in NetCDF input file " << fname << '\n';
                    }
                }
            }
            else {
                // Handle missing 'datasets' attribute
                std::cerr << "NcException: LoadNetCDF: No datasets count supplied in NetCDF input file " << fname << '\n';
            }
        }
        catch (const netCDF::exceptions::NcException& e) {
            // Handle NetCDF exception
            std::cerr << "NcException: LoadNetCDF: " << e.what() << '\n';
            bResult = false;
        }
    }

    // Broadcast the result flag to all processes
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Check if broadcast failed
    if (!bResult) {
        // Shutdown GPU and report error
        getGpu().Shutdown();
        std::cerr << "Error: The MPI broadcast failed." << '\n';
    }

    // Get the size of the data type vector and broadcast it
    uint32_t size = vDataType.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    vDataType.resize(size);
    MPI_Bcast(vDataType.data(), size, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Load DataSetBase objects based on data types
    for (int i = 0; i < vDataType.size(); i++) {
        DataSetBase* pDataSet = nullptr;

        if (getGpu()._id == 0) {
            // Display a message when loading a data set
            std::cout << "LoadNetCDF: Loading " << vDataType[i] << " data set" << '\n';
        }

        // Create DataSetBase objects based on data types
        switch (vDataType[i]) {
        case DataSetEnums::UInt:
            pDataSet = new DataSet<uint32_t>(fname, i);
            break;
        case DataSetEnums::Int:
            pDataSet = new DataSet<long>(fname, i);
            break;
        case DataSetEnums::Float:
            pDataSet = new DataSet<float>(fname, i);
            break;
        case DataSetEnums::Double:
            pDataSet = new DataSet<double>(fname, i);
            break;
        case DataSetEnums::Char:
            pDataSet = new DataSet<char>(fname, i);
            break;
        case DataSetEnums::UChar:
        case DataSetEnums::RGB8:
            pDataSet = new DataSet<uint8_t>(fname, i);
            break;
        default:
            // Handle invalid data type
            std::cerr << "LoadNetCDF: invalid dataset type in binary input file " << fname << '\n';
            getGpu().Shutdown();
            std::exit(-1);
        }

        // Add the loaded DataSetBase pointer to the vector
        vDataSet.push_back(pDataSet);
    }

    // Return the vector of loaded DataSetBase pointers
    return vDataSet;
}

/// <summary>
/// Loads image data from the specified file.
/// </summary>
/// <param name="fname">The path to the image file.</param>
/// <returns>A vector of DataSetBase pointers containing loaded data.</returns>
std::vector<DataSetBase*> LoadImageData(const std::string& fname) {
    // TODO: Implement the loading logic
    return std::vector<DataSetBase*>();
}

/// <summary>
/// Loads CSV data from the specified file.
/// </summary>
/// <param name="fname">The path to the CSV file.</param>
/// <returns>A vector of DataSetBase pointers containing loaded data.</returns>
std::vector<DataSetBase*> LoadCSVData(const std::string& fname) {
    // TODO: Implement the loading logic
    return std::vector<DataSetBase*>();
}

/// <summary>
/// Loads JSON data from the specified file.
/// </summary>
/// <param name="fname">The path to the JSON file.</param>
/// <returns>A vector of DataSetBase pointers containing loaded data.</returns>
std::vector<DataSetBase*> LoadJSONData(const std::string& fname) {
    // TODO: Implement the loading logic
    return std::vector<DataSetBase*>();
}

/// <summary>
/// Loads audio data with the specified name.
/// </summary>
/// <param name="name">The name of the audio data.</param>
/// <returns>A vector of DataSetBase pointers containing loaded data.</returns>
std::vector<DataSetBase*> LoadAudioData(const std::string& name) {
    // TODO: Implement the loading logic
    return std::vector<DataSetBase*>();
}

/// <summary>
/// Loads text data with the specified name.
/// </summary>
/// <param name="name">The name of the text data.</param>
/// <returns>A vector of DataSetBase pointers containing loaded data.</returns>
std::vector<DataSetBase*> LoadTextData(const std::string& name) {
    // TODO: Implement the loading logic
    return std::vector<DataSetBase*>();
}
