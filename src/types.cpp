#include "gpuTypes.h"
#include "types.h"
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>
#include <iostream>
#include <map>
#include "enum.h"
#include "layer.h"
#include "mpi.h"
#include "ncDim.h"
#include "ncDouble.h"
#include "ncException.h"
#include "ncFile.h"
#include "ncFloat.h"
#include "ncGroupAtt.h"
#include "ncInt.h"
#include "ncInt64.h"
#include "ncType.h"
#include "ncUint.h"
#include "ncUint64.h"
#include "ncVar.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <cstdint>
#include <exception>
#include <format>
#include <iosfwd>
#include <new>
#include <utility>
#include <curand.h>

template class DataSet<float>;
template class DataSet<double>;
template class DataSet<unsigned char>;
template class DataSet<char>;
template class DataSet<uint32_t>;
template class DataSet<uint64_t>;
template class DataSet<int32_t>;
template class DataSet<int64_t>;

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

std::ostream& operator<< (std::ostream& out, const TrainingMode& e)
{
    out << sTrainingModeMap[e];
    return out;
}

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

std::ostream& operator<< (std::ostream& out, const ErrorFunction& e)
{
    out << sErrorFunctionMap[e];
    return out;
}

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

std::ostream& operator<< (std::ostream& out, const Activation& a)
{
    out << sActivationMap[a];
    return out;
}

static std::map<WeightInitialization, std::string> sWeightInitializationMap = {
    {WeightInitialization::Xavier,           "Xavier"},
    {WeightInitialization::CaffeXavier,      "CaffeXavier"},
    {WeightInitialization::Gaussian,         "Gaussian"},
    {WeightInitialization::Uniform,          "Uniform"},
    {WeightInitialization::UnitBall,         "UnitBall"},
    {WeightInitialization::Constant,         "Constant"},
    {WeightInitialization::SELU,             "SELU"}
};

std::ostream& operator<< (std::ostream& out, const WeightInitialization& w)
{
    out << sWeightInitializationMap[w];
    return out;
}

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

std::ostream& operator<< (std::ostream& out, const PoolingFunction& a)
{
    out << sPoolingFunctionMap[a];
    return out;
}

static std::map<DataSetEnums::Kind, std::string> sKindMap = {
    {DataSetEnums::Numeric, "Numeric"},
    {DataSetEnums::Image,   "Image"},
    {DataSetEnums::Audio,   "Audio"},
    {DataSetEnums::Text,    "Text"}
};

std::ostream& operator<< (std::ostream& out, DataSetEnums::Kind& k)
{
    out << sKindMap[k];
    return out;
}


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

std::ostream& operator<< (std::ostream& out, DataSetEnums::Attributes& a)
{
    out << sAttributesMap[a];
    return out;
}

static std::map<DataSetEnums::Sharding, std::string> sShardingMap = {
    {DataSetEnums::None,  "None"},
    {DataSetEnums::Model, "Model"},
    {DataSetEnums::Data,  "Data"}
};

std::ostream& operator<< (std::ostream& out, DataSetEnums::Sharding& s)
{
    out << sShardingMap[s];
    return out;
}


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

std::ostream& operator<< (std::ostream& out, DataSetEnums::DataType& t)
{
    out << sDataTypeMap[t];
    return out;
}

static MPI_Datatype getMPIDataType(DataSetEnums::DataType datatype)
{
    MPI_Datatype mpiType;

    switch (datatype)
    {
    case DataSetEnums::UInt:
        mpiType = MPI_UINT32_T;
        std::cout << "Mapping custom data type UInt to MPI_UINT32_T";
        break;

    case DataSetEnums::Int:
        mpiType = MPI_INT32_T;
        std::cout << "Mapping custom data type Int to MPI_INT32_T";
        break;

    case DataSetEnums::ULLInt:
        mpiType = MPI_UINT64_T;
        std::cout << "Mapping custom data type ULLInt to MPI_UINT64_T";
        break;

    case DataSetEnums::LLInt:
        mpiType = MPI_INT64_T;
        std::cout << "Mapping custom data type LLInt to MPI_INT64_T";
        break;

    case DataSetEnums::Float:
        mpiType = MPI_FLOAT;
        std::cout << "Mapping custom data type Float to MPI_FLOAT";
        break;

    case DataSetEnums::Double:
        mpiType = MPI_DOUBLE;
        std::cout << "Mapping custom data type Double to MPI_DOUBLE";
        break;

    default:
        mpiType = MPI_DATATYPE_NULL;
        break;
    }

    return mpiType;
}

static netCDF::NcType getNetCDFDataType(DataSetEnums::DataType datatype)
{
    switch (datatype)
    {
    case DataSetEnums::UInt:
        return netCDF::ncUint;
        std::cout << "Mapping custom data type UInt to netCDF data type ncUint.";
    case DataSetEnums::Int:
        return netCDF::ncInt;
        std::cout << "Mapping custom data type Int to netCDF data type ncInt.";
        break;

    case DataSetEnums::ULLInt:
        return netCDF::ncUint64;
        std::cout << "Mapping custom data type ULLInt to netCDF data type ncUint64.";
        break;

    case DataSetEnums::LLInt:
        return netCDF::ncInt64;
        std::cout << "Mapping custom data type LLInt to netCDF data type ncInt64.";
        break;

    case DataSetEnums::Float:
        return netCDF::ncFloat;
        std::cout << "Mapping custom data type Float to netCDF data type ncFloat.";
        break;

    case DataSetEnums::Double:
        return netCDF::ncDouble;
        std::cout << "Mapping custom data type Double to netCDF data type ncDouble.";
        break;

    default:
        std::cerr << "Unsupported DataType: " << datatype;
        throw std::invalid_argument("Unsupported DataType");
    }
}

int MPI_Bcast_string(std::string& s)
{

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int length = s.size();

    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> buff(length + 1);

    if (MPI_Bcast(buff.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        std::cerr << "MPI_Bcast_string failed on rank " << rank;
    }

    buff[length] = '\0';

    s = buff.data();

    std::cout << "MPI_Bcast_string successful on rank " << rank;

    return MPI_SUCCESS;
}

DataSetDimensions::DataSetDimensions() :
    
    DataSetDimensions(1, 1, 1)
{}

DataSetDimensions::DataSetDimensions(uint32_t width, uint32_t height, uint32_t length) :
    _dimensions(0),
    _width(width),
    _height(height),
    _length(length)
{
    if (width > 1)
    {
        ++_dimensions;
    }

    if (height > 1)
    {
        ++_dimensions;
    }

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

    return dataset;
}

DataSetBase* createDataSet(const DataSetDescriptor& descriptor)
{

    DataSetBase* dataset = nullptr;

    using DataSetEnums::DataType;

    switch (descriptor._dataType) {
    case DataType::UInt:
        dataset = createDataSet<uint32_t>(descriptor);
        break;
    case DataType::Int:
        dataset = createDataSet<int>(descriptor);
        break;
    case DataType::Float:
        dataset = createDataSet<float>(descriptor);
        break;
    case DataType::Double:
        dataset = createDataSet<double>(descriptor);
        break;
    case DataType::Char:
        dataset = createDataSet<char>(descriptor);
        break;
    case DataType::UChar:
    case DataType::RGB8:
        dataset = createDataSet<uint8_t>(descriptor);
        break;
    default:
        std::cerr << "Unsupported data type: " << descriptor._dataType
            << ". DataType must be one of: UInt, Int, Float, Double, Char, UChar, RGB8";
        std::stringstream msg;
        msg << "Unsupported data type: " << descriptor._dataType
            << ". DataType must be one of: UInt, Int, Float, Double, Char, UChar, RGB8";
        throw std::runtime_error(msg.str());
    }

    if (dataset) {
        std::cout << "Created DataSet of type " << descriptor._dataType;
    }

    return dataset;
}


DataSetBase::DataSetBase() :
    
    _name(""),
    
    _attributes(DataSetEnums::None),
    
    _examples(0),
    
    _uniqueExamples(0),
    
    _dimensions(0),
    
    _width(0),
    
    _height(0),
    
    _length(0),
    
    _stride(0),
    
    _sharding(DataSetEnums::Sharding::None),
    
    _minX(0),
    
    _maxX(0),
    
    _sparseDataSize(0),
    
    _sparseTransposedIndices(0),
    
    _sparseDensity(0),
    
    _bDenoising(false),
    
    _pbSparseStart(),
    
    _pbSparseEnd(),
    
    _pbSparseIndex(),
    
    _pbIndex(),
    
    _pbSparseTransposedStart(),
    
    _pbSparseTransposedEnd(),
    
    _pbSparseTransposedIndex(),
    
    _pbSparseTransposedData(),
    
    _batch(0),
    
    _pbDenoisingRandom(),
    
    _bStreaming(false),
    
    _bIndexed(false),
    
    _bDirty(true)
{
}

DataSetBase::DataSetBase(const std::string& name, DataSetEnums::DataType dataType, uint32_t examples,
    uint32_t uniqueExamples, const DataSetDimensions& datasetDim) :

    _name(name),

    _dataType(dataType),

    _attributes(DataSetEnums::None),

    _examples(examples),

    _uniqueExamples(uniqueExamples),

    _localExamples(examples),

    _dimensions(datasetDim._dimensions),

    _width(datasetDim._width),

    _height(datasetDim._height),

    _length(datasetDim._length),

    _stride(0),

    _sharding(DataSetEnums::Sharding::None),

    _minX(0),

    _maxX(0),

    _sparseDataSize(0),

    _sparseTransposedIndices(0),

    _sparseDensity(0),

    _bDenoising(false),

    _pbSparseStart(),

    _pbSparseEnd(),

    _pbSparseIndex(),

    _pbIndex(),

    _pbSparseTransposedStart(),

    _pbSparseTransposedEnd(),

    _pbSparseTransposedIndex(),

    _pbSparseTransposedData(),

    _batch(0),

    _pbDenoisingRandom(),

    _bStreaming(false),

    _bIndexed(false),

    _bDirty(true)
{
}

DataSetBase::~DataSetBase() {}

DataSetDimensions DataSetBase::GetDimensions()
{
    return DataSetDimensions(_width, _height, _length);
}

template<typename T>
auto DataSet<T>::getMemoryUsage() -> std::vector<std::tuple<uint64_t, uint64_t>>
{

    uint64_t cpuMemory = 0;
    uint64_t gpuMemory = 0;

    if (_attributes & DataSetEnums::Sparse)
    {
        cpuMemory += static_cast<uint64_t>(_uniqueExamples) * static_cast<uint64_t>(2) * static_cast<uint64_t>(sizeof(uint64_t));
        gpuMemory += static_cast<uint64_t>(_uniqueExamples) * static_cast<uint64_t>(2) * static_cast<uint64_t>(sizeof(uint64_t));
        cpuMemory += _vSparseIndex.size() * sizeof(uint32_t);
        gpuMemory += _vSparseIndex.size() * sizeof(uint32_t);

        if (!(_attributes & DataSetEnums::Boolean))
        {
            cpuMemory += _vSparseData.size() * sizeof(T);
            gpuMemory += _vSparseData.size() * sizeof(T);
        }
    }
    else
    {
        cpuMemory += _vData.size() * sizeof(T);
        gpuMemory += _vData.size() * sizeof(T);
    }

    if (_bIndexed)
    {
        cpuMemory += _examples * sizeof(uint32_t);
        gpuMemory += _examples * sizeof(uint32_t);
    }

    std::vector<std::tuple<uint64_t, uint64_t>> vResult(getGpu()._numprocs);

    vResult[getGpu()._id] = std::make_tuple(cpuMemory, gpuMemory);

    auto resultSpan = std::span(vResult.data(), vResult.size());
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, resultSpan.data(), sizeof(std::tuple<uint64_t, uint64_t>), MPI_BYTE, MPI_COMM_WORLD);

    std::cout << "CPU Memory Usage: " << cpuMemory << " bytes";
    std::cout << "GPU Memory Usage: " << gpuMemory << " bytes";

    return vResult;
}

template<typename T>
DataSet<T>::DataSet(uint32_t examples, const DataSetDimensions& dim, const std::string& name) :
    DataSetBase(name, DataSetEnums::getDataType<T>(), examples, examples, dim)
{
    _sparseDensity = 1.0f;

    _stride = _width * _height * _length;

    _vData.resize(_stride * _examples);

    _pbData.reset(new GpuBuffer<T>(_vData.size(), false, _bStreaming));
}

template<typename T>
DataSet<T>::DataSet(uint32_t examples, uint32_t uniqueExamples, const DataSetDimensions& dim,
    const std::string& name) :
    DataSetBase(name, DataSetEnums::getDataType<T>(), examples, uniqueExamples, dim)
{
    _sparseDensity = 1.0f;

    _stride = _width * _height * _length;

    _attributes = DataSetEnums::Attributes::Indexed;

    _bIndexed = true;

    _vData.resize(_stride * _uniqueExamples);
    _vIndex.resize(_examples, 0);

    _pbData.reset(new GpuBuffer<T>(_vData.size(), false, _bStreaming));
    _pbIndex.reset(new GpuBuffer<uint32_t>(_vIndex.size(), false, _bStreaming));
}

template<typename T>
DataSet<T>::DataSet(uint32_t examples, float sparseDensity, const DataSetDimensions& dim,
    bool isWeighted, const std::string& name) :
    DataSet(examples, examples,
        (size_t)(((double)(dim._width* dim._height* dim._length* examples))* sparseDensity), dim, false,
        isWeighted, name)
{
    _attributes = DataSetEnums::Attributes::Sparse;

    if (isWeighted) {
        _attributes |= DataSetEnums::Attributes::Weighted;
    }
}

template<typename T>
DataSet<T>::DataSet(uint32_t examples, uint32_t uniqueExamples, size_t sparseDataSize,
    const DataSetDimensions& dim, bool isIndexed, bool isWeighted, const std::string& name) :
    DataSetBase(name, DataSetEnums::getDataType<T>(), examples, uniqueExamples, dim)
{
    _attributes = DataSetEnums::Attributes::Sparse;
    _sparseDataSize = sparseDataSize;

    _vSparseStart.resize(_uniqueExamples, 0);
    _vSparseEnd.resize(_uniqueExamples, 0);
    _vSparseData.resize(_sparseDataSize);
    _vSparseIndex.resize(_sparseDataSize, 0);

    size_t sparseStride = (_sparseDataSize + _uniqueExamples - 1) / _uniqueExamples;

    _vSparseStart[0] = 0;
    _vSparseEnd[0] = sparseStride;
    for (uint32_t i = 1; i < _uniqueExamples; ++i)
    {
        _vSparseStart[i] = _vSparseEnd[static_cast<std::vector<size_t, std::allocator<size_t>>::size_type>(i) - 1];
        _vSparseEnd[i] = _vSparseStart[i] + sparseStride;
    }

    _pbSparseStart.reset(new GpuBuffer<uint64_t>(_vSparseStart.size(), false, _bStreaming));
    _pbSparseEnd.reset(new GpuBuffer<uint64_t>(_vSparseEnd.size(), false, _bStreaming));
    _pbSparseData.reset(new GpuBuffer<T>(_vSparseData.size(), false, _bStreaming));
    _pbSparseIndex.reset(new GpuBuffer<uint32_t>(_vSparseIndex.size(), false, _bStreaming));

    if (isIndexed) {
        _attributes |= DataSetEnums::Attributes::Indexed;
        _bIndexed = true;
        _vIndex.resize(_examples, 0);
        _pbIndex.reset(new GpuBuffer<uint32_t>(_vIndex.size(), false, _bStreaming));
    }

    if (isWeighted)
    {
        _attributes |= DataSetEnums::Attributes::Weighted;
        _vDataWeight.resize(_examples);
        _pbDataWeight.reset(new GpuBuffer<float>(_vDataWeight.size(), false, _bStreaming));
    }
}

template<typename T>
void DataSet<T>::LoadDenseData(const void* srcData) {
    const T* srcDataTyped = static_cast<const T*>(srcData);

    if (_attributes & DataSetEnums::Attributes::Sparse) {
        throw std::runtime_error("Cannot set dense data on a sparse DataSet");
    }
    else {
        std::ranges::copy(srcDataTyped, srcDataTyped + _vData.size(), _vData.data());

        _pbData->Upload(_vData.data());
    }
}

template<typename T>
void DataSet<T>::CopyDenseData(const void* srcData) {

    std::span<const T> srcDataSpan(static_cast<const T*>(srcData), _vData.size());

    if (_attributes & DataSetEnums::Attributes::Sparse) {
        std::cerr << "Cannot set dense data on a sparse DataSet";

        throw std::runtime_error("Cannot set dense data on a sparse DataSet");
    }
    else {
        std::ranges::copy(srcDataSpan, _vData.begin());
    }
}

template<typename T>
void DataSet<T>::LoadSparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd,
    const void* srcSparseData, const uint32_t* srcSparseIndex) {
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    if (_attributes & DataSetEnums::Attributes::Sparse) {
        if (srcSparseStart[0] != 0) {
            throw std::runtime_error("Sparse data should be zero-indexed; srcSparseStart[0] != 0");
        }

        uint64_t dataLength = srcSparseEnd[_uniqueExamples - 1];

        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size()) {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size() << " Required: "
                << dataLength;
            throw std::length_error(msg.str());
        }

        std::ranges::copy(srcSparseStart, srcSparseStart + _uniqueExamples, _vSparseStart.data());
        std::ranges::copy(srcSparseEnd, srcSparseEnd + _uniqueExamples, _vSparseEnd.data());

        std::ranges::copy(srcSparseDataTyped, srcSparseDataTyped + dataLength, _vSparseData.data());
        std::ranges::copy(srcSparseIndex, srcSparseIndex + dataLength, _vSparseIndex.data());

        _pbSparseStart->Upload(_vSparseStart.data());
        _pbSparseEnd->Upload(_vSparseEnd.data());
        _pbSparseIndex->Upload(_vSparseIndex.data());
        _pbSparseData->Upload(_vSparseData.data());
    }
    else {
        throw std::runtime_error("Cannot set sparse data on a non-sparse DataSet");
    }
}

template<typename T>
void DataSet<T>::CopySparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd,
    const void* srcSparseData, const uint32_t* srcSparseIndex) {
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    if (_attributes & DataSetEnums::Attributes::Sparse) {
        if (srcSparseStart[0] != 0) {
            throw std::runtime_error("Sparse data should be zero-indexed; srcSparseStart[0] != 0");
        }

        uint64_t dataLength = srcSparseEnd[_uniqueExamples - 1];

        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size()) {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size() << " Required: "
                << dataLength;
            throw std::length_error(msg.str());
        }

        std::ranges::copy(srcSparseStart, srcSparseStart + _uniqueExamples, _vSparseStart.data());
        std::ranges::copy(srcSparseEnd, srcSparseEnd + _uniqueExamples, _vSparseEnd.data());

        std::ranges::copy(srcSparseDataTyped, srcSparseDataTyped + dataLength, _vSparseData.data());
        std::ranges::copy(srcSparseIndex, srcSparseIndex + dataLength, _vSparseIndex.data());
    }
    else {
        throw std::runtime_error("Cannot set sparse data on a non-sparse DataSet");
    }
}

template<typename T>
void DataSet<T>::LoadSparseData(const long* srcSparseStart, const long* srcSparseEnd,
    const void* srcSparseData, const long* srcSparseIndex) {
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    if (_attributes & DataSetEnums::Attributes::Sparse) {
        if (srcSparseStart[0] != 0) {
            throw std::runtime_error("Sparse data should be zero-indexed; srcSparseStart[0] != 0");
        }

        uint64_t dataLength = static_cast<uint64_t>(_uniqueExamples - 1);

        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size()) {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size() << " Required: "
                << dataLength;
            throw std::length_error(msg.str());
        }

        for (uint32_t i = 0; i < _uniqueExamples; ++i) {
            _vSparseStart[i] = (uint64_t)srcSparseStart[i];
            _vSparseEnd[i] = (uint64_t)srcSparseEnd[i];
        }

        for (uint64_t i = 0; i < dataLength; ++i) {
            _vSparseData[i] = srcSparseDataTyped[i];
            _vSparseIndex[i] = (uint32_t)srcSparseIndex[i];
        }

        _pbSparseStart->Upload(_vSparseStart.data());
        _pbSparseEnd->Upload(_vSparseEnd.data());
        _pbSparseIndex->Upload(_vSparseIndex.data());
        _pbSparseData->Upload(_vSparseData.data());
    }
    else {
        throw std::runtime_error("Cannot set sparse data on a non-sparse DataSet");
    }
}

template<typename T>
void DataSet<T>::CopySparseData(const long* srcSparseStart, const long* srcSparseEnd,
    const void* srcSparseData, const long* srcSparseIndex) {
    const T* srcSparseDataTyped = static_cast<const T*>(srcSparseData);

    if (_attributes & DataSetEnums::Attributes::Sparse) {
        if (srcSparseStart[0] != 0) {
            throw std::runtime_error("Sparse data should be zero-indexed; srcSparseStart[0] != 0");
        }

        uint64_t dataLength = static_cast<uint64_t>(_uniqueExamples) - static_cast<uint64_t>(1);

        if (dataLength > _vSparseData.size() || dataLength > _vSparseIndex.size()) {
            std::stringstream msg;
            msg << "Not enough space to store sparse data. Allocated: " << _vSparseData.size() << " Required: "
                << dataLength;
            throw std::length_error(msg.str());
        }

        for (uint32_t i = 0; i < _uniqueExamples; ++i) {
            _vSparseStart[i] = (uint64_t)srcSparseStart[i];
            _vSparseEnd[i] = (uint64_t)srcSparseEnd[i];
        }

        for (uint64_t i = 0; i < dataLength; ++i) {
            _vSparseData[i] = srcSparseDataTyped[i];
            _vSparseIndex[i] = (uint32_t)srcSparseIndex[i];
        }
    }
    else {
        throw std::runtime_error("Cannot set sparse data on a non-sparse DataSet");
    }
}

template<typename T>
void DataSet<T>::LoadIndexedData(const uint32_t* srcIndexedData) {
    if (_attributes & DataSetEnums::Attributes::Indexed) {
        std::ranges::copy(srcIndexedData, srcIndexedData + _vIndex.size(), _vIndex.data());

        _pbIndex->Upload(_vIndex.data());
    }
    else {
        throw std::runtime_error("Cannot set indexed data on a non-indexed DataSet");
    }
}

template<typename T>
void DataSet<T>::LoadDataWeight(const float* srcWeightData) {
    if (_attributes & DataSetEnums::Attributes::Weighted) {
        std::ranges::copy(srcWeightData, srcWeightData + _vDataWeight.size(), _vDataWeight.data());

        _pbDataWeight->Upload(_vDataWeight.data());
    }
    else {
        throw std::runtime_error("Cannot set weight data on a non-weighted DataSet");
    }
}

template<typename T>
T DataSet<T>::GetDataPoint(uint32_t n, uint32_t x, uint32_t y, uint32_t z) {
    if (_attributes & DataSetEnums::Sparse) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetDataPoint: Attempt to read non-sparse data from a sparse dataset.\n");
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetDataPoint: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (_bIndexed) {
        n = _vIndex[n];
    }

    if ((x >= _width) || (y >= _height) || (z >= _length)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetDataPoint: Invalid data point coordinates (%u, %u, %u) "
                "for dataset dimensions (width: %u, height: %u, length: %u).\n", x, y, z, _width, _height, _length);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    return _vData[(n * _stride) + x + _width * (y + z * _height)];
}

template<typename T>
bool DataSet<T>::SetDataPoint(T v, uint32_t n, uint32_t x, uint32_t y, uint32_t z) {
    if (_attributes & DataSetEnums::Sparse) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetDataPoint: Attempt to read non-sparse data from a sparse dataset.\n");
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetDataPoint: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (_bIndexed) {
        n = _vIndex[n];
    }

    if ((x >= _width) || (y >= _height) || (z >= _length)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetDataPoint: Invalid data point coordinates (%u, %u, %u) "
                "for dataset dimensions (width: %u, height: %u, length: %u).\n", x, y, z, _width, _height, _length);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    _vData[(n * _stride) + x + _width * (y + z * _height)] = v;
    return true;
}

template<typename T>
uint64_t DataSet<T>::GetSparseDataPoints(uint32_t n) {
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseDataPoints: Attempt to read sparse data from a non-sparse dataset.\n");
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseDataPoints: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (_bIndexed) {
        n = _vIndex[n];
    }

    return _vSparseEnd[n] - _vSparseStart[n];
}

template<typename T>
uint32_t DataSet<T>::GetSparseIndex(uint32_t n, uint32_t i) {
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseIndex: Attempt to read sparse data from a non-sparse dataset.\n");
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseIndex: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (_bIndexed) {
        n = _vIndex[n];
    }

    if (i >= _vSparseEnd[n] - _vSparseStart[n]) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseIndex: Sparse index %u is out of range [0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    return _vSparseIndex[_vSparseStart[n] + i];
}

template<typename T>
bool DataSet<T>::SetSparseIndex(uint32_t n, uint32_t i, uint32_t v) {
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseIndex: Attempt to set sparse data index on a non-sparse dataset.\n");
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseIndex: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (_bIndexed) {
        n = _vIndex[n];
    }

    if (i >= _vSparseEnd[n] - _vSparseStart[n]) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseIndex: Sparse index %u is out of range [0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    _vSparseIndex[_vSparseStart[n] + i] = v;
    _bDirty = true;

    return true;
}

template<typename T>
T DataSet<T>::GetSparseDataPoint(uint32_t n, uint32_t i) {
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseDataPoint: Attempt to read sparse data from a non-sparse dataset.\n");
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseDataPoint: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (_bIndexed) {
        n = _vIndex[n];
    }

    if (i >= _vSparseEnd[n] - _vSparseStart[n]) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GetSparseDataPoint: Sparse index %u is out of range [0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    return _vSparseData[_vSparseStart[n] + i];
}

template<typename T>
bool DataSet<T>::SetSparseDataPoint(uint32_t n, uint32_t i, T v) {
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseDataPoint: Attempt to modify sparse data in a non-sparse dataset.\n");
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (n >= _examples) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseDataPoint: Invalid example index %u (must be within [0, %lu)).\n", n, _examples);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    if (_bIndexed) {
        n = _vIndex[n];
    }

    if (i >= _vSparseEnd[n] - _vSparseStart[n]) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetSparseDataPoint: Sparse index %u is out of range [0, %lu).\n", i, _vSparseEnd[n] - _vSparseStart[n]);
        }
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    _vSparseData[_vSparseStart[n] + i] = v;

    _bDirty = true;

    return true;
}

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
            netCDF::NcFile nfc(fname.c_str(), netCDF::NcFile::read);

            bOpened = true;

            std::string nstring = std::to_string(n);

            std::string vname = "name" + nstring;

            netCDF::NcGroupAtt nameAtt = nfc.getAtt(vname);

            if (nameAtt.isNull())
            {
                std::cerr << "NcException: DataSet::DataSet: No dataset name supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            nameAtt.getValues(_name);

            std::cout << "DataSet<T>::DataSet: Name of data set: " << _name << std::endl;

            vname = "dataType" + nstring;

            netCDF::NcGroupAtt dataTypeAtt = nfc.getAtt(vname);

            if (dataTypeAtt.isNull())
            {
                std::cerr << "NcException: DataSet::DataSet: No datatype supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            int dataType;

            dataTypeAtt.getValues(&dataType);

            _dataType = (DataSetEnums::DataType)dataType;

            vname = "attributes" + nstring;

            netCDF::NcGroupAtt attributesAtt = nfc.getAtt(vname);

            if (attributesAtt.isNull())
            {
                std::cerr << "NcException: DataSet::DataSet: No attributes supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            attributesAtt.getValues(&_attributes);

            if (_attributes != 0)
            {
                int tempAtt = _attributes;
                int position = 0;
                std::cout << "DataSet<T>::DataSet: Attributes:";

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

            vname = "examplesDim" + nstring;

            netCDF::NcDim examplesDim = nfc.getDim(vname);

            if (examplesDim.isNull())
            {
                std::cerr << "NcException: DataSet::DataSet: No examples count supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            _examples = examplesDim.getSize();

            if (_examples == 0)
            {
                std::cerr << "NcException: DataSet::DataSet: Zero-valued Examples count in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            vname = "uniqueExamplesDim" + nstring;

            netCDF::NcDim uniqueExamplesDim = nfc.getDim(vname);

            if (uniqueExamplesDim.isNull())
            {
                _uniqueExamples = _examples;
            }
            else
            {
                _uniqueExamples = uniqueExamplesDim.getSize();
            }

            vname = "dimensions" + nstring;

            netCDF::NcGroupAtt dimensionsAtt = nfc.getAtt(vname);

            if (dimensionsAtt.isNull())
            {
                std::cerr << "NcException: DataSet::DataSet: No dimension count supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            dimensionsAtt.getValues(&_dimensions);

            if ((_dimensions < 1) || (_dimensions > 3))
            {
                std::cerr << "NcException: DataSet::DataSet: Invalid dimension count (" << std::to_string(_dimensions) << ") supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            vname = "width" + nstring;

            netCDF::NcGroupAtt widthAtt = nfc.getAtt(vname);

            if (widthAtt.isNull())
            {
                std::cerr << "NcException: DataSet::DataSet: No datapoint width supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }


            if (_dimensions > 1)
            {
                vname = "height" + nstring;

                netCDF::NcGroupAtt heightAtt = nfc.getAtt(vname);

                if (heightAtt.isNull())
                {
                    std::cerr << "NcException: DataSet::DataSet: No datapoint height supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

            }
            else
            {
                _height = 1;
            }

            if (_dimensions > 2)
            {
                vname = "length" + nstring;

                netCDF::NcGroupAtt lengthAtt = nfc.getAtt(vname);

                if (lengthAtt.isNull())
                {
                    std::cerr << "NcException: DataSet::DataSet: No datapoint length supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

            }
            else
            {
                _length = 1;
            }

            std::cerr << "DataSet<T>::DataSet: " << _dimensions << "-dimensional data comprised of (" << _width << ", " << _height << ", " << _length << ") datapoints." << std::endl;

            if ((_width == 0) || (_height == 0) || (_length == 0))
            {
                std::cerr << "NcException: DataSet::DataSet: Invalid dataset dimensions in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            }

            if (_attributes & DataSetEnums::Sparse)
            {
                _vSparseStart.resize(_uniqueExamples);
                _vSparseEnd.resize(_uniqueExamples);

                vname = "sparseDataDim" + nstring;

                netCDF::NcDim sparseDataDim = nfc.getDim(vname);

                if (sparseDataDim.isNull())
                {
                    std::cerr << "NcException: DataSet::DataSet: No sparse data dimensions supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                _sparseDataSize = sparseDataDim.getSize();

                if (_sparseDataSize == 0)
                {
                    std::cerr << "NcException: DataSet::DataSet: Sparse data set with no actual data in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                _vSparseIndex.resize(_sparseDataSize);
                std::cout << "DataSet<T>::DataSet: " << _sparseDataSize << " total datapoints." << std::endl;

                vname = "sparseStart" + nstring;
                netCDF::NcVar sparseStartVar = nfc.getVar(vname);
                vname = "sparseEnd" + nstring;
                netCDF::NcVar sparseEndVar = nfc.getVar(vname);
                vname = "sparseIndex" + nstring;
                netCDF::NcVar sparseIndexVar = nfc.getVar(vname);

                if (sparseStartVar.isNull())
                {
                    std::cerr << "NcException: DataSet::DataSet: No sparse offset start supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                if (sparseEndVar.isNull())
                {
                    std::cerr << "NcException: DataSet::DataSet: No sparse data end supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                if (sparseIndexVar.isNull())
                {
                    std::cerr << "NcException: DataSet::DataSet: No sparse data indices supplied in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                netCDF::NcType vStartType = sparseStartVar.getType();

                if (vStartType == netCDF::ncUint)
                {
                    std::vector<uint32_t> vTempSparseStart(_uniqueExamples);
                    sparseStartVar.getVar((uint32_t*)vTempSparseStart.data());
                    std::ranges::copy(vTempSparseStart.begin(), vTempSparseStart.end(), _vSparseStart.begin());
                }
                else
                    sparseStartVar.getVar((uint64_t*)_vSparseStart.data());

                netCDF::NcType vEndType = sparseEndVar.getType();

                if (vEndType == netCDF::ncUint)
                {
                    std::vector<uint32_t> vTempSparseEnd(_uniqueExamples);
                    sparseEndVar.getVar((uint32_t*)vTempSparseEnd.data());
                    std::ranges::copy(vTempSparseEnd.begin(), vTempSparseEnd.end(), _vSparseEnd.begin());
                }
                else
                    sparseEndVar.getVar((uint64_t*)_vSparseEnd.data());

                sparseIndexVar.getVar((uint32_t*)_vSparseIndex.data());

                if (!(_attributes & DataSetEnums::Boolean))
                {
                    vname = "sparseData" + nstring;

                    netCDF::NcVar sparseDataVar = nfc.getVar(vname);

                    if (sparseDataVar.isNull())
                    {
                        std::cerr << "NcException: DataSet::DataSet: No sparse data located in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                    }

                    _vSparseData.resize(sparseDataDim.getSize());

                    sparseDataVar.getVar(_vSparseData.data());
                }
            }
            else
            {
                _stride = _width * _height * _length;

                vname = "dataDim" + nstring;

                netCDF::NcDim dataDim = nfc.getDim(vname);

                if (dataDim.isNull())
                {
                    std::cerr << "NcException: DataSet::DataSet: No data dimensions located in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                vname = "data" + nstring;

                netCDF::NcVar dataVar = nfc.getVar(vname);

                if (_attributes & DataSetEnums::Boolean)
                {
                    uint64_t size = (uint64_t)_width * (uint64_t)_height * (uint64_t)_length;

                    _vData.resize(dataDim.getSize() * size);

                    memset(_vData.data(), 0, _vData.size() * sizeof(T));

                    std::vector<T> vData(dataDim.getSize());

                    dataVar.getVar(vData.data());

                    for (size_t i = 0; i < static_cast<size_t>(dataDim.getSize()); i++)
                        _vData[i * size + vData[i]] = (T)1.0;
                }
                else
                {
                    _vData.resize(dataDim.getSize());

                    dataVar.getVar(_vData.data());
                }
            }

            if (_attributes & DataSetEnums::Weighted)
            {
                vname = "dataWeight" + nstring;

                netCDF::NcVar DataWeightVar = nfc.getVar(vname);

                if (DataWeightVar.isNull())
                {
                    std::cerr << "NcException: DataSet::DataSet: No data weights located in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                _vDataWeight.resize(_examples);

                DataWeightVar.getVar(_vDataWeight.data());
            }

            if (_attributes & DataSetEnums::Indexed)
            {
                vname = "index" + nstring;

                netCDF::NcVar indexVar = nfc.getVar(vname);

                if (indexVar.isNull())
                {
                    std::cerr << "NcException: DataSet::DataSet: No indexed data located in NetCDF input file " << fname << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                }

                _vIndex.resize(_examples);

                indexVar.getVar(_vIndex.data());
            }

            std::cout << "DataSet<T>::DataSet: " << _examples << " examples." << std::endl;
            std::cout << "DataSet<T>::DataSet: " << _uniqueExamples << " unique examples." << std::endl;
        }
        catch (netCDF::exceptions::NcException& e)
        {
            if (!bOpened)
            {
                std::cout << "Exception: DataSet::DataSet: Error opening NetCDF input file " << fname << std::endl;
            }
            else
            {
                std::cout << "Exception: " << e.what() << std::endl;
            }

            bResult = false;
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult)
    {
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    MPI_Bcast_string(_name);

    MPI_Bcast(&_dataType, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_attributes, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_examples, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_uniqueExamples, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_dimensions, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_width, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_height, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_length, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    MPI_Bcast(&_sparseDataSize, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    if (getGpu()._id != 0)
    {
        _vData.resize(0);
        _vSparseStart.resize(_uniqueExamples, 0);
        _vSparseEnd.resize(_uniqueExamples, 0);
        _vSparseIndex.resize(0);
        _vSparseData.resize(0);
    }

    if (_attributes & DataSetEnums::Indexed)
    {
        _vIndex.resize(_examples);
        MPI_Bcast(_vIndex.data(), _examples, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    }

    if (_attributes & DataSetEnums::Weighted)
    {
        _vDataWeight.resize(_examples);
        MPI_Bcast(_vDataWeight.data(), _examples, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    if (_attributes & DataSetEnums::Sparse)
    {
        CalculateSparseDatapointCounts();
    }
}

template<typename T>
bool DataSet<T>::Rename(const std::string& name) {
    _name = name;
    return true;
}

template<typename T>
bool DataSet<T>::CalculateSparseDatapointCounts() {
    if (_attributes & DataSetEnums::Sparse) {
        uint64_t N = static_cast<uint64_t>(_width) * static_cast<uint64_t>(_height) * static_cast<uint64_t>(_length);

        _vSparseDatapointCount.resize(N);
        _vSparseMaxDatapointCount.resize(N);
        _vSparseMultiDatapointCount.resize(N);

        std::fill(_vSparseDatapointCount.begin(), _vSparseDatapointCount.end(), 0);
        std::fill(_vSparseMaxDatapointCount.begin(), _vSparseMaxDatapointCount.end(), 0);
        std::fill(_vSparseMultiDatapointCount.begin(), _vSparseMultiDatapointCount.end(), 0);

        std::vector<uint32_t> vCount(N, 0);

        std::vector<uint32_t> vExampleCount(_uniqueExamples, 0);

        if (_attributes & DataSetEnums::Indexed) {
            for (size_t i = 0; i < _examples; i++) {
                vExampleCount[_vIndex[i]]++;
            }
        }
        else {
            std::fill(vExampleCount.begin(), vExampleCount.end(), 1);
        }

        for (size_t i = 0; i < _uniqueExamples; i++) {
            uint64_t count = _vSparseEnd[i] - _vSparseStart[i];

            for (size_t j = _vSparseStart[i]; j < _vSparseEnd[i]; j++) {
                try {
                    vCount.at(_vSparseIndex[j])++;
                }
                catch (std::exception& e) {
                    std::cout << "DataSet::CalculateSparseDatapointCounts: vCount address = " << _vSparseIndex[j] << " >= vCount size = " << N << std::endl;
                    std::rethrow_exception(std::current_exception());
                }
            }



            for (size_t j = _vSparseStart[i]; j < _vSparseEnd[i]; j++) {
                uint32_t x = _vSparseIndex[j];

                if (vCount[x] > 0) {
                    _vSparseMaxDatapointCount[x] = std::max(_vSparseMaxDatapointCount[x], vCount[x]);
                    if (vCount[x] > 1)
                        _vSparseMultiDatapointCount[x] += vExampleCount[i];
                    _vSparseDatapointCount[x] += static_cast<size_t>(vExampleCount[i]) * static_cast<size_t>(vCount[x]);
                    vCount[x] = 0;
                }
            }
        }

        size_t batch = 2048;
        size_t active = 0;

        for (size_t i = 0; i < N; i++) {
            size_t size1 = _vSparseDatapointCount[i];
            size1 = std::min(batch, size1);
            active += (_vSparseDatapointCount[i] > 0);
            if (_vSparseMaxDatapointCount[i] > 1) {
                size_t size2 = std::min(static_cast<size_t>(_vSparseMaxDatapointCount[i]) * batch, batch + static_cast<size_t>(_vSparseMaxDatapointCount[i] - 1) * static_cast<size_t>(_vSparseMultiDatapointCount[i]));
                size1 = std::max(size1, size2);
            }
        }

        _sparseDensity = (double_t)_sparseDataSize / (double_t)(_uniqueExamples * N);
        return true;
    }
    else {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::CalculateSparseDatapointCounts: Attempt to calculate sparse datapoint counts on non-sparse dataset %s.\n", _name.c_str());
        }
        return false;
    }
}

template<typename T>
bool DataSet<T>::GenerateSparseTransposedMatrix(uint32_t batch, Layer* pLayer) {
    if (_bDirty) {
        CalculateSparseDatapointCounts();
        _bDirty = false;
    }

    uint64_t NData = static_cast<uint64_t>(_width) * static_cast<uint64_t>(_height) * static_cast<uint64_t>(_length);
    uint32_t Nx, Ny, Nz, Nw;
    std::tie(Nx, Ny, Nz, Nw) = pLayer->GetLocalDimensions();
    uint64_t NLayer = static_cast<uint64_t>(Nx) * Ny * Nz * Nw;
    uint64_t N = max(NData, NLayer);

    _vSparseTransposedStart.resize(N);
    if (!_pbSparseTransposedStart)
        _pbSparseTransposedStart.reset(new GpuBuffer<uint32_t>(N));
    if (!_pbSparseTransposedEnd)
        _pbSparseTransposedEnd.reset(new GpuBuffer<uint32_t>(N));

    _batch = batch;
    uint32_t offset = 0;

    for (size_t i = 0; i < _vSparseDatapointCount.size(); i++) {
        _vSparseTransposedStart[i] = offset;
        size_t size1 = _vSparseDatapointCount[i];

        size1 = std::min((size_t)batch, size1);
        if (_vSparseMaxDatapointCount[i] > 1) {
            size_t size2 = std::min(_vSparseMaxDatapointCount[i] * batch, batch + (_vSparseMaxDatapointCount[i] - 1) * _vSparseMultiDatapointCount[i]);
            size1 = std::max(size1, size2);
        }

        offset += size1;
        offset = ((offset + 31) >> 5) << 5;
    }

    _pbSparseTransposedStart->Upload(_vSparseTransposedStart.data());

    if (offset > _sparseTransposedIndices) {
        _sparseTransposedIndices = offset;
        std::cout << "DataSet::GenerateSparseTransposedMatrix: Allocating " << static_cast<unsigned long long>(_sparseTransposedIndices) * sizeof(uint32_t) << " bytes for sparse transposed weight gradient index matrix " << _name << "." << std::endl;

        _pbSparseTransposedIndex.reset(new GpuBuffer<uint32_t>(_sparseTransposedIndices));

        if (!(_attributes & DataSetEnums::Boolean) || (_attributes & DataSetEnums::Weighted)) {
            std::cout << std::format("DataSet::GenerateSparseTransposedMatrix: Allocating %lu bytes for sparse transposed weight gradient value matrix %s.\n", _sparseTransposedIndices * sizeof(float), _name.c_str());
            _pbSparseTransposedData.reset(new GpuBuffer<float>(_sparseTransposedIndices));
        }
    }

    return true;
}

template<typename T>
bool DataSet<T>::SetDenoising(bool flag) {
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::SetDenoising: Attempt to set denoising on non-sparse data set.\n");
        }
        return false;
    }
    else if (!flag && _bDenoising) {
        _pbDenoisingRandom.reset();
        _bDenoising = false;
    }
    else if (flag && !_bDenoising) {
        _pbDenoisingRandom.reset(new GpuBuffer<float>((uint64_t)_vSparseIndex.size()));
    }
    return true;
}

template<typename T>
bool DataSet<T>::SetStreaming(bool flag) {
    if (!getGpu()._bUnifiedMemory) {
        std::cerr << "DataSet::SetStreaming: Streaming datasets not supported on GPU " << getGpu()._id << std::endl;
        return false;
    }

    if (flag != _bStreaming) {
        _bStreaming = flag && getGpu()._bUnifiedMemory;
        _bDirty = true;
    }

    return true;
}

template<typename T>
bool DataSet<T>::GetStreaming() {
    return _bStreaming;
}

template<typename T>
bool DataSet<T>::GenerateDenoisingData() {
    if (!(_attributes & DataSetEnums::Sparse)) {
        if (getGpu()._id == 0) {
            std::cout << std::format("DataSet::GenerateDenoisingData: Attempt to generate denoising randoms on non-sparse data set.\n");
        }
        return false;
    }

    curandGenerateUniform(getGpu()._RNG, _pbDenoisingRandom->_pDevData, _vSparseIndex.size());

    return true;
}

template<typename T>
bool DataSet<T>::UnShard() {
    if (_sharding == DataSetEnums::Model) {
        if (_attributes & DataSetEnums::Sparse) {
            _pbSparseStart->Download(_vSparseStart.data());
            _pbSparseEnd->Download(_vSparseEnd.data());
            _pbSparseIndex->Download(_vSparseIndex.data());
            _pbSparseStart.reset();
            _pbSparseEnd.reset();
            _pbSparseIndex.reset();

            if (!(_attributes & DataSetEnums::Boolean)) {
                _pbSparseData->Download(_vSparseData.data());
                _pbSparseData.reset();
            }

            int32_t xmin = ((size_t)_width * (size_t)getGpu()._id) / (size_t)getGpu()._numprocs;
            int32_t xmax = ((size_t)_width * ((size_t)getGpu()._id + 1)) / (size_t)getGpu()._numprocs;


            for (auto& index : _vSparseIndex) {
                index -= xmin;
            }

            std::vector<uint32_t> vSparseCount(_uniqueExamples);
            for (uint32_t i = 0; i < _uniqueExamples; i++) {
                vSparseCount[i] = _vSparseEnd[i] - _vSparseStart[i];
            }

            uint64_t datapoints = _vSparseIndex.size();

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

                for (int i = 0; i < static_cast<int>(_uniqueExamples); i++) {
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

                for (uint32_t i = 1; i < static_cast<uint32_t>(getGpu()._numprocs); i++) {
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

                        for (unsigned int j = 0; j < vSparseCount[i]; j++) {
                            vTempSparseIndex[vTempSparseEnd[i]] = vPeerSparseIndex[start];

                            if (!(_attributes & DataSetEnums::Boolean)) {
                                vTempSparseData[vTempSparseEnd[i]] = vPeerSparseData[start];
                            }

                            vTempSparseEnd[i]++;
                            start++;
                        }
                    }
                }

                _vSparseStart = vTempSparseStart;
                _vSparseEnd = vTempSparseEnd;
                _vSparseIndex = vTempSparseIndex;

                if (!(_attributes & DataSetEnums::Boolean)) {
                    _vSparseData = vTempSparseData;
                }

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
            _pbData->Download(_vData.data());
            _pbData.reset();

            if (getGpu()._id == 0) {
                std::vector<T> vTempData(_vData);
                _vData.resize(_uniqueExamples * _width);

                uint32_t xmax = _width / getGpu()._numprocs;
                for (uint64_t i = 0; i < _uniqueExamples; i++) {
                    for (uint64_t j = 0; j < xmax; j++) {
                        _vData[i * _width + j] = vTempData[i * xmax + j];
                    }
                }

                for (int i = 1; i < getGpu()._numprocs; i++) {
                    int xmin = (i * _width) / getGpu()._numprocs;
                    xmax = ((i + 1) * _width) / getGpu()._numprocs;
                    int slice = xmax - xmin;
                    int size = _uniqueExamples * slice;
                    vTempData.resize(size);
                    MPI_Status status;
                    MPI_Recv(vTempData.data(), size, getMPIDataType(_dataType), i, 0, MPI_COMM_WORLD, &status);
                    for (uint32_t j = 0; j < _uniqueExamples; j++) {
                        for (int k = 0; k < slice; k++) {
                            _vData[j * _width + xmin + k] = vTempData[j * slice + k];
                        }
                    }
                }

                _pbData.reset(new GpuBuffer<T>((uint64_t)_vData.size(), false, _bStreaming));
                _pbData->Upload(_vData.data());
            }
            else {
                MPI_Send(_vData.data(), _vData.size(), getMPIDataType(_dataType), 0, 0, MPI_COMM_WORLD);
            }
        }
    }
    else if (_sharding == DataSetEnums::Data) {

		_pbData->Download(_vData.data());
		_pbData.reset();

        if (getGpu()._id == 0) {
			std::vector<T> vTempData(_vData);
			_vData.resize(_uniqueExamples * _width);

			uint32_t xmax = _width / getGpu()._numprocs;
            for (uint64_t i = 0; i < _uniqueExamples; i++) {
                for (uint64_t j = 0; j < xmax; j++) {
					_vData[i * _width + j] = vTempData[i * xmax + j];
				}
			}

            for (int i = 1; i < getGpu()._numprocs; i++) {
				int xmin = (i * _width) / getGpu()._numprocs;
				xmax = ((i + 1) * _width) / getGpu()._numprocs;
				int slice = xmax - xmin;
				int size = _uniqueExamples * slice;
				vTempData.resize(size);
				MPI_Status status;
				MPI_Recv(vTempData.data(), size, getMPIDataType(_dataType), i, 0, MPI_COMM_WORLD, &status);
                for (uint32_t j = 0; j < _uniqueExamples; j++) {
                    for (int k = 0; k < slice; k++) {
						_vData[j * _width + xmin + k] = vTempData[j * slice + k];
					}
				}
			}

			_pbData.reset(new GpuBuffer<T>((uint64_t)_vData.size(), false, _bStreaming));
			_pbData->Upload(_vData.data());
		}
        else {
			MPI_Send(_vData.data(), _vData.size(), getMPIDataType(_dataType), 0, 0, MPI_COMM_WORLD);
		}
    }

    _sharding = DataSetEnums::Sharding::None;

    if (_attributes & DataSetEnums::Indexed) {
        _pbIndex.reset(new GpuBuffer<uint32_t>((uint64_t)_vIndex.size(), false, _bStreaming));
        _pbIndex->Upload(_vIndex.data());
    }

    if (_attributes & DataSetEnums::Weighted) {
        _pbDataWeight.reset(new GpuBuffer<float>((uint64_t)_vDataWeight.size(), false, _bStreaming));
        _pbDataWeight->Upload(_vDataWeight.data());
    }

    return true;
}


template<typename T>
bool DataSet<T>::Shard(DataSetEnums::Sharding sharding) {
    if (sharding == _sharding)
        return true;

    UnShard();

    if (sharding == DataSetEnums::Model) {
        _sharding = DataSetEnums::Model;

        _minX = ((size_t)_width * (size_t)getGpu()._id) / (size_t)getGpu()._numprocs;
        _maxX = ((size_t)_width * (size_t)(getGpu()._id + 1)) / (size_t)getGpu()._numprocs;

        if (_attributes & DataSetEnums::Sparse) {
            if (getGpu()._id == 0) {
                std::cout << std::format("DataSet<T>::Shard: Model Sharding sparse dataset %s across all GPUs.\n", _name.c_str());

                for (size_t i = 1; i < static_cast<size_t>(getGpu()._numprocs); i++) {
                    uint32_t xmin = ((size_t)_width * i) / (size_t)getGpu()._numprocs;
                    uint32_t xmax = ((size_t)_width * (i + 1)) / (size_t)getGpu()._numprocs;

                    std::vector<uint64_t> vLocalSparseStart(_uniqueExamples);
                    std::vector<uint64_t> vLocalSparseEnd(_uniqueExamples);
                    std::vector<uint32_t> vLocalSparseIndex;
                    std::vector<T> vLocalSparseData;

                    for (uint32_t j = 0; j < _uniqueExamples; j++) {
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

                    uint64_t size = vLocalSparseIndex.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseStart.data(), _uniqueExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseEnd.data(), _uniqueExamples, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Send(vLocalSparseIndex.data(), size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD);

                    if (!(_attributes & DataSetEnums::Boolean)) {
                        MPI_Datatype mpiType = getMPIDataType(_dataType);
                        MPI_Send(vLocalSparseData.data(), size, mpiType, i, 0, MPI_COMM_WORLD);
                    }
                }

                std::vector<uint64_t> vTempSparseStart = _vSparseStart;
                std::vector<uint64_t> vTempSparseEnd = _vSparseEnd;
                std::vector<uint32_t> vTempSparseIndex = _vSparseIndex;
                std::vector<T> vTempSparseData = _vSparseData;
                _vSparseIndex.resize(0);
                _vSparseData.resize(0);
                _vSparseStart.resize(_uniqueExamples);
                _vSparseEnd.resize(_uniqueExamples);

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
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                _vSparseStart.resize(_uniqueExamples);
                _vSparseEnd.resize(_uniqueExamples);
                _vSparseIndex.resize(size);
                MPI_Recv(_vSparseStart.data(), _uniqueExamples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(_vSparseEnd.data(), _uniqueExamples, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(_vSparseIndex.data(), size, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, &status);

                if (!(_attributes & DataSetEnums::Boolean)) {
                    MPI_Datatype mpiType = getMPIDataType(_dataType);
                    _vSparseData.resize(size);
                    MPI_Recv(_vSparseData.data(), size, mpiType, 0, 0, MPI_COMM_WORLD, &status);
                }

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
        }
        else {
            if (getGpu()._id == 0) {
                printf("DataSet<T>::Shard: Model Sharding dataset %s across all GPUs.\n", _name.c_str());

                for (size_t i = 1; i < static_cast<size_t>(getGpu()._numprocs); i++) {
                    uint32_t xmin = ((size_t)_width * i) / (size_t)getGpu()._numprocs;
                    uint32_t xmax = ((size_t)_width * (size_t)(i + 1)) / (size_t)getGpu()._numprocs;
                    uint32_t slice = xmax - xmin;
                    std::vector<T> vLocalData(_uniqueExamples * slice);

                    for (size_t j = 0; j < _uniqueExamples; j++) {
                        for (size_t k = 0; k < slice; k++) {
                            vLocalData[j * slice + k] = _vData[j * _width + xmin + k];
                        }
                    }

                    size_t size = vLocalData.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
                    MPI_Datatype mpiType = getMPIDataType(_dataType);
                    MPI_Send(vLocalData.data(), _uniqueExamples * slice, mpiType, i, 0, MPI_COMM_WORLD);
                }

                std::vector<T> vTempData = _vData;
                uint64_t xmax = _width / getGpu()._numprocs;
                _vData.resize(_uniqueExamples * xmax);

                for (uint64_t j = 0; j < _uniqueExamples; j++) {
                    for (uint64_t k = 0; k < xmax; k++) {
                        _vData[j * xmax + k] = vTempData[j * _width + k];
                    }
                }
            }
            else {
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &status);
                _vData.resize(size);
                MPI_Datatype mpiType = getMPIDataType(_dataType);
                MPI_Recv(_vData.data(), size, mpiType, 0, 0, MPI_COMM_WORLD, &status);

                _pbData.reset(new GpuBuffer<T>(_vData.size(), false, _bStreaming));
                _pbData->Upload(_vData.data());
            }
        }
    }

    if (_attributes & DataSetEnums::Indexed) {
        _pbIndex.reset(new GpuBuffer<uint32_t>((uint64_t)_vIndex.size(), false, _bStreaming));
        _pbIndex->Upload(_vIndex.data());
    }
    return true;
}

template<typename T>
bool DataSet<T>::SaveNetCDF(const std::string& fname) {

    bool bResult = true;

    DataSetEnums::Sharding oldSharding = _sharding;
    UnShard();

    if (getGpu()._id == 0) {
        try {
            netCDF::NcFile nfc(fname, netCDF::NcFile::replace);

            nfc.putAtt("datasets", netCDF::ncUint, 1);

            bool bResult = WriteNetCDF(nfc, fname, 0);

            if (!bResult) {
                std::cerr << "SaveNetCDF: Unable to write dataset to NetCDF file " << fname;
            }
        }
        catch (const netCDF::exceptions::NcException& e) {
            std::cerr << "SaveNetCDF: Unable to create or write to NetCDF output file " << fname;
            std::cerr << e.what();
            bResult = false;
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult) {
        getGpu().Shutdown();
        std::cerr << "SaveNetCDF operation failed.";
    }

    Shard(oldSharding);

    return bResult;
}


template <typename T>
bool DataSet<T>::WriteNetCDF(netCDF::NcFile& nfc, const std::string& fname, const uint32_t n) {
    try {
        if (getGpu()._id != 0) {
            return true;
        }

        std::string nstring = std::to_string(n);

        auto handleNetCDFError = [&](const std::string& errMsg) {
            std::cerr << "NcException: " << errMsg << " " << fname << " (" << __FILE__ << ", " << __LINE__ << ")";
            return false;
            };

        auto createAttribute = [&](const std::string& attrName, const std::string& errMsg) {
            return nfc.putAtt(attrName, _name).isNull() ? handleNetCDFError(errMsg) : true;
            };

        auto createDimension = [&](const std::string& dimName, size_t dimSize, const std::string& errMsg) {
            return nfc.addDim(dimName, dimSize).isNull() ? handleNetCDFError(errMsg) : true;
            };

        auto createVariable = [&](const std::string& varName, const std::string& varType, const std::string& dimName, const void* data, const std::string& errMsg) {
            netCDF::NcVar variable = nfc.addVar(varName, varType, dimName);
            if (variable.isNull()) {
                return handleNetCDFError(errMsg);
            }
            variable.putVar(data);
            return true;
            };

        if (!createAttribute("name" + nstring, "Failed to write dataset name to NetCDF file") ||
            !createAttribute("attributes" + nstring, "Failed to write dataset attributes to NetCDF file") ||
            !createAttribute("kind" + nstring, "Failed to write dataset kind to NetCDF file") ||
            !createAttribute("datatype" + nstring, "Failed to write dataset type to NetCDF file") ||
            !createAttribute("dimensions" + nstring, "Failed to write dataset dimensions to NetCDF file") ||
            !createAttribute("width" + nstring, "Failed to write dataset width to NetCDF file")) {
            std::cerr << "Failed to write dataset attributes to NetCDF file " << fname;
            return false;
        }

        if (_dimensions > 1) {
            if (!createAttribute("height" + nstring, "Failed to write dataset height to NetCDF file")) {
                std::cerr << "Failed to write dataset height to NetCDF file " << fname;
                return false;
            }

            if (_dimensions > 2) {
                if (!createAttribute("length" + nstring, "Failed to write dataset length to NetCDF file")) {
                    std::cerr << "Failed to write dataset length to NetCDF file " << fname;
                    return false;
                }
            }
        }

        if (!createDimension("uniqueExamplesDim" + nstring, static_cast<size_t>(_uniqueExamples), "Failed to write dataset unique example count to NetCDF file") ||
            !createDimension("examplesDim" + nstring, static_cast<size_t>(_examples), "Failed to write dataset example count to NetCDF file")) {
            std::cerr << "Failed to create dataset dimensions in NetCDF file " << fname;
            return false;
        }

        if (_attributes & DataSetEnums::Sparse) {
            if (!createDimension("sparseDataDim" + nstring, _vSparseIndex.size(), "Failed to write dataset sparse datapoint count to NetCDF file") ||
                !createVariable("sparseStart" + nstring, "uint", "uniqueExamplesDim" + nstring, _vSparseStart.data(), "Failed to write dataset sparse start variable to NetCDF file") ||
                !createVariable("sparseEnd" + nstring, "uint", "uniqueExamplesDim" + nstring, _vSparseEnd.data(), "Failed to write dataset sparse end variable to NetCDF file") ||
                !createVariable("sparseIndex" + nstring, "uint64", "sparseDataDim" + nstring, _vSparseIndex.data(), "Failed to write dataset sparse index variable to NetCDF file")) {
                std::cerr << "Failed to write dataset sparse data to NetCDF file " << fname;
                return false;
            }

            if (!(_attributes & DataSetEnums::Boolean)) {
                netCDF::NcType sparseType = getNetCDFDataType(_dataType);
                if (!createVariable("sparseData" + nstring, sparseType.getName(), "sparseDataDim" + nstring, _vSparseData.data(), "Failed to write dataset sparse data variable to NetCDF file")) {
                    std::cerr << "Failed to write dataset sparse data to NetCDF file " << fname;
                    return false;
                }
            }
        }

        if (_attributes & DataSetEnums::Weighted) {
            if (!createVariable("dataWeight" + nstring, "float", "uniqueExamplesDim" + nstring, _vDataWeight.data(), "Failed to write data weights to NetCDF file")) {
                std::cerr << "Failed to write data weights to NetCDF file " << fname;
                return false;
            }
        }

        if (_attributes & DataSetEnums::Indexed) {
            if (!createVariable("index" + nstring, "uint32", "examplesDim" + nstring, _vIndex.data(), "Failed to create dataset index variable to NetCDF file")) {
                std::cerr << "Failed to create dataset index variable to NetCDF file " << fname;
                return false;
            }
        }

        return true;
    }
    catch (const netCDF::exceptions::NcException& e) {
        std::cerr << e.what();
        return false;
    }
}

template<typename T>
DataSet<T>::~DataSet() {
}

bool SaveNetCDF(const std::string& fname, std::vector<DataSetBase*>& vDataSet) {
    std::vector<DataSetEnums::Sharding> vSharding;
    vSharding.reserve(vDataSet.size());
    for (auto& dataSet : vDataSet) {
        vSharding.push_back(dataSet->_sharding);
        dataSet->UnShard();
    }

    int mpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    bool bResult = true;
    bool bOpened = false;

    try {
        if (mpiRank == 0) {
            netCDF::NcFile nfc(fname, netCDF::NcFile::replace);
            bOpened = true;

            auto datasetsAtt = nfc.putAtt("datasets", netCDF::ncUint, vDataSet.size());
            if (datasetsAtt.isNull()) {
                throw std::runtime_error("SaveNetCDF: Unable to write datasets attribute to NetCDF file " + fname);
            }

            for (size_t i = 0; i < vDataSet.size(); i++) {
                if (!vDataSet[i]->WriteNetCDF(nfc, fname, i)) {
                    throw std::runtime_error("SaveNetCDF: Unable to write dataset to NetCDF file " + fname);
                }
            }
        }
    }
    catch (const netCDF::exceptions::NcException& e) {
        std::cerr << (bOpened ? e.what() : "SaveNetCDF: Unable to create NetCDF output file " + fname) << '\n';
        bResult = false;
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult) {
        getGpu().Shutdown();
        std::cerr << "Error: The MPI broadcast failed." << '\n';
    }

    for (size_t i = 0; i < vDataSet.size(); i++) {
        vDataSet[i]->Shard(vSharding[i]);
    }

    return bResult;
}

std::vector<DataSetBase*> LoadNetCDF(const std::string& fname) {
    std::vector<DataSetBase*> vDataSet;
    std::vector<DataSetEnums::DataType> vDataType;
    bool bResult = true;

    if (getGpu()._id == 0) {
        try {
            netCDF::NcFile rnc(fname, netCDF::NcFile::read);

            if (!rnc.getAtt("datasets").isNull()) {
                uint32_t datasets;
                rnc.getAtt("datasets").getValues(&datasets);

                for (uint32_t i = 0; i < datasets; i++) {
                    std::string vname = "dataType" + std::to_string(i);

                    if (!rnc.getAtt(vname).isNull()) {
                        uint32_t dataType;
                        rnc.getAtt(vname).getValues(&dataType);

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
                            std::cerr << "LoadNetCDF: Invalid data type in binary input file " << fname << '\n';
                        }
                    }
                    else {
                        std::cerr << "NcException: LoadNetCDF: No " << vname << " attribute located in NetCDF input file " << fname << '\n';
                    }
                }
            }
            else {
                std::cerr << "NcException: LoadNetCDF: No datasets count supplied in NetCDF input file " << fname << '\n';
            }
        }
        catch (const netCDF::exceptions::NcException& e) {
            std::cerr << "NcException: LoadNetCDF: " << e.what() << '\n';
            bResult = false;
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult) {
        getGpu().Shutdown();
        std::cerr << "Error: The MPI broadcast failed." << '\n';
    }

    uint32_t size = vDataType.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    vDataType.resize(size);
    MPI_Bcast(vDataType.data(), size, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    for (std::size_t i = 0; i < vDataType.size(); i++) {
        DataSetBase* pDataSet = nullptr;

        if (getGpu()._id == 0) {
            std::cout << "LoadNetCDF: Loading " << vDataType[i] << " data set" << '\n';
        }

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
            std::cerr << "LoadNetCDF: invalid dataset type in binary input file " << fname << '\n';
            getGpu().Shutdown();
            exit(EXIT_FAILURE);
        }

        vDataSet.push_back(pDataSet);
    }

    return vDataSet;
}

std::vector<DataSetBase*> LoadImageData(const std::string) {
    return std::vector<DataSetBase*>();
}

std::vector<DataSetBase*> LoadCSVData(const std::string) {
    return std::vector<DataSetBase*>();
}

std::vector<DataSetBase*> LoadJSONData(const std::string) {
    return std::vector<DataSetBase*>();
}

std::vector<DataSetBase*> LoadAudioData(const std::string) {
    return std::vector<DataSetBase*>();
}

std::vector<DataSetBase*> LoadTextData(const std::string) {
    return std::vector<DataSetBase*>();
}
