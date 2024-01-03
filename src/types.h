#pragma once

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <netcdf>
#include <tuple>
#include <json/json.h>
#include <cmath>
#include <memory>

class DataSetBase;
class Layer;
class Network;
class Weight;

#define VALIDATION
#ifdef VALIDATION
extern "C"
{
    #include <cblas.h>
}
#endif

template <typename T> struct GpuBuffer;

enum 
{
    DefaultBatch    = 512
};

enum Mode {
    Prediction = 0,

    Training = 1,

    Validation = 2,

    BatchValidation = 3,

    Unspecified = 4
};

enum TrainingMode
{
    SGD = 0,

    Momentum = 1,

    AdaGrad = 2,

    Nesterov = 3,

    RMSProp = 4,

    AdaDelta = 5,

    Adam = 6,

    LAMB = 7,

    RAdam = 8,

    Lookahead = 9,

    SWA = 10,

    Ranger = 11,

    Nadam = 12,

    Adabelief = 13,

    SAM = 14,

    NovoGrad = 15,

    SGDP = 16,

    SGDW = 17,

    Yogi = 18,
};

std::ostream& operator<< (std::ostream& out, const TrainingMode& e);

enum ErrorFunction
{
    L1,

    L2,

    CrossEntropy,

    ScaledMarginalCrossEntropy,

    DataScaledMarginalCrossEntropy,

    Hinge,

    L2Hinge,

    MeanAbsoluteError,

    MeanSquaredError,

    RootMeanSquaredError,

    KullbackLeiblerDivergence,

    JaccardIndex,

    DiceCoefficient,

    HuberLoss,

    LogCosh,

    CosineSimilarity,

    CategoricalCrossEntropy,

    WassersteinDistance,

    TripletMarginLoss,

    EarthMoversDistance,

    FocalLoss,

    SparseCategoricalCrossEntropy,

    LogLoss,

    ExponentialLoss,

    HuberizedHingeLoss,

    WeightedHuberLoss,

    RankingLoss,

    ContrastiveLoss,

    TripletLoss,

    CenterLoss,

    GaussianKLDivergence,

    LogitMarginLoss,
};

std::ostream& operator<< (std::ostream& out, const ErrorFunction& e);

enum Activation {
    Sigmoid,

    Tanh,

    RectifiedLinear,

    Linear,

    ParametricRectifiedLinear,

    SoftPlus,

    SoftSign,

    SoftMax,

    RELUMax,

    LinearMax,

    ExponentialLinear,

    LeakyRectifiedLinear,

    ScaledExponentialLinear,
};

std::ostream& operator<< (std::ostream& out, const Activation& a);

enum WeightInitialization
{
    Xavier,

    CaffeXavier,

    Gaussian,

    Uniform,

    UnitBall,

    Constant,

    SELU,
};
    
std::ostream& operator<< (std::ostream& out, const WeightInitialization& w);
    
enum PoolingFunction {
    None,

    Max,

    Average,

    LRN,

    Maxout,

    DotProduct,

    Cosine,

    Stochastic,

    LCN,

    GlobalTemporal,
};

std::ostream& operator<< (std::ostream& out, const PoolingFunction& p);

#include "Kernels.cuh"
#include "GpuSort.h"
#include "Enum.h"
#include "Weight.h"
#include "Layer.h"
#include "Network.h"


int MPI_Bcast_string(std::string& s);

struct DataSetDimensions
{
    uint32_t _dimensions;
    uint32_t _width;
    uint32_t _height;
    uint32_t _length;

    DataSetDimensions();

    DataSetDimensions(uint32_t width, uint32_t height = 1, uint32_t length = 1);
};

struct DataSetDescriptor
{
    std::string _name;
    DataSetEnums::DataType _dataType;
    uint32_t _attributes;
    DataSetDimensions _dim;
    uint32_t _examples;
    float _sparseDensity;

    static bool isSupported(uint32_t attributes)
    {
        using DataSetEnums::Attributes;

        static const std::vector<Attributes> SUPPORTED_ATTRIBUTES(Attributes::Sparse);
        for (auto mask : SUPPORTED_ATTRIBUTES)
        {
            if (attributes & mask)
            {
                attributes -= mask;
            }
        }
        return attributes == 0;
    }
};

DataSetBase* createDataSet(const DataSetDescriptor &descriptor);

struct DataSetBase {

    std::string _name;

    DataSetEnums::DataType _dataType;

    uint32_t _attributes;

    uint32_t _numPositions;

    uint32_t _examples;

    uint32_t _uniqueExamples;

    uint32_t _localExamples;

    uint32_t _dimensions;

    uint32_t _width;

    uint32_t _height;

    uint32_t _length;

    uint32_t _stride;

    DataSetEnums::Sharding _sharding;

    uint32_t _minX;

    uint32_t _maxX;

    uint64_t _sparseDataSize;

    float _sparseDensity;

    std::vector<uint64_t> _vSparseStart;

    std::unique_ptr<GpuBuffer<uint64_t>> _pbSparseStart;

    std::vector<uint64_t> _vSparseEnd;

    std::unique_ptr<GpuBuffer<uint64_t>> _pbSparseEnd;

    std::vector<uint32_t> _vSparseIndex;

    std::unique_ptr<GpuBuffer<uint32_t>> _pbSparseIndex;

    std::vector<float> _vDataWeight;

    std::unique_ptr<GpuBuffer<float>> _pbDataWeight;

    std::vector<uint32_t> _vIndex;

    std::unique_ptr<GpuBuffer<uint32_t>> _pbIndex;

    std::unique_ptr<GpuBuffer<float>> _pbDenoisingRandom;
    
    std::vector<uint64_t> _vSparseDatapointCount;

    std::vector<uint32_t> _vSparseMaxDatapointCount;

    std::vector<uint32_t> _vSparseMultiDatapointCount;

    std::vector<uint32_t> _vSparseTransposedStart;

    uint64_t _sparseTransposedIndices;

    std::unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedStart;

    std::unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedEnd;

    std::unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedIndex;

    std::unique_ptr<GpuBuffer<float>> _pbSparseTransposedData;

    bool _bDenoising;

    bool _bDirty;

    bool _bStreaming;

    bool _bIndexed;

    uint32_t _batch;

    DataSetBase();

    DataSetDimensions GetDimensions();

    uint32_t GetExamples() { return _examples; };

    uint32_t GetUniqueExamples() { return _uniqueExamples; };

    virtual bool SaveNetCDF(const std::string& fname) = 0;

    virtual bool WriteNetCDF(netCDF::NcFile& nfc, const std::string& fname, const uint32_t n) = 0;

    virtual ~DataSetBase() = 0;

    virtual void RefreshState(uint32_t batch) = 0;

    virtual bool Shard(DataSetEnums::Sharding sharding) = 0;

    virtual bool UnShard() = 0;

    virtual bool SetStreaming(bool flag) = 0;

    virtual bool GetStreaming() = 0;

    virtual std::vector<std::tuple<uint64_t, uint64_t>> getMemoryUsage() = 0;

    virtual bool CalculateSparseDatapointCounts() = 0;

    virtual bool GenerateSparseTransposedMatrix(uint32_t batch, Layer* pLayer) = 0;

    virtual bool CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, Layer* pLayer) = 0;

    virtual bool CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, Layer* pLayer) = 0;

    virtual bool CalculateSparseTransposedWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, float* pDelta, float* pWeightGradient) = 0;

    virtual bool SetDenoising(bool flag) = 0;

    virtual bool GenerateDenoisingData() = 0;

    virtual bool LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual bool LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual bool LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual bool CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, float* pUnit, float beta = 0.0f) = 0;

    virtual bool CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, float* pUnit, float beta = 0.0f) = 0;

    virtual float CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual float CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual float CalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual float CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual float CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual float CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual float CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual float CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual float CalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) = 0;

    virtual bool CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float slope, float alpha, float lambda) = 0;

    virtual bool CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta) = 0;

    virtual bool CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta) = 0;

    virtual bool CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float slope, float alpha, float lambda) = 0;

    virtual bool CalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float slope, float alpha, float lambda) = 0;

    virtual bool CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta) = 0;

    virtual bool CalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta) = 0;

    virtual void LoadDenseData(const void* srcData) = 0;

    virtual void CopyDenseData(const void* srcData) = 0;

    virtual void LoadSparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd, const void* srcSparseData,
        const uint32_t* srcSparseIndex) = 0;

    virtual void CopySparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd, const void* srcSparseData,
        const uint32_t* srcSparseIndex) = 0;

    virtual void LoadSparseData(const long* srcSparseStart, const long* srcSparseEnd, const void* srcSparseData,
        const long* srcSparseIndex) = 0;

    virtual void CopySparseData(const long* srcSparseStart, const long* srcSparseEnd, const void* srcSparseData,
        const long* srcSparseIndex) = 0;

    virtual void LoadIndexedData(const uint32_t* srcIndexedData) = 0;

    virtual void LoadDataWeight(const float* srcWeightData) = 0;

 protected:
    DataSetBase(const std::string &name, DataSetEnums::DataType dataType, uint32_t examples, uint32_t uniqueExamples,
                  const DataSetDimensions &datasetDim);

};

std::ostream& operator<< (std::ostream& out, DataSetEnums::Attributes& a);
std::ostream& operator<< (std::ostream& out, DataSetEnums::Kind& k);
std::ostream& operator<< (std::ostream& out, DataSetEnums::DataType& t);
std::ostream& operator<< (std::ostream& out, DataSetEnums::Sharding& s);

template<typename T> class DataSet : public DataSetBase {
public:
    friend class Network;
    friend class Layer;
    friend std::vector<DataSetBase*> LoadNetCDF(const std::string& fname);
    friend bool SaveNetCDF(const std::string& fname, std::vector<DataSetBase*> vDataSet);

private:
    std::vector<T>                   _vData;
    std::unique_ptr<GpuBuffer<T>>    _pbData;
    std::vector<T>                   _vSparseData;
    std::unique_ptr<GpuBuffer<T>>    _pbSparseData;

    DataSet(const std::string& fname, uint32_t n);
    bool Rename(const std::string& name);
    bool SaveNetCDF(const std::string& fname);
    bool WriteNetCDF(netCDF::NcFile& nfc, const std::string& fname, const uint32_t n);
    void RefreshState(uint32_t batch) {} 
    bool Shard(DataSetEnums::Sharding sharding);
    bool UnShard();
    std::vector<std::tuple<uint64_t, uint64_t> > getMemoryUsage();
    bool CalculateSparseDatapointCounts();
    bool GenerateSparseTransposedMatrix(uint32_t batch, Layer* pLayer);
    bool CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, Layer* pLayer);
    bool CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, Layer* pLayer);
    bool CalculateSparseTransposedWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, float* pDelta, float* pWeightGradient);     
    bool SetStreaming(bool flag);
    bool GetStreaming();  
    bool SetDenoising(bool flag);
    bool GenerateDenoisingData();
    bool LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    bool LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    bool LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    bool CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, float* pUnit, float beta);
    bool CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, float* pUnit, float beta);
    float CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    float CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    float CalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    float CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    float CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    float CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    float CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    float CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    float CalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit);
    bool CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float slope, float alpha, float lambda);
    bool CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta);
    bool CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta);    
    bool CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float slope, float alpha, float lambda);
    bool CalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float slope, float alpha, float lambda);
    bool CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta);
    bool CalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta);

public:
    DataSet(uint32_t examples, const DataSetDimensions& dim, const std::string& name = "");

    DataSet(uint32_t examples, uint32_t uniqueExamples, const DataSetDimensions& dim, const std::string& name = "");

    DataSet(uint32_t examples, float sparseDensity, const DataSetDimensions& dim, bool isWeighted = false, const std::string& name = "");

    DataSet(uint32_t examples, uint32_t uniqueExamples, size_t sparseDataSize, const DataSetDimensions& dim,
        bool isIndexed = false, bool isWeighted = false, const std::string& name = "");

    void LoadDenseData(const void* srcData) override;

    void CopyDenseData(const void* srcData) override;

    void LoadSparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd, const void* srcSparseData,
        const uint32_t* srcSparseIndex) override;

    void CopySparseData(const uint64_t* srcSparseStart, const uint64_t* srcSparseEnd, const void* srcSparseData,
        const uint32_t* srcSparseIndex) override;

    void LoadSparseData(const long* srcSparseStart, const long* srcSparseEnd, const void* srcSparseData,
        const long* srcSparseIndex) override;

    void CopySparseData(const long* srcSparseStart, const long* srcSparseEnd, const void* srcSparseData,
        const long* srcSparseIndex) override;

    void LoadIndexedData(const uint32_t* srcIndexedData) override;

    void LoadDataWeight(const float* srcWeightData) override;

    ~DataSet();

    void Shuffle();

    T GetDataPoint(uint32_t n, uint32_t x, uint32_t y = 0, uint32_t z = 0);

    bool SetDataPoint(T v, uint32_t n, uint32_t x, uint32_t y = 0, uint32_t z = 0);

    uint64_t GetSparseDataPoints(uint32_t n);

    uint32_t GetSparseIndex(uint32_t n, uint32_t i);

    bool SetSparseIndex(uint32_t n, uint32_t i, uint32_t v);

    T GetSparseDataPoint(uint32_t n, uint32_t i);

    bool SetSparseDataPoint(uint32_t n, uint32_t i, T v);
};

template<typename T> bool DataSet<T>::LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    if (_attributes & DataSetEnums::Indexed)  
        kLoadIndexedInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData);        
    else
        kLoadInputUnit(position, batch, stride, pUnit, _pbData->_pDevData);
    return true;
}

template<typename T>
bool DataSet<T>::LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)
            kLoadIndexedSparseInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight);
        else
            kLoadSparseInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kLoadIndexedSparseAnalogInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData);
        else
            kLoadSparseAnalogInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData);
    }
    return true;
}

template<typename T>
bool DataSet<T>::LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)
            kLoadIndexedSparseDenoisedInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData);
        else
            kLoadSparseDenoisedInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kLoadIndexedSparseAnalogDenoisedInputUnit(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData);
        else
            kLoadSparseAnalogDenoisedInputUnit(position, batch, stride, pUnit, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData);
    }
    return true;
}

template<typename T>
bool DataSet<T>::CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, float* pUnit, float beta)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedSparseZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, pUnit, beta);
        else
            invokeSparseZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, pUnit, beta);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedSparseAnalogZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, pUnit, beta);
        else
            invokeSparseAnalogZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, pUnit, beta);
    }
    return true;
}

template<typename T> bool DataSet<T>::CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, float* pUnit, float beta) 
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)          
            invokeIndexedSparseDenoisedZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, pUnit, beta);
        else
            invokeSparseDenoisedZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, pUnit, beta);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)   
            invokeIndexedSparseAnalogDenoisedZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, pUnit, beta);
        else
            invokeSparseAnalogDenoisedZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, pUnit, beta);
    }
    return true;
}

template<typename T>
bool DataSet<T>::CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, Layer* pLayer)
{
    if (_bDirty || (batch != _batch))
    {
        GenerateSparseTransposedMatrix(batch, pLayer);
    }

    _pbSparseTransposedEnd->Copy(_pbSparseTransposedStart->_pDevData);

    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    float* pSparseTransposedData = ((_attributes & DataSetEnums::Weighted) || !(_attributes & DataSetEnums::Boolean)) ? _pbSparseTransposedData->_pDevData : NULL;
    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedSparseTransposedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        else
            invokeSparseTransposedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedSparseTransposedAnalogMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        else
            invokeSparseTransposedAnalogMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
    }

    return true;
}

template<typename T>
bool DataSet<T>::CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, Layer* pLayer)
{
    if (_bDirty || (batch != _batch))
    {
        GenerateSparseTransposedMatrix(batch, pLayer);
    }

    _pbSparseTransposedEnd->Copy(_pbSparseTransposedStart->_pDevData);

    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    float* pSparseTransposedData = ((_attributes & DataSetEnums::Weighted) || !(_attributes & DataSetEnums::Boolean)) ? _pbSparseTransposedData->_pDevData : NULL;
    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedSparseTransposedDenoisedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        else
            invokeSparseTransposedDenoisedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedSparseTransposedAnalogDenoisedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        else
            invokeSparseTransposedAnalogDenoisedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
    }


    return true;
}


template<typename T>
bool DataSet<T>::CalculateSparseTransposedWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, float* pDelta, float* pWeightGradient)
{
    if ((_attributes & DataSetEnums::Boolean) && !(_attributes & DataSetEnums::Weighted))
        invokeSparseTransposedWeightGradient(alpha, beta, m, n, _pbSparseTransposedStart->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pDelta, pWeightGradient);
    else
        invokeSparseTransposedAnalogWeightGradient(alpha, beta, m, n, _pbSparseTransposedStart->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, _pbSparseTransposedData->_pDevData, pDelta, pWeightGradient);
    return true;
}

template<typename T>
float DataSet<T>::CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseL1Error(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
            else
            {
                return invokeSparseL1Error(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseAnalogL1Error(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
            else
            {
                return invokeSparseAnalogL1Error(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return invokeIndexedL1Error(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return invokeL1Error(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T>
float DataSet<T>::CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseL2Error(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
            else
            {
                return invokeSparseL2Error(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseAnalogL2Error(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
            else
            {
                return invokeSparseAnalogL2Error(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return invokeIndexedL2Error(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return invokeL2Error(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T>
float DataSet<T>::CalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseL2HingeError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
            else
            {
                return invokeSparseL2HingeError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    bSparseIgnoreZero);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseAnalogL2HingeError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
            else
            {
                return invokeSparseAnalogL2HingeError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData,
                    _pbSparseEnd->_pDevData,
                    _pbSparseIndex->_pDevData,
                    pDataWeight,
                    _pbSparseData->_pDevData,
                    bSparseIgnoreZero);
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return invokeIndexedL2HingeError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return invokeL2HingeError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T>
float DataSet<T>::CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
        {
            return invokeIndexedSparseCrossEntropyError(position, batch, stride, pUnit,
                _pbIndex->_pDevData,
                _pbSparseStart->_pDevData,
                _pbSparseEnd->_pDevData,
                _pbSparseIndex->_pDevData,
                pDataWeight,
                bSparseIgnoreZero);
        }
        else
        {
            return invokeSparseCrossEntropyError(position, batch, stride, pUnit,
                _pbSparseStart->_pDevData,
                _pbSparseEnd->_pDevData,
                _pbSparseIndex->_pDevData,
                pDataWeight,
                bSparseIgnoreZero);
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return invokeIndexedCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return invokeCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T>
float DataSet<T>::CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
        {
            return invokeIndexedSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                _pbIndex->_pDevData,
                _pbSparseStart->_pDevData,
                _pbSparseEnd->_pDevData,
                _pbSparseIndex->_pDevData,
                pDataWeight,
                bSparseIgnoreZero);
        }
        else
        {
            return invokeSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                _pbSparseStart->_pDevData,
                _pbSparseEnd->_pDevData,
                _pbSparseIndex->_pDevData,
                pDataWeight,
                bSparseIgnoreZero);
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return invokeIndexedScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return invokeScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T>
float DataSet<T>::CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseMultinomialCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    pDataWeight);
            }
            else
            {
                return invokeSparseMultinomialCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseAnalogMultinomialCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    pDataWeight, _pbSparseData->_pDevData);
            }
            else
            {
                return invokeSparseAnalogMultinomialCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight,
                    _pbSparseData->_pDevData);
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return invokeIndexedMultinomialCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return invokeMultinomialCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T>
float DataSet<T>::CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    pDataWeight);
            }
            else
            {
                return invokeSparseMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    pDataWeight, _pbSparseData->_pDevData);
            }
            else
            {
                return invokeSparseAnalogMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight,
                    _pbSparseData->_pDevData);
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return invokeIndexedMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return invokeMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T>
float DataSet<T>::CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    pDataWeight, bSparseIgnoreZero);
            }
            else
            {
                return invokeSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    pDataWeight, bSparseIgnoreZero);
            }
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return invokeIndexedSparseDataScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    _pbSparseData->_pDevData, bSparseIgnoreZero);
            }
            else
            {
                return invokeSparseDataScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                    _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    _pbSparseData->_pDevData, bSparseIgnoreZero);
            }
        }
    }
    else
    {
        std::cout << "unsupported data format of this cost function" << '\n';
        getGpu().Shutdown();
        exit(-1);
    }
}

template<typename T>
float DataSet<T>::CalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Indexed)
        return invokeIndexedHingeError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
    else
        return invokeHingeError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
}

template<typename T>
bool DataSet<T>::CalculateL1OutputDelta(Activation activation,
    uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta,
    float slope, float alpha, float lambda)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedSparseL1OutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
        else
            invokeSparseL1OutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight,
                bSparseIgnoreZero, slope, alpha, lambda);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedL1OutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
        else
            invokeL1OutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
    }
    return true;
}

template<typename T>
bool DataSet<T>::CalculateCrossEntropyOutputDelta(Activation activation,
    uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedSparseCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                pDataWeight, bSparseIgnoreZero);
        else
            invokeSparseCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight,
                bSparseIgnoreZero);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            invokeCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbData->_pDevData, pDataWeight);
    }
    return true;
}

template<typename T>
bool DataSet<T>::CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation,
    uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedSparseScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero);
        else
            invokeSparseScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            invokeScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbData->_pDevData, pDataWeight);
    }
    return true;
}

template<typename T>
bool DataSet<T>::CalculateOutputDelta(Activation activation,
    uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta,
    float slope, float alpha, float lambda)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse) {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean) {
            if (_attributes & DataSetEnums::Indexed)
                invokeIndexedSparseOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                    _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
            else
                invokeSparseOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                    _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight,
                    bSparseIgnoreZero, slope, alpha, lambda);
        }
        else {
            if (_attributes & DataSetEnums::Indexed)
                invokeIndexedSparseAnalogOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                    _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    pDataWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
            else
                invokeSparseAnalogOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                    _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight,
                    _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
        }
    }
    else {
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
        else
            invokeOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
    }
    return true;
}


template<typename T>
bool DataSet<T>::CalculateL2HingeOutputDelta(Activation activation,
    uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta,
    float slope, float alpha, float lambda)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Sparse) {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean) {
            if (_attributes & DataSetEnums::Indexed)
                invokeIndexedSparseL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                    _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
            else
                invokeSparseL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                    _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight,
                    bSparseIgnoreZero, slope, alpha, lambda);
        }
        else {
            if (_attributes & DataSetEnums::Indexed)
                invokeIndexedSparseAnalogL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                    _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                    pDataWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
            else
                invokeSparseAnalogL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                    _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight,
                    _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
        }
    }
    else {
        if (_attributes & DataSetEnums::Indexed)
            invokeIndexedL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
        else
            invokeL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
    }
    return true;
}

template<typename T>
bool DataSet<T>::CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation,
    uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta)
{
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
        {
            invokeIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                _pbSparseData->_pDevData, bSparseIgnoreZero);
        }
        else
        {
            invokeSparseDataScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                _pbSparseData->_pDevData, bSparseIgnoreZero);
        }
    }
    else
    {
        std::cout << "unsupported data format of this cost function" << '\n';
        getGpu().Shutdown();
        exit(-1);
    }
    return true;
}

template<typename T>
bool DataSet<T>::CalculateHingeOutputDelta(Activation activation,
    uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : nullptr;
    if (_attributes & DataSetEnums::Indexed)
        invokeIndexedHingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
    else
        invokeHingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight);
    return true;
}

std::vector<DataSetBase*> LoadNetCDF(const std::string& fname);
bool SaveNetCDF(const std::string& fname, std::vector<DataSetBase*> vDataset);
std::vector<DataSetBase*> LoadImageData(const std::string& fname);
std::vector<DataSetBase*> LoadCSVData(const std::string& fname);
std::vector<DataSetBase*> LoadJSONData(const std::string& fname);
std::vector<DataSetBase*> LoadAudioData(const std::string& name);
