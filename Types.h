#pragma once

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <netcdf>
#ifndef __NVCC__
#include <tuple>
#include <json/json.h>
#endif
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


static const float VERSION       = 0.9f;
static const float MIN_ERROR        = 1.0e-12f;
static const float MIN_ACTIVATION   = 0.000001f;
static const float MAX_ACTIVATION   = 0.999999f;

template <typename T> struct GpuBuffer;

enum 
{
    DefaultBatch    = 512
};

enum Mode {
    Prediction = 0,
    Training = 1,
    Validation = 2,
    Unspecified = 3
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

    std::string                          _name;
    DataSetEnums::DataType        _dataType;
    uint32_t                        _attributes;
    uint32_t                        _examples;
    uint32_t                        _uniqueExamples;
    uint32_t                        _localExamples;
    uint32_t                        _dimensions;
    uint32_t                        _width;
    uint32_t                        _height;
    uint32_t                        _length;
    uint32_t                        _stride;
    DataSetEnums::Sharding        _sharding;
    uint32_t                        _minX;
    uint32_t                        _maxX;
    uint64_t                        _sparseDataSize;
    float                         _sparseDensity;
    std::vector<uint64_t>                _vSparseStart;
    std::unique_ptr<GpuBuffer<uint64_t>> _pbSparseStart;
    std::vector<uint64_t>                _vSparseEnd;
    std::unique_ptr<GpuBuffer<uint64_t>> _pbSparseEnd;
    std::vector<uint32_t>                _vSparseIndex;
    std::unique_ptr<GpuBuffer<uint32_t>> _pbSparseIndex;
    std::vector<float>                 _vDataWeight;
    std::unique_ptr<GpuBuffer<float>>  _pbDataWeight;
    std::vector<uint32_t>                _vIndex;
    std::unique_ptr<GpuBuffer<uint32_t>> _pbIndex;
    std::unique_ptr<GpuBuffer<float>>  _pbDenoisingRandom;
    
    std::vector<uint64_t>                _vSparseDatapointCount;
    std::vector<uint32_t>                _vSparseMaxDatapointCount;
    std::vector<uint32_t>                _vSparseMultiDatapointCount;
    std::vector<uint32_t>                _vSparseTransposedStart;
    uint64_t                        _sparseTransposedIndices;
    std::unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedStart;
    std::unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedEnd;
    std::unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedIndex;
    std::unique_ptr<GpuBuffer<float>>  _pbSparseTransposedData;    

    bool                            _bDenoising;
    bool                            _bDirty;
    bool                            _bStreaming;
    bool                            _bIndexed;
    uint32_t                        _batch;

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
    virtual std::vector<std::tuple<uint64_t, uint64_t> > getMemoryUsage() = 0;
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
    virtual bool CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, float* pUnit, float beta = (float)0.0) = 0;
    virtual bool CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, float* pUnit, float beta = (float)0.0) = 0;
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

    virtual void LoadDenseData(const void *srcData) = 0;

    virtual void CopyDenseData(const void *srcData) = 0;

    virtual void LoadSparseData(const uint64_t *srcSparseStart, const uint64_t *srcSparseEnd, const void *srcSparseData,
                                const uint32_t *srcSparseIndex) = 0;

    virtual void CopySparseData(const uint64_t *srcSparseStart, const uint64_t *srcSparseEnd, const void *srcSparseData,
                                const uint32_t *srcSparseIndex) = 0;

    virtual void LoadSparseData(const long *srcSparseStart, const long *srcSparseEnd, const void *srcSparseData,
                                const long *srcSparseIndex) = 0;

    virtual void CopySparseData(const long *srcSparseStart, const long *srcSparseEnd, const void *srcSparseData,
                                const long *srcSparseIndex) = 0;

    virtual void LoadIndexedData(const uint32_t *srcIndexedData) = 0;

    virtual void LoadDataWeight(const float *srcWeightData) = 0;

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
    DataSet(uint32_t examples, const DataSetDimensions &dim, const std::string &name = "");

    DataSet(uint32_t examples, uint32_t uniqueExamples, const DataSetDimensions &dim, const std::string &name = "");

    DataSet(uint32_t examples, float sparseDensity, const DataSetDimensions &dim, bool isWeighted = false, const std::string &name = "");

    DataSet(uint32_t examples, uint32_t uniqueExamples, size_t sparseDataSize, const DataSetDimensions &dim,
              bool isIndexed = false, bool isWeighted = false, const std::string &name = "");

    void LoadDenseData(const void *srcData) override;

    void CopyDenseData(const void *srcData) override;

    void LoadSparseData(const uint64_t *srcSparseStart, const uint64_t *srcSparseEnd, const void *srcSparseData,
                        const uint32_t *srcSparseIndex) override;

    void CopySparseData(const uint64_t *srcSparseStart, const uint64_t *srcSparseEnd, const void *srcSparseData,
                        const uint32_t *srcSparseIndex) override;

    void LoadSparseData(const long *srcSparseStart, const long *srcSparseEnd, const void *srcSparseData,
                        const long *srcSparseIndex) override;

    void CopySparseData(const long *srcSparseStart, const long *srcSparseEnd, const void *srcSparseData,
                        const long *srcSparseIndex) override;

    void LoadIndexedData(const uint32_t *srcIndexedData) override;

    void LoadDataWeight(const float *srcWeightData) override;

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

template<typename T> bool DataSet<T>::LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) 
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

template<typename T> bool DataSet<T>::LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit) 
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

template<typename T> bool DataSet<T>::CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, float* pUnit, float beta) 
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL; 
    if (_attributes & DataSetEnums::Boolean)
    {       
        if (_attributes & DataSetEnums::Indexed)        
            kCalculateIndexedSparseZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, pUnit, beta);
        else
            kCalculateSparseZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, pUnit, beta);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)  
            kCalculateIndexedSparseAnalogZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, pUnit, beta);
        else
            kCalculateSparseAnalogZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, pUnit, beta);
    }
    return true;
}

template<typename T> bool DataSet<T>::CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, float* pUnit, float beta) 
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    if (_attributes & DataSetEnums::Boolean)
    {
        if (_attributes & DataSetEnums::Indexed)          
            kCalculateIndexedSparseDenoisedZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, pUnit, beta);
        else
            kCalculateSparseDenoisedZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, pUnit, beta);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)   
            kCalculateIndexedSparseAnalogDenoisedZ(position, batch, stride, pWeight, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, pUnit, beta);
        else
            kCalculateSparseAnalogDenoisedZ(position, batch, stride, pWeight, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, pUnit, beta);
    }
    return true;
}

template<typename T> bool DataSet<T>::CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, Layer* pLayer)
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
            kCalculateIndexedSparseTransposedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        else
            kCalculateSparseTransposedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)   
            kCalculateIndexedSparseTransposedAnalogMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData); 
        else
            kCalculateSparseTransposedAnalogMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData); 
    }
    
    return true;
}

template<typename T> bool DataSet<T>::CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, Layer* pLayer)
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
            kCalculateIndexedSparseTransposedDenoisedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
        else
            kCalculateSparseTransposedDenoisedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed) 
            kCalculateIndexedSparseTransposedAnalogDenoisedMatrix(position, batch, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);  
        else
            kCalculateSparseTransposedAnalogDenoisedMatrix(position, batch, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, _pbDenoisingRandom->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pSparseTransposedData);  
    }
    
#if 0    
    vector<uint32_t> vSparseTransposedStart(53120);        
    vector<uint32_t> vSparseTransposedEnd(53120);
    _pbSparseTransposedStart->Download(&vSparseTransposedStart[0]);
    _pbSparseTransposedEnd->Download(&vSparseTransposedEnd[0]);
    for (uint32_t i = 0; i < 53120; i++)
    printf("%6u %9u %9u %9u %9u\n", i, vSparseTransposedStart[i], vSparseTransposedEnd[i], vSparseTransposedEnd[i] - vSparseTransposedStart[i], (uint32_t)_vSparseDatapointCount[i]);
    exit(-1);
#endif       
    return true;
}


template<typename T> bool DataSet<T>::CalculateSparseTransposedWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, float* pDelta, float* pWeightGradient)
{    
    if ((_attributes & DataSetEnums::Boolean) && !(_attributes & DataSetEnums::Weighted))
        kCalculateSparseTransposedWeightGradient(alpha, beta, m, n, _pbSparseTransposedStart->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, pDelta, pWeightGradient);
    else
        kCalculateSparseTransposedAnalogWeightGradient(alpha, beta, m, n, _pbSparseTransposedStart->_pDevData, _pbSparseTransposedEnd->_pDevData, _pbSparseTransposedIndex->_pDevData, _pbSparseTransposedData->_pDevData, pDelta, pWeightGradient);               
    return true;
}

template<typename T> float DataSet<T>::CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;    
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean)
        {
           if (_attributes & DataSetEnums::Indexed) 
           {
                return kCalculateIndexedSparseL1Error(position, batch, stride, pUnit,
                       _pbIndex->_pDevData,
                       _pbSparseStart->_pDevData, 
                       _pbSparseEnd->_pDevData, 
                       _pbSparseIndex->_pDevData,
                       pDataWeight,
                       bSparseIgnoreZero);
           }
           else
           {
                return kCalculateSparseL1Error(position, batch, stride, pUnit, 
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
                return kCalculateIndexedSparseAnalogL1Error(position, batch, stride, pUnit,
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
                return kCalculateSparseAnalogL1Error(position, batch, stride, pUnit, 
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
            return kCalculateIndexedL1Error(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateL1Error(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T> float DataSet<T>::CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;    
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;        
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseL2Error(position, batch, stride, pUnit,
                       _pbIndex->_pDevData,
                       _pbSparseStart->_pDevData, 
                       _pbSparseEnd->_pDevData, 
                       _pbSparseIndex->_pDevData,
                       pDataWeight,
                       bSparseIgnoreZero);                
            }
            else
            {
                return kCalculateSparseL2Error(position, batch, stride, pUnit, 
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
                return kCalculateIndexedSparseAnalogL2Error(position, batch, stride, pUnit, 
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
                return kCalculateSparseAnalogL2Error(position, batch, stride, pUnit, 
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
            return kCalculateIndexedL2Error(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateL2Error(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);            
    }
}




template<typename T> float DataSet<T>::CalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL; 
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;        
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseL2HingeError(position, batch, stride, pUnit,
                       _pbIndex->_pDevData,
                       _pbSparseStart->_pDevData, 
                       _pbSparseEnd->_pDevData, 
                       _pbSparseIndex->_pDevData,
                       pDataWeight,
                       bSparseIgnoreZero);                
            }
            else
            {
                return kCalculateSparseL2HingeError(position, batch, stride, pUnit, 
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
                return kCalculateIndexedSparseAnalogL2HingeError(position, batch, stride, pUnit, 
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
                return kCalculateSparseAnalogL2HingeError(position, batch, stride, pUnit, 
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
            return kCalculateIndexedL2HingeError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateL2HingeError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);            
    }
}

template<typename T> float DataSet<T>::CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;  
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)   
        {
            return kCalculateIndexedSparseCrossEntropyError(position, batch, stride, pUnit,
                   _pbIndex->_pDevData,
                   _pbSparseStart->_pDevData, 
                   _pbSparseEnd->_pDevData, 
                   _pbSparseIndex->_pDevData,
                   pDataWeight,
                   bSparseIgnoreZero);             
        }
        else
        {
            return kCalculateSparseCrossEntropyError(position, batch, stride, pUnit,
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
            return kCalculateIndexedCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else    
            return kCalculateCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T> float DataSet<T>::CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL; 
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;  
        if (_attributes & DataSetEnums::Indexed)         
        {
            return kCalculateIndexedSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                   _pbIndex->_pDevData,
                   _pbSparseStart->_pDevData, 
                   _pbSparseEnd->_pDevData, 
                   _pbSparseIndex->_pDevData,
                   pDataWeight,
                   bSparseIgnoreZero);
        }
    else
        {
            return kCalculateSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
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
            return kCalculateIndexedScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T> float DataSet<T>::CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL; 
    if (_attributes & DataSetEnums::Sparse)
    {    
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)         
            {            
                return kCalculateIndexedSparseMultinomialCrossEntropyError(position, batch, stride, pUnit,
                        _pbIndex->_pDevData,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pDataWeight);
            }
            else
            {            
                return kCalculateSparseMultinomialCrossEntropyError(position, batch, stride, pUnit,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pDataWeight);
            }                
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)         
            {   
                return kCalculateIndexedSparseAnalogMultinomialCrossEntropyError(position, batch, stride, pUnit,
                        _pbIndex->_pDevData,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pDataWeight,
                        _pbSparseData->_pDevData);
            }
            else
            {
                return kCalculateSparseAnalogMultinomialCrossEntropyError(position, batch, stride, pUnit,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pDataWeight,
                        _pbSparseData->_pDevData);                
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)    
            return kCalculateIndexedMultinomialCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateMultinomialCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T> float DataSet<T>::CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL; 
    if (_attributes & DataSetEnums::Sparse)   
    {
        if (_attributes & DataSetEnums::Boolean)
        {           
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                        _pbIndex->_pDevData,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pDataWeight);
            }
            else
            {
                return kCalculateSparseMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pDataWeight);                
            }  
        }
        else
        {
            if (_attributes & DataSetEnums::Indexed)
            {            
                return kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                        _pbIndex->_pDevData,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pDataWeight,
                        _pbSparseData->_pDevData);
            }
            else
            {
                return kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pDataWeight,
                        _pbSparseData->_pDevData);
            }
        }
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            return kCalculateIndexedMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            return kCalculateMultinomialScaledMarginalCrossEntropyError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
    }
}

template<typename T> float DataSet<T>::CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Boolean)
        {
            if (_attributes & DataSetEnums::Indexed)
            {
                return kCalculateIndexedSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit, 
                        _pbIndex->_pDevData,
                        _pbSparseStart->_pDevData, 
                        _pbSparseEnd->_pDevData, 
                        _pbSparseIndex->_pDevData,
                        pDataWeight,
                        bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
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
                return kCalculateIndexedSparseDataScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                                _pbIndex->_pDevData,
                                _pbSparseStart->_pDevData,
                                _pbSparseEnd->_pDevData,
                                _pbSparseIndex->_pDevData,
                                _pbSparseData->_pDevData, bSparseIgnoreZero);
            }
            else
            {
                return kCalculateSparseDataScaledMarginalCrossEntropyError(position, batch, stride, pUnit,
                                _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                                _pbSparseData->_pDevData, bSparseIgnoreZero);
            }
        }
    }
    else
    {
        std::cout << "unsupported data format of this cost function" << std::endl;
        getGpu().Shutdown();
        exit(-1);
    }
}

template<typename T> float DataSet<T>::CalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL; 
    if (_attributes & DataSetEnums::Indexed)
        return kCalculateIndexedHingeError(position, batch, stride, pUnit, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
    else
        return kCalculateHingeError(position, batch, stride, pUnit, _pbData->_pDevData, pDataWeight);
}

template<typename T> bool DataSet<T>::CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float slope, float alpha, float lambda)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    if (_attributes & DataSetEnums::Sparse)
    {        
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
        else
            kCalculateSparseL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
        else
            kCalculateL1OutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
    }
    return true;
}

template<typename T> bool DataSet<T>::CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;  
    if (_attributes & DataSetEnums::Sparse)
    {      
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero);
        else
            kCalculateSparseCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero);

    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            kCalculateCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight);
    }
    return true;
}

template<typename T> bool DataSet<T>::CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero);
        else
            kCalculateSparseScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero);
    }
    else
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
        else
            kCalculateScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight);
    }
    return true;
}

template<typename T> bool DataSet<T>::CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float slope, float alpha, float lambda)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    if (_attributes & DataSetEnums::Sparse) {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;        
        if (_attributes & DataSetEnums::Boolean) 
        {
            if (_attributes & DataSetEnums::Indexed)
                kCalculateIndexedSparseOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
        } 
        else 
        {
            if (_attributes & DataSetEnums::Indexed)
                kCalculateIndexedSparseAnalogOutputDelta(activation, position, batch, stride, pUnit,  pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseAnalogOutputDelta(activation, position, batch, stride, pUnit,  pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
        }
    } 
    else 
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
        else
            kCalculateOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
    }
    return true;
}


template<typename T> bool DataSet<T>::CalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float slope, float alpha, float lambda)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    if (_attributes & DataSetEnums::Sparse) {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;        
        if (_attributes & DataSetEnums::Boolean) 
        {
            if (_attributes & DataSetEnums::Indexed)
                kCalculateIndexedSparseL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, bSparseIgnoreZero, slope, alpha, lambda);
        } 
        else 
        {
            if (_attributes & DataSetEnums::Indexed)
                kCalculateIndexedSparseAnalogL2HingeOutputDelta(activation, position, batch, stride, pUnit,  pDelta, _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
            else
                kCalculateSparseAnalogL2HingeOutputDelta(activation, position, batch, stride, pUnit,  pDelta, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData, pDataWeight, _pbSparseData->_pDevData, bSparseIgnoreZero, slope, alpha, lambda);
        }
    } 
    else 
    {
        if (_attributes & DataSetEnums::Indexed)
            kCalculateIndexedL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
        else
            kCalculateL2HingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight, slope, alpha, lambda);
    }
    return true;
}

template<typename T> bool DataSet<T>::CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation,
                uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta)
{
    if (_attributes & DataSetEnums::Sparse)
    {
        bool bSparseIgnoreZero = _attributes & DataSetEnums::SparseIgnoreZero;
        if (_attributes & DataSetEnums::Indexed)
        {
            kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                             _pbIndex->_pDevData, _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                            _pbSparseData->_pDevData, bSparseIgnoreZero);
        }
        else
        {
            kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta(activation, position, batch, stride, pUnit, pDelta,
                            _pbSparseStart->_pDevData, _pbSparseEnd->_pDevData, _pbSparseIndex->_pDevData,
                            _pbSparseData->_pDevData, bSparseIgnoreZero);
        }
    } 
    else 
    {
        std::cout << "unsupported data format of this cost function" << std::endl;
        getGpu().Shutdown();
        exit(-1);
    }
    return true;
}

template<typename T> bool DataSet<T>::CalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta)
{
    float* pDataWeight = (_attributes & DataSetEnums::Weighted) ? _pbDataWeight->_pDevData : NULL;
    if (_attributes & DataSetEnums::Indexed)
        kCalculateIndexedHingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbIndex->_pDevData, _pbData->_pDevData, pDataWeight);
    else
        kCalculateHingeOutputDelta(activation, position, batch, stride, pUnit, pDelta, _pbData->_pDevData, pDataWeight);
    return true;
}

std::vector<DataSetBase*> LoadNetCDF(const std::string& fname);
bool SaveNetCDF(const std::string& fname, std::vector<DataSetBase*> vDataset);
std::vector<DataSetBase*> LoadImageData(const std::string& fname);
std::vector<DataSetBase*> LoadCSVData(const std::string& fname);
std::vector<DataSetBase*> LoadJSONData(const std::string& fname);
std::vector<DataSetBase*> LoadAudioData(const std::string& name);
