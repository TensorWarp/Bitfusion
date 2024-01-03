#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <span>
#include <ncFile.h>
#include "Types.h"
#include <functional>

using ActivationFunction = std::function<void(void*, uint64_t)>;

class LayerDescriptor;

class Layer {
public:
    friend class Network;
    friend class Weight;
    friend Network* LoadNeuralNetworkNetCDF(const std::string& fname, int batch);
    enum Kind
    {
        Input,
        Hidden,
        Output,
        Target,
    };

    static std::pair<Layer::Kind, std::string> _sKindPair[];
    static const std::map<Kind, std::string> _sKindMap;

    enum Type
    {
        FullyConnected,
        Convolutional,
        Pooling,
        LSTM
    };

    static std::pair<Layer::Type, std::string> _sTypePair[];
    static const std::map<Type, std::string> _sTypeMap;

    enum Attributes
    {
        None = 0x0,
        Sparse = 0x1,
        Denoising = 0x2,
        BatchNormal = 0x4,
    };

    static std::pair<Layer::Attributes, std::string> _sAttributesPair[];
    static const std::map<Attributes, std::string> _sAttributesMap;

    enum Parallelization {
        Data,
        Model,
        Serial,
    };

    static std::pair<Layer::Parallelization, std::string> _sParallelizationPair[];
    static const std::map<Parallelization, std::string> _sParallelizationMap;


private:
    const std::string _name;
    const Kind _kind;
    const Type _type;
    const uint32_t _attributes;
    PoolingFunction _poolingFunction;
    std::string _dataSet;
    DataSetBase* _pDataSet;
    std::vector<std::string> _vSource;
    std::vector<std::string> _vSkip;
    uint32_t _Nx;
    uint32_t _Ny;
    uint32_t _Nz;
    uint32_t _Nw;
    uint32_t _stride;
    uint32_t _localStride;
    uint32_t _maxLocalStride;
    uint32_t _strideBN;
    uint32_t _batch;
    uint32_t _localBatch;
    uint32_t _deltaUpdateCount;
    uint32_t _unitUpdateCount;
    uint32_t _dimensions;
    uint32_t _minX;
    uint32_t _maxX;
    WeightInitialization _weightInit;
    float _weightInitScale;
    float _biasInit;
    float _RELUSlope;
    float _ELUAlpha;
    float _SELULambda;
    bool _bBatchNormalization;
    const uint32_t _kernelX;
    const uint32_t _kernelY;
    const uint32_t _kernelZ;
    const uint32_t _kernelStrideX;
    const uint32_t _kernelStrideY;
    const uint32_t _kernelStrideZ;
    const uint32_t _kernelPaddingX;
    const uint32_t _kernelPaddingY;
    const uint32_t _kernelPaddingZ;
    const uint32_t _kernelDimensions;
    const Activation _activation;
    const float _pDropout;
    bool _bSparse;
    bool _bFastSparse;
    float _sparsenessPenalty_p;
    float _sparsenessPenalty_beta;
    const bool _bDenoising;
    float _weightNorm;
    float _deltaNorm;
    Parallelization _parallelization;
    bool _bTransposeParallelization;
    bool _bDirty;
    cudnnTensorDescriptor_t _scaleBiasMeanVarDescBN;
    cudnnTensorDescriptor_t _tensorDescriptorBN;
    cudnnTensorDescriptor_t _tensorDescriptor;
    cudnnTensorDescriptor_t _oddBatchTensorDescriptor;
    uint32_t _oddBatch;
    cudnnPoolingDescriptor_t _poolingDescriptor;
    cudnnLRNDescriptor_t _LRNDescriptor;
    std::vector<Layer*> _vIncomingLayer;
    std::vector<Weight*> _vIncomingWeight;
    std::vector<Layer*> _vOutgoingLayer;
    std::vector<Weight*> _vOutgoingWeight;
    std::vector<Layer*> _vIncomingLargerLayer;
    std::vector<Weight*> _vIncomingLargerWeight;
    std::vector<Layer*> _vOutgoingLargerLayer;
    std::vector<Weight*> _vOutgoingLargerWeight;
    std::vector<Layer*> _vIncomingSkip;
    std::vector<Layer*> _vOutgoingSkip;
    std::vector<float> _vUnit;
    std::vector<float> _vDelta;
    std::vector<float> _vBuffer1;
    std::vector<float> _vBuffer2;
    std::unique_ptr<GpuBuffer<float>> _pbUnit;
    std::unique_ptr<GpuBuffer<float>> _pbDelta;
    std::unique_ptr<GpuBuffer<float>> _pbDropout;
    std::unique_ptr<GpuBuffer<float>> _pbBuffer1;
    std::unique_ptr<GpuBuffer<float>> _pbBuffer2;
    std::unique_ptr<GpuBuffer<float>> _pbDeltaBN;
    std::unique_ptr<GpuBuffer<float>> _pbScaleGradientBN;
    std::unique_ptr<GpuBuffer<float>> _pbBiasGradientBN;
    std::unique_ptr<GpuBuffer<float>> _pbUnitBN;
    std::unique_ptr<GpuBuffer<float>> _pbScaleBN;
    std::unique_ptr<GpuBuffer<float>> _pbBiasBN;
    std::unique_ptr<GpuBuffer<float>> _pbScaleVelocityBN;
    std::unique_ptr<GpuBuffer<float>> _pbBiasVelocityBN;
    std::unique_ptr<GpuBuffer<float>> _pbScaleGradientVelocityBN;
    std::unique_ptr<GpuBuffer<float>> _pbBiasGradientVelocityBN;
    std::unique_ptr<GpuBuffer<float>> _pbRunningMeanBN;
    std::unique_ptr<GpuBuffer<float>> _pbRunningVarianceBN;
    std::unique_ptr<GpuBuffer<float>> _pbSaveMeanBN;
    std::unique_ptr<GpuBuffer<float>> _pbSaveInvVarianceBN;
    uint32_t _bnCalls;
    int32_t _priority;
    Layer(LayerDescriptor& l, uint32_t batch);
    ~Layer();
    void DestroyCudnnDescriptors();
    void DestroyCudnnDescriptor(cudnnTensorDescriptor_t& descriptor);
    void ResetBatchNormalizationBuffers();
    void DestroyPoolingDescriptor();
    void Allocate(bool validate);
    void Deallocate();
    void SetBatch(uint32_t batch);
    void RefreshParallelization();
    void RefreshState(Network* pNetwork, TrainingMode trainingMode, bool validate);
    void LoadPredictionBatch(uint32_t position, uint32_t batch);
    void LoadTrainingBatch(uint32_t position, uint32_t batch);
    void LoadValidationBatch(uint32_t position, uint32_t batch);
    void ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining = false);
    void ForwardPropagateFullyConnected(uint32_t position, uint32_t batch, bool bTraining);
    void ForwardPropagateConvolutional(uint32_t position, uint32_t batch, bool bTraining);
    void ForwardPropagatePooling(uint32_t position, uint32_t batch, bool bTraining);
    void CalculateActivation(uint32_t batch);
    float CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef);
    void CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef);
    void BackPropagate(uint32_t position, uint32_t batch);
    void BackPropagateConvolutional(uint32_t position, uint32_t batch);
    void BackPropagatePooling(uint32_t position, uint32_t batch);
    void CalculateDropout(uint32_t batch);
    void BackPropagateFullyConnected(uint32_t position, uint32_t batch);
    void UpdateWeights(TrainingMode trainingMode, uint32_t batch, float alpha, float lambda, float lambda1, float mu, float mu1, float t);
    void GenerateDenoisingData();
    void Reduce(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride, uint32_t updateCount);
    void Gather(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride);
    void ClearUpdates();
    void Dump(std::string fname, float* pData);
    bool WriteNetCDF(netCDF::NcFile& nc, uint32_t index);
    void NamedEntityRecognition(uint32_t batch, uint32_t sequenceLength, uint32_t numEntities);
    float* GetIncomingUnitBuffer()
    {
        if (_bBatchNormalization)
            return _pbUnitBN ? _pbUnitBN->_pDevData : NULL;
        else
            return _pbUnit ? _pbUnit->_pDevData : NULL;
    }
    float* GetUnitBuffer() { return _pbUnit ? _pbUnit->_pDevData : NULL; }
    float* GetIncomingDeltaBuffer()
    {
        if (_bBatchNormalization)
            return _pbDeltaBN ? _pbDeltaBN->_pDevData : NULL;
        else
            return _pbDelta ? _pbDelta->_pDevData : NULL;
    }
    float* GetDeltaBuffer() { return _pbDelta ? _pbDelta->_pDevData : NULL; }
    uint64_t GetBufferSize() { return _batch * _stride; }
    cudnnTensorDescriptor_t getTensorDescriptor(uint32_t batch);
    cudnnTensorDescriptor_t getTensorDescriptorBN(uint32_t batch);

public:
    const std::string& GetName() const;

    const std::string& GetDataSetName() const;

    Layer::Kind GetKind() const;
    Layer::Type GetType() const;
    uint32_t GetAttributes() const;

    DataSetBase* GetDataSet() const;

    uint32_t GetNumDimensions() const;

    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetDimensions() const;
    virtual std::vector<double> forward(const std::vector<double>& input) const = 0;
    virtual std::vector<double> backward(const std::vector<double>& error) const = 0;
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetLocalDimensions() const;
    std::tuple<uint32_t, uint32_t, uint32_t> GetKernelDimensions() const;
    std::tuple<uint32_t, uint32_t, uint32_t> GetKernelStride() const;
    bool GetUnits(std::vector<float>& vUnit);
    bool GetUnits(float* pUnit);
    bool GetUnits(std::span<float> units);
    bool SetUnits(const std::vector<float>& vUnit);
    bool GetDeltas(std::vector<float>& vUnit);
    bool GetDeltas(float* pUnit);
    bool SetDeltas(const std::vector<float>& vUnit);

};


std::ostream& operator<< (std::ostream& out, Layer::Kind& k);
std::ostream& operator<< (std::ostream& out, Layer::Type& t);
std::ostream& operator<< (std::ostream& out, Layer::Parallelization& p);
std::ostream& operator<< (std::ostream& out, Layer::Attributes& a);

struct LayerDescriptor
{
    std::string _name;
    Layer::Kind _kind;
    Layer::Type _type;
    PoolingFunction _poolingFunction;
    std::string _dataSet;
    std::vector<std::string> _vSource;
    std::vector<std::string> _vSkip;
    uint32_t _Nx;
    uint32_t _Ny;
    uint32_t _Nz;
    uint32_t _Nw;
    uint32_t _dimensions;
    bool _bDimensionsProvided;
    WeightInitialization _weightInit;
    float _weightInitScale;
    float _biasInit;
    uint32_t _kernelX;
    uint32_t _kernelY;
    uint32_t _kernelZ;
    uint32_t _kernelStrideX;
    uint32_t _kernelStrideY;
    uint32_t _kernelStrideZ;
    uint32_t _kernelPaddingX;
    uint32_t _kernelPaddingY;
    uint32_t _kernelPaddingZ;
    uint32_t _kernelDimensions;
    std::vector<float> _vScaleBN;
    std::vector<float> _vBiasBN;
    std::vector<float> _vRunningMeanBN;
    std::vector<float> _vRunningVarianceBN;
    float _weightNorm;
    float _deltaNorm;
    float _pDropout;
    Activation _activation;
    float _sparsenessPenalty_p;
    float _sparsenessPenalty_beta;
    uint32_t _attributes;
    float _RELUSlope;
    float _ELUAlpha;
    float _SELULambda;
    LayerDescriptor();
};

struct MinMaxSpan {
    uint32_t minX;
    uint32_t maxX;
    uint32_t span;
};

bool LoadLayerDescriptorNetCDF(const std::string& fname, netCDF::NcFile& nc, uint32_t index, LayerDescriptor& ld);
std::ostream& operator<< (std::ostream& out, LayerDescriptor& d);
uint32_t MPI_Bcast_LayerDescriptor(LayerDescriptor& d);
