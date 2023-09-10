#pragma once

#include <memory>
#include <cudnn.h>
#include <string>
#include <map>
#include <vector>
#include <netcdf>

class Weight {
public:
    enum Transform
    {
        Convolution,
        Linear
    };
    static std::pair<Weight::Transform, std::string> _sTransformPair[];
    static std::map<Weight::Transform, std::string> _sTransformMap;

private:
    friend class Network;
    friend class Layer;
    friend Network* LoadNeuralNetworkNetCDF(const std::string& fname, uint32_t batch);

    Layer&                        _inputLayer;
    Layer&                        _outputLayer;
    const bool                      _bShared;
    const bool                      _bTransposed;
    Transform                       _transform;
    bool                            _bLocked;
    Weight*                       _pSharedWeight;
    uint32_t                        _sharingCount;
    uint32_t                        _updateCount;
    uint32_t                        _dimensionality;
    uint64_t                        _width;
    uint64_t                        _height;
    uint64_t                        _length;
    uint64_t                        _depth;
    uint64_t                        _breadth;
    uint32_t                        _widthStride;
    uint32_t                        _heightStride;
    uint32_t                        _lengthStride;
    uint64_t                        _size;
    uint64_t                        _biasSize;
    uint64_t                        _localSize;
    uint64_t                        _localBiasSize;
    float                         _norm;
    cudnnTensorDescriptor_t         _convBiasTensor;
    cudnnFilterDescriptor_t         _convFilterDesc;
    cudnnConvolutionDescriptor_t    _convDesc;
    int                             _convPad[3];
    cudnnConvolutionFwdAlgo_t       _convFWAlgo;
    cudnnConvolutionBwdFilterAlgo_t _convBWWeightAlgo;
    cudnnConvolutionBwdDataAlgo_t   _convBWDeltaAlgo;
    std::vector<float>                 _vWeight;
    std::vector<float>                 _vBias;
    std::unique_ptr<GpuBuffer<float>> _pbWeight;
    std::unique_ptr<GpuBuffer<float>> _pbBias;
    std::unique_ptr<GpuBuffer<float>> _pbWeightGradient;
    std::unique_ptr<GpuBuffer<float>> _pbBiasGradient;
    std::unique_ptr<GpuBuffer<float>> _pbWeightVelocity;
    std::unique_ptr<GpuBuffer<float>> _pbBiasVelocity;
    std::unique_ptr<GpuBuffer<float>> _pbWeightGradientVelocity;
    std::unique_ptr<GpuBuffer<float>> _pbBiasGradientVelocity;
    Weight(Layer& inputLayer, Layer& outputLayer, bool bShared = false, bool bTransposed = false, bool bLocked = false, float maxNorm = 0.0f);
    void initializeLayers();
    void initializeConvolution();
    void initializeLinear();
    void setWeightValues(const std::vector<std::vector<float>>& values);
    void randomizeWeightMatrix();
    std::vector<std::vector<float>> _weightMatrix;
    ~Weight();
    void ClearSharedGradient();
    void ClearGradient();
    float CalculateRegularizationError(float lambda, float lambda1);
    void ClearVelocity();
    void Randomize();
    void Lock();
    void Unlock();
    void Dump(std::string fname, float* pBuffer);
    void RefreshState(Network* pNetwork, TrainingMode trainingMode);
    void UpdateWeights(TrainingMode trainingMode, uint32_t batch, float alpha, float lambda, float lambda1, float mu, float mu1, float t);
    bool WriteNetCDF(netCDF::NcFile& nc, uint32_t index, float* pWeight = NULL, float* pBias = NULL);
    float* GetWeightBuffer() { return _pbWeight ? _pbWeight->_pDevData : NULL; }
    float* GetWeightGradientBuffer() { return _pbWeightGradient ? _pbWeightGradient->_pDevData : NULL; }
    uint64_t GetBufferSize() { return _localSize; }
public:
    bool CopyWeights(const Weight* pWeight);
    bool SetWeights(const std::vector<float>& vWeight);
    bool SetBiases(const std::vector<float>& vBias);
    bool GetWeights(std::vector<float>& vWeight);
    bool GetBiases(std::vector<float>& vBias);      
    bool GetDimensions(std::vector<uint64_t>& dimensions);
    bool SetNorm(float norm) { _norm = norm; return true; };
};


struct WeightDescriptor
{
    std::string                  _inputLayer;
    std::string                  _outputLayer;
    uint64_t                _width;
    uint64_t                _height;
    uint64_t                _length;
    uint64_t                _depth;
    uint64_t                _breadth;
    std::vector<float>         _vWeight;
    std::vector<float>         _vBias;
    bool                    _bShared;
    bool                    _bTransposed;
    bool                    _bLocked;
    float                 _norm;
    std::string                  _sourceInputLayer;
    std::string                  _sourceOutputLayer;

    WeightDescriptor();
};

bool LoadWeightDescriptorNetCDF(const std::string& fname, netCDF::NcFile& nc, uint32_t index, WeightDescriptor& wd);
std::ostream& operator<< (std::ostream& out, WeightDescriptor& d);
uint32_t MPI_Bcast_WeightDescriptor(WeightDescriptor& d);
