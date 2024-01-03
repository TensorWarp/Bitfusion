#pragma once

#include <memory>
#include <map>
#include "GpuSort.h"
#include "GpuTypes.h"
#include "Layer.h"
#include "Types.h"
#include "Weight.h"
#include <stdio.h>
#include <cstdint>
#include <exception>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct NetworkDescriptor;

class Network {
public:
    friend class Layer;

    friend class Weight;

    friend void GpuContext::SetNeuralNetwork(Network* pNetwork);

    enum Kind {
        FeedForward,

        AutoEncoder,
    };

    static std::pair<Network::Kind, std::string> _sKindPair[];

    static std::map<Network::Kind, std::string> _sKindMap;

private:

    friend Network* LoadNeuralNetworkJSON(const std::string& fname, const uint32_t batch, const std::vector<DataSetBase*>& vDataSet);

    friend Network* LoadNeuralNetworkNetCDF(const std::string& fname, const uint32_t batch);

    friend Network* ImportAutoEncoder(const std::string& fname, uint32_t batch);

    std::string _name;

    uint32_t _batch;

    uint32_t _localBatch;

    uint32_t _position;
    uint32_t _localPosition;

    bool _bExamplesFound;

    bool _bAllDataLoaded;

    uint32_t _examples;

    const Kind _kind;

    ErrorFunction _errorFunction;

    TrainingMode _trainingMode;

    Mode _mode;

    uint32_t _epochs;

    uint32_t _indices;

    uint32_t _batches;

    float _decay;

    float _LRN_k;

    uint32_t _LRN_n;

    float _LRN_alpha;

    float _LRN_beta;

    float _RELUSlope;

    float _ELUAlpha;

    float _SELULambda;

    uint32_t _maxout_k;
    bool _bSparsenessPenalty;

    float _sparsenessPenalty_p;

    float _sparsenessPenalty_beta;

    bool _bDenoising;

    float _denoising_p;

    float _deltaBoost_one;

    float _deltaBoost_zero;

    float _SMCE_oneTarget;

    float _SMCE_zeroTarget;

    float _SMCE_oneScale;

    float _SMCE_zeroScale;

    bool _bShuffleIndices;

    uint32_t _shuffleIndices;

    uint32_t* _pShuffleIndex;
    std::unique_ptr<GpuBuffer<uint32_t>> _pbShuffleIndex;

    std::unique_ptr<GpuSort<uint32_t, uint32_t>> _pShuffleIndexSort;

    std::string _checkpoint_name;

    int32_t _checkpoint_interval;

    int32_t _checkpoint_epochs;

    std::vector<Layer*> _vLayer;

    std::vector<Layer*> _vInputLayer;

    std::vector<Layer*> _vOutputLayer;

    std::vector<Weight*> _vWeight;

    std::vector<Weight*> _vSharedWeight;

    std::vector<DataSetBase*> _vData;

    std::vector<Layer*> _vFPOrder;

    std::vector<Layer*> _vBPOrder;

    std::map<std::string, Layer*> _mLayer;

    bool _bDirty;

    bool _bClearVelocity;
    size_t _scratchBufferSize;

    std::shared_ptr<float[]> _pbScratchBuffer;

    uint32_t _maxStride;

    uint32_t _sendIndex;

    uint32_t _receiveIndex;

    std::unique_ptr<GpuBuffer<float>> _pbP2PBuffer[2];

    float* _pPeerBuffer[2];

    std::unique_ptr<float[]> _pCPUBuffer;

    size_t _CUDNNWorkspaceSize;

    size_t _maxCUDNNWorkspaceSize;

    std::unique_ptr<GpuBuffer<uint8_t>> _pbCUDNNWorkspace;

    bool _verbose;

public:
    ~Network();

    void ClearDataSets();

    void ValidateLayerDimensionMatch(const Layer* layer, const Layer* otherLayer, bool checkDimensionMatch = false) const;

    void LoadDataSets(std::vector<DataSetBase*> datasets);

    void ConnectLayers(NetworkDescriptor& d);

    void ValidateLayerExistence(const Layer* layer) const;

    void ConnectSkipLayers(Layer* layer, Layer* skipLayer);

    void ConnectPoolingLayers(Layer* layer, Layer* sourceLayer);

    void HandleException(const std::exception& e, const std::string& errorMessage);

    void InitializeWeights(NetworkDescriptor& d);

    void Randomize();

    bool Validate();
    
    float Train(uint32_t epochs = 1, float alpha = (float)0.1, float lambda = (float)0.001, float lambda1 = (float)0.0, float mu = (float)0.1, float mu1 = 0.0);
    
    void ApplyWeightDecayAndMomentum(float lambda, float mu);

    void PredictBatch(uint32_t layers = 0);

    void SaveBatch(std::string fname);

    void DumpBatch(FILE* fp);

    void SaveLayer(const std::string& fname, const std::string& layer);

    void DumpLayer(FILE* fp, const std::string& layer);

    void SaveBatch(const std::string& fname);

    void SaveWeights(const std::string& fname, const std::string& inputLayer, const std::string& outputLayer);

    bool LockWeights(const std::string& inputLayer, const std::string& outputLayer);

    bool UnlockWeights(const std::string& inputLayer, const std::string& outputLayer);

    void SetBatch(uint32_t batch);

    void SetPosition(uint32_t position);

    bool SetDecay(float decay);
    void SetTrainingMode(TrainingMode mode);

    void SetShuffleIndices(bool bShuffleIndices);

    void SetCPUValidate(bool bValidate);

    void SetClearVelocity(bool bClear) { _bClearVelocity = bClear; };

    bool SaveNetCDF(const std::string& fname);

    class LayerNotFoundException : public std::runtime_error {
    public:
        LayerNotFoundException(const std::string& layerName)
            : std::runtime_error("Layer not found: " + layerName) {}
    };

    class DimensionMismatchException : public std::runtime_error {
    public:
        DimensionMismatchException(const std::string& layerName1, const std::string& layerName2)
            : std::runtime_error("Dimension mismatch between layers: " + layerName1 + " and " + layerName2) {}
    };

    unsigned int GetBatch() const;

    uint32_t GetExamples() const;

    uint32_t GetPosition() const;

    Weight* GetWeight(const std::string& inputLayer, const std::string& outputLayer) const;

    uint64_t GetBufferSize(const std::string& layer) const;

    Layer* GetLayer(const std::string& layer) const;

    std::vector<const Layer*>::iterator GetLayers(Layer::Kind layerKind, std::vector<const Layer*>& layers) const;

    std::vector<std::string> GetLayers() const;

    const std::string& GetName() const;

    std::tuple<float, uint32_t, float, float> GetLRN() const;

    std::tuple<float> GetDecay() const;

    std::tuple<uint32_t> GetMaxout() const;

    std::tuple<float, float> GetSparsenessPenalty() const;

    std::tuple<float> GetDenoising() const;

    std::tuple<float, float> GetDeltaBoost() const;

    std::tuple<float, float, float, float> GetSMCE() const;

    std::tuple<bool> GetShuffleIndices() const;

    std::tuple<std::string, int32_t> GetCheckPoint() const;

    bool GetDebugLevel() const { return _verbose; }

    float* GetUnitBuffer(const std::string& layer);

    float* GetDeltaBuffer(const std::string& layer);

    float* GetWeightBuffer(const std::string& inputLayer, const std::string& outputLayer);

    std::shared_ptr<float[]> GetScratchBuffer(size_t size = 0);

    float* GetP2PSendBuffer();

    float* GetP2PReceiveBuffer();

    float* GetP2PCPUBuffer();

    float* GetPeerBuffer() const;

    float* GetPeerBackBuffer() const;

    bool P2P_Bcast(void* pBuffer, size_t size);

    bool P2P_Allreduce(float* pBuffer, size_t size);

    bool SetLRN(float k = (float)2.0, uint32_t n = 5, float alpha = (float)0.0001, float beta = (float)0.75);

    bool SetMaxout(uint32_t k = 2);

    bool SetSparsenessPenalty(float p = 0.0f, float beta = 0.0f);

    bool SetDenoising(float p = 0.0f);

    bool SetDeltaBoost(float one = 1.0f, float zero = 1.0f);

    bool SetSMCE(float oneTarget = 0.9f, float zeroTarget = 0.1f, float oneScale = 1.0f, float zeroScale = 1.0f);

    bool SetCheckpoint(std::string_view name, int32_t interval) noexcept;

    void SetDebugLevel(bool verbose) { _verbose = verbose; }

private:
    void CalculateDerivedLayerDimensions(NetworkDescriptor& d);
    void CalculatePropagationOrder();

    bool GenerateNetworkGraph();

    void AllocatePeerBuffers();

    void DeallocatePeerBuffers();

    void SwapPeerBuffers();

    void LoadBatch();

    void PredictTrainingBatch(uint32_t layers = 0);

    void PredictValidationBatch(uint32_t layers = 0);

    void RefreshShuffleBuffers();

    void ShuffleIndices();

    std::tuple<float, float> CalculateError(float lambda, float lambda1);

    void ProcessTrainingBatch(uint32_t pos, float alpha, float lambda, float lambda1);

    void ApplyGradientClipping(float threshold);

    void ClearUpdates();

    void BackPropagate();

    void UpdateWeights(float alpha, float lambda, float lambda1, float mu, float mu1);

    void CalculateExamples(const std::string& layer, uint32_t k, GpuBuffer<float>* pbKey, GpuBuffer<unsigned int>* pbValue);

    Network(NetworkDescriptor& nd, uint32_t batch = DefaultBatch);

    void InitializeLayers(NetworkDescriptor& d);

    void RefreshState();

    void Shuffle();

    void SetCUDNNWorkspace(size_t size);
};

std::ostream& operator<< (std::ostream& out, Network::Kind& k);

struct NetworkDescriptor
{
    std::string _name;

    Network::Kind _kind;

    ErrorFunction _errorFunction;

    std::vector<LayerDescriptor> _vLayerDescriptor;

    std::vector<WeightDescriptor> _vWeightDescriptor;

    bool _bShuffleIndices;

    float _decay;

    uint32_t _maxout_k;

    float _LRN_k;

    uint32_t _LRN_n;

    float _LRN_alpha;

    float _LRN_beta;

    float _RELUSlope;

    float _ELUAlpha;

    float _SELULambda;

    bool _bSparsenessPenalty;

    float _sparsenessPenalty_p;

    float _sparsenessPenalty_beta;

    bool _bDenoising;

    float _denoising_p;

    float _deltaBoost_one;

    float _deltaBoost_zero;

    float _SMCE_oneTarget;

    float _SMCE_zeroTarget;

    float _SMCE_oneScale;

    float _SMCE_zeroScale;

    std::string _checkpoint_name;

    int32_t _checkpoint_interval;

    int32_t _checkpoint_epochs;

    bool _bConvLayersCalculated;

    NetworkDescriptor();
};

std::ostream& operator<< (std::ostream& out, NetworkDescriptor& d);

Network* LoadNeuralNetworkNetCDF(const std::string& fname, const uint32_t batch = DefaultBatch);

Network* LoadNeuralNetworkJSON(const std::string& fname, const uint32_t batch = DefaultBatch, const std::vector<DataSetBase*>& vDataSet = std::vector<DataSetBase*>());

bool SaveNeuralNetworkJSON(const Network& net, const std::string& fname);

bool SaveNeuralNetworkNetCDF(const Network& net, const std::string& jname);

Network* ImportAutoEncoder(const std::string& fname, uint32_t batch = DefaultBatch);
