#pragma once

#include <memory>
#include <cudnn.h>
#include <string>
#include <map>
#include <vector>
#include <netcdf>

class Weight {
public:
    enum Transform {
        Convolution,
        Linear
    };
    static std::pair<Weight::Transform, std::string> _sTransformPair[];
    static std::map<Weight::Transform, std::string> _sTransformMap;

private:
    friend class Network;
    friend class Layer;
    friend Network* LoadNeuralNetworkNetCDF(const std::string& fname, uint32_t batch);

    Layer& _inputLayer;
    Layer& _outputLayer;
    const bool _bShared;
    const bool _bTransposed;
    Transform _transform;
    bool _bLocked;
    Weight* _pSharedWeight;
    uint32_t _sharingCount;
    uint32_t _updateCount;
    uint32_t _dimensionality;
    uint64_t _width;
    uint64_t _height;
    uint64_t _length;
    uint64_t _depth;
    uint64_t _breadth;
    uint32_t _widthStride;
    uint32_t _heightStride;
    uint32_t _lengthStride;
    uint64_t _size;
    uint64_t _biasSize;
    uint64_t _localSize;
    uint64_t _localBiasSize;
    float _norm;
    cudnnTensorDescriptor_t _convBiasTensor;
    cudnnFilterDescriptor_t _convFilterDesc;
    cudnnConvolutionDescriptor_t _convDesc;
    int _convPad[3];
    bool _bQuantized;
    float _minValue;
    float _maxValue;
    float* _data;
    bool _bSerialized;
    cudnnConvolutionFwdAlgo_t _convFWAlgo;
    cudnnConvolutionBwdFilterAlgo_t _convBWWeightAlgo;
    cudnnConvolutionBwdDataAlgo_t _convBWDeltaAlgo;
    std::vector<float> _vWeight;
    std::vector<float> _vBias;
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
    void UpdateWeights(TrainingMode trainingMode);
    void ClearVelocity();
    void Randomize();
    void Lock();
    void Unlock();
    void RefreshState(Network* pNetwork, TrainingMode trainingMode);
    bool WriteNetCDF(netCDF::NcFile& nc, uint32_t index, float* pWeight, float* pBias);
    float* GetWeightBuffer() { return _pbWeight ? _pbWeight->_pDevData : nullptr; }
    float* GetWeightGradientBuffer() { return _pbWeightGradient ? _pbWeightGradient->_pDevData : nullptr; }
    uint64_t GetBufferSize() const { return _localSize; }

public:
    bool CopyWeights(const Weight* pWeight);
    bool SetWeights(const std::vector<float>& vWeight);
    bool SetBiases(const std::vector<float>& vBias);
    bool GetWeights(std::vector<float>& vWeight);
    void ApplyAdaptiveLearningRate(float learningRateDecay, float mu);
    bool GetBiases(std::vector<float>& vBias);
    bool GetDimensions(std::vector<uint64_t>& dimensions) const;
    void copySingleProcessor(float* pBuffer);
    void copyMultipleProcessors(float* pBuffer);
    void processOutgoingBiggerThanIncoming(float* pWeight);
    void processIncomingBiggerThanOutgoing(float* pWeight);
    void writeToOutput(const std::vector<float>& data, const std::filesystem::path& outputPath);
    void Dump(const std::filesystem::path& filename, float* pBuffer);
    bool SetNorm(float norm) { _norm = norm; return true; };
    void NormalizeWeights();
    void PruneWeights(float threshold);
    void QuantizeWeights(int numBits);
    void DequantizeWeights();
    bool SerializeWeights(const std::string& filename);
    void VisualizeWeights();
    void ApplyRegularization(float lambda);
    void InitializeWeightsWithPretrainedModel(const std::string& pretrainedModelFile);
    void UpdateWeightAverages(float momentum);
    void ClipWeightGradients(float gradientClipThreshold);
    float CalculateRegularizationLoss(float lambda);
    void SaveWeightSnapshot(const std::string& snapshotFilename);
    void LoadWeightSnapshot(const std::string& snapshotFilename);
    void ApplyGradientNoise(float noiseMagnitude);
    void AdjustLearningRate(float newLearningRate);
    void ScaleWeights(float scaleFactor);
    void WeightQuantization(float quantizationLevels);
    void ApplyGradientClipping(float gradientClipValue);
    void ApplyMomentum(float momentumCoefficient);
    void ApplyWeightSmoothing(float smoothingFactor);
    void ApplyWeightPerturbation(float perturbationStrength);
    void RegularizeWeights(float regularizationStrength);
    void InitializeWeightsRandomly();
    void ApplyWeightTying(const Weight& sourceWeight);
    void ApplyWeightMixing(const Weight& weight1, const Weight& weight2, float mixingFactor);
    void ApplyWeightSparsityMask(const std::vector<bool>& sparsityMask);
    void ComputeWeightSimilarity(const Weight& otherWeight);
    void ApplyWeightNormalization(float targetMean, float targetStdDev);
    void ApplyWeightFreezingSchedule(float freezeFraction, int epochThreshold);
    void ApplyWeightQuantizationAwareTraining();
    void ApplyWeightReplication(int replicationFactor);
    void ApplyGradientAveraging(const std::vector<Weight*>& gradientWeights);
    void ApplyWeightInterpolation(const Weight& weight1, const Weight& weight2, float interpolationFactor);
    void ApplyLayerNormalization(float epsilon);
    void ApplyWeightDropout(float dropoutRate);
    void ApplyWeightAugmentation(const Weight& augmentationWeight);
    void ApplyDynamicWeightClipping(float minClip, float maxClip);
    void ApplyQuantizationErrorMinimization(float targetQuantizationError);
    void ApplyWeightShuffling();
    void ApplyWeightGrouping(const std::vector<Weight*>& groupedWeights);
    void ApplyWeightVisualization(const std::string& visualizationType);
    void ApplyWeightRounding(int numDecimalPlaces);
    void ApplyWeightMerging(const std::vector<Weight*>& mergeWeights);
    void ApplyWeightDecomposition(const std::vector<Weight*>& decompositionWeights);
    void ApplyWeightDithering(float ditheringFactor);
    void ApplyWeightEnsemble(const std::vector<Weight*>& ensembleWeights);
    void ApplyKnowledgeDistillation(const Weight& teacherWeight, float temperature);
    void InitializeWeightsFromDistribution(const std::string& distributionType, float mean, float stddev);
    void ShareWith(Weight& otherWeight);
    void Unshare();
    void ApplyWeightDecay(float weightDecayRate);
    void ApplyDropout(float dropoutRate);
    void ApplyWeightConstraints(float minConstraint, float maxConstraint);
    void VisualizeWeightHistogram();
    void VisualizeWeightHeatmap();
    float CalculateWeightSparsity();
    void Freeze();
    void Unfreeze();
    void SetRegularizationHyperparameters(float lambda, float lambda1);
    void MergeWeights(const std::vector<Weight*>& weightsToMerge);
    std::vector<Weight*> SplitWeights(int numSplits);
};

struct WeightDescriptor {
    std::string _inputLayer;
    std::string _outputLayer;
    uint64_t _width;
    uint64_t _height;
    uint64_t _length;
    uint64_t _depth;
    uint64_t _breadth;
    std::vector<float> _vWeight;
    std::vector<float> _vBias;
    bool _bShared;
    bool _bTransposed;
    bool _bLocked;
    float _norm;
    std::string _sourceInputLayer;
    std::string _sourceOutputLayer;

    WeightDescriptor();
};

bool LoadWeightDescriptorNetCDF(const std::string& fname, netCDF::NcFile& nc, uint32_t index, WeightDescriptor& wd);
std::ostream& operator<< (std::ostream& out, WeightDescriptor& d);
uint32_t MPI_Bcast_WeightDescriptor(WeightDescriptor& d);
