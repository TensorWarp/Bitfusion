#pragma once

#ifndef __NVCC__

#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <map>
#include <tuple>

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
        Pooling
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
    void InitializeDescriptors();

    void InitializeBatchNormalization(LayerDescriptor& d);

    void InitializePoolingDescriptor();
    ~Layer();
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
    void ApplyAttentionMask(float* pMask, uint32_t batch);
    void MultiHeadAttention(uint32_t batch, uint32_t headCount, uint32_t querySize, uint32_t keySize, uint32_t valueSize);
    void CalculateAttentionScores(uint32_t batch, uint32_t headCount, uint32_t querySize, uint32_t keySize);
    float CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef);
    void CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef);
    void BackPropagate(uint32_t position, uint32_t batch);
    void BackPropagateConvolutional(uint32_t position, uint32_t batch);
    void BackPropagatePooling(uint32_t position, uint32_t batch);
    void ApplyLayerNormalization(float* pInput, uint32_t batch);
    void AddResidualConnections(float* pResidual, uint32_t batch);

    void PositionalEncoding(uint32_t maxSeqLength, uint32_t batchSize);

    void ApplyPositionwiseFeedforward(uint32_t batch, uint32_t hiddenSize);
    void CalculatePositionwiseFeedforward(uint32_t batch, uint32_t hiddenSize);

    void LayerNormalization(uint32_t batch);

    void MaskedMultiHeadAttentionForDecoder(uint32_t batch, uint32_t headCount, uint32_t querySize, uint32_t keySize, uint32_t valueSize, float* pMask);
    void CrossAttentionForDecoder(uint32_t batch, uint32_t headCount, uint32_t querySize, uint32_t keySize, uint32_t valueSize);
    void DecoderPositionwiseFeedforwardLayer(uint32_t batch, uint32_t hiddenSize, float dropoutProbability);
    void ComputeDecoderOutput(uint32_t batch, uint32_t headCount, uint32_t querySize, uint32_t keySize, uint32_t valueSize, uint32_t hiddenSize, float dropoutProbability);
    void ComputeDecoderPositionwiseFeedforwardOutput(uint32_t batch, uint32_t hiddenSize);
    void ComputeDecoderFinalOutput(uint32_t batch, uint32_t headCount, uint32_t valueSize, uint32_t hiddenSize, float dropoutProbability);

    void BeamSearch(uint32_t beamSize);
    void ApplyLayerDropout(float dropoutProbability);
    void UpdateLearningRateSchedule(float* pLearningRate, uint32_t globalStep);
    void ApplyWeightDecay(float weightDecayRate);
    void MergeSubwordEmbeddings(uint32_t batchSize, uint32_t subwordEmbeddingSize);
    void ExtractFeaturesForVisualization(uint32_t batch);
    void CalculateDropout(uint32_t batch);

    void CalculateSparsenessPenalty(float p, float beta, uint32_t batch);
    void CalculateHadamardProduct(uint32_t batch, float scale);
    void NormalizeDeltas(float deltaNorm, uint32_t batch);
    void BatchNormalization(uint32_t batch);
    void MatrixMultiplication(float* pA, float* pB, float* pC, int m, int n, int k, int lda, int ldb, int ldc);
    void BackPropagateFullyConnected(uint32_t position, uint32_t batch);
    void UpdateWeights(TrainingMode trainingMode, uint32_t batch, float alpha, float lambda, float lambda1, float mu, float mu1, float t);
    void GenerateDenoisingData();
    void Reduce(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride, uint32_t updateCount);
    void Gather(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride);
    void ClearUpdates();
    void Dump(std::string fname, float* pData);
    bool WriteNetCDF(netCDF::NcFile& nc, uint32_t index);

    void ApplyMaskedSoftmax(uint32_t batch, uint32_t seqLength, float* pMask);
    void ApplyGatedLinearUnit(uint32_t batch);
    void ComputeSelfAttention(uint32_t batch, uint32_t headCount, uint32_t querySize, uint32_t keySize, uint32_t valueSize);
    void ApplyLayerNormalizationAndResidual(float* pResidual, uint32_t batch);
    void GenerateRandomData(uint32_t dataSize);
    void ApplyCustomRegularization(float regularizationStrength);
    void ComputeGradientNorms(uint32_t batch);
    void ApplyCustomActivationFunction();
    void MaskedMultiHeadAttention(uint32_t batch, uint32_t headCount, uint32_t querySize, uint32_t keySize, uint32_t valueSize, float* pMask);
    void FeedForwardLayer(uint32_t batch, uint32_t hiddenSize, float dropoutProbability);
    void LayerNormalizationAndResidualConnection(float* pResidual, uint32_t batch);
    void ComputeAttentionHeads(uint32_t batch, uint32_t headCount, uint32_t querySize, uint32_t keySize, uint32_t valueSize);
    void ApplyLayerDropoutForAttention(float dropoutProbability);
    void ApplyLayerDropoutForFFN(float dropoutProbability);
    void ComputeMultiHeadAttentionOutput(uint32_t batch, uint32_t headCount, uint32_t valueSize);
    void PositionwiseFeedforwardLayer(uint32_t batch, uint32_t hiddenSize, float dropoutProbability);
    void ComputePositionwiseFeedforwardOutput(uint32_t batch, uint32_t hiddenSize);
    void ComputeEncoderOutput(uint32_t batch, uint32_t headCount, uint32_t querySize, uint32_t keySize, uint32_t valueSize, uint32_t hiddenSize, float dropoutProbability);
    void MultiHeadAttentionForDecoder(uint32_t batch, uint32_t headCount, uint32_t querySize, uint32_t keySize, uint32_t valueSize);
    void ComputeDecoderAttentionOutput(uint32_t batch, uint32_t headCount, uint32_t valueSize);

    void CrossModalAttention(uint32_t batch, uint32_t headCount, uint32_t querySize, uint32_t keySize, uint32_t valueSize);
    void ComputeCrossModalAttentionOutput(uint32_t batch, uint32_t headCount, uint32_t valueSize);
    void CrossModalPositionwiseFeedforwardLayer(uint32_t batch, uint32_t hiddenSize, float dropoutProbability);
    void ComputeCrossModalPositionwiseFeedforwardOutput(uint32_t batch, uint32_t hiddenSize);

    void ApplyTokenEmbeddings(float* tokenEmbeddings, uint32_t batch, uint32_t sequenceLength);
    void ComputeTokenEmbeddingOutput(uint32_t batch, uint32_t sequenceLength, uint32_t embeddingSize);

    void ApplyPositionalEmbeddings(float* positionalEmbeddings, uint32_t batch, uint32_t sequenceLength);
    void ComputePositionalEmbeddingOutput(uint32_t batch, uint32_t sequenceLength, uint32_t embeddingSize);

    void MaskedLanguageModeling(uint32_t batch, uint32_t sequenceLength, uint32_t vocabSize);
    void ComputeLanguageModelingOutput(uint32_t batch, uint32_t sequenceLength, uint32_t vocabSize);

    void TranslationModeling(uint32_t batch, uint32_t sourceSeqLength, uint32_t targetSeqLength, uint32_t sourceEmbeddingSize, uint32_t targetEmbeddingSize);
    void ComputeTranslationModelingOutput(uint32_t batch, uint32_t targetSeqLength, uint32_t vocabSize);

    void DependencyParsing(uint32_t batch, uint32_t sequenceLength, uint32_t numLabels);
    void ComputeDependencyParsingOutput(uint32_t batch, uint32_t sequenceLength, uint32_t numLabels);

    void SemanticRoleLabeling(uint32_t batch, uint32_t sequenceLength, uint32_t numRoles);
    void ComputeSemanticRoleLabelingOutput(uint32_t batch, uint32_t sequenceLength, uint32_t numRoles);

    void SentimentAnalysis(uint32_t batch, uint32_t sequenceLength, uint32_t numClasses);
    void ComputeSentimentAnalysisOutput(uint32_t batch, uint32_t sequenceLength, uint32_t numClasses);

    void NamedEntityRecognition(uint32_t batch, uint32_t sequenceLength, uint32_t numEntities);
    void ComputeNamedEntityRecognitionOutput(uint32_t batch, uint32_t sequenceLength, uint32_t numEntities);
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

    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetLocalDimensions() const;
    std::tuple<uint32_t, uint32_t, uint32_t> GetKernelDimensions() const;
    std::tuple<uint32_t, uint32_t, uint32_t> GetKernelStride() const;
    bool GetUnits(std::vector<float>& vUnit);
    bool GetUnits(float* pUnit);
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
#endif
