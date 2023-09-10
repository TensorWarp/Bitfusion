#include "GpuTypes.h"
#include "Types.h"
#include "Kernels.cuh"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <future>
#include <execution>
#include "ThreadPool.h"
#include <optional>

using namespace netCDF;
using namespace netCDF::exceptions;

WeightDescriptor::WeightDescriptor() :
    _width(1),
    _height(1),
    _length(1),
    _breadth(1),
    _depth(1),
    _bShared(false),
    _bTransposed(false),
    _bLocked(false),
    _norm((float)0.0)
{

}

static void DumpTensor(cudnnTensorDescriptor_t t)
{
    cudnnDataType_t dataType;
    int ndims;
    std::vector<int> vDim(16);
    std::vector<int> vStride(16);
    cudnnStatus_t cudnnStatus = cudnnGetTensorNdDescriptor(t, 8, &dataType, &ndims, vDim.data(), vStride.data());
    CUDNNERROR(cudnnStatus, "cudnnGetTensorNdDescriptor error");    
    std::cout << "Tensor:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType: " << dataType << std::endl;
    for (int i = 0; i < ndims; i++)
        std::cout << i << " " << vDim[i] << " " << vStride[i] << std::endl;
    std::cout << std::endl;
    
}

static void DumpFilter(cudnnFilterDescriptor_t f)
{
    cudnnDataType_t dataType;
    cudnnTensorFormat_t format;
    int ndims;
    std::vector<int> vDim(16);
    cudnnStatus_t cudnnStatus = cudnnGetFilterNdDescriptor(f, 5, &dataType, &format, &ndims, vDim.data());
    CUDNNERROR(cudnnStatus, "cudnnGetFilterNdDescriptor error");        
    std::cout << "Filter:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType: " << dataType << std::endl;
    std::cout << "Format:   " << format << std::endl;    
    for (int i = 0; i < ndims; i++)
        std::cout << i << " " << vDim[i] << " " << std::endl;
    std::cout << std::endl;
    
}

static void DumpConvolution(cudnnConvolutionDescriptor_t c)
{
    cudnnDataType_t dataType;
    cudnnConvolutionMode_t mode;
    int ndims;
    std::vector<int> vPad(16);
    std::vector<int> vStride(16);
    std::vector<int> vUpscale(16);        
    cudnnStatus_t cudnnStatus = cudnnGetConvolutionNdDescriptor(c, 5, &ndims, vPad.data(), vStride.data(), vUpscale.data(), &mode, &dataType);
    CUDNNERROR(cudnnStatus, "cudnnGetConvolutionNdDescriptor error");      
    std::cout << "Convolution:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType:      " << dataType << std::endl;
    std::cout << "Mode:          " << mode << std::endl;    
    for (int i = 0; i < ndims; i++)
        std::cout << i << " " << vPad[i] << " " << vStride[i] << " " << vUpscale[i] << std::endl;
    std::cout << std::endl;
    
}


std::optional<std::string> GetAttribute(netCDF::NcFile& nc, const std::string& attrName) {
    try {
        NcGroupAtt attribute = nc.getAtt(attrName);
        if (!attribute.isNull()) {
            std::string value;
            attribute.getValues(value);
            return value;
        }
    }
    catch (NcException& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    return std::nullopt;
}

bool LoadWeightDescriptorNetCDF(const std::string& fname, netCDF::NcFile& nc, uint32_t index, WeightDescriptor& wd) {
    bool bResult = true;

    if (getGpu()._id == 0) {
        std::string wstring = "weight" + std::to_string(index) + "_";

        auto inputLayerOpt = GetAttribute(nc, wstring + "inputLayer");
        auto outputLayerOpt = GetAttribute(nc, wstring + "outputLayer");
        auto normOpt = GetAttribute(nc, wstring + "norm");
        auto bSharedOpt = GetAttribute(nc, wstring + "bShared");
        auto sourceInputLayerOpt = GetAttribute(nc, wstring + "sourceInputLayer");
        auto sourceOutputLayerOpt = GetAttribute(nc, wstring + "sourceOutputLayer");
        auto bTransposedOpt = GetAttribute(nc, wstring + "bTransposed");
        auto bLockedOpt = GetAttribute(nc, wstring + "bLocked");
        auto widthOpt = GetAttribute(nc, wstring + "width");
        auto heightOpt = GetAttribute(nc, wstring + "height");
        auto lengthOpt = GetAttribute(nc, wstring + "length");
        auto depthOpt = GetAttribute(nc, wstring + "depth");
        auto breadthOpt = GetAttribute(nc, wstring + "breadth");

        if (inputLayerOpt && outputLayerOpt && normOpt && bSharedOpt && bLockedOpt && widthOpt && heightOpt &&
            lengthOpt && depthOpt && breadthOpt) {
            wd._inputLayer = *inputLayerOpt;
            wd._outputLayer = *outputLayerOpt;
            wd._norm = std::stof(*normOpt);
            wd._bShared = std::stoi(*bSharedOpt) != 0;
            wd._bLocked = std::stoi(*bLockedOpt);
            wd._width = std::stoi(*widthOpt);
            wd._height = std::stoi(*heightOpt);
            wd._length = std::stoi(*lengthOpt);
            wd._depth = std::stoi(*depthOpt);
            wd._breadth = std::stoi(*breadthOpt);

            if (wd._bShared) {
                if (sourceInputLayerOpt && sourceOutputLayerOpt && bTransposedOpt) {
                    wd._sourceInputLayer = *sourceInputLayerOpt;
                    wd._sourceOutputLayer = *sourceOutputLayerOpt;
                    wd._bTransposed = std::stoi(*bTransposedOpt) != 0;
                }
                else {
                    std::cerr << "Exception: Missing shared weight attributes." << std::endl;
                    bResult = false;
                }
            }

            NcDim biasDim = nc.getDim(wstring + "biasDim");
            NcVar biasVar = nc.getVar(wstring + "bias");
            wd._vBias.resize(biasDim.getSize());
            biasVar.getVar(wd._vBias.data());

            if (!wd._bShared) {
                NcDim weightDim = nc.getDim(wstring + "weightDim");
                NcVar weightVar = nc.getVar(wstring + "weights");
                wd._vWeight.resize(weightDim.getSize());
                weightVar.getVar(wd._vWeight.data());
            }
        }
        else {
            std::cerr << "Exception: Missing required weight attributes." << std::endl;
            bResult = false;
        }

#if 0
        printf("Weights %d %lu %lu\n", index, _vWeight.size(), _vBias.size());
        for (int i = 0; i < 20; i++)
            printf("%3d %16.8f %16.8f\n", i, _vWeight[i], _vBias[i]);
#endif
    }

    return bResult;
}

// MPI broadcast function with exception handling
template <typename T>
void safe_MPI_Bcast(T* data, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    if (MPI_Bcast(data, count, datatype, root, comm) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Bcast failed");
    }
}

uint32_t MPI_Bcast_WeightDescriptor(std::shared_ptr<WeightDescriptor> d) {
    try {
        MPI_Bcast_string(d->_inputLayer);
        MPI_Bcast_string(d->_outputLayer);
        safe_MPI_Bcast(&d->_bShared, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        safe_MPI_Bcast(&d->_bTransposed, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        safe_MPI_Bcast(&d->_bLocked, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        safe_MPI_Bcast(&d->_norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast_string(d->_sourceInputLayer);
        MPI_Bcast_string(d->_sourceOutputLayer);
        safe_MPI_Bcast(&d->_width, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        safe_MPI_Bcast(&d->_height, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        safe_MPI_Bcast(&d->_length, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        safe_MPI_Bcast(&d->_depth, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        safe_MPI_Bcast(&d->_breadth, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        uint64_t weights = d->_vWeight.size();
        safe_MPI_Bcast(&weights, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        d->_vWeight.resize(weights);
        safe_MPI_Bcast(d->_vWeight.data(), static_cast<int>(weights), MPI_FLOAT, 0, MPI_COMM_WORLD);

        uint64_t biases = d->_vBias.size();
        safe_MPI_Bcast(&biases, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        d->_vBias.resize(biases);
        safe_MPI_Bcast(d->_vBias.data(), static_cast<int>(biases), MPI_FLOAT, 0, MPI_COMM_WORLD);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in MPI_Bcast_WeightDescriptor: " << e.what() << std::endl;
        return 1;
    }
}

std::ostream& operator<< (std::ostream& out, WeightDescriptor& d)
{
    if (getGpu()._id == 0)
    {
        out << "Input Layer:        " << d._inputLayer << std::endl;
        out << "Output Layer:       " << d._outputLayer << std::endl;
        out << "Width               " << d._width << std::endl;
        out << "Height              " << d._height << std::endl;
        out << "Length              " << d._length << std::endl;
        out << "Depth               " << d._depth << std::endl;   
        out << "Breadth             " << d._breadth << std::endl;
        out << "bShared:            " << std::boolalpha << d._bShared << std::endl;
        out << "bTransposed:        " << std::boolalpha << d._bTransposed << std::endl;
        if (d._bShared)
        {
            out << "sourceInputLayer:   " << d._sourceInputLayer << std::endl;
            out << "sourceOutputLayer:  " << d._sourceOutputLayer << std::endl;
        }
        out << "bLocked:            " << std::boolalpha << d._bLocked << std::endl;
        out << "norm:               " << d._norm << std::endl;
    }
    return out;
}


Weight::Weight(Layer& inputLayer, Layer& outputLayer, bool bShared, bool bTransposed, bool bLocked, float norm) :
    _inputLayer(inputLayer),
    _outputLayer(outputLayer),
    _dimensionality(2),
    _width(1),
    _height(1),
    _length(1),
    _depth(1),
    _breadth(1),
    _sharingCount(1),
    _updateCount(0),
    _bShared(bShared),
    _bTransposed(bTransposed),
    _bLocked(bLocked),
    _norm(norm),
    _pSharedWeight(nullptr),
    _pbWeight(),
    _pbBias(),
    _pbWeightGradient(),
    _pbBiasGradient(),
    _pbWeightVelocity(),
    _pbBiasVelocity(),
    _pbWeightGradientVelocity()
{
    initializeLayers();

    if (_outputLayer._type == Layer::Type::Convolutional)
    {
        _transform = Convolution;
        initializeConvolution();
    }
    else
    {
        _transform = Linear;
        initializeLinear();
    }
}

void Weight::initializeLayers()
{
    _inputLayer._vOutgoingLayer.push_back(&_outputLayer);
    _outputLayer._vIncomingLayer.push_back(&_inputLayer);
    _inputLayer._vOutgoingWeight.push_back(this);
    _outputLayer._vIncomingWeight.push_back(this);
}

void Weight::initializeConvolution()
{
    _transform = Convolution;

    cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&_convBiasTensor);
    CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to create tensor descriptor");
    cudnnStatus = cudnnCreateFilterDescriptor(&_convFilterDesc);
    CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to create filter descriptor");
    cudnnStatus = cudnnCreateConvolutionDescriptor(&_convDesc);
    CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to create convolution descriptor");

    std::vector<int> vFilterDim(5, 1);
    switch (_outputLayer._dimensions)
    {
    case 2:
        vFilterDim[0] = _outputLayer._Ny;
        vFilterDim[1] = _inputLayer._Ny;
        vFilterDim[2] = _inputLayer._kernelX;
        _dimensionality = 3;
        break;

    case 3:
        vFilterDim[0] = _outputLayer._Nz;
        vFilterDim[1] = _inputLayer._Nz;
        vFilterDim[2] = _outputLayer._kernelY;
        vFilterDim[3] = _outputLayer._kernelX;
        _dimensionality = 4;
        break;

    case 4:
        vFilterDim[0] = _outputLayer._Nw;
        vFilterDim[1] = _inputLayer._Nw;
        vFilterDim[2] = _outputLayer._kernelZ;
        vFilterDim[3] = _outputLayer._kernelY;
        vFilterDim[4] = _outputLayer._kernelX;
        _dimensionality = 5;
        break;
    }
    cudnnStatus = cudnnSetFilterNdDescriptor(_convFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, _outputLayer._dimensions + 1, vFilterDim.data());
    CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to set filter descriptor");

    _width = vFilterDim[0];
    _height = vFilterDim[1];
    _length = vFilterDim[2];
    _depth = vFilterDim[3];
    _breadth = vFilterDim[4];

    std::vector<int> vConvPad(3, 0);
    std::vector<int> vConvStride(3, 1);
    std::vector<int> vConvUpscale(3, 1);
    switch (_outputLayer._dimensions)
    {
    case 2:
        vConvPad[0] = _outputLayer._kernelPaddingX;
        vConvStride[0] = _outputLayer._kernelStrideX;
        break;

    case 3:
        vConvPad[0] = _outputLayer._kernelPaddingY;
        vConvStride[0] = _outputLayer._kernelStrideY;
        vConvPad[1] = _outputLayer._kernelPaddingX;
        vConvStride[1] = _outputLayer._kernelStrideX;
        break;

    case 4:
        vConvPad[0] = _outputLayer._kernelPaddingZ;
        vConvStride[0] = _outputLayer._kernelStrideZ;
        vConvPad[1] = _outputLayer._kernelPaddingY;
        vConvStride[1] = _outputLayer._kernelStrideY;
        vConvPad[2] = _outputLayer._kernelPaddingX;
        vConvStride[2] = _outputLayer._kernelStrideX;
        break;
    }
    cudnnStatus = cudnnSetConvolutionNdDescriptor(_convDesc, _outputLayer._kernelDimensions, vConvPad.data(), vConvStride.data(), vConvUpscale.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    CUDNNERROR(cudnnStatus, "Weight::Weight: cudnnSetConvolutionNdDescriptor failed.");

    std::vector<int> vBiasDim(5, 1);
    std::vector<int> vBiasStride(5, 1);
    vBiasDim[1] = vFilterDim[0];
    cudnnStatus = cudnnSetTensorNdDescriptor(_convBiasTensor, CUDNN_DATA_FLOAT, _outputLayer._dimensions + 1, vBiasDim.data(), vBiasStride.data());
    CUDNNERROR(cudnnStatus, "Weight::Weight: Unable to set bias tensor descriptor");

    _size = vFilterDim[0] * vFilterDim[1] * _outputLayer._kernelX * _outputLayer._kernelY * _outputLayer._kernelZ;
    _biasSize = vFilterDim[0];
    _localSize = _size;
    _localBiasSize = _biasSize;

    if (getGpu()._id == 0)
    {
        printf("Weight::Weight: Allocating %" PRIu64 " bytes (%d x %d x %u", _localSize * sizeof(float), vFilterDim[0], vFilterDim[1], _outputLayer._kernelX);
        if (_outputLayer._dimensions >= 3)
            printf(" x %u", _outputLayer._kernelY);
        if (_outputLayer._dimensions >= 4)
            printf(" x %u", _outputLayer._kernelZ);
        printf(") for convolutional weights between layers %s and %s\n", _inputLayer._name.c_str(), _outputLayer._name.c_str());
    }
}

void Weight::initializeLinear() {
    _transform = Linear;

    uint32_t outgoingSize = _outputLayer._stride * 3;
    uint32_t incomingSize = _inputLayer._stride * 2;

    if (outgoingSize > incomingSize) {
        _inputLayer._vOutgoingLargerLayer.push_back(&_outputLayer);
        _inputLayer._vOutgoingLargerWeight.push_back(this);
        _width = _outputLayer._localStride;
        _height = _inputLayer._stride;
    }
    else {
        _outputLayer._vIncomingLargerLayer.push_back(&_inputLayer);
        _outputLayer._vIncomingLargerWeight.push_back(this);
        _width = _outputLayer._stride;
        _height = _inputLayer._localStride;
    }
    _localSize = _width * _height * 4 * 2 * 3;
    _localBiasSize = _outputLayer._localStride;
    _size = _outputLayer._stride * _inputLayer._stride * 4 * 2 * 3;
    _biasSize = _outputLayer._stride;

    if (getGpu()._id == 0)
        std::cout << "Weight::Weight: Allocating " << _localSize * sizeof(float) << " bytes (" << _width << ", " << _height << ") for fully connected weights between layers " << _inputLayer._name << " and " << _outputLayer._name << std::endl;
}

void Weight::setWeightValues(const std::vector<std::vector<float>>& values) {
    if (values.size() != _width || values[0].size() != _height) {
        std::cerr << "Error: Invalid weight matrix dimensions." << std::endl;
        return;
    }

    _weightMatrix = values;
}

void Weight::randomizeWeightMatrix() {
    _weightMatrix.resize(_width, std::vector<float>(_height));
    for (uint32_t i = 0; i < _width; ++i) {
        for (uint32_t j = 0; j < _height; ++j) {
            _weightMatrix[i][j] = static_cast<float>(rand()) / (RAND_MAX)-0.5f;
        }
    }
}

Weight::~Weight()
{
}

void Weight::ClearVelocity()
{
    cudaMemset(_pbWeightVelocity->_pDevData, 0, _localSize * sizeof(float));
    cudaMemset(_pbBiasVelocity->_pDevData, 0, _localBiasSize * sizeof(float));
    if (_pbWeightGradientVelocity)
        cudaMemset(_pbWeightGradientVelocity->_pDevData, 0, _localSize * sizeof(float));
    if (_pbBiasGradientVelocity)
        cudaMemset(_pbBiasGradientVelocity->_pDevData, 0, _localBiasSize * sizeof(float));
}

void Weight::ClearGradient()
{
    cudaMemset(_pbWeightGradient->_pDevData, 0, _localSize * sizeof(float));
}

void Weight::Randomize()
{
    if (!_bShared)
    {
        float scale, bias;        
        switch (_outputLayer._weightInit)
        {
        case CaffeXavier:
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale               = _outputLayer._weightInitScale * 2.0f * sqrtf(3.0f / _outputLayer._stride);
            bias                = 0.5f * scale;                 
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
            break;
            
        case Xavier:
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale               = _outputLayer._weightInitScale * sqrtf(6.0f / (_outputLayer._stride + _inputLayer._stride));
            bias                = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
            break;
     
        case Uniform:
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale               = 2.0f * _outputLayer._weightInitScale;
            bias                = 0.5f * scale;                 
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);  
            break;
            
        case Gaussian:
            curandGenerateNormal(getGpu()._RNG, _pbWeight->_pDevData, _localSize, 0.0f, _outputLayer._weightInitScale);
            break;        
            
        case UnitBall:      
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale               = _outputLayer._weightInitScale;              
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, 0.0f);     
            break;
            
        case SELU:
            curandGenerateNormal(getGpu()._RNG, _pbWeight->_pDevData, _localSize, 0.0f, 1.0f / _inputLayer._stride);
            break;
          
        case Constant:
            cudaMemset(_pbWeight->_pDevData, 0, _localSize * sizeof(float));
            kScaleAndBias(_pbWeight->_pDevData, _localSize, (float)0.0, _outputLayer._weightInitScale); 
            break;
        };
    }
        
    cudaMemset(_pbBias->_pDevData, 0, _localBiasSize * sizeof(float));
    kScaleAndBias(_pbBias->_pDevData, _localBiasSize, (float)0.0, -_outputLayer._biasInit); 
}

void Weight::Lock()
{
    _bLocked                = true;
}

void Weight::Unlock()
{
    _bLocked                = false;
}

template<typename T>
concept GpuBufferType = std::is_same_v<T, GpuBuffer<float>>;

template<GpuBufferType BufferType>
void ResetBufferIfNeeded(std::unique_ptr<BufferType>& buffer, size_t size)
{
    if (!buffer)
        buffer = std::make_unique<BufferType>(size);
}

template<TrainingMode Mode, GpuBufferType BufferType>
void ResetBufferBasedOnMode(std::unique_ptr<BufferType>& buffer, size_t size)
{
    if constexpr (Mode != TrainingMode::SGD)
        ResetBufferIfNeeded(buffer, size);
    else
        buffer.reset();
}

void Weight::RefreshState(Network* pNetwork, TrainingMode mode)
{
    const size_t MAX_THREADS = 4;
    ThreadPool pool(MAX_THREADS);

    std::mutex cv_m;
    std::condition_variable cv;
    bool gpu_ready = false;

    auto checkGPUStatus = [&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        {
            std::lock_guard<std::mutex> lock(cv_m);
            gpu_ready = true;
        }
        cv.notify_one();
        };

    auto refreshBuffers = [this, &mode]() {
        ResetBufferBasedOnMode<TrainingMode::AdaDelta>(_pbWeightGradientVelocity, _localSize);
        ResetBufferBasedOnMode<TrainingMode::AdaDelta>(_pbBiasGradientVelocity, _localBiasSize);
        ResetBufferBasedOnMode<TrainingMode::Adam>(_pbWeightGradientVelocity, _localSize);
        ResetBufferBasedOnMode<TrainingMode::Adam>(_pbBiasGradientVelocity, _localBiasSize);
        ResetBufferBasedOnMode<TrainingMode::SGD>(_pbWeightVelocity, _localSize);
        ResetBufferBasedOnMode<TrainingMode::SGD>(_pbBiasVelocity, _localBiasSize);
        };

    auto cudnnProcessing = [this, pNetwork]() {
        if (_outputLayer._type == Layer::Type::Convolutional)
        {
            std::cout << "Getting algorithm between " << _inputLayer._name << " and " << _outputLayer._name << "\n";
            size_t workspaceSize;
            cudnnConvolutionFwdAlgoPerf_t convolutionAlgo;
            auto cudnnStatus = cudnnGetConvolutionForwardAlgorithm_v7(getGpu()._cuDNNHandle, _inputLayer._tensorDescriptor, _convFilterDesc, _convDesc, _outputLayer._tensorDescriptor, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, 0, &convolutionAlgo);
            CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionForwardAlgorithm failed.");
            auto _convFWAlgo = convolutionAlgo.algo;
            cudnnStatus = cudnnGetConvolutionForwardWorkspaceSize(getGpu()._cuDNNHandle, _inputLayer._tensorDescriptor, _convFilterDesc, _convDesc, _outputLayer._tensorDescriptor, _convFWAlgo, &workspaceSize);
            CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionForwardWorkspaceSize failed.");
            pNetwork->SetCUDNNWorkspace(workspaceSize);
            cudnnConvolutionBwdFilterAlgoPerf_t _convBWWeightAlgoPerf;
            cudnnStatus = cudnnGetConvolutionBackwardFilterAlgorithm_v7(getGpu()._cuDNNHandle, _inputLayer._tensorDescriptor, _outputLayer._tensorDescriptor, _convDesc, _convFilterDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, 0, &_convBWWeightAlgoPerf);
            CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardFilterAlgorithm failed.");
            auto _convBWWeightAlgo = _convBWWeightAlgoPerf.algo;
            cudnnStatus = cudnnGetConvolutionBackwardFilterWorkspaceSize(getGpu()._cuDNNHandle, _inputLayer._tensorDescriptor, _outputLayer._tensorDescriptor, _convDesc, _convFilterDesc, _convBWWeightAlgo, &workspaceSize);
            CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardFilterWorkspaceSize failed.");
            pNetwork->SetCUDNNWorkspace(workspaceSize);
            cudnnConvolutionBwdDataAlgoPerf_t perfData;
            cudnnStatus = cudnnGetConvolutionBackwardDataAlgorithm_v7(getGpu()._cuDNNHandle, _convFilterDesc, _outputLayer._tensorDescriptor, _convDesc, _inputLayer._tensorDescriptor, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, 0, &perfData);
            CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardDataAlgorithm failed.");
            cudnnStatus = cudnnGetConvolutionBackwardDataWorkspaceSize(getGpu()._cuDNNHandle, _convFilterDesc, _outputLayer._tensorDescriptor, _convDesc, _inputLayer._tensorDescriptor, _convBWDeltaAlgo, &workspaceSize);
            CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionBackwardDataWorkspaceSize failed.");
            pNetwork->SetCUDNNWorkspace(workspaceSize);
            std::vector<int> vOutputDim(8, 1);
            cudnnStatus = cudnnGetConvolutionNdForwardOutputDim(_convDesc, _inputLayer._tensorDescriptor, _convFilterDesc, _outputLayer._dimensions + 1, vOutputDim.data());
            CUDNNERROR(cudnnStatus, "Weight::Refresh: cudnnGetConvolutionNdForwardOutputDim failed.");
            size_t dim = std::accumulate(vOutputDim.begin(), vOutputDim.end(), 1, std::multiplies<int>());
            if (dim != static_cast<unsigned long long>(_outputLayer._maxLocalStride) * _outputLayer._localBatch)
            {
                if (getGpu()._id == 0)
                    std::cout << ("Output layer {}: has incorrectly calculated dimensions for cuDNN.\n", _outputLayer._name);
                getGpu().Shutdown();
        }
        }
};

    auto monitorGPUMemory = [&]() {
        bool memoryOk = true;
        if (!memoryOk) {
        }
        };

    if (mode == TrainingMode::SGD) {
        pool.enqueue(checkGPUStatus);
        pool.enqueue(refreshBuffers);
    }
    else {
        pool.enqueue(refreshBuffers);
        pool.enqueue(cudnnProcessing);
        }
    pool.enqueue(monitorGPUMemory);

    if (mode == TrainingMode::SGD) {
        {
            std::unique_lock<std::mutex> lock(cv_m);
            cv.wait(lock, [&]() { return gpu_ready; });
        }
    }

    std::async(std::launch::async, refreshBuffers).wait();
    std::async(std::launch::async, cudnnProcessing).wait();
    std::async(std::launch::async, checkGPUStatus).wait();
}

float Weight::CalculateRegularizationError(float lambda, float lambda1)
{
    if (_bShared)
        return 0;
    else
        return kCalculateRegularizationError(lambda, lambda1, _pbWeight->_pDevData, _localSize);
}

void Weight::UpdateWeights(TrainingMode trainingMode, uint32_t batch, float alpha, float lambda, float lambda1, float mu, float mu1, float t)
{
    cublasStatus_t cstatus;

    if (_bLocked)
        return; 

    if (!_bShared)
    {
        switch (trainingMode)
        {
            case SGD:
                kSGDUpdateWeights(alpha, lambda, lambda1, _localSize, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                
            case Momentum:
                kMomentumUpdateWeights(alpha, lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                        
            case AdaGrad:
                kAdaGradUpdateWeights(alpha, lambda, lambda1, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                        
            case Nesterov:
                kNesterovUpdateWeights(alpha, lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                        
            case RMSProp:
                kRMSPropUpdateWeights(alpha, lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;

            case AdaDelta:
                kAdaDeltaUpdateWeights(lambda, lambda1, mu, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeightGradientVelocity->_pDevData, _pbWeight->_pDevData);
                break;     

            case Adam:
                kAdamUpdateWeights(alpha, lambda, lambda1, mu, mu1, t, _localSize, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeightGradientVelocity->_pDevData, _pbWeight->_pDevData);
                break;   
        }
    }

    if (_transform == Linear)
    {
        switch (trainingMode)
        {
            case SGD:
                kSGDUpdateBiases(alpha, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBias->_pDevData);
                break;

            case Momentum:
                kMomentumUpdateBiases(alpha, mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;
                    
            case AdaGrad:
                kAdaGradUpdateBiases(alpha, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;
                    
            case Nesterov:
                kNesterovUpdateBiases(alpha, mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;
                    
            case RMSProp:
                kRMSPropUpdateBiases(alpha, mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;
                
            case AdaDelta:
                kAdaDeltaUpdateBiases(mu, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;                         

            case Adam:
                kAdamUpdateBiases(alpha, mu, mu1, t, batch, _localBiasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break; 
        }
    }
    else
    {
        switch (trainingMode)
        {
            case SGD:
                kSGDUpdateWeights(alpha, (float)0.0, (float)0.0, _localBiasSize, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case Momentum:
                kMomentumUpdateWeights(alpha, (float)0.0, (float)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;
                    
            case AdaGrad:
                kAdaGradUpdateWeights(alpha, (float)0.0, (float)0.0, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;
                        
            case Nesterov:
                kNesterovUpdateWeights(alpha, (float)0.0, (float)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;
                        
            case RMSProp:
                kRMSPropUpdateWeights(alpha, (float)0.0, (float)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case AdaDelta:
                kAdaDeltaUpdateWeights((float)0.0, (float)0.0, mu, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;

            case Adam:
                kAdamUpdateWeights(alpha, (float)0.0, (float)0.0, mu, mu1, t, _localBiasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;                                 
        }       
    }
#if 0
        if (_width < 1024)
        {
            _pbBias->Download(&_vBias[0]);
            for (int i = 0; i < _width; i++)
                printf("%3d %16.8f\n", i, _vBias[i]);
        }
#endif
          
    if ((_norm > (float)0.0) && (!_bShared))
    {
        if (getGpu()._numprocs == 1)
            kNormalizeWeights(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData);
        else
        {
            float* pMagnitude                 = getGpu()._pNetwork->GetScratchBuffer(_outputLayer._stride);
            kCalculateWeightMagnitudes(_outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, _outputLayer._stride);
            kNormalizeWeightMagnitudes(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);       
        }
    }
}

bool Weight::WriteNetCDF(netCDF::NcFile& nc, uint32_t index, float* pWeight, float* pBias)
{
    bool bResult                = true;
    if (getGpu()._id == 0)
    {
        std::string wstring          = "weight" + std::to_string(index) + "_";
        nc.putAtt(wstring + "inputLayer", _inputLayer._name);
        nc.putAtt(wstring + "outputLayer", _outputLayer._name);

        nc.putAtt(wstring + "width", ncUint64, (unsigned long long int)_width);  
        nc.putAtt(wstring + "height", ncUint64, (unsigned long long int)_height);
        nc.putAtt(wstring + "length", ncUint64, (unsigned long long int)_length);
        nc.putAtt(wstring + "depth", ncUint64, (unsigned long long int)_depth);
        nc.putAtt(wstring + "breadth", ncUint64, (unsigned long long int)_breadth);  

        nc.putAtt(wstring + "bShared", ncUint, (uint32_t)_bShared);
        nc.putAtt(wstring + "bLocked", ncUint, (uint32_t)_bLocked);
        nc.putAtt(wstring + "norm", ncFloat, _norm);
        
        NcDim biasDim           = nc.addDim(wstring + "biasDim", _biasSize);
        NcVar biasVar           = nc.addVar(wstring + "bias", "float", biasDim.getName());
        if (pBias == NULL)
            pBias               = _vBias.data();
        biasVar.putVar(pBias);  
        if (_bShared)
        {
            nc.putAtt(wstring + "bTransposed", ncUint, (uint32_t)_bTransposed);
            nc.putAtt(wstring + "sourceInputLayer", _pSharedWeight->_inputLayer._name);
            nc.putAtt(wstring + "sourceOutputLayer", _pSharedWeight->_outputLayer._name);
        }
        else
        {

#if 0
        printf("Weights %d %lu %lu\n", index, _vWeight.size(), _vBias.size());
        for (int i = 0; i < 20; i++)
            printf("%3d %16.8f %16.8f\n", i, _vWeight[i], _vBias[i]);
#endif
            NcDim weightDim     = nc.addDim(wstring + "weightDim", _size);            
            NcVar weightVar     = nc.addVar(wstring + "weights", "float", weightDim.getName());            
            if (!pWeight)
                pWeight         = _vWeight.data();
            weightVar.putVar(pWeight);
        }
    }

    return bResult;
}

bool Weight::CopyWeights(const Weight* pSrcWeight)
{
    bool bValid                 = true;
    Weight* pDstWeight = _bShared ? _pSharedWeight : this;

    if (!pSrcWeight)
    {
        if (getGpu()._id == 0)
            printf("Weight::CopyWeights: Invalid weight pointer.\n");
        return false;
    }
    
    pSrcWeight = _bShared ? pSrcWeight->_pSharedWeight : pSrcWeight;
    if ((pSrcWeight->_width != pDstWeight->_width) || (pSrcWeight->_height != pDstWeight->_height) || (pSrcWeight->_length != pDstWeight->_length))
    {
        if (getGpu()._id == 0)
        {
            printf("Weight::CopyWeights: Mismatched weight dimensions (%" PRIu64 " x %" PRIu64 " x %" PRIu64") versus (%" PRIu64 " x %" PRIu64 " x %" PRIu64 ").\n", pDstWeight->_width, pDstWeight->_height, pDstWeight->_length,
            pSrcWeight->_width, pSrcWeight->_height, pSrcWeight->_length);
        }
        bValid                  = false;        
    }
    else
    {
        pDstWeight->_vWeight    = pSrcWeight->_vWeight;
        _vBias                  = pSrcWeight->_vBias;
        if (pDstWeight->_pbWeight != NULL)
            pDstWeight->_pbWeight->Upload(pDstWeight->_vWeight.data());
        if (_pbBias != NULL)
            _pbBias->Upload(_vBias.data());
    }
    return bValid;
}

bool Weight::SetWeights(const std::vector<float>& vWeight)
{
    bool bValid                 = true;
    Weight* pWeight = _bShared ? _pSharedWeight : this;
    
    if (getGpu()._numprocs == 1)
    {
        if (vWeight.size() < pWeight->_vWeight.size())
        {
            if (getGpu()._id == 0)
            {
                printf("Weight::SetWeights: Input vector smaller than weight vector.\n");
            }
            bValid                  = false;        
        }
        else
        {
            if (vWeight.size() > pWeight->_vWeight.size())
                std::copy(vWeight.data(), vWeight.data() + pWeight->_vWeight.size(), pWeight->_vWeight.data());
            else
                pWeight->_vWeight       = vWeight;
            if (pWeight->_pbWeight != NULL)
                pWeight->_pbWeight->Upload(_vWeight.data());
        }
    }
    else
    {
        
    }
    return bValid;
}

bool Weight::SetBiases(const std::vector<float>& vBias)
{
    bool bValid                 = true;

    if (vBias.size() < _vBias.size())
    {
        if (getGpu()._id == 0)
        {
            printf("Weight::SetBiases: Input vector smaller than bias vector.\n");
        }
        bValid                  = false;        
    }
    else
    {
        if (vBias.size() > _vBias.size())
            std::copy(vBias.data(), vBias.data() + _vBias.size(), _vBias.data());
        else
            _vBias       = vBias;
        if (_pbBias != NULL)
            _pbBias->Upload(_vBias.data());
    }
    return bValid;
}

bool Weight::GetWeights(std::vector<float>& vWeight)
{
    bool bValid                 = true;

    if (vWeight.size() < _vWeight.size())
    {
        vWeight.resize(_vWeight.size());
    }

    if (_pbWeight != NULL)
    {
        _pbWeight->Download(vWeight.data());
    }
    else
    {
        vWeight = _vWeight;
    }
    return bValid;
}

bool Weight::GetBiases(std::vector<float>& vBias)
{
    bool bValid                 = true;

    if (getGpu()._numprocs == 1)
    {

        if (vBias.size() < _vBias.size())
        {
            vBias.resize(_vBias.size());
        }

        if (_pbBias != NULL)
        {
            _pbBias->Download(vBias.data());
        }
        else
        {
            vBias = _vBias;
        }
    }
    else
    {
        
    }
    return bValid;
}

bool Weight::GetDimensions(std::vector<uint64_t>& dimensions)
{
  if (_dimensionality < 2 || _dimensionality > 5) {
      printf("Weight::GetDimensions: _dimensionality = %u\n", _dimensionality);
      return false;
  }
  if (_dimensionality >= 1) dimensions.push_back(_width);
  if (_dimensionality >= 2) dimensions.push_back(_height);
  if (_dimensionality >= 3) dimensions.push_back(_length);
  if (_dimensionality >= 4) dimensions.push_back(_depth);
  if (_dimensionality == 5) dimensions.push_back(_breadth);
  return true;
}

void Weight::Dump(std::string fname, float* pBuffer)
{
    std::vector<float> vWeight;

    if (getGpu()._numprocs == 1)
    {
        vWeight.resize(_localSize);
        cudaMemcpy(vWeight.data(), pBuffer, _localSize * sizeof(float), cudaMemcpyDefault);
    }
    else
    {
        if (getGpu()._id == 0)
            vWeight.resize(_outputLayer._stride * _inputLayer._stride);        
        uint32_t outgoingSize       = _outputLayer._stride * 3;               
        uint32_t incomingSize       = _inputLayer._stride * 2;     
        cudaMemcpy(_vWeight.data(), pBuffer, _localSize * sizeof(float), cudaMemcpyDefault);

        if (getGpu()._id == 0)
        {
            float* pWeight            = vWeight.data();                    
            if (outgoingSize > incomingSize)
            {
                cudaMemcpy2D(pWeight, _outputLayer._stride * sizeof(float), _vWeight.data(), _outputLayer._localStride * sizeof(float), _outputLayer._localStride * sizeof(float), _inputLayer._stride, cudaMemcpyDefault);
                pWeight                += _outputLayer._localStride;
                for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                {                        
                    uint64_t size;
                    MPI_Status status;                
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    std::vector<float> vTemp(size);
                    MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                    uint64_t lstride    = size / _inputLayer._stride;
                    float* pSrcWeight = vTemp.data();
                    float* pDstWeight = pWeight;
                    for (uint32_t j = 0; j < _inputLayer._stride; j++)
                    {
                        memcpy(pDstWeight, pSrcWeight, lstride * sizeof(float));
                        pSrcWeight     += lstride;
                        pDstWeight     += _outputLayer._stride;
                    }                          
                    pWeight            += lstride;
                }
            }
            else
            {
                cudaMemcpy(pWeight, _vWeight.data(), _outputLayer._stride * _inputLayer._localStride * sizeof(float), cudaMemcpyDefault);
                pWeight                += _outputLayer._stride * _inputLayer._localStride;
                for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                {
                    uint64_t size;
                    MPI_Status status;                
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(pWeight, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                    pWeight            += size;
                }                        
            }
        }              
        else
        {
            uint64_t size               = _vWeight.size();
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(_vWeight.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);                  
        }

    }

    if (getGpu()._id == 0)
    {
        FILE* fp                        = fopen(fname.c_str(), "w");
        float* pData                  = vWeight.data();
        for (int i = 0; i < _inputLayer._stride; i++)
        {
            for (int j = 0; j < _outputLayer._stride; j++)
            {
                fprintf(fp, "%12.9f ", *pData);
                pData++;
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}
