#include "GpuTypes.h"
#include "Types.h"
#include "Kernels.cuh"
#include <format>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

using namespace netCDF;
using namespace netCDF::exceptions;

void DumpP(const char *name, float *p, int stride) {
    std::cout << name << ":  ";
    std::vector<float> data(stride);
    cudaMemcpy(data.data(), p, stride*sizeof(float), cudaMemcpyDefault);
    for (auto i : data) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}

/// <summary>
/// Constructor for the Layer class.
/// </summary>
/// <param name="d">The LayerDescriptor for the layer.</param>
/// <param name="batch">The batch size.</param>
Layer::Layer(LayerDescriptor& d, uint32_t batch)
    : _name(d._name),                                                               // Name of the layer
    _kind(d._kind),                                                                 // Kind of the layer
    _type(d._type),                                                                 // Type of the layer
    _attributes(d._attributes),                                                     // Attributes of the layer
    _poolingFunction(d._poolingFunction),                                           // Pooling function used by the layer
    _dataSet(d._dataSet),                                                           // Data set associated with the layer
    _pDataSet(nullptr),                                                             // Pointer to the data set (initially nullptr)
    _vSource(d._vSource),                                                           // Source vector for the layer
    _vSkip(d._vSkip),                                                               // Skip vector for the layer
    _pbUnit(),                                                                      // Unit parameter
    _pbDelta(),                                                                     // Delta parameter
    _pbDropout(),                                                                   // Dropout parameter
    _pbDeltaBN(),                                                                   // Delta batch normalization parameter
    _pbScaleGradientBN(),                                                           // Scale gradient batch normalization parameter
    _pbScaleGradientVelocityBN(),                                                   // Scale gradient velocity batch normalization parameter
    _pbBiasGradientBN(),                                                            // Bias gradient batch normalization parameter
    _pbBiasGradientVelocityBN(),                                                    // Bias gradient velocity batch normalization parameter
    _pbUnitBN(),                                                                    // Batch normalization parameter
    _pbScaleBN(),                                                                   // Batch normalization parameter
    _pbBiasBN(),                                                                    // Bias batch normalization parameter
    _pbRunningMeanBN(),                                                             // Running mean batch normalization parameter
    _pbRunningVarianceBN(),                                                         // Running variance batch normalization parameter
    _pbSaveMeanBN(),                                                                // Saved mean batch normalization parameter
    _pbSaveInvVarianceBN(),                                                         // Saved inverse variance batch normalization parameter
    _Nx(d._Nx),                                                                     // Dimension Nx
    _Ny(d._Ny),                                                                     // Dimension Ny
    _Nz(d._Nz),                                                                     // Dimension Nz
    _Nw(d._Nw),                                                                     // Dimension Nw
    _strideBN(0),                                                                   // Batch normalization stride
    _dimensions(d._dimensions),                                                     // Dimensions of the layer
    _weightInit(d._weightInit),                                                     // Weight initialization method
    _weightInitScale(d._weightInitScale),                                           // Weight initialization scale
    _biasInit(d._biasInit),                                                         // Bias initialization
    _kernelX(d._kernelX),                                                           // Kernel size along X-axis
    _kernelY(d._kernelY),                                                           // Kernel size along Y-axis
    _kernelZ(d._kernelZ),                                                           // Kernel size along Z-axis
    _kernelStrideX(d._kernelStrideX),                                               // Kernel stride along X-axis
    _kernelStrideY(d._kernelStrideY),                                               // Kernel stride along Y-axis
    _kernelStrideZ(d._kernelStrideZ),                                               // Kernel stride along Z-axis
    _kernelPaddingX(d._kernelPaddingX),                                             // Kernel padding along X-axis
    _kernelPaddingY(d._kernelPaddingY),                                             // Kernel padding along Y-axis
    _kernelPaddingZ(d._kernelPaddingZ),                                             // Kernel padding along Z-axis
    _kernelDimensions(d._kernelDimensions),                                         // Dimensions of the kernel
    _weightNorm(d._weightNorm),                                                     // Weight normalization
    _deltaNorm(d._deltaNorm),                                                       // Delta normalization
    _pDropout(d._pDropout),                                                         // Pointer to dropout
    _activation(d._activation),                                                     // Activation function
    _oddBatch(0),                                                                   // Odd batch flag
    _bSparse(d._attributes& Layer::Attributes::Sparse),                             // Sparse flag
    _sparsenessPenalty_p(d._sparsenessPenalty_p),                                   // Sparseness penalty (p)
    _sparsenessPenalty_beta(d._sparsenessPenalty_beta),                             // Sparseness penalty (beta)
    _bDenoising(d._attributes& Layer::Attributes::Denoising),                       // Denoising flag
    _bFastSparse(false),                                                            // Fast sparse flag (initially false)
    _bDirty(true),                                                                  // Dirty flag (initially true)
    _bnCalls(0),                                                                    // Batch normalization calls count
    _priority(-1),                                                                  // Priority level (initially -1)
    _deltaUpdateCount(0),                                                           // Delta update count
    _unitUpdateCount(0),                                                            // Unit update count
    _batch(batch),                                                                  // Batch size
    _localBatch(batch),                                                             // Local batch size
    _RELUSlope(d._RELUSlope),                                                       // ReLU slope
    _ELUAlpha(d._ELUAlpha),                                                         // ELU alpha
    _SELULambda(d._SELULambda),                                                     // SELU lambda
    _bBatchNormalization(d._attributes& Layer::Attributes::BatchNormalization) {    // Batch normalization flag
    // Initialize layer descriptors
    InitializeDescriptors();

    // Initialize batch normalization if enabled
    if (_bBatchNormalization) {
        InitializeBatchNormalization(d);
    }

    // Initialize pooling descriptor if the layer type is pooling
    if (_type == Layer::Type::Pooling) {
        InitializePoolingDescriptor();
    }
}

/// <summary>
/// Initializes the descriptors for the Layer based on its properties.
/// </summary>
void Layer::InitializeDescriptors() {
    try {
        // Calculate the stride based on the layer dimensions
        _stride = _Nx * _Ny * _Nz * _Nw;

        // Determine the parallelization strategy based on the layer type
        if (_type == FullyConnected)
            _parallelization = Model;
        else
            _parallelization = Data;

        // Calculate the minimum and maximum X-dimensions for the current GPU process
        _minX = ((size_t)_Nx * (size_t)getGpu()._id) / (size_t)getGpu()._numprocs;
        _maxX = ((size_t)_Nx * (size_t)(getGpu()._id + 1)) / (size_t)getGpu()._numprocs;

        // Calculate the local stride and maximum local stride
        _localStride = (_maxX - _minX) * _Ny * _Nz * _Nw;
        _maxLocalStride = (((size_t)_Nx + getGpu()._numprocs - 1) / (size_t)getGpu()._numprocs) * _Ny * _Nz * _Nw;

        // Create tensor descriptors for Pooling and Convolutional layers, if applicable
        if ((_type == Layer::Type::Pooling) || (_type == Layer::Type::Convolutional)) {
            cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&_tensorDescriptor);
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                throw std::runtime_error("Layer::InitializeDescriptors: Unable to create _tensorDescriptor");
            }

            cudnnStatus = cudnnCreateTensorDescriptor(&_oddBatchTensorDescriptor);
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                throw std::runtime_error("Layer::InitializeDescriptors: Unable to create _oddBatchTensorDescriptor");
            }
        }
    }
    catch (const std::exception& e) {
        // Handle the exception
        std::cerr << "Error in Layer::InitializeDescriptors: " << e.what() << std::endl;
    }
}

/// <summary>
/// Initializes the batch normalization for the Layer based on the provided LayerDescriptor.
/// </summary>
/// <param name="d">The LayerDescriptor for the layer.</param>
void Layer::InitializeBatchNormalization(LayerDescriptor& d) {
    // Declarations for CUDA and cuDNN status
    cudaError_t status;
    cudnnStatus_t cudnnStatus;

    // Create tensor descriptors for scale/bias and batch normalization
    cudnnStatus = cudnnCreateTensorDescriptor(&_scaleBiasMeanVarDescBN);
    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Layer::InitializeBatchNormalization: Unable to create _scaleBiasMeanVarDescBN");
    }

    cudnnStatus = cudnnCreateTensorDescriptor(&_tensorDescriptorBN);
    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Layer::InitializeBatchNormalization: Unable to create _tensorDescriptorBN");
    }

    // Determine the stride for batch normalization based on layer type
    if (_type == Layer::Type::Convolutional)
        _strideBN = _Nz;
    else
        _strideBN = _localStride;

    // Create GpuBuffers for batch normalization parameters
    _pbScaleGradientBN.reset(new GpuBuffer<float>(_strideBN));
    _pbBiasGradientBN.reset(new GpuBuffer<float>(_strideBN));
    _pbScaleBN.reset(new GpuBuffer<float>(_strideBN));
    _pbBiasBN.reset(new GpuBuffer<float>(_strideBN));
    _pbRunningMeanBN.reset(new GpuBuffer<float>(_strideBN));
    _pbRunningVarianceBN.reset(new GpuBuffer<float>(_strideBN));
    _pbSaveMeanBN.reset(new GpuBuffer<float>(_strideBN));
    _pbSaveInvVarianceBN.reset(new GpuBuffer<float>(_strideBN));

    // Print allocation information if GPU ID is 0
    if (getGpu()._id == 0) {
        std::cout << std::format("Layer::InitializeBatchNormalization: Allocating {} bytes of BN scale diff for layer {}\n",
            _strideBN * sizeof(float), _name)
            << std::format("Layer::InitializeBatchNormalization: Allocating {} bytes of BN bias diff for layer {}\n",
                _strideBN * sizeof(float), _name)
            << std::format("Layer::InitializeBatchNormalization: Allocating {} bytes of BN scale for layer {}\n",
                _strideBN * sizeof(float), _name)
            << std::format("Layer::InitializeBatchNormalization: Allocating {} bytes of BN bias for layer {}\n",
                _strideBN * sizeof(float), _name)
            << std::format("Layer::InitializeBatchNormalization: Allocating {} bytes of BN running mean for layer {}\n",
                _strideBN * sizeof(float), _name)
            << std::format("Layer::InitializeBatchNormalization: Allocating {} bytes of BN running variance for layer {}\n",
                _strideBN * sizeof(float), _name)
            << std::format("Layer::InitializeBatchNormalization: Allocating {} bytes of BN saving mean for layer {}\n",
                _strideBN * sizeof(float), _name)
            << std::format("Layer::InitializeBatchNormalization: Allocating {} bytes of BN saving InvVariance for layer {}\n",
                _strideBN * sizeof(float), _name);
    }

    // Initialize scale, bias, running mean, running variance, and related buffers
    if (d._vScaleBN.size() != 0) {
        status = cudaMemcpy(_pbScaleBN->_pDevData, d._vScaleBN.data(), _strideBN * sizeof(float), cudaMemcpyHostToDevice);
    }
    else {
        std::vector<float> ones(_strideBN);
        for (int i = 0; i < _strideBN; ++i)
            ones[i] = 1;
        status = cudaMemcpy(_pbScaleBN->_pDevData, ones.data(), _strideBN * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error("Layer::InitializeBatchNormalization: cudaMemcpy failed on _pbScaleBN");
    }

    if (d._vBiasBN.size() != 0) {
        status = cudaMemcpy(_pbBiasBN->_pDevData, d._vBiasBN.data(), _strideBN * sizeof(float), cudaMemcpyHostToDevice);
    }
    else {
        status = cudaMemset(_pbBiasBN->_pDevData, 0, _strideBN * sizeof(float));
    }
    if (status != cudaSuccess) {
        throw std::runtime_error("Layer::InitializeBatchNormalization: cudaMemcpy failed on _pbBiasBN");
    }

    if (d._vRunningMeanBN.size() != 0) {
        status = cudaMemcpy(_pbRunningMeanBN->_pDevData, d._vRunningMeanBN.data(), _strideBN * sizeof(float), cudaMemcpyHostToDevice);
    }
    else {
        status = cudaMemset(_pbRunningMeanBN->_pDevData, 0, _strideBN * sizeof(float));
    }
    if (status != cudaSuccess) {
        throw std::runtime_error("Layer::InitializeBatchNormalization: cudaMemcpy failed on _pbRunningMeanBN");
    }

    if (d._vRunningVarianceBN.size() != 0) {
        status = cudaMemcpy(_pbRunningVarianceBN->_pDevData, d._vRunningVarianceBN.data(), _strideBN * sizeof(float), cudaMemcpyHostToDevice);
    }
    else {
        status = cudaMemset(_pbRunningVarianceBN->_pDevData, 0, _strideBN * sizeof(float));
    }
    if (status != cudaSuccess) {
        throw std::runtime_error("Layer::InitializeBatchNormalization: cudaMemcpy failed on _pbRunningVarianceBN");
    }

    status = cudaMemset(_pbScaleGradientBN->_pDevData, 0, _strideBN * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("Layer::InitializeBatchNormalization: cudaMemset failed on _pbScaleGradientBN");
    }

    status = cudaMemset(_pbBiasGradientBN->_pDevData, 0, _strideBN * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("Layer::InitializeBatchNormalization: cudaMemset failed on _pbBiasGradientBN");
    }

    status = cudaMemset(_pbSaveMeanBN->_pDevData, 0, _strideBN * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("Layer::InitializeBatchNormalization: cudaMemset failed on _pbSaveMeanBN");
    }

    status = cudaMemset(_pbSaveInvVarianceBN->_pDevData, 0, _strideBN * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("Layer::InitializeBatchNormalization: cudaMemset failed on _pbSaveInvVarianceBN");
    }
}

Layer::~Layer()
{
    Deallocate();
    if ((_type == Layer::Type::Pooling) || (_type == Layer::Type::Convolutional))
    {
        cudnnStatus_t cudnnStatus       = cudnnDestroyTensorDescriptor(_tensorDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _tensorDescriptor");        
        cudnnStatus                     = cudnnDestroyTensorDescriptor(_oddBatchTensorDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _oddBatchTensorDescriptor");  
    }

    if (_bBatchNormalization)
    {
        cudnnStatus_t cudnnStatus       = cudnnDestroyTensorDescriptor(_scaleBiasMeanVarDescBN);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _scaleBiasMeanVarDescBN");        
        cudnnStatus                     = cudnnDestroyTensorDescriptor(_tensorDescriptorBN);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _tensorDescriptorBN");        
        _pbScaleBN.reset();
        _pbBiasBN.reset();
        _pbScaleGradientBN.reset();
        _pbBiasGradientBN.reset();
        _pbRunningMeanBN.reset();
        _pbRunningVarianceBN.reset();
        _pbSaveMeanBN.reset();
        _pbSaveInvVarianceBN.reset();
    }
    
    if (_type == Layer::Type::Pooling)
    {
        cudnnStatus_t cudnnStatus       = cudnnDestroyPoolingDescriptor(_poolingDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to destroy _poolingDescriptor");
        
        if (_poolingFunction == PoolingFunction::LRN)
        {
            cudnnStatus_t cudnnStatus   = cudnnDestroyLRNDescriptor(_LRNDescriptor);
            CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _LRNDescriptor");
        }
    }
}

void Layer::Deallocate()
{
    if (getGpu()._id == 0)
        printf("Layer::Deallocate: Deallocating all data for layer %s\n", _name.c_str());

    _pbUnit.reset();
    _pbUnitBN.reset();
    _pbDelta.reset();
    _pbDeltaBN.reset();
    _pbDropout.reset();
    _pbBuffer1.reset();
    _pbBuffer2.reset();
    _pbScaleVelocityBN.reset();
    _pbScaleGradientVelocityBN.reset();
    _pbBiasVelocityBN.reset();
    _pbBiasGradientVelocityBN.reset();
}

bool Layer::GetUnits(std::vector<float>& vUnit)
{
    bool bValid = true;
    
    if (_pbUnit)
    {
        if (vUnit.size() < _stride)
        {
            vUnit.resize(_stride);
        }
    
        _pbUnit->Download(vUnit.data());
    }
    else
    {
        printf("Layer::GetUnits: Unit data not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}

bool Layer::GetUnits(float* pUnit)
{
    bool bValid = true;
    
    if (_pbUnit)
    {
        if (pUnit == NULL)
        {
            printf("Layer::GetUnits: Download pointer invalid.\n");
            bValid = false;
        }
        else
        {
            _pbUnit->Download(pUnit);
        }
    }
    else
    {
        printf("Layer::GetUnits: Unit data not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}

bool Layer::GetDeltas(std::vector<float>& vDelta)
{
    bool bValid = true;
    
    if (_pbDelta)
    {
        if (vDelta.size() < _stride)
        {
            vDelta.resize(_stride);
        }
    
        _pbDelta->Download(vDelta.data());
    }
    else
    {
        printf("Layer::GetDeltas: Deltas not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}

bool Layer::GetDeltas(float* pDelta)
{
    bool bValid = true;
    
    if (_pbDelta)
    {
        if (pDelta == NULL)
        {
            printf("Layer::GetDeltas: Download pointer invalid.\n");
            bValid = false;
        }
        else
        {
            _pbDelta->Download(pDelta);
        }
    }
    else
    {
        printf("Layer::GetDeltas: Deltas not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}

bool Layer::SetUnits(const std::vector<float>& vUnit)
{
    bool bValid = true;
    
    if (_pbUnit)
    {
        if (vUnit.size() < _stride)
        {
            printf("Layer::SetUnits: Input unit data too small to set all units.\n");
            bValid = false;
        }
    
        _pbUnit->Upload(vUnit.data());
    }
    else
    {
        printf("Layer::SetUnits: Unit data not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}


bool Layer::SetDeltas(const std::vector<float>& vDelta)
{
    bool bValid = true;
    
    if (_pbDelta)
    {
        if (vDelta.size() < _stride)
        {
            printf("Layer::SetDeltas: Input delta data too small to set all deltas.\n");
            bValid = false;
        }
    
        _pbDelta->Upload(vDelta.data());
    }
    else
    {
        printf("Layer::SetDeltas: Deltas not yet allocated.\n");
        bValid = false;
    }
    
    return bValid;    
}

cudnnTensorDescriptor_t Layer::getTensorDescriptor(uint32_t batch)
{
    if (batch == _batch)
    {
        return _tensorDescriptor;
    }
    
    else if (batch != _oddBatch)
    {
        cudnnStatus_t cudnnStatus;
        std::vector<int> vDimensions(5, 1);
        std::vector<int> vStride(5, 1);
        switch (_dimensions)
        {
            case 2:
                vDimensions[0]      = batch;
                vDimensions[1]      = _Ny;
                vDimensions[2]      = _Nx;
                vStride[2]          = 1;
                vStride[1]          = _Nx;
                vStride[0]          = _Nx * _Ny;                
                cudnnStatus         = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
            
            case 3:
                cudnnStatus         = cudnnSetTensor4dDescriptor(_oddBatchTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _Nx);
                break;
                
            case 4:
                vDimensions[0]      = batch;
                vDimensions[1]      = _Nw;
                vDimensions[2]      = _Nz;
                vDimensions[3]      = _Ny;
                vDimensions[4]      = _Nx;
                vStride[4]          = 1;
                vStride[3]          = _Nx;
                vStride[2]          = _Nx * _Ny;
                vStride[1]          = _Nx * _Ny * _Nz;
                vStride[0]          = _Nx * _Ny * _Nz * _Nw;                                             
                cudnnStatus         = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
        }
        CUDNNERROR(cudnnStatus, "Layer::getTensorDescriptor: Unable to set oddBatchTensorDescriptor");
        _oddBatch = batch;
    }

    return _oddBatchTensorDescriptor;
}

const std::string& Layer::GetName() const {
  return _name;
}

const std::string& Layer::GetDataSetName() const {
    return _dataSet;
}
Layer::Kind Layer::GetKind() const {
  return _kind;
}

Layer::Type Layer::GetType() const {
  return _type;
}

uint32_t Layer::GetAttributes() const {
  return _attributes;
}

DataSetBase* Layer::GetDataSet() const {
  return _pDataSet;
}

uint32_t Layer::GetNumDimensions() const {
  return _dimensions;
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> Layer::GetDimensions() const
{
    return std::make_tuple(_Nx, _Ny, _Nz, _Nw);
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> Layer::GetLocalDimensions() const
{
    return std::make_tuple(_maxX - _minX, _Ny, _Nz, _Nw);
}

std::tuple<uint32_t, uint32_t, uint32_t> Layer::GetKernelDimensions() const
{
    return std::make_tuple(_kernelX, _kernelY, _kernelZ);
}

std::tuple<uint32_t, uint32_t, uint32_t> Layer::GetKernelStride() const
{
    return std::make_tuple(_kernelStrideX, _kernelStrideY, _kernelStrideZ);
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

void Layer::Allocate(bool validate)
{
    Deallocate();
    uint64_t size                   = (uint64_t)_maxLocalStride * (uint64_t)_localBatch; 
    
    if ((_type == Layer::Type::Pooling) && (_poolingFunction == PoolingFunction::Cosine))
    {
        _vBuffer1.resize(size);
        _pbBuffer1.reset(new GpuBuffer<float>(size));
        if (getGpu()._id == 0)
            printf("Layer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of auxilliary buffer 1 data for layer %s\n", size * sizeof(float), _maxLocalStride, _localBatch, _name.c_str());
        _vBuffer2.resize(size);
        _pbBuffer2.reset(new GpuBuffer<float>(size));
        if (getGpu()._id == 0)
            printf("Layer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of auxilliary buffer 2 data for layer %s\n", size * sizeof(float), _maxLocalStride, _localBatch, _name.c_str());
    }
        
    else if ((_type == Layer::Type::Pooling) || (_type == Layer::Type::Convolutional))
    {
        cudnnStatus_t cudnnStatus;
        std::vector<int> vDimensions(5, 1);
        std::vector<int> vStride(5, 1);
        switch (_dimensions)
        {
            case 2:
                vDimensions[0]      = _localBatch;
                vDimensions[1]      = _Ny;
                vDimensions[2]      = _Nx;
                vStride[2]          = 1;
                vStride[1]          = _Nx;
                vStride[0]          = _Nx * _Ny;                
                cudnnStatus         = cudnnSetTensorNdDescriptor(_tensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
            
            case 3:
                cudnnStatus         = cudnnSetTensor4dDescriptor(_tensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _localBatch, _Nz, _Ny, _Nx);
                break;
                
            case 4:
                vDimensions[0]      = _localBatch;
                vDimensions[1]      = _Nw;
                vDimensions[2]      = _Nz;
                vDimensions[3]      = _Ny;
                vDimensions[4]      = _Nx;
                vStride[4]          = 1;
                vStride[3]          = _Nx;
                vStride[2]          = _Nx * _Ny;
                vStride[1]          = _Nx * _Ny * _Nz;
                vStride[0]          = _Nx * _Ny * _Nz * _Nw;                           
                cudnnStatus         = cudnnSetTensorNdDescriptor(_tensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
        }
        CUDNNERROR(cudnnStatus, "Layer::Allocate: Unable to set tensor descriptor");
        DumpTensor(_tensorDescriptor);
    }

    if (!_bSparse || !_bFastSparse || (_kind != Input)
        || (_bSparse && (_kind == Input) && validate)
    )
    {
        _vUnit.resize(size);
        _pbUnit.reset(new GpuBuffer<float>(size));
        if (getGpu()._id == 0)
            printf("Layer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of unit data for layer %s\n", size * sizeof(float), _maxLocalStride, _localBatch, _name.c_str());
    }

    if (_kind != Input)
    {
        _vDelta.resize(size);
        _pbDelta.reset(new GpuBuffer<float>(size));
        if (getGpu()._id == 0)       
            printf("Layer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of delta data for layer %s\n", size * sizeof(float), _maxLocalStride, _localBatch, _name.c_str());
        
        if (_bBatchNormalization)
        {
            _pbUnitBN.reset(new GpuBuffer<float>(size));
            _pbDeltaBN.reset(new GpuBuffer<float>(size));            
        }        
        
    }
    
    if (_pDropout > (float)0.0)
    {
        _pbDropout.reset(new GpuBuffer<float>(size));
        if (getGpu()._id == 0)        
            printf("Layer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of dropout data for layer %s\n", size * sizeof(float), _maxLocalStride, _localBatch, _name.c_str());
    } 
    _bDirty                         = false;
}

void Layer::SetBatch(uint32_t batch)
{
    if (batch != _batch)
    {
        _batch                      = batch;
        if (_parallelization == Layer::Parallelization::Data)
            _localBatch             = batch / getGpu()._numprocs;
        else
            _localBatch             = batch;
        _bDirty                     = true;
    }
}

void Layer::RefreshParallelization()
{
    uint32_t convolutionalInputs = 0;
    uint32_t fullyConnectedInputs = 0;
    uint32_t poolingInputs = 0;
    uint32_t convolutionalOutputs = 0;
    uint32_t fullyConnectedOutputs = 0;
    uint32_t poolingOutputs = 0;    
    
    for (auto l : _vIncomingLayer)
    {
        switch (l->_type)
        {
            case Layer::Type::Pooling:
                poolingInputs++;
                break;
            
            case Layer::Type::FullyConnected:
                fullyConnectedInputs++;
                break;
                
            case Layer::Type::Convolutional:
                convolutionalInputs++;
                break;
        }
    }
    
    for (auto l : _vOutgoingLayer)
    {
        switch (l->_type)
        {
            case Layer::Type::Pooling:
                poolingOutputs++;
                break;
                
            case Layer::Type::FullyConnected:
                fullyConnectedOutputs++;
                break;
                
            case Layer::Type::Convolutional:
                convolutionalOutputs++;
                break;
        }
    }
    
    switch (_kind)
    {
        case Layer::Kind::Input:
            if (convolutionalOutputs > 0)
                _parallelization = Layer::Parallelization::Data;
            else
                _parallelization = Layer::Parallelization::Model;
            break;
    
        case Layer::Kind::Output:
            if (convolutionalInputs > 0)
                _parallelization = Layer::Parallelization::Data;
            else
                _parallelization = Layer::Parallelization::Model;
            break;
        
        case Layer::Hidden:
            if (_type == Layer::Type::FullyConnected)
            {    
                _parallelization = Layer::Parallelization::Model;
                if (convolutionalOutputs > 0)
                    _bTransposeParallelization = true;
            }
            
            else if (_type == Layer::Type::Pooling)
            {
                if (convolutionalInputs > 0)
                {
                    _parallelization = Layer::Parallelization::Data;
                    if (fullyConnectedOutputs > 0)
                        _bTransposeParallelization = true;
                }
                else
                {
                    _parallelization = Layer::Parallelization::Model;
                    if (convolutionalOutputs > 0)
                        _bTransposeParallelization = true;                
                }
            }
            
            else
            {
                _parallelization = Layer::Parallelization::Data;                
                 if (fullyConnectedOutputs > 0)
                    _bTransposeParallelization = true;
            }
            break;
    }
}

void Layer::RefreshState(Network* pNetwork, TrainingMode trainingMode, bool validate)
{
    if (_bDirty)
    {
        _bFastSparse                = false;
        if ((_kind == Input) && (_pDataSet != NULL) && (_bSparse))
        {
            if (_pDataSet->_sparseDensity > (float)0.1)
            {
                 if (getGpu()._id == 0)
                    printf("Layer::RefreshState: Sparse density per (%.2f) is too high to use fast sparse kernels on input layer %s\n", _pDataSet->_sparseDensity, _name.c_str());                 
            }
            else
            {
                _bFastSparse        = true;
            }
        }
        
        if (getGpu()._numprocs > 1)
            RefreshParallelization();

        Allocate(validate);
        
        if (_bBatchNormalization)
        {
            if (trainingMode != TrainingMode::SGD)
            {
                if (!_pbScaleVelocityBN)
                    _pbScaleVelocityBN.reset(new GpuBuffer<float>(_localStride));
                if (!_pbBiasVelocityBN)
                    _pbBiasVelocityBN.reset(new GpuBuffer<float>(_localStride));

                if ((trainingMode == TrainingMode::AdaDelta) || (trainingMode == TrainingMode::Adam))
                {
                    if (!_pbScaleGradientVelocityBN)
                        _pbScaleGradientVelocityBN.reset(new GpuBuffer<float>(_localStride));
                    if (!_pbBiasGradientVelocityBN)
                        _pbBiasGradientVelocityBN.reset(new GpuBuffer<float>(_localStride));
                }
                else
                {
                    _pbScaleGradientVelocityBN.reset();
                    _pbScaleGradientVelocityBN.reset();
                }
            }
            else
            {
                _pbScaleVelocityBN.reset();
                _pbBiasVelocityBN.reset();
                _pbScaleGradientVelocityBN.reset();
                _pbBiasGradientVelocityBN.reset();
            }
        } 

        if ((_kind != Hidden) && (_pDataSet != NULL))
        {
            if (_parallelization == Layer::Parallelization::Model)
            {
                _pDataSet->Shard(DataSetEnums::Model);
            }
            else if (_parallelization == Layer::Parallelization::Data)
            {
                _pDataSet->Shard(DataSetEnums::Data);
            }
        }
        _bDirty                     = false;
    }

    if ((_kind == Input) && _pDataSet)
        _pDataSet->SetDenoising(_bDenoising);
        
    if ((_type == Layer::Type::Pooling) && (_poolingFunction == PoolingFunction::LRN))
    {
        cudnnStatus_t status = cudnnSetLRNDescriptor(_LRNDescriptor,
                                                    pNetwork->_LRN_n,
                                                    pNetwork->_LRN_alpha,
                                                    pNetwork->_LRN_beta,
                                                    pNetwork->_LRN_k);
        CUDNNERROR(status, "Layer::RefreshState: unable to set LRN descriptor");
    }
}

void Layer::ClearUpdates()
{
    _unitUpdateCount                = 0;
    _deltaUpdateCount               = 0;
    _bnCalls                        = 0;
}

void Layer::LoadPredictionBatch(uint32_t position, uint32_t batch)
{

    if (_kind == Input)
    {
        if (!_bSparse)
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
        else if (!_bFastSparse)
        {
            _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
    }
}

void Layer::LoadTrainingBatch(uint32_t position, uint32_t batch)
{
    if (_kind == Input)
    {
        if (_bSparse)
        {
            if (_bFastSparse)
            {
                if (_bDenoising)
                {
                    _pDataSet->CalculateSparseTransposedDenoisedMatrix(position, batch, this);
                }
                else
                {
                    _pDataSet->CalculateSparseTransposedMatrix(position, batch, this);
                }
            }
            else
            {
                if (_bDenoising)
                {
                    _pDataSet->LoadSparseDenoisedInputUnit(position, batch, _localStride, _pbUnit->_pDevData);    
                }
                else
                {
                    _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);  
                }               
            }
        }
        else
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
            
            if (_pDropout > (float)0.0)
                CalculateDropout(batch);    
        }
    }
}

void Layer::LoadValidationBatch(uint32_t position, uint32_t batch)
{
    if (_kind == Input)
    {
        if (_bSparse)
        {
            _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
            _pDataSet->CalculateSparseTransposedMatrix(position, batch, this);
        }
        else
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
    }
}

void Layer::GenerateDenoisingData()
{
    if (_pDataSet)
        _pDataSet->GenerateDenoisingData();
}

void Layer::ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining)
{
    
    switch (_type)
    {
        case FullyConnected:
            ForwardPropagateFullyConnected(position, batch, bTraining);
            break;
            
        case Convolutional:
            ForwardPropagateConvolutional(position, batch, bTraining);
            break;
            
        case Pooling:
            ForwardPropagatePooling(position, batch, bTraining);
            break;                        
        
    }
}
    
    
void Layer::ForwardPropagateFullyConnected(uint32_t position, uint32_t batch, bool bTraining)
{    
    if (getGpu()._numprocs == 1)
    {
        if (_kind != Input)
        {         
            switch (_vIncomingLayer.size())
            {
                case 0:
                    cudaMemset(GetIncomingUnitBuffer(), 0, _stride * batch * sizeof(float));
                    break;
                    
                case 1:
                    kClearUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, _stride, batch);
                    break; 
                    
                case 2:
                    kClearDualSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[1]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;                   
                    
                case 3:
                    kClearTripleSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                                    _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                                    _vIncomingWeight[2]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;      

                case 4:
                    kClearQuadSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[2]->_pbBias->_pDevData, 
                                                                  _vIncomingWeight[3]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;                  
                    
                default:
                    if (getGpu()._id == 0)
                        printf("Layer::ForwardPropagate: Too many input layers for network layer %s\n", _name.c_str());          
                    getGpu().Shutdown();
                    exit(-1);
                    break; 
            }
        
        
            const float sgemm_beta                = (float)1.0;
            for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
            {
                if (_vIncomingLayer[i]->_bFastSparse)
                {
                    float* pWeight                = _vIncomingWeight[i]->_bShared ? 
                                                      _vIncomingWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingWeight[i]->_pbWeight->_pDevData;
                    if (bTraining && _vIncomingLayer[i]->_bDenoising)
                        _vIncomingLayer[i]->_pDataSet->CalculateSparseDenoisedZ(position, batch, _stride, pWeight, GetIncomingUnitBuffer(), sgemm_beta);  
                    else
                        _vIncomingLayer[i]->_pDataSet->CalculateSparseZ(position, batch, _stride, pWeight, GetIncomingUnitBuffer(), sgemm_beta);
                }
                else      
                {
                    const float sgemm_alpha       = (float)1.0;
                    cublasStatus_t cstatus;
                    float* pA                     = _vIncomingLayer[i]->GetUnitBuffer();
                    float* pB                     = _vIncomingWeight[i]->_bShared ? 
                                                      _vIncomingWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingWeight[i]->_pbWeight->_pDevData;
                    float* pC                     = GetIncomingUnitBuffer();
                    int m                           = batch;
                    int n                           = _localStride;
                    int k                           = _vIncomingLayer[i]->_stride;
                    int lda                         = _vIncomingWeight[i]->_bTransposed ? k : n;
                    int ldb                         = k;
                    int ldc                         = n;

                    cstatus                         =
                                                    cublasSgemm(getGpu()._cuBLASHandle, 
                                                    _vIncomingWeight[i]->_bTransposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    n,
                                                    m,
                                                    k,
                                                    &sgemm_alpha,
                                                    pB,
                                                    lda,
                                                    pA,
                                                    ldb,
                                                    &sgemm_beta,
                                                    pC,
                                                    ldc);  

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("Layer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }
            }

            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _stride);
            }
            
            if (_bBatchNormalization)
            {
                float alpha = 1;
                float beta = 0;
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");        
                if (bTraining) {
                    cudnnStatus = cudnnBatchNormalizationForwardTraining(
                            getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            1.0/(_bnCalls + 1), 
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON,
                            _pbSaveMeanBN->_pDevData,
                            _pbSaveInvVarianceBN->_pDevData);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardTraining Failed");
                    ++_bnCalls;
                } else {
                    cudnnStatus = cudnnBatchNormalizationForwardInference(
                            getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardInference Failed");
                }
            }
           
            CalculateActivation(batch);
            
            if (bTraining && (_pDropout > (float)0.0))
                CalculateDropout(batch);             
       
#if 0
        string fname = "activation_" + _name;
        Dump(fname, _pbUnit->_pDevData);
#endif              
        }       
    }
    else
    {
        if (_kind != Input)
        {              
            if (_vIncomingLargerLayer.size() > 0)
            {
                float sgemm_beta                  = (float)0.0;
                for (uint32_t i = 0; i < _vIncomingLargerLayer.size(); i++)
                {
                    Layer* pInputLayer            = _vIncomingLargerLayer[i];
                    float* pWeight                = _vIncomingLargerWeight[i]->_bShared ? 
                                                      _vIncomingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingLargerWeight[i]->_pbWeight->_pDevData;                                           

                    if (pInputLayer->_bFastSparse)
                    {
                        if (bTraining && pInputLayer->_bDenoising)
                            pInputLayer->_pDataSet->CalculateSparseDenoisedZ(position, batch, _stride, pWeight, getGpu()._pNetwork->GetP2PSendBuffer(), sgemm_beta);  
                        else
                            pInputLayer->_pDataSet->CalculateSparseZ(position, batch, _stride, pWeight, getGpu()._pNetwork->GetP2PSendBuffer(), sgemm_beta);  
                    }
                    else
                    {
                
                        const float sgemm_alpha   = (float)1.0;

                        float* pA                 = pWeight;
                        float* pB                 = pInputLayer->GetUnitBuffer();
                        float* pC                 = getGpu()._pNetwork->GetP2PSendBuffer();
                        int m                       = _stride;
                        int n                       = batch;
                        int k                       = pInputLayer->_localStride;
                        int lda                     = _stride;
                        int ldb                     = pInputLayer->_localStride;
                        int ldc                     = _stride;

                        cublasStatus_t cstatus      =
                                                    cublasSgemm(getGpu()._cuBLASHandle, 
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    m,
                                                    n,
                                                    k,
                                                    &sgemm_alpha,
                                                    pA,
                                                    lda,
                                                    pB,
                                                    ldb,
                                                    &sgemm_beta,
                                                    pC,
                                                    ldc);  

                        if (cstatus != CUBLAS_STATUS_SUCCESS)
                        {
                            if (getGpu()._id == 0)
                                printf("Layer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                            getGpu().Shutdown();
                            exit(-1);
                        }                                     
                    }
                    
                    sgemm_beta                      = (float)1.0;
                }

                Reduce(batch, _stride, GetIncomingUnitBuffer(), _localStride, _unitUpdateCount);
                _unitUpdateCount++;
            }
            
            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _localStride);
            }            
                   
            switch (_vIncomingLayer.size())
            {
                case 0:
                    break;
                
                case 1:
                    kAddBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, _localStride, batch);
                    break; 
                        
                case 2:
                    kAddDualBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[1]->_pbBias->_pDevData, _localStride, batch);
                    break;                   
                        
                case 3:
                    kAddTripleBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                            _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                            _vIncomingWeight[2]->_pbBias->_pDevData, _localStride, batch);
                    break;      

                case 4:
                    kAddQuadBias(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[2]->_pbBias->_pDevData, 
                                                          _vIncomingWeight[3]->_pbBias->_pDevData, _localStride, batch);
                    break;                  
                        
                default:
                    if (getGpu()._id == 0)
                        printf("Layer::ForwardPropagate: Too many input layers for network layer %s\n", _name.c_str());
                    getGpu().Shutdown();
                    exit(-1);
                    break; 
            }    
            
            if (_bBatchNormalization)
            {
                float alpha = 1;
                float beta = 0;
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");        
                if (bTraining) {
                    cudnnStatus = cudnnBatchNormalizationForwardTraining(
                            getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            1.0/(_bnCalls + 1), 
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON,
                            _pbSaveMeanBN->_pDevData,
                            _pbSaveInvVarianceBN->_pDevData);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardTraining Failed");
                } else {
                    cudnnStatus = cudnnBatchNormalizationForwardInference(
                            getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_PER_ACTIVATION,
                            &alpha,
                            &beta,
                            _tensorDescriptorBN,
                            GetIncomingUnitBuffer(),
                            _tensorDescriptorBN,
                            GetUnitBuffer(),
                            _scaleBiasMeanVarDescBN,
                            _pbScaleBN->_pDevData,
                            _pbBiasBN->_pDevData,
                            _pbRunningMeanBN->_pDevData,
                            _pbRunningVarianceBN->_pDevData,
                            CUDNN_BN_MIN_EPSILON);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardInference Failed");
                }
            }
                                      
            CalculateActivation(batch);   
            
            if (bTraining && (_pDropout > (float)0.0))
                CalculateDropout(batch);  
        }
        
#if 0
        string fname = "activation_" + _name;
        Dump(fname, _pbUnit->_pDevData);
#endif                                      
        if (_vOutgoingLargerLayer.size() > 0)
        {  
        
            if (_bFastSparse)
            {
                for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
                {
                    Layer* pOutputLayer       = _vOutgoingLargerLayer[i];
                    float* pWeight            = _vOutgoingLargerWeight[i]->_bShared ? 
                                                  _vOutgoingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                  _vOutgoingLargerWeight[i]->_pbWeight->_pDevData;
                    const float sgemm_beta    = (pOutputLayer->_unitUpdateCount == 0) ? (float)0.0 : (float)1.0;
                    
                    if (bTraining && _bDenoising)
                        _pDataSet->CalculateSparseDenoisedZ(position, batch, pOutputLayer->_localStride, pWeight, pOutputLayer->GetIncomingUnitBuffer(), sgemm_beta);  
                    else
                        _pDataSet->CalculateSparseZ(position, batch, pOutputLayer->_localStride, pWeight, pOutputLayer->GetIncomingUnitBuffer(), sgemm_beta);
                }
            }
            else
            {
        
                Gather(batch, _stride, GetUnitBuffer(), _localStride);

                for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
                {
                    Layer* pOutputLayer       = _vOutgoingLargerLayer[i];
                    Weight* pWeight           = _vOutgoingLargerWeight[i];     
                    Weight* pSrcWeight        = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
                    float* pA                 = pSrcWeight->_pbWeight->_pDevData;
                    float* pB                 = getGpu()._pNetwork->GetP2PSendBuffer();
                    float* pC                 = pOutputLayer->GetIncomingUnitBuffer();
                    
                    int m                       = pOutputLayer->_localStride;
                    int n                       = batch;
                    int k                       = _stride;
                    int lda                     = pOutputLayer->_localStride;
                    int ldb                     = _stride;
                    int ldc                     = pOutputLayer->_localStride;
                    const float sgemm_alpha   = 1.0;
                    const float sgemm_beta    = (pOutputLayer->_unitUpdateCount == 0) ? (float)0.0 : (float)1.0;
            
                    cublasStatus_t cstatus      = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("Layer::ForwardPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }
                        
                    pOutputLayer->_unitUpdateCount++;
                }
            }
        }
    }
    
#if 0
    _pbUnit->Download(_vUnit.data());
    MPI_Barrier(MPI_COMM_WORLD);
    if (getGpu()._id == 0)
        cout << _name << " ";
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < getGpu()._numprocs; i++)
    {
        if (i == getGpu()._id)
        {
            for (auto f : _vUnit)
                printf("%8.4f ", f);
            printf("\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    cout << endl;
    exit(-1);
#endif    
}

/// <summary>
/// This function performs forward propagation for a convolutional layer.
/// </summary>
/// <param name="position">The position of the layer.</param>
/// <param name="batch">The batch size.</param>
/// <param name="bTraining">A flag indicating whether training mode is enabled.</param>
void Layer::ForwardPropagateConvolutional(uint32_t position, uint32_t batch, bool bTraining)
{
    // Check if the layer kind is not Input and the number of GPU processes is 1
    if (_kind != Layer::Kind::Input && getGpu()._numprocs == 1)
    {
        constexpr float alpha = 1.0f;                   // Define a constant alpha with a value of 1.0
        float beta = 0.0f;                              // Initialize a beta variable with a value of 0.0

        // Iterate over incoming layers and perform convolution forward propagation
        for (auto i = 0u; i < _vIncomingLayer.size(); ++i)
        {
            auto* pLayer = _vIncomingLayer[i];
            auto* pWeight = _vIncomingWeight[i]->_bShared ? _vIncomingWeight[i]->_pSharedWeight : _vIncomingWeight[i];

            // Perform convolution forward operation using cuDNN
            cudnnStatus_t cudnnStatus = cudnnConvolutionForward(getGpu()._cuDNNHandle,
                &alpha,
                pLayer->getTensorDescriptor(batch),
                pLayer->GetUnitBuffer(),
                pWeight->_convFilterDesc,
                pWeight->_pbWeight->_pDevData,
                pWeight->_convDesc,
                pWeight->_convFWAlgo,
                getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                getGpu()._pNetwork->_CUDNNWorkspaceSize,
                &beta,
                getTensorDescriptor(batch),
                GetIncomingUnitBuffer());
            CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: cudnnConvolutionForward Failed");

            // Add bias to the result using cuDNN
            cudnnStatus = cudnnAddTensor(getGpu()._cuDNNHandle,
                &alpha,
                _vIncomingWeight[i]->_convBiasTensor,
                _vIncomingWeight[i]->_pbBias->_pDevData,
                &alpha,
                getTensorDescriptor(batch),
                GetIncomingUnitBuffer());
            CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: cudnnAddTensor Failed");
            beta = 1.0f;                            // Update beta to 1.0 for subsequent iterations
        }

        // Add skip connections by calling kAddBuffers function
        for (const auto& l : _vIncomingSkip)
        {
            kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _stride);
        }

        // Check if batch normalization is enabled
        if (_bBatchNormalization)
        {
            constexpr float alphaBN = 1.0f;         // Define constant alpha for batch normalization
            constexpr float betaBN = 0.0f;          // Define constant beta for batch normalization

            cudnnStatus_t cudnnStatus;

            // Set tensor descriptors for batch normalization
            cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _Nx);
            CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: unable to create _tensorDescriptorBN");
            cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, 1, 1);
            CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: unable to create _scaleBiasMeanVarDescBN");

            // Perform batch normalization based on training or inference mode
            if (bTraining)
            {
                cudnnStatus = cudnnBatchNormalizationForwardTraining(
                    getGpu()._cuDNNHandle,
                    CUDNN_BATCHNORM_SPATIAL,
                    &alphaBN,
                    &betaBN,
                    _tensorDescriptorBN,
                    GetIncomingUnitBuffer(),
                    _tensorDescriptorBN,
                    GetUnitBuffer(),
                    _scaleBiasMeanVarDescBN,
                    _pbScaleBN->_pDevData,
                    _pbBiasBN->_pDevData,
                    1.0 / (_bnCalls + 1),
                    _pbRunningMeanBN->_pDevData,
                    _pbRunningVarianceBN->_pDevData,
                    CUDNN_BN_MIN_EPSILON,
                    _pbSaveMeanBN->_pDevData,
                    _pbSaveInvVarianceBN->_pDevData);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: cudnnBatchNormalizationForwardTraining Failed");
                ++_bnCalls;
            }
            else
            {
                cudnnStatus = cudnnBatchNormalizationForwardInference(
                    getGpu()._cuDNNHandle,
                    CUDNN_BATCHNORM_SPATIAL,
                    &alphaBN,
                    &betaBN,
                    _tensorDescriptorBN,
                    GetIncomingUnitBuffer(),
                    _tensorDescriptorBN,
                    GetUnitBuffer(),
                    _scaleBiasMeanVarDescBN,
                    _pbScaleBN->_pDevData,
                    _pbBiasBN->_pDevData,
                    _pbRunningMeanBN->_pDevData,
                    _pbRunningVarianceBN->_pDevData,
                    CUDNN_BN_MIN_EPSILON);
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateConvolutional: cudnnBatchNormalizationForwardInference Failed");
            }
        }

        // Calculate the activation function for the current batch
        CalculateActivation(batch);

        // If training and dropout rate is greater than 0, apply dropout
        if (bTraining && (_pDropout > 0.0f))
            CalculateDropout(batch);
    }
}

/// <summary>
/// Forward propagates the pooling operation for this layer.
/// </summary>
/// <param name="position">The position.</param>
/// <param name="batch">The batch size.</param>
/// <param name="bTraining">True if the network is in training mode; otherwise, false.</param>
void Layer::ForwardPropagatePooling(uint32_t position, uint32_t batch, bool bTraining)
{
    // Check if the layer is not of type "Input" because pooling is not applied to input layers.
    if (_kind != Layer::Kind::Input)
    {
        // Initialize alpha and beta values used in cuDNN pooling operations.
        float alpha = 1.0f;
        float beta = 0.0f;

        // Loop over incoming layers connected to this layer.
        for (auto* pLayer : _vIncomingLayer)
        {
            cudnnStatus_t cudnnStatus;

            // Determine the type of pooling operation based on _poolingFunction.
            switch (_poolingFunction)
            {
            case PoolingFunction::Max:
            case PoolingFunction::Average:
                // Perform cuDNN pooling operation (Max or Average).
                cudnnStatus = cudnnPoolingForward(getGpu()._cuDNNHandle,
                    _poolingDescriptor,
                    &alpha,
                    pLayer->getTensorDescriptor(batch),
                    pLayer->GetUnitBuffer(),
                    &beta,
                    getTensorDescriptor(batch),
                    GetIncomingUnitBuffer());
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagatePooling: cudnnPoolingForward Failed");
                break;

            case PoolingFunction::LRN:
                // Perform cuDNN LRN (Local Response Normalization) operation.
                cudnnStatus = cudnnLRNCrossChannelForward(getGpu()._cuDNNHandle,
                    _LRNDescriptor,
                    CUDNN_LRN_CROSS_CHANNEL_DIM1,
                    &alpha,
                    pLayer->getTensorDescriptor(batch),
                    pLayer->GetUnitBuffer(),
                    &beta,
                    getTensorDescriptor(batch),
                    GetIncomingUnitBuffer());
                CUDNNERROR(cudnnStatus, "Layer::ForwardPropagatePooling: cudnnLRNCrossChannelForward Failed");
                break;

            case PoolingFunction::Cosine:
                // Calculate cosine similarity between this layer and another incoming layer.
                if (auto* p0Layer = _vIncomingLayer.front(); pLayer != p0Layer)
                {
                    uint32_t offset = static_cast<uint32_t>(std::distance(_vIncomingLayer.begin(), std::find(_vIncomingLayer.begin(), _vIncomingLayer.end(), pLayer)));
                    kCalculateCosine(p0Layer->GetUnitBuffer(), pLayer->GetUnitBuffer(), batch, pLayer->_localStride,
                        GetIncomingUnitBuffer() + offset,
                        _pbBuffer1->_pDevData + offset,
                        _pbBuffer2->_pDevData + offset,
                        _localStride);
                }
                break;

            case PoolingFunction::DotProduct:
                // Calculate dot product between this layer and another incoming layer.
                if (auto* p0Layer = _vIncomingLayer.front(); pLayer != p0Layer)
                {
                    uint32_t offset = static_cast<uint32_t>(std::distance(_vIncomingLayer.begin(), std::find(_vIncomingLayer.begin(), _vIncomingLayer.end(), pLayer)));
                    kCalculateDotProduct(p0Layer->GetUnitBuffer(), pLayer->GetUnitBuffer(), batch, pLayer->_localStride,
                        GetIncomingUnitBuffer() + offset,
                        _localStride);
                }
                break;

            case PoolingFunction::Maxout:
                // Perform maxout pooling operation.
                if (beta != 0.0f)
                {
                    kCalculateMaxout(pLayer->GetUnitBuffer(), batch * _localStride, GetIncomingUnitBuffer());
                }
                else
                {
                    // Copy the data from the incoming layer to this layer (maxout pooling).
                    cudaError_t status = cudaMemcpy(GetIncomingUnitBuffer(), pLayer->GetUnitBuffer(), batch * _localStride * sizeof(float), cudaMemcpyDefault);
                    RTERROR(status, "Layer::ForwardPropagate: Error calling cudaMemcpy for maxout pooling.");
                }
                break;
            }
            beta = 1.0f; // Set beta to 1.0 for subsequent iterations.
        }

        // Loop over incoming skip layers and add their buffers to this layer's buffer.
        for (auto* l : _vIncomingSkip)
        {
            kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), batch * _stride);
        }
    }
}

/// <summary>
/// Calculate the activation for the layer.
/// </summary>
/// <param name="batch">The batch size.</param>
void Layer::CalculateActivation(uint32_t batch)
{
    // Calculate the size based on batch and local stride
    uint64_t size = (uint64_t)batch * (uint64_t)_localStride;

    // Switch based on the activation type
    switch (_activation)
    {
    case Sigmoid:
        // Calculate sigmoid activation
        kCalculateSigmoidActivation(GetUnitBuffer(), size);
        break;

    case Tanh:
        // Calculate tanh activation
        kCalculateTanhActivation(GetUnitBuffer(), size);
        break;

    case RectifiedLinear:
        // Calculate ReLU activation
        kCalculateRELUActivation(GetUnitBuffer(), size);
        break;

    case LeakyRectifiedLinear:
        // Calculate Leaky ReLU activation with specified slope
        kCalculateLRELUActivation(GetUnitBuffer(), size, _RELUSlope);
        break;

    case ExponentialLinear:
        // Calculate ELU activation with specified alpha
        kCalculateELUActivation(GetUnitBuffer(), size, _ELUAlpha);
        break;

    case ScaledExponentialLinear:
        // Calculate SELU activation with specified alpha and lambda
        kCalculateSELUActivation(GetUnitBuffer(), size, _ELUAlpha, _SELULambda);
        break;

    case SoftMax:
        // Calculate SoftMax activation
        kCalculateSoftMaxActivation(GetUnitBuffer(), batch, _localStride);
        break;

    case Linear:
        // No activation needed for the Linear case
        break;
    }
}

void Layer::CalculateDropout(uint32_t batch)
{
    float lambda              = (_activation == ScaledExponentialLinear) ? _SELULambda : (float)1.0;
    float alpha               = -lambda * _ELUAlpha;
    float q                   = (float)1.0 - _pDropout;
    float a                   = (float)1.0 / sqrt(q + alpha * alpha * _pDropout * q);
    float b                   = -a * _pDropout * alpha;
    float target              = (_activation == Sigmoid) ? (float)0.5 : (float)0.0;


    
    switch (_activation)
    {
        case ExponentialLinear:
        case ScaledExponentialLinear:
            kCalculateScaledBiasedDropout(GetUnitBuffer(), _pbDropout->_pDevData, batch, _localStride, _pDropout, alpha, a, b);
            break;
            
        default:
            kCalculateDropout(GetUnitBuffer(), _pbDropout->_pDevData, batch, _localStride, _pDropout, target);
            break;
    }
}

float Layer::CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef)
{
    if (_kind != Output)
    {
        if (getGpu()._id == 0)
            printf("Layer::CalculateError: Attempt to calculate error on non-output layer %s.\n", _name.c_str());
        getGpu().Shutdown();
        exit(-1);
    }

    switch (ef)
    {
        case L1:
            return _pDataSet->CalculateL1Error(position, batch, _localStride, GetUnitBuffer());

        case L2:
            return _pDataSet->CalculateL2Error(position, batch, _localStride, GetUnitBuffer());
            
        case L2Hinge:
            return _pDataSet->CalculateL2HingeError(position, batch, _localStride, GetUnitBuffer());            

        case Hinge:
            return _pDataSet->CalculateHingeError(position, batch, _localStride, GetUnitBuffer());              

        case CrossEntropy:
            if (_activation == SoftMax)
                return _pDataSet->CalculateMultinomialCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
            else
                return _pDataSet->CalculateCrossEntropyError(position, batch, _localStride, GetUnitBuffer());

        case ScaledMarginalCrossEntropy:
            if (_activation == SoftMax)
                return _pDataSet->CalculateMultinomialScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
            else        
                return _pDataSet->CalculateScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());

        case DataScaledMarginalCrossEntropy:
            if (_activation == SoftMax)
            {
                std::cout << "unsupported combination of activation with cost function" << std::endl;
                getGpu().Shutdown();
                exit(-1);
            }
            else
            {
                return _pDataSet->CalculateDataScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
            }
    }
    
    return (float)0.0;
}

void Layer::CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef)
{
    if (_kind != Output)
    {
        if (getGpu()._id == 0)
            printf("Layer::CalculateOutputDelta: Attempt to calculate output delta on non-output layer %s.\n", _name.c_str());
        getGpu().Shutdown();
        exit(-1);
    }

    switch (ef)
    {
        case L1:
            _pDataSet->CalculateL1OutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;

        case CrossEntropy:
            _pDataSet->CalculateCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        case ScaledMarginalCrossEntropy:
            _pDataSet->CalculateScaledMarginalCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        case L2:
            _pDataSet->CalculateOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;
            
        case L2Hinge:
            _pDataSet->CalculateL2HingeOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;            

        case Hinge:
            _pDataSet->CalculateHingeOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;            

        case DataScaledMarginalCrossEntropy:
            _pDataSet->CalculateDataScaledMarginalCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        default:
            std::cout << "Unsupported cost function" << std::endl;
            exit(2);
    }
    
    
    
    if (_deltaNorm > (float)0.0)
    {
        if (getGpu()._numprocs == 1)
            kNormalizeDeltas(_deltaNorm, batch, _localStride, GetDeltaBuffer());
        else
        {
            float* pMagnitude                 = getGpu()._pNetwork->GetScratchBuffer(batch);
            kCalculateDeltaMagnitudes(batch, _localStride, GetDeltaBuffer(), pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, batch);
            kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, GetDeltaBuffer(), pMagnitude);            
        }
    }
}


void Layer::BackPropagate(uint32_t position, uint32_t batch)
{
    
    switch (_type)
    {
        case FullyConnected:
            BackPropagateFullyConnected(position, batch);
            break;
            
        case Convolutional:
            BackPropagateConvolutional(position, batch);
            break;
            
        case Pooling:
            BackPropagatePooling(position, batch);
            break;                        
        
    }
}

void Layer::BackPropagateConvolutional(uint32_t position, uint32_t batch)
{
    if (getGpu()._numprocs == 1)
    {
        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                float p       = (_sparsenessPenalty_p > (float)0.0)   ? _sparsenessPenalty_p     : getGpu()._pNetwork->_sparsenessPenalty_p;
                float beta    = (_sparsenessPenalty_beta > (float)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
            }   

            float scale                           = (float)1.0 / ((float)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            
            if (_deltaNorm > (float)0.0)
            {            
                kNormalizeDeltas(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer());
            }
            
            if (_bBatchNormalization)
            {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _Nx);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateConvolutional: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, 1, 1);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateConvolutional: unable to create _scaleBiasMeanVarDescBN");        
                float alpha = 1;
                float beta = 0;
                cudnnStatus = cudnnBatchNormalizationBackward(
                        getGpu()._cuDNNHandle,
                        CUDNN_BATCHNORM_SPATIAL,
                        &alpha,
                        &beta,
                        &alpha,
                        &beta,
                        _tensorDescriptorBN,
                        GetIncomingUnitBuffer(),
                        _tensorDescriptorBN,
                        GetIncomingDeltaBuffer(),
                        _tensorDescriptorBN,
                        GetDeltaBuffer(),
                        _scaleBiasMeanVarDescBN,
                        _pbScaleBN->_pDevData,
                        _pbScaleGradientBN->_pDevData,
                        _pbBiasGradientBN->_pDevData,
                        CUDNN_BN_MIN_EPSILON,
                        _pbSaveMeanBN->_pDevData,
                        _pbSaveInvVarianceBN->_pDevData);
                CUDNNERROR(cudnnStatus, "Layer:BackPropagateConvolutional cudnnBatchNormalizationBackward Failed");
            }
        }


        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
        {
            Layer* pInputLayer                = _vIncomingLayer[i];

            Weight* pWeight                   = _vIncomingWeight[i];     
            Weight* pSrcWeight                = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
            float gradient_alpha              = -(float)1.0 / (pSrcWeight->_sharingCount * (float)batch);            

            cudnnStatus_t cudnnStatus;
            if (!pWeight->_bLocked)
            {
                float beta                    = (pSrcWeight->_updateCount == 0) ? (float)0.0 : (float)1.0;
                cudnnStatus                     = cudnnConvolutionBackwardFilter(getGpu()._cuDNNHandle,
                                                                                 &gradient_alpha,
                                                                                 pInputLayer->getTensorDescriptor(batch),
                                                                                 pInputLayer->GetUnitBuffer(),
                                                                                 getTensorDescriptor(batch),
                                                                                 GetDeltaBuffer(),
                                                                                 pSrcWeight->_convDesc,
                                                                                 pSrcWeight->_convBWWeightAlgo,
                                                                                 getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                                                                                 getGpu()._pNetwork->_CUDNNWorkspaceSize,
                                                                                 &beta,
                                                                                 pSrcWeight->_convFilterDesc,
                                                                                 pSrcWeight->_pbWeightGradient->_pDevData);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateConvolutional: cudnnConvolutionBackwardFilter Failed"); 
                
                beta                            = (float)0.0;
                cudnnStatus                     = cudnnConvolutionBackwardBias(getGpu()._cuDNNHandle,
                                                                           &gradient_alpha,
                                                                           getTensorDescriptor(batch),
                                                                           GetDeltaBuffer(),
                                                                           &beta,
                                                                           pWeight->_convBiasTensor,
                                                                           pWeight->_pbBiasGradient->_pDevData);                
                

                pSrcWeight->_updateCount++;
            }
     
            if (pInputLayer->_kind != Input)
            {
                float delta_alpha             = (float)1.0;                
                float beta                    = (pInputLayer->_deltaUpdateCount == 0) ? (float)0.0 : (float)1.0;
                cudnnStatus                     = cudnnConvolutionBackwardData(getGpu()._cuDNNHandle,
                                                                               &delta_alpha,
                                                                               pSrcWeight->_convFilterDesc,
                                                                               pSrcWeight->_pbWeight->_pDevData,
                                                                               getTensorDescriptor(batch),
                                                                               GetDeltaBuffer(),
                                                                               pSrcWeight->_convDesc, 
                                                                               pSrcWeight->_convBWDeltaAlgo,
                                                                               getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                                                                               getGpu()._pNetwork->_CUDNNWorkspaceSize,
                                                                               &beta,
                                                                               pInputLayer->getTensorDescriptor(batch),
                                                                               pInputLayer->GetIncomingDeltaBuffer());
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateConvolutional: cudnnConvolutionBackwardData Failed");

                pInputLayer->_deltaUpdateCount++; 
            }
        }    
        
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride * sizeof(float), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }
    }
}

void Layer::BackPropagatePooling(uint32_t position, uint32_t batch)
{
    {
        float pooling_alpha                   = (float)1.0;
        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
        {
            Layer* pInputLayer                = _vIncomingLayer[i];

            if (pInputLayer->_kind != Input)
            {
                cudnnStatus_t cudnnStatus;
                float beta                    = (pInputLayer->_deltaUpdateCount == 0) ? (float)0.0 : (float)1.0;
                switch (_poolingFunction)
                {
                    case Max:
                    case Average:
                        cudnnStatus             = cudnnPoolingBackward(getGpu()._cuDNNHandle,
                                                                       _poolingDescriptor,
                                                                       &pooling_alpha,
                                                                       getTensorDescriptor(batch),
                                                                       GetUnitBuffer(),
                                                                       getTensorDescriptor(batch),
                                                                       GetDeltaBuffer(),
                                                                       pInputLayer->getTensorDescriptor(batch),
                                                                       pInputLayer->GetUnitBuffer(),
                                                                       &beta,
                                                                       pInputLayer->getTensorDescriptor(batch),
                                                                       pInputLayer->GetIncomingDeltaBuffer());                                                                         
                        CUDNNERROR(cudnnStatus, "Layer::BackPropagatePooling: cudnnPoolingBackward Failed");

                        pInputLayer->_deltaUpdateCount++;                           
                        break;

                    case LRN:
                        cudnnStatus             = cudnnLRNCrossChannelBackward(getGpu()._cuDNNHandle,
                                                                                _LRNDescriptor,
                                                                                CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                                                                &pooling_alpha,
                                                                                getTensorDescriptor(batch),
                                                                                GetUnitBuffer(),
                                                                                getTensorDescriptor(batch),
                                                                                GetDeltaBuffer(),
                                                                                pInputLayer->getTensorDescriptor(batch),
                                                                                pInputLayer->GetUnitBuffer(),
                                                                                &beta,
                                                                                pInputLayer->getTensorDescriptor(batch),
                                                                                pInputLayer->GetIncomingDeltaBuffer());                      
                        CUDNNERROR(cudnnStatus, "Layer::BackPropagatePooling: cudnnLRNCrossChannelBackward Failed");

                        pInputLayer->_deltaUpdateCount++;   
                        break;
                        
                    case Maxout:
                        kCalculateMaxoutDelta(GetUnitBuffer(), GetDeltaBuffer(), batch * _localStride, beta, pInputLayer->GetUnitBuffer(), pInputLayer->GetIncomingDeltaBuffer());
                        pInputLayer->_deltaUpdateCount++;                         
                        break;

                    case Cosine:
                        if (i != 0)
                        {
                            Layer* p0Layer    = _vIncomingLayer[0];
                            float beta0       = (p0Layer->_deltaUpdateCount == 0) ? (float)0.0 : (float)1.0;                                        
                            uint32_t offset     = i - 1;
                            float* pDPIn      = GetUnitBuffer() + offset;
                            float* pDPDeltaIn = GetDeltaBuffer() + offset;                            
                            float* pAIn       = _pbBuffer1->_pDevData + offset;
                            float* pBIn       = _pbBuffer2->_pDevData + offset;
                            kCalculateCosineDelta(pDPDeltaIn, pDPIn, pAIn, pBIn, 
                            p0Layer->GetUnitBuffer(), pInputLayer->GetUnitBuffer(), batch, _localStride, 
                            p0Layer->GetIncomingDeltaBuffer(), beta0, 
                            pInputLayer->GetIncomingDeltaBuffer(), beta, 
                            pInputLayer->_localStride);

                            p0Layer->_deltaUpdateCount++;
                            pInputLayer->_deltaUpdateCount++; 
                        }                            
                        break;
                        
                    case DotProduct:
                        if (i != 0)
                        {
                            Layer* p0Layer    = _vIncomingLayer[0];
                            float beta0       = (p0Layer->_deltaUpdateCount == 0) ? (float)0.0 : (float)1.0;                                                 
                            uint32_t offset     = i - 1;
                            float* pDPDeltaIn = GetDeltaBuffer() + offset;
                            kCalculateDotProductDelta(pDPDeltaIn, p0Layer->GetUnitBuffer(), pInputLayer->GetUnitBuffer(), batch, _localStride, 
                            p0Layer->GetIncomingDeltaBuffer(), beta0, 
                            pInputLayer->GetIncomingDeltaBuffer(), beta, 
                            pInputLayer->_localStride);

                            p0Layer->_deltaUpdateCount++;
                            pInputLayer->_deltaUpdateCount++; 
                        }                            
                        break;                        

                }
            }
        }    
        
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride * sizeof(float), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }
    }
}

void Layer::BackPropagateFullyConnected(uint32_t position, uint32_t batch)
{    
    if (getGpu()._numprocs == 1)
    {
        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                float p       = (_sparsenessPenalty_p > (float)0.0)   ? _sparsenessPenalty_p     : getGpu()._pNetwork->_sparsenessPenalty_p;
                float beta    = (_sparsenessPenalty_beta > (float)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
            }   

            float scale                           = (float)1.0 / ((float)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            
            if (_deltaNorm > (float)0.0)
            {            
                kNormalizeDeltas(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer());
            }
            
            if (_bBatchNormalization)
            {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateFullyConnected: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");        
                float alpha = 1;
                float beta = 0;
                cudnnStatus = cudnnBatchNormalizationBackward(
                        getGpu()._cuDNNHandle,
                        CUDNN_BATCHNORM_PER_ACTIVATION,
                        &alpha,
                        &beta,
                        &alpha,
                        &beta,
                        _tensorDescriptorBN,
                        GetIncomingUnitBuffer(),
                        _tensorDescriptorBN,
                        GetIncomingDeltaBuffer(),
                        _tensorDescriptorBN,
                        GetDeltaBuffer(),
                        _scaleBiasMeanVarDescBN,
                        _pbScaleBN->_pDevData,
                        _pbScaleGradientBN->_pDevData,
                        _pbBiasGradientBN->_pDevData,
                        CUDNN_BN_MIN_EPSILON,
                        _pbSaveMeanBN->_pDevData,
                        _pbSaveInvVarianceBN->_pDevData);
                CUDNNERROR(cudnnStatus, "Layer:BackPropagateFullyConnected cudnnBatchNormalizationBackward Failed");
            }
        }

#if 0
        if (_kind == Hidden)
        {
            string fname = "delta_" + _name;
            Dump(fname, _pbDelta->_pDevData);
        }
#endif 
        
        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
        {
            Layer* pInputLayer                = _vIncomingLayer[i];
            cublasStatus_t cstatus;
            Weight* pWeight                   = _vIncomingWeight[i];     
            Weight* pSrcWeight                = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

            if (!pWeight->_bLocked)
            {
                float* pDelta                 = GetDeltaBuffer();
                float* pUnit                  = pInputLayer->GetUnitBuffer();
                float* pA                     = pWeight->_bTransposed ? pDelta                    : pUnit;
                float* pB                     = pWeight->_bTransposed ? pUnit                     : pDelta;
                int m                           = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;
                int n                           = pWeight->_bTransposed ? _localStride              : pInputLayer->_localStride;
                int k                           = batch;
                int lda                         = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;
                int ldb                         = pWeight->_bTransposed ? _localStride              : pInputLayer->_localStride;
                int ldc                         = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;

                float sgemm_alpha             = -(float)1.0 / (pSrcWeight->_sharingCount * (float)batch);
                float sgemm_beta              = (pSrcWeight->_updateCount == 0) ? (float)0.0 : (float)1.0;
                float* pC                     = pSrcWeight->_pbWeightGradient->_pDevData;
                
                if ((pInputLayer->_kind == Layer::Kind::Input) && pInputLayer->_bFastSparse && !pWeight->_bTransposed)
                {
                    pInputLayer->_pDataSet->CalculateSparseTransposedWeightGradient(sgemm_alpha, sgemm_beta, n, m, pB, pC);
                }
                else
                {
                    cstatus                 = cublasSgemm(getGpu()._cuBLASHandle, 
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_T,
                                              m,
                                              n,
                                              k,
                                              &sgemm_alpha,
                                              pB,
                                              lda,
                                              pA,
                                              ldb,
                                              &sgemm_beta,
                                              pC,
                                              ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("Layer::BackPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }
                
                pSrcWeight->_updateCount++;
            }
     
            if (pInputLayer->_kind != Input)
            {
                float sgemm_alpha         = (float)1.0;
                float sgemm_beta          = (pInputLayer->_deltaUpdateCount == 0) ? (float)0.0 : (float)1.0;
                int m                       = pInputLayer->_localStride;
                int n                       = batch;  
                
                
                float* pA                 = GetDeltaBuffer();
                float* pB                 = pWeight->_bShared ? 
                                              pSrcWeight->_pbWeight->_pDevData :
                                              pWeight->_pbWeight->_pDevData;

                float* pC                 = pInputLayer->GetIncomingDeltaBuffer();
                int k                       = _localStride;
                int lda                     = pWeight->_bTransposed ? pInputLayer->_localStride : k;
                int ldb                     = k;
                int ldc                     = pInputLayer->_localStride;
                
                
                cstatus                     = cublasSgemm(getGpu()._cuBLASHandle, 
                                            pWeight->_bTransposed ? CUBLAS_OP_N : CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            m,
                                            n,
                                            k,
                                            &sgemm_alpha,
                                            pB,
                                            lda,
                                            pA,
                                            ldb,
                                            &sgemm_beta,
                                            pC,
                                            ldc);   

                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("Layer::BackPropagate: SGEMM failure, aborting.\n");
                    getGpu().Shutdown();
                    exit(-1);
                }
                
                pInputLayer->_deltaUpdateCount++; 
            }
        }    
        
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride * sizeof(float), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }
    }
    else
    {
        if (_vOutgoingLargerLayer.size() > 0)
        {
            Gather(batch, _stride, GetUnitBuffer(), _localStride);

            for (int i = 0; i < _vOutgoingLargerLayer.size(); i++)
            {
                Layer* pOutputLayer           = _vOutgoingLargerLayer[i];
                Weight* pWeight               = _vOutgoingLargerWeight[i];     
                Weight* pSrcWeight            = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
                
                float* pA                     = pOutputLayer->GetDeltaBuffer();
                float* pB                     = getGpu()._pNetwork->GetP2PSendBuffer();
                float* pC                     = pSrcWeight->_pbWeightGradient->_pDevData;
                int m                           = pOutputLayer->_localStride;
                int n                           = _stride;
                int k                           = batch;
                int lda                         = pOutputLayer->_localStride;
                int ldb                         = _stride;
                int ldc                         = pOutputLayer->_localStride;

                float sgemm_alpha             = -(float)1.0 / (pSrcWeight->_sharingCount * (float)batch);
                float sgemm_beta              = (pSrcWeight->_updateCount == 0) ? (float)0.0 : (float)1.0;               
                
                cublasStatus_t cstatus          = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_T,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("Layer::BackPropagate: SGEMM failure, aborting.\n");
                    getGpu().Shutdown();
                    exit(-1);
                }

                pSrcWeight->_updateCount++;
            }  

            float sgemm_beta                  = (float)0.0;              
            for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
            {
                Layer* pOutputLayer           = _vOutgoingLargerLayer[i];
                const float sgemm_alpha       = (float)1.0;
                float* pA                     = _vOutgoingLargerWeight[i]->_bShared ? 
                                                  _vOutgoingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                  _vOutgoingLargerWeight[i]->_pbWeight->_pDevData;
                float* pB                     = pOutputLayer->GetDeltaBuffer();
                float* pC                     = getGpu()._pNetwork->GetP2PSendBuffer();
                int m                           = _stride;
                int n                           = batch;
                int k                           = pOutputLayer->_localStride;
                int lda                         = pOutputLayer->_localStride;
                int ldb                         = pOutputLayer->_localStride;
                int ldc                         = _stride;

                cublasStatus_t cstatus          =
                                                cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);  

                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("Layer::BackPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                    getGpu().Shutdown();
                    exit(-1);
                }
#if 0
                float* pD = pOutputLayer->_vDelta.data();
                float* pW = _vOutgoingWeight[i]->_vWeight.data();
                
                pOutputLayer->_pbDelta->Download(pD);
                _vOutgoingLargerWeight[i]->_pbWeight->Download(pW);
                pW += pOutputLayer->_localStride;
                float sum = 0.0f;
                for (int j = 0; j < pOutputLayer->_localStride; j++)
                {
                    sum += (*pD) * (*pW);
                    pD++;
                    pW++;
                }
                MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                if (getGpu()._id == 0)
                    printf("ZAG %16.12f\n", sum);
                MPI_Barrier(MPI_COMM_WORLD);  
#endif
                        
                sgemm_beta                      = (float)1.0;
            }


            Reduce(batch, _stride, GetIncomingDeltaBuffer(), _localStride, _deltaUpdateCount);
            _deltaUpdateCount++;
        }


        
        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                float p       = (_sparsenessPenalty_p > (float)0.0)   ? _sparsenessPenalty_p     : getGpu()._pNetwork->_sparsenessPenalty_p;
                float beta    = (_sparsenessPenalty_beta > (float)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);                
            }   

            float scale                           = (float)1.0 / ((float)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            
            if (_deltaNorm > (float)0.0)
            {            
                float* pMagnitude             = getGpu()._pNetwork->GetScratchBuffer(batch);
                kCalculateDeltaMagnitudes(batch, _localStride, GetIncomingDeltaBuffer(), pMagnitude);
                getGpu()._pNetwork->P2P_Allreduce(pMagnitude, batch);
                kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer(), pMagnitude);
            }
            
            if (_bBatchNormalization)
            {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateFullyConnected: unable to create _tensorDescriptorBN");        
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
                CUDNNERROR(cudnnStatus, "Layer::BackPropagateFullyConnected: unable to create _scaleBiasMeanVarDescBN");        
                float alpha = 1;
                float beta = 0;
                cudnnStatus = cudnnBatchNormalizationBackward(
                        getGpu()._cuDNNHandle,
                        CUDNN_BATCHNORM_PER_ACTIVATION,
                        &alpha,
                        &beta,
                        &alpha,
                        &beta,
                        _tensorDescriptorBN,
                        GetIncomingUnitBuffer(),
                        _tensorDescriptorBN,
                        GetIncomingDeltaBuffer(),
                        _tensorDescriptorBN,
                        GetDeltaBuffer(),
                        _scaleBiasMeanVarDescBN,
                        _pbScaleBN->_pDevData,
                        _pbScaleGradientBN->_pDevData,
                        _pbBiasGradientBN->_pDevData,
                        CUDNN_BN_MIN_EPSILON,
                        _pbSaveMeanBN->_pDevData,
                        _pbSaveInvVarianceBN->_pDevData);
                CUDNNERROR(cudnnStatus, "Layer:BackPropagateFullyConnected cudnnBatchNormalizationBackward Failed");
            }
        }

        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), batch * _localStride * sizeof(float), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }          

        if (_vIncomingLargerLayer.size() > 0)
        {
            Gather(batch, _stride, GetDeltaBuffer(), _localStride);   
                 
            for (int i = 0; i < _vIncomingLargerLayer.size(); i++)
            {
                Layer* pInputLayer            = _vIncomingLargerLayer[i];
                Weight* pWeight               = _vIncomingLargerWeight[i];     
                Weight* pSrcWeight            = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
                
                float* pA                     = getGpu()._pNetwork->GetP2PSendBuffer();
                float* pC                     = pSrcWeight->_pbWeightGradient->_pDevData;
                int m                           = _stride;
                int n                           = pInputLayer->_localStride;
                int k                           = batch;
                int lda                         = _stride;
                int ldb                         = pInputLayer->_localStride;
                int ldc                         = _stride;

                float sgemm_alpha             = -(float)1.0 / (pSrcWeight->_sharingCount * (float)batch);
                float sgemm_beta              = (pSrcWeight->_updateCount == 0) ? (float)0.0 : (float)1.0;
                
                if ((pInputLayer->_kind == Layer::Kind::Input) && pInputLayer->_bFastSparse)
                {
                    pInputLayer->_pDataSet->CalculateSparseTransposedWeightGradient(sgemm_alpha, sgemm_beta, n, m, pA, pC);
                }
                else
                { 
                    float* pB                 = pInputLayer->GetUnitBuffer();          
                    cublasStatus_t cstatus      = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_T,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("Layer::BackPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }

                
                pSrcWeight->_updateCount++;
               
                if (pInputLayer->_kind != Input)
                {
                    sgemm_alpha                 = (float)1.0;
                    sgemm_beta                  = (pInputLayer->_deltaUpdateCount == 0) ? (float)0.0 : (float)1.0;
                    pA                          = pSrcWeight->_pbWeight->_pDevData;
                    float* pB                 = getGpu()._pNetwork->GetP2PSendBuffer();
                    pC                          = pInputLayer->GetIncomingDeltaBuffer();
                    m                           = pInputLayer->_localStride;
                    n                           = batch;
                    k                           = _stride;                           
                    lda                         = _stride;
                    ldb                         = _stride;
                    ldc                         = pInputLayer->_localStride;
                    cublasStatus_t cstatus      = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("Layer::BackPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }

                    pInputLayer->_deltaUpdateCount++;
                }
            }
        }
    }
    
    
#if 0
    Weight* pWeight                       = _vIncomingWeight[0];
    vector<float> vLocalWeightGradient(pWeight->_size);
    pWeight->_pbWeightGradient->Download(vLocalWeightGradient.data());
    for (int i = 0; i < getGpu()._numprocs; i++)
    {
        if (i == getGpu()._id)
        {
            uint32_t count = 0;
            while (count < pWeight->_size)
            {
                for (int j = 0; j < pWeight->_outputLayer._stride; j++)
                {
                    printf("%8.4f ", vLocalWeightGradient[count++]);
                }
                printf("\n");
           }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }   
    if (getGpu()._id == 0)
        cout << endl;
#endif   
}

void Layer::UpdateWeights(TrainingMode trainingMode, uint32_t batch, float alpha, float lambda, float lambda1, float mu, float mu1, float t)
{
    if (_bBatchNormalization)
    {
        switch (trainingMode)
        {
            case SGD:
                kSGDUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kSGDUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                
            case Momentum:
                kMomentumUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kMomentumUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                        
            case AdaGrad:
                kAdaGradUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kAdaGradUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                        
            case Nesterov:
                kNesterovUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kNesterovUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;
                        
            case RMSProp:
                kRMSPropUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
                kRMSPropUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
                break;

            case AdaDelta:
                kAdaDeltaUpdateWeights(lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleGradientVelocityBN->_pDevData, _pbScaleBN->_pDevData);
                kAdaDeltaUpdateWeights(lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasGradientVelocityBN->_pDevData, _pbBiasBN->_pDevData);
                break;     

            case Adam:
                kAdamUpdateWeights(-alpha, lambda, lambda1, mu, mu1, t, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleGradientVelocityBN->_pDevData, _pbScaleBN->_pDevData);
                kAdamUpdateWeights(-alpha, lambda, lambda1, mu, mu1, t, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasGradientVelocityBN->_pDevData, _pbBiasBN->_pDevData);
                break;   
        }
    }
}

void Layer::Reduce(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride, uint32_t updateCount)
{

    if (getGpu()._numprocs > 1)
    {
        uint32_t stages                             = getGpu()._numprocs - 1;
        uint64_t pos                                = (getGpu()._id + 1) % getGpu()._numprocs; 
        uint32_t minX                               = (stride * pos) / getGpu()._numprocs;
        uint32_t maxX                               = (stride * (pos + 1)) / getGpu()._numprocs;
        uint32_t span                               = maxX - minX;
        float* pSendBuffer                        = getGpu()._pNetwork->GetP2PSendBuffer();

        if (getGpu()._bP2P)
        {
            float* pReceiveBuffer                 = getGpu()._pNetwork->GetP2PReceiveBuffer();
            float* pPeerBuffer                    = getGpu()._pNetwork->GetPeerBuffer();

            for (uint32_t i = 0; i < stages; i++)
            {
                kCopy2D(pPeerBuffer + minX, stride, pSendBuffer + minX, stride, span, batch);
                cudaDeviceSynchronize();       
                MPI_Barrier(MPI_COMM_WORLD);
        
                pos                                 = (pos + 1) % getGpu()._numprocs;
                minX                                = (stride * pos) / getGpu()._numprocs;
                maxX                                = (stride * (pos + 1)) / getGpu()._numprocs;
                span                                = maxX - minX;
                kAddBuffers2D(pSendBuffer + minX, stride, pReceiveBuffer + minX, stride, span, batch);
            }
        }
        else
        {
            float* pCPUBuffer                     = getGpu()._pNetwork->GetP2PCPUBuffer();
            cudaError_t status                      = cudaMemcpy(pCPUBuffer, pSendBuffer, batch * stride * sizeof(float), cudaMemcpyDefault);
            RTERROR(status, "Layer::Reduce1: cudaMemcpy download failed " + getGpu()._id );
            MPI_Allreduce(MPI_IN_PLACE, pCPUBuffer, batch * stride, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            status = cudaMemcpy(pSendBuffer, pCPUBuffer, batch * stride * sizeof(float), cudaMemcpyDefault);
            RTERROR(status, "Layer::Reduce: cudaMemcpy upload failed" + getGpu()._id );
            minX                                    = (stride * getGpu()._id) / getGpu()._numprocs;
            maxX                                    = (stride * (getGpu()._id + 1)) / getGpu()._numprocs;
            span                                    = maxX - minX;            
        }

        if (updateCount > 0) 
        {
            kAddBuffers2D(pBuffer, localStride, pSendBuffer + minX, stride, span, batch);
        }
        else 
        {
            kCopy2D(pBuffer, localStride, pSendBuffer + minX, stride, span, batch);
        }
    }
}

void Layer::Gather(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride)
{
    if (getGpu()._numprocs > 1)
    {
        uint32_t stages                                 = getGpu()._numprocs - 1;
        uint64_t pos                                    = getGpu()._id;
        float* pSendBuffer                            = getGpu()._pNetwork->GetP2PSendBuffer();
        uint32_t minX                                   = (stride * pos) / getGpu()._numprocs;
        uint32_t maxX                                   = (stride * (pos + 1)) / getGpu()._numprocs;
        uint32_t span                                   = maxX - minX;

        if (getGpu()._bP2P)
        {
            float* pPeerBuffer                        = getGpu()._pNetwork->GetPeerBackBuffer();

            cudaDeviceSynchronize();  
            MPI_Barrier(MPI_COMM_WORLD);

            kCopy2D(pSendBuffer + minX, stride, pBuffer, localStride, span, batch); 

            for (uint32_t i = 0; i < stages; i++)
            {                    
                kCopy2D(pPeerBuffer + minX, stride, pSendBuffer + minX, stride, span, batch);
                cudaDeviceSynchronize();  
                MPI_Barrier(MPI_COMM_WORLD);
                pos                                     = (pos + 1) % getGpu()._numprocs;
                minX                                    = (stride * pos) / getGpu()._numprocs;
                maxX                                    = (stride * (pos + 1)) / getGpu()._numprocs;
                span                                    = maxX - minX;
            }
        }
        else
        {
            float* pCPUBuffer                        = getGpu()._pNetwork->GetP2PCPUBuffer();

            cudaError_t status                         = cudaMemcpy2D(pCPUBuffer + minX, stride * sizeof(float), pBuffer, localStride * sizeof(float), localStride * sizeof(float), batch, cudaMemcpyDefault);
            RTERROR(status, "Layer::Gather: cudaMemcpy download failed");


            for (uint32_t i = 0; i < getGpu()._numprocs; i++)
            {
                uint32_t minX                          = (stride * i) / getGpu()._numprocs;
                uint32_t maxX                          = (stride * (i + 1)) / getGpu()._numprocs;
                uint32_t span                          = maxX - minX;
                MPI_Datatype spanType;
                MPI_Type_vector(batch, span, stride, MPI_FLOAT, &spanType);
                MPI_Type_commit(&spanType);
                MPI_Bcast(pCPUBuffer + minX, 1, spanType, i, MPI_COMM_WORLD);
                MPI_Type_free(&spanType);
            }
 
            status                                     = cudaMemcpy(pSendBuffer, pCPUBuffer, batch * stride * sizeof(float), cudaMemcpyDefault);
            RTERROR(status, "Layer::Gather: cudaMemcpy upload failed");
        }
    }
}

/// <summary>
/// Dump the layer's data to a file.
/// </summary>
/// <param name="fname">The name of the output file.</param>
/// <param name="pBuffer">A pointer to the float buffer containing the data.</param>
void Layer::Dump(std::string fname, float* pBuffer)
{
    // Create a vector "vData" to hold float data, with a size of "_batch * _stride".
    std::vector<float> vData(_batch * _stride);

    // Check if the number of GPU processors is equal to 1.
    if (getGpu()._numprocs == 1)
    {
        // If there is only one processor, copy data from "pBuffer" to "vData" using CUDA memory copy.
        cudaMemcpy(vData.data(), pBuffer, _batch * _stride * sizeof(float), cudaMemcpyDefault);
    }
    else
    {
        // If there is more than one processor, perform data communication.
        if (getGpu()._id == 0)
        {
            // For the master GPU (ID 0), prepare to receive data from other GPUs.
            float* pData = vData.data();
            cudaMemcpy2D(pData, _stride * sizeof(float), pBuffer, _localStride * sizeof(float), _localStride * sizeof(float), _batch, cudaMemcpyDefault);
            pData += _localStride;

            // Iterate through all other GPUs to receive and reconstruct data.
            for (uint32_t i = 1; i < getGpu()._numprocs; i++)
            {
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                std::vector<float> vTemp(size);
                MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                uint64_t lstride = size / _batch;
                float* pSrc = vTemp.data();
                float* pDst = pData;

                // Reconstruct data received from other GPUs.
                for (uint32_t j = 0; j < _batch; j++)
                {
                    memcpy(pDst, pSrc, lstride * sizeof(float));
                    pSrc += lstride;
                    pDst += _stride;
                }
                pData += lstride;
            }
        }
        else
        {
            // For non-master GPUs, prepare data to be sent to the master GPU.
            uint64_t size = _batch * _localStride;
            std::vector<float> vLocalData(size);
            cudaMemcpy(vLocalData.data(), pBuffer, size * sizeof(float), cudaMemcpyDefault);
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(vLocalData.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    // If the current GPU ID is 0 (master GPU),
    if (getGpu()._id == 0)
    {
        // Open an output file with the specified file name.
        std::ofstream outputFile(fname);
        float* pData = vData.data();

        // Iterate over the data and write it to the output file with formatting.
        for (int i = 0; i < _batch; i++)
        {
            outputFile << std::setw(4) << i << " ";
            for (int j = 0; j < _stride; j++)
            {
                outputFile << std::fixed << std::setprecision(9) << *pData << " ";
                pData++;
            }
            outputFile << "\n";
        }

        // Close the output file.
        outputFile.close();
    }
}

/// <summary>
/// Defines a static constant map that associates Layer kinds with their string representations.
/// </summary>
const std::map<Layer::Kind, std::string> Layer::_sKindMap = {
    {Layer::Kind::Input, "Input"},
    {Layer::Kind::Hidden, "Hidden"},
    {Layer::Kind::Output, "Output"},
    {Layer::Kind::Target, "Target"}
};

/// <summary>
/// Defines a static constant map that associates Layer types with their string representations.
/// </summary>
const std::map<Layer::Type, std::string> Layer::_sTypeMap = {
    {Layer::Type::FullyConnected, "FullyConnected"},
    {Layer::Type::Convolutional, "Convolutional"},
    {Layer::Type::Pooling, "Pooling"}
};

/// <summary>
/// Defines a static constant map that associates Layer attributes with their string representations.
/// </summary>
const std::map<Layer::Attributes, std::string> Layer::_sAttributesMap = {
    {Layer::Attributes::None, "None"},
    {Layer::Attributes::Sparse, "Sparse"},
    {Layer::Attributes::Denoising, "Denoising"},
    {Layer::Attributes::BatchNormalization, "BatchNormalization"}
};

/// <summary>
/// Defines a static constant map that associates Layer parallelization options with their string representations.
/// </summary>
const std::map<Layer::Parallelization, std::string> Layer::_sParallelizationMap = {
    {Layer::Parallelization::Data, "Data"},
    {Layer::Parallelization::Model, "Model"},
    {Layer::Parallelization::Serial, "Serial"}
};

/// <summary>
/// Overload the << operator for Layer::Kind to allow printing a Layer kind to an output stream.
/// </summary>
/// <param name="out">The output stream.</param>
/// <param name="k">The Layer::Kind to be printed.</param>
/// <returns>The output stream.</returns>
std::ostream& operator<< (std::ostream& out, Layer::Kind k) {
    // Use the _sKindMap to get the string representation of the Layer kind and output it to the stream.
    out << Layer::_sKindMap.at(k);
    return out;
}

/// <summary>
/// Overload the << operator for Layer::Type to allow printing a Layer type to an output stream.
/// </summary>
/// <param name="out">The output stream.</param>
/// <param name="t">The Layer::Type to be printed.</param>
/// <returns>The output stream.</returns>
std::ostream& operator<< (std::ostream& out, Layer::Type t) {
    // Use the _sTypeMap to get the string representation of the Layer type and output it to the stream.
    out << Layer::_sTypeMap.at(t);
    return out;
}

/// <summary>
/// Overload the << operator for Layer::Parallelization to allow printing a Layer parallelization option to an output stream.
/// </summary>
/// <param name="out">The output stream.</param>
/// <param name="p">The Layer::Parallelization option to be printed.</param>
/// <returns>The output stream.</returns>
std::ostream& operator<< (std::ostream& out, Layer::Parallelization p) {
    // Use the _sParallelizationMap to get the string representation of the parallelization option and output it to the stream.
    out << Layer::_sParallelizationMap.at(p);
    return out;
}

/// <summary>
/// Constructor for the LayerDescriptor class.
/// </summary>
LayerDescriptor::LayerDescriptor() :
    _kind(Layer::Kind::Hidden),                 // The kind of layer.
    _type(Layer::Type::FullyConnected),         // The type of layer.
    _poolingFunction(None),                     // The pooling function used (if any).
    _Nx(1),                                     // The number of units in the x dimension.
    _Ny(1),                                     // The number of units in the y dimension.
    _Nz(1),                                     // The number of units in the z dimension.
    _Nw(1),                                     // The number of units in the w dimension.
    _dimensions(1),                             // The number of dimensions for the layer.
    _bDimensionsProvided(true),                 // Indicates if dimensions are provided.
    _weightInit(Xavier),                        // The weight initialization method.
    _weightInitScale((float)1.0),               // The scale for weight initialization.
    _biasInit((float)0.0),                      // The bias initialization value.
    _kernelX(1),                                // The size of the kernel in the x dimension.
    _kernelY(1),                                // The size of the kernel in the y dimension.
    _kernelZ(1),                                // The size of the kernel in the z dimension.
    _kernelStrideX(1),                          // The stride in the x dimension for the kernel.
    _kernelStrideY(1),                          // The stride in the y dimension for the kernel.
    _kernelStrideZ(1),                          // The stride in the z dimension for the kernel.
    _kernelPaddingX(0),                         // The padding in the x dimension for the kernel.
    _kernelPaddingY(0),                         // The padding in the y dimension for the kernel.
    _kernelPaddingZ(0),                         // The padding in the z dimension for the kernel.
    _kernelDimensions(1),                       // The number of dimensions for the kernel.
    _weightNorm((float)0.0),                    // The weight normalization factor.
    _deltaNorm((float)0.0),                     // The delta normalization factor.
    _pDropout((float)0.0),                      // The dropout probability.
    _activation(Activation::Sigmoid),           // The activation function.
    _sparsenessPenalty_p((float)0.0),           // The sparseness penalty factor (p).
    _sparsenessPenalty_beta((float)0.0),        // The sparseness penalty factor (beta).
    _RELUSlope(NAN),                            // The slope for the ReLU activation function.
    _ELUAlpha(NAN),                             // The alpha parameter for the ELU activation function.
    _SELULambda(NAN),                           // The lambda parameter for the SELU activation function.
    _attributes(Layer::Attributes::None)        // Additional attributes for the layer.
{
}

/// <summary>
/// Load a layer descriptor from a NetCDF file.
/// </summary>
/// <param name="fname">The name of the NetCDF file.</param>
/// <param name="nc">A reference to the NetCDF file.</param>
/// <param name="index">The index of the layer.</param>
/// <param name="ld">A reference to the LayerDescriptor object to populate.</param>
/// <returns>True if the layer descriptor is successfully loaded, false otherwise.</returns>
bool LoadLayerDescriptorNetCDF(const std::string& fname, netCDF::NcFile& nc, uint32_t index, LayerDescriptor& ld) {
    // Check if this function is running on GPU 0, return early if not
    if (getGpu()._id != 0)
        return true;

    // Create a string to represent the attribute name for this layer
    std::string lstring = "layer" + std::to_string(index) + "_";

    // Lambda function to check and retrieve an attribute value from the NetCDF file
    auto checkAttribute = [&nc, &fname, &lstring](const std::string& attributeName, auto& value) {
        try {
            // Attempt to retrieve the attribute from the NetCDF file
            auto attribute = nc.getAtt(lstring + attributeName);
            if (!attribute.isNull()) {
                attribute.getValues(&value);
            }
            else {
                // Handle the case when the attribute is missing
                std::cerr << "NcException Layer::Layer: No " << attributeName << " supplied in NetCDF input file " << fname << " " << __FILE__ << " " << __LINE__ << std::endl;
            }
        }
        catch (const netCDF::exceptions::NcException& e) {
            // Handle NetCDF exceptions
            std::cerr << "NcException Layer::Layer: " << e.what() << std::endl;
        }
        };

    // Check and retrieve various attributes for the layer descriptor
    checkAttribute("name", ld._name);
    checkAttribute("kind", ld._kind);
    checkAttribute("type", ld._type);
    checkAttribute("weightInit", ld._weightInit);
    checkAttribute("weightInitScale", ld._weightInitScale);
    checkAttribute("biasInit", ld._biasInit);
    checkAttribute("weightNorm", ld._weightNorm);
    checkAttribute("deltaNorm", ld._deltaNorm);
    checkAttribute("pDropout", ld._pDropout);
    checkAttribute("activation", ld._activation);
    checkAttribute("RELUSlope", ld._RELUSlope);
    checkAttribute("ELUAlpha", ld._ELUAlpha);
    checkAttribute("SELULambda", ld._SELULambda);
    checkAttribute("sparsenessPenalty_p", ld._sparsenessPenalty_p);
    checkAttribute("sparsenessPenalty_beta", ld._sparsenessPenalty_beta);
    checkAttribute("Nx", ld._Nx);
    checkAttribute("Ny", ld._Ny);
    checkAttribute("Nz", ld._Nz);
    checkAttribute("Nw", ld._Nw);
    checkAttribute("dimensions", ld._dimensions);
    checkAttribute("kernelX", ld._kernelX);
    checkAttribute("kernelY", ld._kernelY);
    checkAttribute("kernelZ", ld._kernelZ);
    checkAttribute("kernelStrideX", ld._kernelStrideX);
    checkAttribute("kernelStrideY", ld._kernelStrideY);
    checkAttribute("kernelStrideZ", ld._kernelStrideZ);
    checkAttribute("kernelPaddingX", ld._kernelPaddingX);
    checkAttribute("kernelPaddingY", ld._kernelPaddingY);
    checkAttribute("kernelPaddingZ", ld._kernelPaddingZ);
    checkAttribute("kernelDimensions", ld._kernelDimensions);

    // Lambda function to check and retrieve attributes representing lists of strings
    auto checkSourcesOrSkips = [&nc, &fname, &lstring](const std::string& attributeName, std::vector<std::string>& vec) {
        uint32_t count = 0;
        try {
            // Attempt to retrieve the count attribute
            auto att = nc.getAtt(lstring + attributeName);
            if (!att.isNull()) {
                att.getValues(&count);
                // Loop through and retrieve individual source attributes
                for (uint32_t i = 0; i < count; i++) {
                    auto nstring = std::to_string(i);
                    auto sourceAtt = nc.getAtt(lstring + attributeName + nstring);
                    if (!sourceAtt.isNull()) {
                        std::string source;
                        sourceAtt.getValues(source);
                        vec.push_back(source);
                    }
                    else {
                        // Handle the case when an individual source attribute is missing
                        std::cerr << "NcException Layer::Layer: No " << attributeName << " attributes supplied in NetCDF input file " << fname << " " << __FILE__ << " " << __LINE__ << std::endl;
                    }
                }
            }
            else {
                // Handle the case when the count attribute is missing
                std::cerr << "NcException Layer::Layer: No " << attributeName << " supplied in NetCDF input file " << fname << " " << __FILE__ << " " << __LINE__ << std::endl;
            }
        }
        catch (const netCDF::exceptions::NcException& e) {
            // Handle NetCDF exceptions
            std::cerr << "NcException Layer::Layer: " << e.what() << std::endl;
        }
        };

    // Check and retrieve lists of source and skip attributes
    checkSourcesOrSkips("sources", ld._vSource);
    checkSourcesOrSkips("skips", ld._vSkip);

    // Return true to indicate success
    return true;
}

/// <summary>
/// Output stream operator for LayerDescriptor class.
/// </summary>
/// <param name="out">The output stream.</param>
/// <param name="d">The LayerDescriptor object to output.</param>
/// <returns>The output stream.</returns>
std::ostream& operator<<(std::ostream& out, const LayerDescriptor& d)
{
    // Output the name, kind, and type of the layer with proper formatting.
    out << std::left << std::setw(20) << "Name:" << d._name << '\n'
        << std::setw(20) << "Kind:" << d._kind << '\n'
        << std::setw(20) << "Type:" << d._type << '\n';

    // If the layer is not of type Pooling, output its pooling function.
    if (d._type != Layer::Type::Pooling)
        out << std::setw(20) << "Pooling Function:" << d._poolingFunction << '\n';

    // Output the dimensions of the layer.
    out << std::setw(20) << "Nx:" << d._Nx << '\n'
        << std::setw(20) << "Ny:" << d._Ny << '\n'
        << std::setw(20) << "Nz:" << d._Nz << '\n'
        << std::setw(20) << "Nw:" << d._Nw << '\n';

    // If the layer is not of type FullyConnected, output its kernel parameters.
    if (d._type != Layer::Type::FullyConnected)
    {
        out << std::setw(20) << "kernelX:" << d._kernelX << '\n'
            << std::setw(20) << "kernelY:" << d._kernelY << '\n'
            << std::setw(20) << "kernelZ:" << d._kernelZ << '\n'
            << std::setw(20) << "kernelStrideX:" << d._kernelStrideX << '\n'
            << std::setw(20) << "kernelStrideY:" << d._kernelStrideY << '\n'
            << std::setw(20) << "kernelStrideZ:" << d._kernelStrideZ << '\n'
            << std::setw(20) << "kernelPaddingX:" << d._kernelPaddingX << '\n'
            << std::setw(20) << "kernelPaddingY:" << d._kernelPaddingY << '\n'
            << std::setw(20) << "kernelPaddingZ:" << d._kernelPaddingZ << '\n'
            << std::setw(20) << "kernelDimensions:" << d._kernelDimensions << '\n';
    }

    // If the layer is not of type Pooling, output additional properties.
    if (d._type != Layer::Type::Pooling)
    {
        out << std::setw(20) << "pDropout:" << d._pDropout << '\n'
            << std::setw(20) << "weightInit:" << d._weightInit << '\n'
            << std::setw(20) << "weightInitScale:" << d._weightInitScale << '\n'
            << std::setw(20) << "biasInit:" << d._biasInit << '\n'
            << std::setw(20) << "weightNorm:" << d._weightNorm << '\n'
            << std::setw(20) << "deltaNorm:" << d._deltaNorm << '\n'
            << std::setw(20) << "activation:" << d._activation << '\n'
            << std::setw(20) << "RELUSlope:" << d._RELUSlope << '\n'
            << std::setw(20) << "ELUAlpha:" << d._ELUAlpha << '\n'
            << std::setw(20) << "SELULambda:" << d._SELULambda << '\n'
            << std::setw(20) << "Sparse:" << std::boolalpha << ((d._attributes & Layer::Attributes::Sparse) != 0) << '\n'
            << std::setw(20) << "batchNormalization:" << std::boolalpha << ((d._attributes & Layer::Attributes::BatchNormalization) != 0) << '\n';

        // If the layer is of type FullyConnected, output sparseness penalty parameters if they are greater than 0.
        if (d._type == Layer::Type::FullyConnected)
        {
            if (d._sparsenessPenalty_p > 0.0)
                out << std::setw(20) << "sparsenessPenalty_p:" << d._sparsenessPenalty_p << '\n';
            if (d._sparsenessPenalty_beta > 0.0)
                out << std::setw(20) << "sparsenessPenalty_beta:" << d._sparsenessPenalty_beta << '\n';
        }

        // If the layer's kind is not Hidden, output the DataSet property.
        if (d._kind != Layer::Kind::Hidden)
            out << std::setw(20) << "DataSet:" << d._dataSet << '\n';
    }

    // Output sources connected to the layer.
    for (size_t i = 0; i < d._vSource.size(); i++)
    {
        out << "source " << std::setw(3) << i << ":" << d._vSource[i] << '\n';
    }

    // Output skip connections connected to the layer.
    for (size_t i = 0; i < d._vSkip.size(); i++)
    {
        out << "skip " << std::setw(3) << i << ":" << d._vSkip[i] << '\n';
    }

    // Return the output stream.
    return out;
}

uint32_t MPI_Bcast_LayerDescriptor(LayerDescriptor& d)
{
    MPI_Bcast_string(d._name);
    MPI_Bcast(&d._kind, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._type, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._poolingFunction, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&d._Nx, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Ny, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Nz, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Nw, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._dimensions, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bDimensionsProvided, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._pDropout, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightInit, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightInitScale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._biasInit, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightNorm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._deltaNorm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._activation, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_beta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);    
    MPI_Bcast(&d._attributes, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._RELUSlope, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._ELUAlpha, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._SELULambda, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast_string(d._dataSet);
    size_t size                         = d._vSource.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSource.resize(size);
    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSource[i]);
    size                                = d._vSkip.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSkip.resize(size);
    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSkip[i]);        
    return 0;
}

bool Layer::WriteNetCDF(NcFile& nc, uint32_t index)
{
    bool bResult                        = true;
    if (getGpu()._id == 0)
    {
        std::string lstring                  = "layer" + std::to_string(index) + "_";
        nc.putAtt(lstring + "name", _name);
        nc.putAtt(lstring + "kind", ncUint, _kind);
        nc.putAtt(lstring + "type", ncUint, _type);
        nc.putAtt(lstring + "poolingfunction", ncUint, _poolingFunction);
        nc.putAtt(lstring + "dataSet", _dataSet);
        nc.putAtt(lstring + "Nx", ncUint, _Nx);
        nc.putAtt(lstring + "Ny", ncUint, _Ny);
        nc.putAtt(lstring + "Nz", ncUint, _Nz);
        nc.putAtt(lstring + "Nw", ncUint, _Nw);
        nc.putAtt(lstring + "dimensions", ncUint, _dimensions);
        nc.putAtt(lstring + "kernelX", ncUint, _kernelX);
        nc.putAtt(lstring + "kernelY", ncUint, _kernelY);
        nc.putAtt(lstring + "kernelZ", ncUint, _kernelZ);
        nc.putAtt(lstring + "kernelDimensions", ncUint, _kernelDimensions);
        nc.putAtt(lstring + "kernelStrideX", ncUint, _kernelStrideX);
        nc.putAtt(lstring + "kernelStrideY", ncUint, _kernelStrideY);
        nc.putAtt(lstring + "kernelStrideZ", ncUint, _kernelStrideZ);
        nc.putAtt(lstring + "kernelPaddingX", ncUint, _kernelPaddingX);
        nc.putAtt(lstring + "kernelPaddingY", ncUint, _kernelPaddingY);
        nc.putAtt(lstring + "kernelPaddingZ", ncUint, _kernelPaddingZ);
        nc.putAtt(lstring + "pDropout", ncFloat, _pDropout);
        nc.putAtt(lstring + "weightInit", ncUint, _weightInit);
        nc.putAtt(lstring + "weightInitScale", ncFloat, _weightInitScale);
        nc.putAtt(lstring + "biasInit", ncFloat, _biasInit);
        nc.putAtt(lstring + "weightNorm", ncFloat, _weightNorm);
        nc.putAtt(lstring + "deltaNorm", ncFloat, _deltaNorm);
        nc.putAtt(lstring + "activation", ncUint, _activation);
        nc.putAtt(lstring + "sparsenessPenalty_p", ncFloat, _sparsenessPenalty_p);
        nc.putAtt(lstring + "sparsenessPenalty_beta", ncFloat, _sparsenessPenalty_beta);
        nc.putAtt(lstring + "RELUSlope", ncFloat, _RELUSlope);
        nc.putAtt(lstring + "ELUAlpha", ncFloat, _ELUAlpha);
        nc.putAtt(lstring + "SELULambda", ncFloat, _SELULambda);
                
        uint32_t attributes             = 0;
        if (_bSparse)
            attributes                 |= Layer::Attributes::Sparse;
        if (_bDenoising)
            attributes                 |= Layer::Attributes::Denoising;
        if (_bBatchNormalization)
            attributes                 |= Layer::Attributes::BatchNormalization;
        nc.putAtt(lstring + "attributes", ncUint, attributes);
        nc.putAtt(lstring + "sources", ncUint, (uint32_t)_vSource.size());
        for (size_t i = 0; i < _vSource.size(); i++)
        {
            std::string nstring             = std::to_string(i);
            nc.putAtt(lstring + "source" + nstring, _vSource[i]);
        }
        nc.putAtt(lstring + "skips", ncUint, (uint32_t)_vSkip.size());        
        for (size_t i = 0; i < _vSkip.size(); i++)
        {
            std::string nstring             = std::to_string(i);
            nc.putAtt(lstring + "skip" + nstring, _vSkip[i]);
        }

        if (_bBatchNormalization)
        {
            std::vector<float>  bndata(_strideBN);
            size_t bytes = _strideBN * sizeof(float);
            NcDim bnDim   = nc.addDim(lstring + "bnDim", _strideBN);

            cudaMemcpy(bndata.data(), _pbScaleBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar scaleVar  = nc.addVar(lstring + "scaleBN", "float", bnDim.getName());
            scaleVar.putVar(bndata.data());

            cudaMemcpy(bndata.data(), _pbBiasBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar biasVar  = nc.addVar(lstring + "biasBN", "float", bnDim.getName());
            biasVar.putVar(bndata.data());

            cudaMemcpy(bndata.data(), _pbRunningMeanBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar runningMeanVar  = nc.addVar(lstring + "runningMeanBN", "float", bnDim.getName());
            runningMeanVar.putVar(bndata.data());

            cudaMemcpy(bndata.data(), _pbRunningVarianceBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
            NcVar runningVarianceVar  = nc.addVar(lstring + "runningVarianceBN", "float", bnDim.getName());
            runningVarianceVar.putVar(bndata.data());
        }
    }
    else
        bResult                     = false;

    return bResult;
}
