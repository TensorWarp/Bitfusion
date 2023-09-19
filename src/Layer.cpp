#include "GpuTypes.h"
#include "Types.h"
#include "Kernels.cuh"
#include <format>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <format>

void DumpP(const char* name, float* p, int stride) {
    // Allocate memory for the data on the CPU
    std::vector<float> data(stride);

    // Copy data from GPU to CPU memory
    cudaMemcpy(data.data(), p, stride * sizeof(float), cudaMemcpyDefault);

    // Use std::format for improved output
    std::cout << std::format("{}:  ", name);

    // Use range-based for loop to iterate through the data
    for (auto i : data) {
        std::cout << std::format("{:.2f}, ", i);
    }

    std::cout << '\n';
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
    _bBatchNormalization(d._attributes& Layer::Attributes::BatchNormal) {           // Batch normalization flag
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
        std::cerr << "Error in Layer::InitializeDescriptors: " << e.what() << '\n';
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

/// <summary>
/// Destructor for the Layer class.
/// </summary>
Layer::~Layer()
{
     // Deallocate any resources held by this layer.
    Deallocate();

    // Destroy tensor descriptors if the layer type is Pooling or Convolutional.
    if ((_type == Layer::Type::Pooling) || (_type == Layer::Type::Convolutional))
    {
        cudnnStatus_t cudnnStatus = cudnnDestroyTensorDescriptor(_tensorDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _tensorDescriptor");
        cudnnStatus = cudnnDestroyTensorDescriptor(_oddBatchTensorDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _oddBatchTensorDescriptor");
    }

    // Destroy batch normalization-related descriptors and reset associated smart pointers if batch normalization is enabled.
    if (_bBatchNormalization)
    {
        cudnnStatus_t cudnnStatus = cudnnDestroyTensorDescriptor(_scaleBiasMeanVarDescBN);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _scaleBiasMeanVarDescBN");
        cudnnStatus = cudnnDestroyTensorDescriptor(_tensorDescriptorBN);
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

    // Destroy pooling descriptor and, if applicable, LRN descriptor if the layer type is Pooling.
    if (_type == Layer::Type::Pooling)
    {
        cudnnStatus_t cudnnStatus = cudnnDestroyPoolingDescriptor(_poolingDescriptor);
        CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to destroy _poolingDescriptor");

        if (_poolingFunction == PoolingFunction::LRN)
        {
            cudnnStatus = cudnnDestroyLRNDescriptor(_LRNDescriptor);
            CUDNNERROR(cudnnStatus, "Layer::~Layer: unable to delete _LRNDescriptor");
        }
    }
}

void Layer::Deallocate()
{
    if (getGpu()._id == 0)
        std::cout << std::format("Layer::Deallocate: Deallocating all data for layer %s\n", _name.c_str());

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

/// <summary>
/// Retrieves the unit values for this layer and stores them in a provided vector.
/// </summary>
/// <param name="vUnit">A vector where unit values will be stored.</param>
/// <returns>True if the operation is successful, otherwise false.</returns>
bool Layer::GetUnits(std::vector<float>& vUnit)
{
    bool bValid = true;
    
    if (_pbUnit)
    {
        if (vUnit.size() < _stride)
        {
             // Resize the vector if its size is insufficient.
            vUnit.resize(_stride);
        }
    
         // Download unit data to the provided vector.
        _pbUnit->Download(vUnit.data());
    }
    else
    {
        // Handle the case when unit data buffer is not allocated.
        std::cout << std::format("Layer::GetUnits: Unit data not yet allocated.\n");
        bValid = false; // Set the flag to false.
    }
    
     // Return the flag indicating the success of the operation.
    return bValid;  
}

/// <summary>
/// Retrieves the unit values for this layer and stores them in a provided float pointer.
/// </summary>
/// <param name="pUnit">A pointer where unit values will be stored.</param>
/// <returns>True if the operation is successful, otherwise false.</returns>
bool Layer::GetUnits(float* pUnit)
{
    bool bValid = true;
    
    if (_pbUnit)
    {
        if (pUnit == nullptr)
        {
            // Check if the provided pointer is invalid.
            std::cout << std::format("Layer::GetUnits: Download pointer invalid.\n");
            bValid = false; // Set the flag to false.
        }
        else
        {
             // Download unit data to the provided pointer.
            _pbUnit->Download(pUnit);
        }
    }
    else
    {
        // Handle the case when unit data buffer is not allocated.
        std::cout << std::format("Layer::GetUnits: Unit data not yet allocated.\n");
        bValid = false; // Set the flag to false.
    }
    
     // Return the flag indicating the success of the operation. 
    return bValid;   
}

/// <summary>
/// Retrieves the delta values for this layer and stores them in a provided vector.
/// </summary>
/// <param name="vDelta">A vector where delta values will be stored.</param>
/// <returns>True if the operation is successful, otherwise false.</returns>
bool Layer::GetDeltas(std::vector<float>& vDelta)
{
    bool bValid = true;
    
    if (_pbDelta)
    {
        if (vDelta.size() < _stride)
        {
             // Resize the vector if its size is insufficient.
            vDelta.resize(_stride);
        }
        
         // Download delta data to the provided vector.
        _pbDelta->Download(vDelta.data());
    }
    else
    {
        // Handle the case when delta data buffer is not allocated.
        std::cout << std::format("Layer::GetDeltas: Deltas not yet allocated.\n");
        
         // Set the flag to false.
        bValid = false;
    }
    
     // Return the flag indicating the success of the operation.
    return bValid;    
}

/// <summary>
/// Retrieves the delta values for this layer and stores them in a provided float pointer.
/// </summary>
/// <param name="pDelta">A pointer where delta values will be stored.</param>
/// <returns>True if the operation is successful, otherwise false.</returns>
bool Layer::GetDeltas(float* pDelta)
{
    bool bValid = true;
    
    if (_pbDelta)
    {
        if (pDelta == nullptr)
        {
            // Check if the provided pointer is invalid.
            std::cout << ("Layer::GetDeltas: Download pointer invalid.\n");
            
             // Set the flag to false.
            bValid = false;
        }
        else
        {
             // Download delta data to the provided pointer.
            _pbDelta->Download(pDelta);
        }
    }
    else
    {
        // Handle the case when delta data buffer is not allocated.
        std::cout << ("Layer::GetDeltas: Deltas not yet allocated.\n");
        
         // Set the flag to false.
        bValid = false;
    }
    
     // Return the flag indicating the success of the operation. 
    return bValid;   
}

/// <summary>
/// Sets the unit values for this layer.
/// </summary>
/// <param name="vUnit">A vector containing unit values to be set.</param>
/// <returns>True if the operation is successful, otherwise false.</returns>
bool Layer::SetUnits(const std::vector<float>& vUnit)
{
     // Initialize a boolean flag as true.
    bool bValid = true;
    
    if (_pbUnit)
    {
        if (vUnit.size() < _stride)
        {
            // Check if the input unit data size is insufficient.
            std::cout << ("Layer::SetUnits: Input unit data too small to set all units.\n");
            
             // Set the flag to false.
            bValid = false;
        }
    
         // Upload unit data to the associated buffer.
        _pbUnit->Upload(vUnit.data());
    }
    else
    {
        // Handle the case when unit data buffer is not allocated.
        std::cout << ("Layer::SetUnits: Unit data not yet allocated.\n");
        
         // Set the flag to false.
        bValid = false;
    }
    
    return bValid; // Return the flag indicating the success of the operation.    
}

/// <summary>
/// Sets the delta values for this layer.
/// </summary>
/// <param name="vDelta">A vector containing delta values to be set.</param>
/// <returns>True if the operation is successful, otherwise false.</returns>
bool Layer::SetDeltas(const std::vector<float>& vDelta)
{
     // Initialize a boolean flag as true.
    bool bValid = true;
    
    if (_pbDelta)
    {
        if (vDelta.size() < _stride)
        {
            // Check if the input delta data size is insufficient.
            std::cout << std::format("Layer::SetDeltas: Input delta data too small to set all deltas.\n");
            
             // Set the flag to false.
            bValid = false;
        }
    
         // Upload delta data to the associated buffer.
        _pbDelta->Upload(vDelta.data());
    }
    else
    {
        // Handle the case when delta data buffer is not allocated.
        std::cout << std::format("Layer::SetDeltas: Deltas not yet allocated.\n");
        bValid = false; // Set the flag to false.
    }
    
     // Return the flag indicating the success of the operation.
    return bValid;    
}

/// <summary>
/// Gets the tensor descriptor for the specified batch size.
/// </summary>
/// <param name="batch">The batch size.</param>
/// <returns>The cudnnTensorDescriptor for the specified batch size.</returns>
cudnnTensorDescriptor_t Layer::getTensorDescriptor(uint32_t batch)
{
    // Check if the requested batch size matches the current batch size.
    if (batch == _batch)
    {
         // If it matches, return the existing tensor descriptor.
        return _tensorDescriptor;
    }
    else if (batch != _oddBatch)
    {
        cudnnStatus_t cudnnStatus;
        std::vector<int> vDimensions(5, 1);
        std::vector<int> vStride(5, 1);

        // Depending on the dimensions of the tensor (2D, 3D, or 4D), configure the descriptor.
        switch (_dimensions)
        {
            case 2:
                vDimensions[0] = batch;
                vDimensions[1] = _Ny;
                vDimensions[2] = _Nx;
                vStride[2] = 1;
                vStride[1] = _Nx;
                vStride[0] = _Nx * _Ny;
                // Set the tensor descriptor for 2D data with specified dimensions and strides.
                cudnnStatus = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;

            case 3:
                // Set the tensor descriptor for 3D data with specified dimensions and format.
                cudnnStatus = cudnnSetTensor4dDescriptor(_oddBatchTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _Nx);
                break;

            case 4:
                vDimensions[0] = batch;
                vDimensions[1] = _Nw;
                vDimensions[2] = _Nz;
                vDimensions[3] = _Ny;
                vDimensions[4] = _Nx;
                vStride[4] = 1;
                vStride[3] = _Nx;
                vStride[2] = _Nx * _Ny;
                vStride[1] = _Nx * _Ny * _Nz;
                vStride[0] = _Nx * _Ny * _Nz * _Nw;
                // Set the tensor descriptor for 4D data with specified dimensions and strides.
                cudnnStatus = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
        }

        // Check and handle any errors during descriptor creation.
        CUDNNERROR(cudnnStatus, "Layer::getTensorDescriptor: Unable to set oddBatchTensorDescriptor");
        
         // Update the stored oddBatch size.
        _oddBatch = batch;
    }

     // Return the computed tensor descriptor.
    return _oddBatchTensorDescriptor;
}

/// <summary>
/// Get the name of the layer.
/// </summary>
/// <returns>The name of the layer as a constant reference to a string.</returns>
const std::string& Layer::GetName() const {
  return _name;
}

/// <summary>
/// Get the dataset name associated with the layer.
/// </summary>
/// <returns>The dataset name as a constant reference to a string.</returns>
const std::string& Layer::GetDataSetName() const {
    return _dataSet;
}

/// <summary>
/// Get the kind of the layer.
/// </summary>
/// <returns>The kind of the layer (e.g., input, hidden, output).</returns>
Layer::Kind Layer::GetKind() const {
  return _kind;
}

/// <summary>
/// Get the type of the layer.
/// </summary>
/// <returns>The type of the layer (e.g., fully connected, convolutional).</returns>
Layer::Type Layer::GetType() const {
  return _type;
}

/// <summary>
/// Get the attributes associated with the layer.
/// </summary>
/// <returns>The attributes of the layer as an unsigned 32-bit integer.</returns>
uint32_t Layer::GetAttributes() const {
  return _attributes;
}

/// <summary>
/// Get the dataset associated with the layer.
/// </summary>
/// <returns>A pointer to the dataset associated with the layer.</returns>
DataSetBase* Layer::GetDataSet() const {
  return _pDataSet;
}

/// <summary>
/// Get the number of dimensions for the layer's data.
/// </summary>
/// <returns>The number of dimensions as an unsigned 32-bit integer.</returns>
uint32_t Layer::GetNumDimensions() const {
  return _dimensions;
}

/// <summary>
/// Get the dimensions of the layer's data as a tuple (Nx, Ny, Nz, Nw).
/// </summary>
/// <returns>A tuple containing the dimensions of the layer's data.</returns>
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> Layer::GetDimensions() const
{
    return std::make_tuple(_Nx, _Ny, _Nz, _Nw);
}

/// <summary>
/// Get the local dimensions of the layer's data as a tuple (maxX - minX, Ny, Nz, Nw).
/// </summary>
/// <returns>A tuple containing the local dimensions of the layer's data.</returns>
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> Layer::GetLocalDimensions() const
{
    return std::make_tuple(_maxX - _minX, _Ny, _Nz, _Nw);
}

/// <summary>
/// Get the kernel dimensions used by the layer as a tuple (kernelX, kernelY, kernelZ).
/// </summary>
/// <returns>A tuple containing the kernel dimensions used by the layer.</returns>
std::tuple<uint32_t, uint32_t, uint32_t> Layer::GetKernelDimensions() const
{
    return std::make_tuple(_kernelX, _kernelY, _kernelZ);
}

/// <summary>
/// Get the kernel stride used by the layer as a tuple (kernelStrideX, kernelStrideY, kernelStrideZ).
/// </summary>
/// <returns>A tuple containing the kernel stride used by the layer.</returns>
std::tuple<uint32_t, uint32_t, uint32_t> Layer::GetKernelStride() const
{
    return std::make_tuple(_kernelStrideX, _kernelStrideY, _kernelStrideZ);
}

/// <summary>
/// Dumps information about a CUDA tensor descriptor, including data type, number of dimensions, dimensions, and strides.
/// </summary>
/// <param name="t">The CUDA tensor descriptor to be dumped.</param>
static void DumpTensor(cudnnTensorDescriptor_t t) {
    cudnnDataType_t dataType;
    int ndims;
    
     // Array to store dimensions.
    std::vector<int> vDim(16);
    
     // Array to store strides.
    std::vector<int> vStride(16);
    cudnnStatus_t cudnnStatus = cudnnGetTensorNdDescriptor(t, 8, &dataType, &ndims, vDim.data(), vStride.data());

    // Display tensor information.
    std::cout << ("Tensor:   {} dimensions\n", ndims);
    std::cout << ("DataType: {}\n", dataType);

    // Display dimension and stride information.
    for (int i = 0; i < ndims; i++) {
        std::cout << ("{} {} {}\n", i, vDim[i], vStride[i]);
    }

    std::cout << '\n';
}

/// <summary>
/// Allocates memory and GPU buffers for the layer.
/// </summary>
/// <param name="validate">A boolean flag indicating whether validation is required.</param>
void Layer::Allocate(bool validate)
{
    // Deallocate any existing resources before allocating new ones
    Deallocate();

    // Calculate the size of buffers based on the layer's properties
    uint64_t size = static_cast<uint64_t>(_maxLocalStride) * static_cast<uint64_t>(_localBatch);

    // Check if the layer type is Pooling with Cosine function
    if ((_type == Layer::Type::Pooling) && (_poolingFunction == PoolingFunction::Cosine))
    {
        // Allocate memory and create GPU buffers for auxiliary buffer 1
        _vBuffer1.resize(size);
        _pbBuffer1 = std::make_unique<GpuBuffer<float>>(size);

        // Print allocation information (only for GPU id 0)
        if (getGpu()._id == 0)
            std::cout << std::format("Layer::Allocate: Allocating {} bytes ({}, {}) of auxiliary buffer 1 data for layer {}\n", size * sizeof(float), _maxLocalStride, _localBatch, _name);

        // Allocate memory and create GPU buffers for auxiliary buffer 2
        _vBuffer2.resize(size);
        _pbBuffer2 = std::make_unique<GpuBuffer<float>>(size);

        // Print allocation information (only for GPU id 0)
        if (getGpu()._id == 0)
            std::cout << std::format("Layer::Allocate: Allocating {} bytes ({}, {}) of auxiliary buffer 2 data for layer {}\n", size * sizeof(float), _maxLocalStride, _localBatch, _name);
    }
    // Check if the layer type is Pooling or Convolutional
    else if ((_type == Layer::Type::Pooling) || (_type == Layer::Type::Convolutional))
    {
        cudnnStatus_t cudnnStatus;
        std::vector<int> vDimensions(5, 1);
        std::vector<int> vStride(5, 1);

        // Set tensor descriptor based on the layer's dimensions
        switch (_dimensions)
        {
        case 2:
            vDimensions[0] = _localBatch;
            vDimensions[1] = _Ny;
            vDimensions[2] = _Nx;
            vStride[2] = 1;
            vStride[1] = _Nx;
            vStride[0] = _Nx * _Ny;
            cudnnStatus = cudnnSetTensorNdDescriptor(_tensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
            break;
        case 3:
            cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _localBatch, _Nz, _Ny, _Nx);
            break;
        case 4:
            vDimensions[0] = _localBatch;
            vDimensions[1] = _Nw;
            vDimensions[2] = _Nz;
            vDimensions[3] = _Ny;
            vDimensions[4] = _Nx;
            vStride[4] = 1;
            vStride[3] = _Nx;
            vStride[2] = _Nx * _Ny;
            vStride[1] = _Nx * _Ny * _Nz;
            vStride[0] = _Nx * _Ny * _Nz * _Nw;
            cudnnStatus = cudnnSetTensorNdDescriptor(_tensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
            break;
        }
        // Check for errors while setting the tensor descriptor
        CUDNNERROR(cudnnStatus, "Layer::Allocate: Unable to set tensor descriptor");

        // Dump the tensor descriptor information
        DumpTensor(_tensorDescriptor);
    }

    // Allocate memory and create GPU buffers for unit data
    if (!_bSparse || !_bFastSparse || (_kind != Input) || (_bSparse && (_kind == Input) && validate))
    {
        _vUnit.resize(size);
        _pbUnit = std::make_unique<GpuBuffer<float>>(size);

        // Print allocation information (only for GPU id 0)
        if (getGpu()._id == 0)
            std::cout << std::format("Layer::Allocate: Allocating {} bytes ({}, {}) of unit data for layer {}\n", size * sizeof(float), _maxLocalStride, _localBatch, _name);
    }

    // Allocate memory and create GPU buffers for delta data
    if (_kind != Input)
    {
        _vDelta.resize(size);
        _pbDelta = std::make_unique<GpuBuffer<float>>(size);

        // Print allocation information (only for GPU id 0)
        if (getGpu()._id == 0)
            std::cout << std::format("Layer::Allocate: Allocating {} bytes ({}, {}) of delta data for layer {}\n", size * sizeof(float), _maxLocalStride, _localBatch, _name);

        // Check if batch normalization is enabled and allocate buffers
        if (_bBatchNormalization)
        {
            _pbUnitBN = std::make_unique<GpuBuffer<float>>(size);
            _pbDeltaBN = std::make_unique<GpuBuffer<float>>(size);
        }
    }

    // Allocate memory and create GPU buffers for dropout data if dropout is applied
    if (_pDropout > 0.0f)
    {
        _pbDropout = std::make_unique<GpuBuffer<float>>(size);

        // Print allocation information (only for GPU id 0)
        if (getGpu()._id == 0)
            std::cout << std::format("Layer::Allocate: Allocating {} bytes ({}, {}) of dropout data for layer {}\n", size * sizeof(float), _maxLocalStride, _localBatch, _name);
    }

    // Mark the layer as not dirty (allocation is complete)
    _bDirty = false;
}

void Layer::SetBatch(uint32_t batch)
{
    // Check if the new batch value is different from the current one
    if (batch != _batch)
    {
        // Update the current batch value
        _batch = batch;

        // Calculate the local batch size based on parallelization mode
        if (_parallelization == Layer::Parallelization::Data)
        {
            _localBatch = batch / getGpu()._numprocs;
        }
        else
        {
            // If not using data parallelism, use the same batch size
            _localBatch = batch;
        }

        // Mark the layer as dirty (requires re-computation)
        _bDirty = true;
    }
}

/// <summary>
/// Refreshes the parallelization settings for the layer based on its connections with other layers.
/// </summary>
void Layer::RefreshParallelization()
{
    uint32_t convolutionalInputs = 0;
    uint32_t fullyConnectedInputs = 0;
    uint32_t poolingInputs = 0;
    uint32_t convolutionalOutputs = 0;
    uint32_t fullyConnectedOutputs = 0;
    uint32_t poolingOutputs = 0;    
    
    // Count the types of incoming layers.
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
    
    // Count the types of outgoing layers.
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
    
    // Determine parallelization settings based on the layer's kind and type.
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

/// <summary>
/// Refreshes the state of the layer, including various internal variables and memory allocations.
/// </summary>
/// <param name="pNetwork">A pointer to the network to which this layer belongs.</param>
/// <param name="trainingMode">The training mode for the network (e.g., SGD, AdaDelta, Adam).</param>
/// <param name="validate">A boolean flag indicating whether to validate the layer's state.</param>
void Layer::RefreshState(Network* pNetwork, TrainingMode trainingMode, bool validate)
{
    // Check if the layer state is marked as dirty and needs to be refreshed
    if (_bDirty)
    {
        _bFastSparse = false;

        // Check if this is an input layer with sparse data and determine if fast sparse kernels can be used
        if ((_kind == Input) && (_pDataSet != nullptr) && (_bSparse))
        {
            // If the sparse density is too high, fast sparse kernels cannot be used
            if (_pDataSet->_sparseDensity > 0.1f)
            {
                if (getGpu()._id == 0)
                    std::cout << ("Layer::RefreshState: Sparse density per (%.2f) is too high to use fast sparse kernels on input layer %s\n", _pDataSet->_sparseDensity, _name.c_str());
            }
            else
            {
                // Enable fast sparse kernels for this layer
                _bFastSparse = true;
            }
        }

        // If multiple GPUs are available, refresh parallelization settings
        if (getGpu()._numprocs > 1)
            RefreshParallelization();

        // Allocate memory for the layer and its associated buffers
        Allocate(validate);

        // Check if batch normalization is enabled for this layer
        if (_bBatchNormalization)
        {
            if (trainingMode != TrainingMode::SGD)
            {
                // Allocate memory for scale and bias velocity buffers
                _pbScaleVelocityBN = std::make_unique<GpuBuffer<float>>(_localStride);
                _pbBiasVelocityBN = std::make_unique<GpuBuffer<float>>(_localStride);

                // Allocate memory for scale and bias gradient velocity buffers if using AdaDelta or Adam
                if ((trainingMode == TrainingMode::AdaDelta) || (trainingMode == TrainingMode::Adam))
                {
                    _pbScaleGradientVelocityBN = std::make_unique<GpuBuffer<float>>(_localStride);
                    _pbBiasGradientVelocityBN = std::make_unique<GpuBuffer<float>>(_localStride);
                }
                else
                {
                    // Reset gradient velocity buffers if not using AdaDelta or Adam
                    _pbScaleGradientVelocityBN.reset();
                    _pbScaleGradientVelocityBN.reset();
                }
            }
            else
            {
                // Reset all velocity and gradient velocity buffers if not using batch normalization
                _pbScaleVelocityBN.reset();
                _pbBiasVelocityBN.reset();
                _pbScaleGradientVelocityBN.reset();
                _pbBiasGradientVelocityBN.reset();
            }
        }

        // If the layer is not of type Hidden and is associated with a dataset, shard the dataset based on parallelization settings
        if ((_kind != Hidden) && (_pDataSet != nullptr))
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

        // Reset the dirty flag to indicate that the layer state has been refreshed
        _bDirty = false;
    }

    // If this is an Input layer and is associated with a dataset, set denoising based on the denoising flag
    if ((_kind == Input) && _pDataSet)
    {
        _pDataSet->SetDenoising(_bDenoising);
    }

    // If the layer type is Pooling and the pooling function is LRN, set the LRN descriptor
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

/// <summary>
/// Clears update counts and batch normalization calls for the layer.
/// </summary>
void Layer::ClearUpdates()
{
    _unitUpdateCount = 0;   // Reset the unit update count.
    _deltaUpdateCount = 0;  // Reset the delta update count.
    _bnCalls = 0;           // Reset the batch normalization calls count.
}

/// <summary>
/// Loads a prediction batch of data into the input layer.
/// </summary>
/// <param name="position">The position in the dataset to load from.</param>
/// <param name="batch">The batch index to load.</param>
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

void Layer::LoadTrainingBatch(uint32_t position, uint32_t batch) {
    // Check if the layer kind is 'Input'
    if (_kind == Input) {
        // Check if the layer uses sparse data
        if (_bSparse) {
            // Check if fast sparse computation is enabled
            if (_bFastSparse) {
                // Check if denoising is enabled for sparse data
                if (_bDenoising) {
                    // Calculate sparse transposed denoised matrix
                    _pDataSet->CalculateSparseTransposedDenoisedMatrix(position, batch, this);
                }
                else {
                    // Calculate sparse transposed matrix
                    _pDataSet->CalculateSparseTransposedMatrix(position, batch, this);
                }
            }
            else {
                // Check if denoising is enabled for sparse data
                if (_bDenoising) {
                    // Load sparse denoised input unit
                    _pDataSet->LoadSparseDenoisedInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
                }
                else {
                    // Load sparse input unit
                    _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
                }
            }
        }
        else {
            // Load regular input unit
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);

            // Check if dropout is enabled
            if (_pDropout > 0.0f) {
                // Calculate dropout
                CalculateDropout(batch);
            }
        }
    }
}

void Layer::LoadValidationBatch(uint32_t position, uint32_t batch) {
    // Check if the layer kind is 'Input'
    if (_kind == Input) {
        if (_bSparse) {
            // Load sparse input unit
            _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
            // Calculate sparse transposed matrix
            _pDataSet->CalculateSparseTransposedMatrix(position, batch, this);
        }
        else {
            // Load regular input unit
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
    }
}

void Layer::GenerateDenoisingData() {
    // Check if _pDataSet exists and generate denoising data if available
    if (_pDataSet) {
        _pDataSet->GenerateDenoisingData();
    }
}

/// <summary>
/// Performs forward propagation for the layer.
/// </summary>
/// <param name="position">The position for which forward propagation is performed.</param>
/// <param name="batch">The batch index.</param>
/// <param name="bTraining">A boolean indicating whether forward propagation is done during training.</param>
void Layer::ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining)
{
    // Perform forward propagation based on the layer type.
    switch (_type)
    {
    case FullyConnected:
        /// <summary>
        /// Performs forward propagation for a fully connected layer.
        /// </summary>
        ForwardPropagateFullyConnected(position, batch, bTraining);
        break;

    case Convolutional:
        /// <summary>
        /// Performs forward propagation for a convolutional layer.
        /// </summary>
        ForwardPropagateConvolutional(position, batch, bTraining);
        break;

    case Pooling:
        /// <summary>
        /// Performs forward propagation for a pooling layer.
        /// </summary>
        ForwardPropagatePooling(position, batch, bTraining);
        break;
    }
}
    
/// <summary>
/// Performs forward propagation for a fully connected layer.
/// </summary>
/// <param name="position">The position for which forward propagation is performed.</param>
/// <param name="batch">The batch index.</param>
/// <param name="bTraining">A boolean indicating whether forward propagation is done during training.</param>
void Layer::ForwardPropagateFullyConnected(uint32_t position, uint32_t batch, bool bTraining)
{
    if (getGpu()._numprocs == 1)
    {
        if (_kind != Input)
        {
            switch (_vIncomingLayer.size())
            {
            case 0:
                // Initialize the incoming unit buffer with zeros.
                cudaMemset(GetIncomingUnitBuffer(), 0, _stride * static_cast<unsigned long long>(batch) * sizeof(float));
                break;

            case 1:
                // Clear the unit buffer with bias values from one incoming layer.
                kClearUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData, _stride, batch);
                break;

            case 2:
                // Clear the unit buffer with bias values from two incoming layers.
                kClearDualSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData,
                    _vIncomingWeight[1]->_pbBias->_pDevData, _stride, batch);
                break;

            case 3:
                // Clear the unit buffer with bias values from three incoming layers.
                kClearTripleSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData,
                    _vIncomingWeight[1]->_pbBias->_pDevData,
                    _vIncomingWeight[2]->_pbBias->_pDevData, _stride, batch);
                break;

            case 4:
                // Clear the unit buffer with bias values from four incoming layers.
                kClearQuadSourceUnit(GetIncomingUnitBuffer(), _vIncomingWeight[0]->_pbBias->_pDevData,
                    _vIncomingWeight[1]->_pbBias->_pDevData,
                    _vIncomingWeight[2]->_pbBias->_pDevData,
                    _vIncomingWeight[3]->_pbBias->_pDevData, _stride, batch);
                break;

            default:
                if (getGpu()._id == 0)
                    std::cout << ("Layer::ForwardPropagate: Too many input layers for network layer %s\n", _name.c_str());
                getGpu().Shutdown();
                std::exit(-1);
                break;
            }

            const float sgemm_beta = (float)1.0;
            for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
            {
                if (_vIncomingLayer[i]->_bFastSparse)
                {
                    float* pWeight = _vIncomingWeight[i]->_bShared ?
                        _vIncomingWeight[i]->_pSharedWeight->_pbWeight->_pDevData :
                        _vIncomingWeight[i]->_pbWeight->_pDevData;
                    if (bTraining && _vIncomingLayer[i]->_bDenoising)
                        _vIncomingLayer[i]->_pDataSet->CalculateSparseDenoisedZ(position, batch, _stride, pWeight, GetIncomingUnitBuffer(), sgemm_beta);
                    else
                        _vIncomingLayer[i]->_pDataSet->CalculateSparseZ(position, batch, _stride, pWeight, GetIncomingUnitBuffer(), sgemm_beta);
                }
                else
                {
                    const float sgemm_alpha = (float)1.0;
                    cublasStatus_t cstatus;
                    float* pA = _vIncomingLayer[i]->GetUnitBuffer();
                    float* pB = _vIncomingWeight[i]->_bShared ?
                        _vIncomingWeight[i]->_pSharedWeight->_pbWeight->_pDevData :
                        _vIncomingWeight[i]->_pbWeight->_pDevData;
                    float* pC = GetIncomingUnitBuffer();
                    int m = batch;
                    int n = _localStride;
                    int k = _vIncomingLayer[i]->_stride;
                    int lda = _vIncomingWeight[i]->_bTransposed ? k : n;
                    int ldb = k;
                    int ldc = n;

                    cstatus =
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
                            std::cout << ("Layer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                        getGpu().Shutdown();
                        std::exit(-1);
                    }
                }
            }

            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), static_cast<uint64_t>(batch) * _stride);
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
                        1.0 / (_bnCalls + 1),
                        _pbRunningMeanBN->_pDevData,
                        _pbRunningVarianceBN->_pDevData,
                        CUDNN_BN_MIN_EPSILON,
                        _pbSaveMeanBN->_pDevData,
                        _pbSaveInvVarianceBN->_pDevData);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardTraining Failed");
                    ++_bnCalls;
                }
                else {
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
    }
    else
    {
        if (_kind != Input)
        {
            if (_vIncomingLargerLayer.size() > 0)
            {
                float sgemm_beta = (float)0.0;
                for (uint32_t i = 0; i < _vIncomingLargerLayer.size(); i++)
                {
                    Layer* pInputLayer = _vIncomingLargerLayer[i];
                    float* pWeight = _vIncomingLargerWeight[i]->_bShared ?
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

                        const float sgemm_alpha = (float)1.0;

                        float* pA = pWeight;
                        float* pB = pInputLayer->GetUnitBuffer();
                        float* pC = getGpu()._pNetwork->GetP2PSendBuffer();
                        int m = _stride;
                        int n = batch;
                        int k = pInputLayer->_localStride;
                        int lda = _stride;
                        int ldb = k;
                        int ldc = _stride;

                        cublasStatus_t cstatus =
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
                                std::cout << ("Layer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                            getGpu().Shutdown();
                            std::exit(-1);
                        }
                    }

                    sgemm_beta = (float)1.0;
                }

                Reduce(batch, _stride, GetIncomingUnitBuffer(), _localStride, _unitUpdateCount);
                _unitUpdateCount++;
            }

            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), static_cast<uint64_t>(batch) * _localStride);
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
                    std::cout << ("Layer::ForwardPropagate: Too many input layers for network layer %s\n", _name.c_str());
                getGpu().Shutdown();
                std::exit(-1);
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
                        1.0 / (_bnCalls + 1),
                        _pbRunningMeanBN->_pDevData,
                        _pbRunningVarianceBN->_pDevData,
                        CUDNN_BN_MIN_EPSILON,
                        _pbSaveMeanBN->_pDevData,
                        _pbSaveInvVarianceBN->_pDevData);
                    CUDNNERROR(cudnnStatus, "Layer::ForwardPropagateFullyConnected: cudnnBatchNormalizationForwardTraining Failed");
                }
                else {
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
        std::string fname = "activation_" + _name;
        Dump(fname, _pbUnit->_pDevData);
#endif

        if (_vOutgoingLargerLayer.size() > 0)
        {

            if (_bFastSparse)
            {
                for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
                {
                    Layer* pOutputLayer = _vOutgoingLargerLayer[i];
                    float* pWeight = _vOutgoingLargerWeight[i]->_bShared ?
                        _vOutgoingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData :
                        _vOutgoingLargerWeight[i]->_pbWeight->_pDevData;
                    const float sgemm_beta = (pOutputLayer->_unitUpdateCount == 0) ? (float)0.0 : (float)1.0;

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
                    Layer* pOutputLayer = _vOutgoingLargerLayer[i];
                    Weight* pWeight = _vOutgoingLargerWeight[i];
                    Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
                    float* pA = pSrcWeight->_pbWeight->_pDevData;
                    float* pB = getGpu()._pNetwork->GetP2PSendBuffer();
                    float* pC = pOutputLayer->GetIncomingUnitBuffer();

                    int m = pOutputLayer->_localStride;
                    int n = batch;
                    int k = _stride;
                    int lda = pOutputLayer->_localStride;
                    int ldb = _stride;
                    int ldc = pOutputLayer->_localStride;
                    const float sgemm_alpha = 1.0;
                    const float sgemm_beta = (pOutputLayer->_unitUpdateCount == 0) ? (float)0.0 : (float)1.0;

                    cublasStatus_t cstatus = cublasSgemm(getGpu()._cuBLASHandle,
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
                            std::cout << ("Layer::ForwardPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        std::exit(-1);
                    }

                    pOutputLayer->_unitUpdateCount++;
                }
            }
        }
    }

#if 0
    _pbUnit->Download(_vUnit.data());
    MPI_Barrier(MPI_COMM_WORLD);
    if (getGpu()._id == 0) {
        char sLog[256];
        sprintf(sLog, "fp_%s_%04d.txt", _name.c_str(), getGpu()._id);
        FILE* pLog = fopen(sLog, "w");
        for (int i = 0; i < _localStride; i++) {
            for (int j = 0; j < batch; j++) {
                fprintf(pLog, "%.6f ", _vUnit[i + j * _localStride]);
            }
            fprintf(pLog, "\n");
                    }
        fclose(pLog);
                }
    MPI_Barrier(MPI_COMM_WORLD);
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
            kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), static_cast<uint64_t>(batch) * _stride);
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
                    kCalculateMaxout(pLayer->GetUnitBuffer(), static_cast<size_t>(batch) * _localStride, GetIncomingUnitBuffer());
                }
                else
                {
                    // Copy the data from the incoming layer to this layer (maxout pooling).
                    cudaError_t status = cudaMemcpy(GetIncomingUnitBuffer(), pLayer->GetUnitBuffer(), static_cast<unsigned long long>(batch) * _localStride * sizeof(float), cudaMemcpyDefault);
                    RTERROR(status, "Layer::ForwardPropagate: Error calling cudaMemcpy for maxout pooling.");
                }
                break;
            }
            beta = 1.0f; // Set beta to 1.0 for subsequent iterations.
        }

        // Loop over incoming skip layers and add their buffers to this layer's buffer.
        for (auto* l : _vIncomingSkip)
        {
            kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), static_cast<uint64_t>(batch) * _stride);
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

/// <summary>
/// Applies dropout regularization to the layer for a specific batch.
/// </summary>
/// <param name="batch">The batch index for which dropout is applied.</param>
void Layer::CalculateDropout(uint32_t batch)
{
    // Calculate the lambda and alpha values based on the activation function.
    float lambda = (_activation == ScaledExponentialLinear) ? _SELULambda : (float)1.0;
    float alpha = -lambda * _ELUAlpha;

    // Calculate the dropout probability and related parameters.
    float q = (float)1.0 - _pDropout;
    float a = (float)1.0 / sqrt(q + alpha * alpha * _pDropout * q);
    float b = -a * _pDropout * alpha;

    // Determine the target value based on the activation function.
    float target = (_activation == Sigmoid) ? (float)0.5 : (float)0.0;

    // Apply dropout based on the activation function.
    switch (_activation)
    {
    case ExponentialLinear:
    case ScaledExponentialLinear:
        /// <summary>
        /// Applies scaled and biased dropout using the given parameters.
        /// </summary>
        kCalculateScaledBiasedDropout(GetUnitBuffer(), _pbDropout->_pDevData, batch, _localStride, _pDropout, alpha, a, b);
        break;

    default:
        /// <summary>
        /// Applies dropout using the given target value.
        /// </summary>
        kCalculateDropout(GetUnitBuffer(), _pbDropout->_pDevData, batch, _localStride, _pDropout, target);
        break;
    }
}


/// <summary>
/// Calculates the error for a specific position in the layer.
/// </summary>
/// <param name="position">The position for which the error is calculated.</param>
/// <param name="batch">The batch index.</param>
/// <param name="ef">The error function to use.</param>
/// <returns>The calculated error as a floating-point value.</returns>
float Layer::CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef)
{
    if (_kind != Output)
    {
        if (getGpu()._id == 0)
            std::cout << std::format("Layer::CalculateError: Attempt to calculate error on non-output layer %s.\n", _name.c_str());
        getGpu().Shutdown();
        std::exit(-1);
    }

    switch (ef)
    {
    case L1:
        /// <summary>
        /// Calculate the L1 error for the given position and batch.
        /// </summary>
        return _pDataSet->CalculateL1Error(position, batch, _localStride, GetUnitBuffer());

    case L2:
        /// <summary>
        /// Calculate the L2 error for the given position and batch.
        /// </summary>
        return _pDataSet->CalculateL2Error(position, batch, _localStride, GetUnitBuffer());

    case L2Hinge:
        /// <summary>
        /// Calculate the L2 Hinge error for the given position and batch.
        /// </summary>
        return _pDataSet->CalculateL2HingeError(position, batch, _localStride, GetUnitBuffer());

    case Hinge:
        /// <summary>
        /// Calculate the Hinge error for the given position and batch.
        /// </summary>
        return _pDataSet->CalculateHingeError(position, batch, _localStride, GetUnitBuffer());

    case CrossEntropy:
        if (_activation == SoftMax)
        {
            /// <summary>
            /// Calculate the Multinomial Cross-Entropy error for the given position and batch
            /// when the activation function is SoftMax.
            /// </summary>
            return _pDataSet->CalculateMultinomialCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
        }
        else
        {
            /// <summary>
            /// Calculate the Cross-Entropy error for the given position and batch
            /// when the activation function is not SoftMax.
            /// </summary>
            return _pDataSet->CalculateCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
        }

    case ScaledMarginalCrossEntropy:
        if (_activation == SoftMax)
        {
            /// <summary>
            /// Calculate the Multinomial Scaled Marginal Cross-Entropy error for the given position and batch
            /// when the activation function is SoftMax.
            /// </summary>
            return _pDataSet->CalculateMultinomialScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
        }
        else
        {
            /// <summary>
            /// Calculate the Scaled Marginal Cross-Entropy error for the given position and batch
            /// when the activation function is not SoftMax.
            /// </summary>
            return _pDataSet->CalculateScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
        }

    case DataScaledMarginalCrossEntropy:
        if (_activation == SoftMax)
        {
            std::cout << "unsupported combination of activation with cost function" << '\n';
            getGpu().Shutdown();
            std::exit(-1);
        }
        else
        {
            /// <summary>
            /// Calculate the Data Scaled Marginal Cross-Entropy error for the given position and batch
            /// when the activation function is not SoftMax.
            /// </summary>
            return _pDataSet->CalculateDataScaledMarginalCrossEntropyError(position, batch, _localStride, GetUnitBuffer());
        }
    }

    return (float)0.0;
}

void Layer::CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef)
{
    // Check if this layer is not an output layer
    if (_kind != Output)
    {
        // Handle the error condition where output delta calculation is attempted on a non-output layer
        if (getGpu()._id == 0)
            std::cout << std::format("Layer::CalculateOutputDelta: Attempt to calculate output delta on non-output layer %s.\n", _name.c_str());
        getGpu().Shutdown();
        std::exit(-1);
    }

    // Choose the appropriate method for calculating output delta based on the error function (ef)
    switch (ef)
    {
        case L1:
            // Calculate output delta using L1 loss function
            _pDataSet->CalculateL1OutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;

        case CrossEntropy:
            // Calculate output delta using Cross-Entropy loss function
            _pDataSet->CalculateCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        case ScaledMarginalCrossEntropy:
            // Calculate output delta using Scaled Marginal Cross-Entropy loss function
            _pDataSet->CalculateScaledMarginalCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        case L2:
            // Calculate output delta using L2 loss function
            _pDataSet->CalculateOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;

        case L2Hinge:
            // Calculate output delta using L2 Hinge loss function
            _pDataSet->CalculateL2HingeOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
            break;

        case Hinge:
            // Calculate output delta using Hinge loss function
            _pDataSet->CalculateHingeOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        case DataScaledMarginalCrossEntropy:
            // Calculate output delta using Data-Scaled Marginal Cross-Entropy loss function
            _pDataSet->CalculateDataScaledMarginalCrossEntropyOutputDelta(_activation, position, batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer());
            break;

        default:
            // Handle the case where an unsupported cost function is specified
            std::cout << "Unsupported cost function" << '\n';
            std::exit(2);
    }

    // Check if delta normalization is enabled
    if (_deltaNorm > (float)0.0)
    {
        // Normalize delta values based on specified normalization method
        if (getGpu()._numprocs == 1)
            kNormalizeDeltas(_deltaNorm, batch, _localStride, GetDeltaBuffer());
        else
        {
            // Calculate delta magnitudes and perform distributed delta normalization in a multi-GPU setup
            float* pMagnitude = getGpu()._pNetwork->GetScratchBuffer(batch);
            kCalculateDeltaMagnitudes(batch, _localStride, GetDeltaBuffer(), pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, batch);
            kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, GetDeltaBuffer(), pMagnitude);
        }
    }
}

void Layer::BackPropagate(uint32_t position, uint32_t batch)
{
    // Determine the layer type and call the corresponding backpropagation function
    
    switch (_type)
    {
        case FullyConnected:
            // If the layer type is FullyConnected, invoke the FullyConnected backpropagation function
            BackPropagateFullyConnected(position, batch);
            break;
            
        case Convolutional:
            // If the layer type is Convolutional, invoke the Convolutional backpropagation function
            BackPropagateConvolutional(position, batch);
            break;
            
        case Pooling:
            // If the layer type is Pooling, invoke the Pooling backpropagation function
            BackPropagatePooling(position, batch);
            break;                        
    }
}

void Layer::BackPropagateConvolutional(uint32_t position, uint32_t batch)
{
    // Check if there is only one GPU (single-GPU case)
    if (getGpu()._numprocs == 1)
    {
        // Check if the current layer is of kind "Hidden"
        if (_kind == Hidden)
        {
            // Apply sparseness penalty if enabled
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                // Calculate sparseness penalty terms based on specified parameters
                float p = (_sparsenessPenalty_p > (float)0.0) ? _sparsenessPenalty_p : getGpu()._pNetwork->_sparsenessPenalty_p;
                float beta = (_sparsenessPenalty_beta > (float)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
            }

            // Calculate the Hadamard product of activation derivatives and incoming delta
            float scale = (float)1.0 / ((float)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, static_cast<uint64_t>(batch) * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);

            // Normalize incoming delta values if specified
            if (_deltaNorm > (float)0.0)
            {
                kNormalizeDeltas(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer());
            }

            // Apply batch normalization if enabled
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

        // Iterate through incoming layers and perform backpropagation
        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
        {
            Layer* pInputLayer = _vIncomingLayer[i];

            Weight* pWeight = _vIncomingWeight[i];
            Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
            float gradient_alpha = -(float)1.0 / (pSrcWeight->_sharingCount * (float)batch);

            cudnnStatus_t cudnnStatus;
            if (!pWeight->_bLocked)
            {
                // Calculate gradients for convolutional filters (weights)
                float beta = (pSrcWeight->_updateCount == 0) ? (float)0.0 : (float)1.0;
                cudnnStatus = cudnnConvolutionBackwardFilter(getGpu()._cuDNNHandle,
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

                // Calculate gradients for convolutional biases
                beta = (float)0.0;
                cudnnStatus = cudnnConvolutionBackwardBias(getGpu()._cuDNNHandle,
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
                // Calculate delta values for the input layer
                float delta_alpha = (float)1.0;
                float beta = (pInputLayer->_deltaUpdateCount == 0) ? (float)0.0 : (float)1.0;
                cudnnStatus = cudnnConvolutionBackwardData(getGpu()._cuDNNHandle,
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

        // Handle skip connections (if any)
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                // Add incoming delta to the skip-connected layer's delta
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<uint64_t>(batch) * _localStride);
            }
            else
            {
                // Copy incoming delta to the skip-connected layer's delta
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<unsigned long long>(batch) * _localStride * sizeof(float), cudaMemcpyDefault);
            }

            l->_deltaUpdateCount++;
        }
    }
}

/// <summary>
/// Backpropagates through the pooling layer.
/// </summary>
/// <param name="position">The position parameter description.</param>
/// <param name="batch">The batch parameter description.</param>
void Layer::BackPropagatePooling(uint32_t position, uint32_t batch)
{
    // Define a constant for the pooling alpha value
    const float pooling_alpha = 1.0f;

    // Loop through the incoming layers
    for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
    {
        Layer* pInputLayer = _vIncomingLayer[i];

        // Check if the current input layer is not of kind 'Input'
        if (pInputLayer->_kind != Input)
        {
            cudnnStatus_t cudnnStatus;

            // Calculate the beta value based on delta update count
            const float beta = (pInputLayer->_deltaUpdateCount == 0) ? 0.0f : 1.0f;

            // Perform different operations based on the pooling function
            switch (_poolingFunction)
            {
            case Max:
            case Average:
                // Perform backward pooling using cuDNN
                cudnnStatus = cudnnPoolingBackward(getGpu()._cuDNNHandle,
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

                // Increment the delta update count
                pInputLayer->_deltaUpdateCount++;
                break;

            case LRN:
                // Perform LRN backward using cuDNN
                cudnnStatus = cudnnLRNCrossChannelBackward(getGpu()._cuDNNHandle,
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

                // Increment the delta update count
                pInputLayer->_deltaUpdateCount++;
                break;

            case Maxout:
                // Calculate Maxout delta
                kCalculateMaxoutDelta(GetUnitBuffer(), GetDeltaBuffer(), static_cast<size_t>(batch) * _localStride, beta, pInputLayer->GetUnitBuffer(), pInputLayer->GetIncomingDeltaBuffer());

                // Increment the delta update count
                pInputLayer->_deltaUpdateCount++;
                break;

            case Cosine:
                if (i != 0)
                {
                    Layer* p0Layer = _vIncomingLayer[0];
                    const float beta0 = (p0Layer->_deltaUpdateCount == 0) ? 0.0f : 1.0f;
                    const uint32_t offset = i - 1;
                    float* pDPIn = GetUnitBuffer() + offset;
                    float* pDPDeltaIn = GetDeltaBuffer() + offset;
                    float* pAIn = _pbBuffer1->_pDevData + offset;
                    float* pBIn = _pbBuffer2->_pDevData + offset;

                    // Calculate Cosine delta
                    kCalculateCosineDelta(pDPDeltaIn, pDPIn, pAIn, pBIn,
                        p0Layer->GetUnitBuffer(), pInputLayer->GetUnitBuffer(), batch, _localStride,
                        p0Layer->GetIncomingDeltaBuffer(), beta0,
                        pInputLayer->GetIncomingDeltaBuffer(), beta,
                        pInputLayer->_localStride);

                    // Increment delta update counts
                    p0Layer->_deltaUpdateCount++;
                    pInputLayer->_deltaUpdateCount++;
                }
                break;

            case DotProduct:
                if (i != 0)
                {
                    Layer* p0Layer = _vIncomingLayer[0];
                    const float beta0 = (p0Layer->_deltaUpdateCount == 0) ? 0.0f : 1.0f;
                    const uint32_t offset = i - 1;
                    float* pDPDeltaIn = GetDeltaBuffer() + offset;

                    // Calculate Dot Product delta
                    kCalculateDotProductDelta(pDPDeltaIn, p0Layer->GetUnitBuffer(), pInputLayer->GetUnitBuffer(), batch, _localStride,
                        p0Layer->GetIncomingDeltaBuffer(), beta0,
                        pInputLayer->GetIncomingDeltaBuffer(), beta,
                        pInputLayer->_localStride);

                    // Increment delta update counts
                    p0Layer->_deltaUpdateCount++;
                    pInputLayer->_deltaUpdateCount++;
                }
                break;
            }
        }
    }

    // Process incoming skip layers
    for (auto l : _vIncomingSkip)
    {
        if (l->_deltaUpdateCount > 0)
        {
            // Add buffers if delta update count is greater than 0
            kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<uint64_t>(batch) * _localStride);
        }
        else
        {
            // Otherwise, perform a memory copy
            cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<unsigned long long>(batch) * _localStride * sizeof(float), cudaMemcpyDefault);
        }

        // Increment delta update count for skip layer
        l->_deltaUpdateCount++;
    }
}

/// <summary>
/// Calculate the sparseness penalty for the layer's activations.
/// </summary>
/// <param name="p">Sparseness target.</param>
/// <param name="beta">Sparseness penalty coefficient.</param>
/// <param name="batch">The batch size for sparseness penalty calculation.</param>
void Layer::CalculateSparsenessPenalty(float p, float beta, uint32_t batch) {
    if (_bSparse && getGpu()._data._bSparsenessPenalty) {
        kCalculateSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
    }
}

/// <summary>
/// Calculate the Hadamard product of the layer's activations and deltas with scale factor.
/// </summary>
/// <param name="batch">The batch size for Hadamard product calculation.</param>
/// <param name="scale">Scale factor for the Hadamard product.</param>
void Layer::CalculateHadamardProduct(uint32_t batch, float scale) {
    kCalculateHadamardProduct(_activation, static_cast<uint64_t>(batch) * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);
}

/// <summary>
/// Normalize the deltas of the layer if a normalization threshold is specified.
/// </summary>
/// <param name="deltaNorm">Normalization threshold for deltas.</param>
/// <param name="batch">The batch size for delta normalization.</param>
void Layer::NormalizeDeltas(float deltaNorm, uint32_t batch) {
    if (deltaNorm > 0.0) {
        kNormalizeDeltas(deltaNorm, batch, _localStride, GetIncomingDeltaBuffer());
    }
}

/// <summary>
/// Apply batch normalization to the layer's activations.
/// </summary>
/// <param name="batch">The batch size for batch normalization.</param>
void Layer::BatchNormalization(uint32_t batch) {
    cudnnStatus_t cudnnStatus;

    // Set tensor descriptors for batch normalization
    cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
    CUDNNERROR(cudnnStatus, "Layer::BatchNormalization: Unable to create _tensorDescriptorBN");

    cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
    CUDNNERROR(cudnnStatus, "Layer::BatchNormalization: Unable to create _scaleBiasMeanVarDescBN");

    float alpha = 1;
    float beta = 0;

    // Perform batch normalization backward pass
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

    CUDNNERROR(cudnnStatus, "Layer::BatchNormalization: cudnnBatchNormalizationBackward Failed");
}

/// <summary>
/// Perform matrix multiplication using cuBLAS.
/// </summary>
/// <param name="pA">Pointer to matrix A.</param>
/// <param name="pB">Pointer to matrix B.</param>
/// <param name="pC">Pointer to the result matrix C.</param>
/// <param name="m">Number of rows in matrix A and C.</param>
/// <param name="n">Number of columns in matrix B and C.</param>
/// <param name="k">Number of columns in matrix A and rows in matrix B.</param>
/// <param name="lda">Leading dimension of matrix A.</param>
/// <param name="ldb">Leading dimension of matrix B.</param>
/// <param name="ldc">Leading dimension of matrix C.</param>
void Layer::MatrixMultiplication(float* pA, float* pB, float* pC, int m, int n, int k, int lda, int ldb, int ldc) {
    // Define alpha and beta values for cuBLAS SGEMM function
    float sgemm_alpha = 1.0;
    float sgemm_beta = 0.0;

    // Perform matrix multiplication using cuBLAS SGEMM function
    cublasStatus_t cstatus = cublasSgemm(getGpu()._cuBLASHandle,
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

    // Check for cuBLAS operation status and handle errors
    if (cstatus != CUBLAS_STATUS_SUCCESS) {
        if (getGpu()._id == 0)
            printf("Layer::BackPropagate: SGEMM failure, aborting.\n");
        getGpu().Shutdown();
        exit(-1);
    }
}

/// <summary>
/// Backpropagates the error for a fully connected layer.
/// </summary>
/// <param name="position">Position of the layer.</param>
/// <param name="batch">Batch size.</param>
void Layer::BackPropagateFullyConnected(uint32_t position, uint32_t batch) {
    // Check if there is only one GPU process
    if (getGpu()._numprocs == 1) {
        // Check if the layer is of kind "Hidden"
        if (_kind == Hidden) {
            // Calculate sparseness penalty parameters
            float p = (_sparsenessPenalty_p > 0.0) ? _sparsenessPenalty_p : getGpu()._pNetwork->_sparsenessPenalty_p;
            float beta = (_sparsenessPenalty_beta > 0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;

            // Calculate sparseness penalty term
            CalculateSparsenessPenalty(p, beta, batch);

            // Calculate scale factor for dropout
            float scale = 1.0 / (1.0 - _pDropout);

            // Calculate Hadamard product with dropout
            CalculateHadamardProduct(batch, scale);

            // Normalize deltas based on the specified delta norm
            NormalizeDeltas(_deltaNorm, batch);

            // Apply batch normalization if enabled
            if (_bBatchNormalization) {
                BatchNormalization(batch);
            }
        }

        // Iterate over incoming layers
        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++) {
            Layer* pInputLayer = _vIncomingLayer[i];
            cublasStatus_t cstatus;
            Weight* pWeight = _vIncomingWeight[i];
            Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

            // Check if the weight is not locked
            if (!pWeight->_bLocked) {
                float* pDelta = GetDeltaBuffer();
                float* pUnit = pInputLayer->GetUnitBuffer();
                float* pA = pWeight->_bTransposed ? pDelta : pUnit;
                float* pB = pWeight->_bTransposed ? pUnit : pDelta;
                int m = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;
                int n = pWeight->_bTransposed ? _localStride : pInputLayer->_localStride;
                int k = batch;
                int lda = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;
                int ldb = pWeight->_bTransposed ? _localStride : pInputLayer->_localStride;
                int ldc = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;

                // Calculate matrix multiplication between input and deltas
                float sgemm_alpha = -(float)1.0 / (pSrcWeight->_sharingCount * (float)batch);
                float sgemm_beta = (pSrcWeight->_updateCount == 0) ? (float)0.0 : (float)1.0;
                float* pC = pSrcWeight->_pbWeightGradient->_pDevData;

                // If input layer is of kind "Input" and supports fast sparse operations, use a specialized sparse multiplication
                if ((pInputLayer->_kind == Layer::Kind::Input) && pInputLayer->_bFastSparse && !pWeight->_bTransposed) {
                    pInputLayer->_pDataSet->CalculateSparseTransposedWeightGradient(sgemm_alpha, sgemm_beta, n, m, pB, pC);
                }
                else {
                    // Perform a regular dense matrix multiplication
                    MatrixMultiplication(pA, pB, pC, m, n, k, lda, ldb, ldc);
                }

                // Increment the weight update count
                pSrcWeight->_updateCount++;
            }

            // Check if the input layer is not of kind "Input"
            if (pInputLayer->_kind != Input) {
                float sgemm_alpha = (float)1.0;
                float sgemm_beta = (pInputLayer->_deltaUpdateCount == 0) ? (float)0.0 : (float)1.0;
                int m = pInputLayer->_localStride;
                int n = batch;

                // Get input data, weight data, and delta data
                float* pA = GetDeltaBuffer();
                float* pB = pWeight->_bShared ?
                    pSrcWeight->_pbWeight->_pDevData :
                    pWeight->_pbWeight->_pDevData;

                float* pC = pInputLayer->GetIncomingDeltaBuffer();
                int k = _localStride;
                int lda = pWeight->_bTransposed ? pInputLayer->_localStride : k;
                int ldb = k;
                int ldc = pInputLayer->_localStride;

                // Calculate matrix multiplication between deltas and weights
                MatrixMultiplication(pA, pB, pC, m, n, k, lda, ldb, ldc);

                // Increment the delta update count for the input layer
                pInputLayer->_deltaUpdateCount++;
            }
        }

        // Iterate over incoming skip connections
        for (auto l : _vIncomingSkip) {
            // Check if the delta update count is greater than zero
            if (l->_deltaUpdateCount > 0) {
                // Add the incoming delta to the skip connection
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<uint64_t>(batch) * _localStride);
            }
            else {
                // Copy the incoming delta to the skip connection
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<unsigned long long>(batch) * _localStride * sizeof(float), cudaMemcpyDefault);
            }

            // Increment the delta update count for the skip connection
            l->_deltaUpdateCount++;
        }
    }
    else {
        // Check if there are outgoing larger layers
        if (_vOutgoingLargerLayer.size() > 0) {
            // Gather the data from outgoing layers
            Gather(batch, _stride, GetUnitBuffer(), _localStride);

            // Iterate over outgoing larger layers
            for (int i = 0; i < _vOutgoingLargerLayer.size(); i++) {
                Layer* pOutputLayer = _vOutgoingLargerLayer[i];
                Weight* pWeight = _vOutgoingLargerWeight[i];
                Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

                // Get the necessary parameters for matrix multiplication
                float* pA = pOutputLayer->GetDeltaBuffer();
                float* pB = getGpu()._pNetwork->GetP2PSendBuffer();
                float* pC = pSrcWeight->_pbWeightGradient->_pDevData;
                int m = pOutputLayer->_localStride;
                int n = _stride;
                int k = batch;
                int lda = pOutputLayer->_localStride;
                int ldb = _stride;
                int ldc = pOutputLayer->_localStride;

                // Calculate matrix multiplication
                float sgemm_alpha = -(float)1.0 / (pSrcWeight->_sharingCount * (float)batch);
                float sgemm_beta = (pSrcWeight->_updateCount == 0) ? (float)0.0 : (float)1.0;
                MatrixMultiplication(pA, pB, pC, m, n, k, lda, ldb, ldc);

                // Increment the weight update count
                pSrcWeight->_updateCount++;
            }

            // Initialize the beta value for reduction
            float sgemm_beta = (float)0.0;

            // Iterate over outgoing larger layers to reduce the data
            for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++) {
                Layer* pOutputLayer = _vOutgoingLargerLayer[i];
                const float sgemm_alpha = (float)1.0;
                float* pA = _vOutgoingLargerWeight[i]->_bShared ?
                    _vOutgoingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData :
                    _vOutgoingLargerWeight[i]->_pbWeight->_pDevData;
                float* pB = pOutputLayer->GetDeltaBuffer();
                float* pC = getGpu()._pNetwork->GetP2PSendBuffer();
                int m = _stride;
                int n = batch;
                int k = pOutputLayer->_localStride;
                int lda = pOutputLayer->_localStride;
                int ldb = pOutputLayer->_localStride;
                int ldc = _stride;

                // Calculate matrix multiplication with beta value
                MatrixMultiplication(pA, pB, pC, m, n, k, lda, ldb, ldc);

                // Update the beta value
                sgemm_beta = (float)1.0;
            }

            // Reduce the gathered data
            Reduce(batch, _stride, GetIncomingDeltaBuffer(), _localStride, _deltaUpdateCount);

            // Increment the delta update count for the layer
            _deltaUpdateCount++;
        }

        // Check if the layer is of kind "Hidden"
        if (_kind == Hidden) {
            // Calculate sparseness penalty parameters
            float p = (_sparsenessPenalty_p > 0.0) ? _sparsenessPenalty_p : getGpu()._pNetwork->_sparsenessPenalty_p;
            float beta = (_sparsenessPenalty_beta > 0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;

            // Calculate sparseness penalty term
            CalculateSparsenessPenalty(p, beta, batch);

            // Calculate scale factor for dropout
            float scale = 1.0 / (1.0 - _pDropout);

            // Calculate Hadamard product with dropout
            CalculateHadamardProduct(batch, scale);

            // Check if delta normalization is enabled
            if (_deltaNorm > 0.0) {
                // Calculate magnitude of delta values
                float* pMagnitude = getGpu()._pNetwork->GetScratchBuffer(batch);
                kCalculateDeltaMagnitudes(batch, _localStride, GetIncomingDeltaBuffer(), pMagnitude);

                // Perform all-reduce operation for magnitude values
                getGpu()._pNetwork->P2P_Allreduce(pMagnitude, batch);

                // Normalize delta values based on magnitude
                kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer(), pMagnitude);
            }

            // Apply batch normalization if enabled
            if (_bBatchNormalization) {
                BatchNormalization(batch);
            }
        }

        // Iterate over incoming skip connections
        for (auto l : _vIncomingSkip) {
            // Check if the delta update count is greater than zero
            if (l->_deltaUpdateCount > 0) {
                // Add the incoming delta to the skip connection
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<uint64_t>(batch) * _localStride);
            }
            else {
                // Copy the incoming delta to the skip connection
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<unsigned long long>(batch) * _localStride * sizeof(float), cudaMemcpyDefault);
            }

            // Increment the delta update count for the skip connection
            l->_deltaUpdateCount++;
        }

        // Check if there are incoming larger layers
        if (_vIncomingLargerLayer.size() > 0) {
            // Gather delta data from incoming layers
            Gather(batch, _stride, GetDeltaBuffer(), _localStride);

            // Iterate over incoming larger layers
            for (int i = 0; i < _vIncomingLargerLayer.size(); i++) {
                Layer* pInputLayer = _vIncomingLargerLayer[i];
                Weight* pWeight = _vIncomingLargerWeight[i];
                Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

                // Get the necessary parameters for matrix multiplication
                float* pA = getGpu()._pNetwork->GetP2PSendBuffer();
                float* pC = pSrcWeight->_pbWeightGradient->_pDevData;
                int m = _stride;
                int n = pInputLayer->_localStride;
                int k = batch;
                int lda = _stride;
                int ldb = pInputLayer->_localStride;
                int ldc = _stride;

                // Calculate matrix multiplication
                float sgemm_alpha = -(float)1.0 / (pSrcWeight->_sharingCount * (float)batch);
                float sgemm_beta = (pSrcWeight->_updateCount == 0) ? (float)0.0 : (float)1.0;

                // If input layer is of kind "Input" and supports fast sparse operations, use a specialized sparse multiplication
                if ((pInputLayer->_kind == Layer::Kind::Input) && pInputLayer->_bFastSparse) {
                    pInputLayer->_pDataSet->CalculateSparseTransposedWeightGradient(sgemm_alpha, sgemm_beta, n, m, pA, pC);
                }
                else {
                    // Perform a regular dense matrix multiplication
                    float* pB = pInputLayer->GetUnitBuffer();
                    MatrixMultiplication(pA, pB, pC, m, n, k, lda, ldb, ldc);
                }

                // Increment the weight update count
                pSrcWeight->_updateCount++;

                // Check if the input layer is not of kind "Input"
                if (pInputLayer->_kind != Input) {
                    sgemm_alpha = 1.0;
                    sgemm_beta = (pInputLayer->_deltaUpdateCount == 0) ? 0.0 : 1.0;
                    pA = pSrcWeight->_pbWeight->_pDevData;
                    float* pB = getGpu()._pNetwork->GetP2PSendBuffer();
                    pC = pInputLayer->GetIncomingDeltaBuffer();
                    m = pInputLayer->_localStride;
                    n = batch;
                    k = _stride;
                    lda = _stride;
                    ldb = _stride;
                    ldc = pInputLayer->_localStride;

                    // Calculate matrix multiplication between deltas and weights
                    MatrixMultiplication(pA, pB, pC, m, n, k, lda, ldb, ldc);

                    // Increment the delta update count for the input layer
                    pInputLayer->_deltaUpdateCount++;
                }
            }
        }
    }
}

/// <summary>
/// Updates the weights of the layer.
/// </summary>
/// <param name="trainingMode">The training mode to use.</param>
/// <param name="batch">The batch size.</param>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The regularization lambda.</param>
/// <param name="lambda1">Another regularization lambda.</param>
/// <param name="mu">The momentum parameter (if applicable).</param>
/// <param name="mu1">Another momentum parameter (if applicable).</param>
/// <param name="t">The time step (if applicable).</param>
void Layer::UpdateWeights(TrainingMode trainingMode, uint32_t batch, float alpha, float lambda, float lambda1, float mu, float mu1, float t)
{
    // Check if batch normalization is enabled for this layer.
    if (_bBatchNormalization)
    {
        // Use a switch statement to handle different training modes.
        switch (trainingMode)
        {
        case SGD:
            // Update weights using Stochastic Gradient Descent (SGD) for scale and bias parameters.
            kSGDUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
            kSGDUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
            break;

        case Momentum:
            // Update weights using Momentum optimization for scale and bias parameters.
            kMomentumUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
            kMomentumUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
            break;

        case AdaGrad:
            // Update weights using AdaGrad optimization for scale and bias parameters.
            kAdaGradUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
            kAdaGradUpdateWeights(-alpha, lambda, lambda1, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
            break;

        case Nesterov:
            // Update weights using Nesterov Accelerated Gradient (NAG) for scale and bias parameters.
            kNesterovUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
            kNesterovUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
            break;

        case RMSProp:
            // Update weights using RMSProp optimization for scale and bias parameters.
            kRMSPropUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleBN->_pDevData);
            kRMSPropUpdateWeights(-alpha, lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasBN->_pDevData);
            break;

        case AdaDelta:
            // Update weights using AdaDelta optimization for scale and bias parameters.
            kAdaDeltaUpdateWeights(lambda, lambda1, mu, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleGradientVelocityBN->_pDevData, _pbScaleBN->_pDevData);
            kAdaDeltaUpdateWeights(lambda, lambda1, mu, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasGradientVelocityBN->_pDevData, _pbBiasBN->_pDevData);
            break;

        case Adam:
            // Update weights using Adam optimization for scale and bias parameters.
            kAdamUpdateWeights(-alpha, lambda, lambda1, mu, mu1, t, _localStride, _pbScaleVelocityBN->_pDevData, _pbScaleGradientBN->_pDevData, _pbScaleGradientVelocityBN->_pDevData, _pbScaleBN->_pDevData);
            kAdamUpdateWeights(-alpha, lambda, lambda1, mu, mu1, t, _localStride, _pbBiasVelocityBN->_pDevData, _pbBiasGradientBN->_pDevData, _pbBiasGradientVelocityBN->_pDevData, _pbBiasBN->_pDevData);
            break;

        default:
            // Throw an exception if an invalid training mode is provided.
            throw std::runtime_error("Invalid training mode: " + std::to_string(trainingMode));
        }
    }
}

/// <summary>
/// Calculates the minimum and maximum X spans for a given ID, number of processors, and stride.
/// </summary>
/// <param name="id">The current processor ID.</param>
/// <param name="numprocs">The total number of processors.</param>
/// <param name="stride">The stride value.</param>
/// <returns>A MinMaxSpan struct containing the minimum X, maximum X, and span values.</returns>
MinMaxSpan calcMinXMaxXSpan(uint32_t id, uint32_t numprocs, uint32_t stride)
{
    // Calculate the position within the processor grid.
    uint64_t pos = (static_cast<uint64_t>(id) + 1) % numprocs;

    // Calculate the minimum X coordinate based on the position.
    uint32_t minX = (stride * pos) / numprocs;

    // Calculate the maximum X coordinate based on the position.
    uint32_t maxX = (stride * (pos + 1)) / numprocs;

    // Calculate the span of X values.
    uint32_t span = maxX - minX;

    // Return a MinMaxSpan struct with the calculated values.
    return { minX, maxX, span };
}

/// <summary>
/// Copies data from the source to the destination using CUDA and MPI synchronization.
/// </summary>
/// <param name="src">The source data pointer.</param>
/// <param name="dest">The destination data pointer.</param>
/// <param name="offset">The offset for copying.</param>
/// <param name="stride">The stride value.</param>
/// <param name="span">The span of data to copy.</param>
/// <param name="batch">The batch size.</param>
void copyData(float* src, float* dest, uint32_t offset, uint32_t stride, uint32_t span, uint32_t batch)
{
    // Call the kCopy2D function to perform data copying using CUDA.
    kCopy2D(src + offset, stride, dest + offset, stride, span, batch);

    // Synchronize CUDA devices.
    cudaDeviceSynchronize();

    // Synchronize MPI processes using MPI_Barrier.
    MPI_Barrier(MPI_COMM_WORLD);
}

/// <summary>
/// Reduces data across multiple processors in a Layer using CUDA, MPI, and P2P communication.
/// </summary>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride value.</param>
/// <param name="pBuffer">The data buffer to reduce.</param>
/// <param name="localStride">The local stride value.</param>
/// <param name="updateCount">The update count.</param>
void Layer::Reduce(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride, uint32_t updateCount)
{
    // Check if there's only one processor; no reduction needed.
    if (getGpu()._numprocs <= 1) {
        return;
    }

    // Calculate the number of stages in the reduction process.
    const uint32_t stages = getGpu()._numprocs - 1;

    // Get a pointer to the P2P send buffer from the GPU object.
    float* pSendBuffer = getGpu()._pNetwork->GetP2PSendBuffer();

    if (getGpu()._bP2P)
    {
        // When P2P communication is enabled.

        // Get pointers to P2P receive and peer buffers from the GPU object.
        float* pReceiveBuffer = getGpu()._pNetwork->GetP2PReceiveBuffer();
        float* pPeerBuffer = getGpu()._pNetwork->GetPeerBuffer();

        for (uint32_t i = 0; i < stages; i++)
        {
            // Calculate the minimum and maximum X spans for the current processor.
            MinMaxSpan minMaxSpan = calcMinXMaxXSpan(getGpu()._id, getGpu()._numprocs, stride);

            // Copy data from the peer buffer to the send buffer.
            copyData(pPeerBuffer, pSendBuffer, minMaxSpan.minX, stride, minMaxSpan.span, batch);

            // Calculate the new position for the current processor.
            uint64_t pos = (static_cast<uint64_t>(getGpu()._id) + 1) % getGpu()._numprocs;

            // Recalculate the minimum and maximum X spans based on the new position.
            minMaxSpan = calcMinXMaxXSpan(pos, getGpu()._numprocs, stride);

            // Copy data from the send buffer to the receive buffer.
            copyData(pSendBuffer, pReceiveBuffer, minMaxSpan.minX, stride, minMaxSpan.span, batch);
        }
    }
    else
    {
        // When P2P communication is not enabled.

        // Create a CPU buffer for data transfer.
        std::vector<float> pCPUBuffer(batch * stride);

        // Copy data from the P2P send buffer to the CPU buffer.
        cudaError_t status = cudaMemcpy(pCPUBuffer.data(), pSendBuffer, pCPUBuffer.size() * sizeof(float), cudaMemcpyDefault);

        // Check for cudaMemcpy errors.
        if (status != cudaSuccess) {
            throw std::runtime_error(std::format("Layer::Reduce1: cudaMemcpy download failed {}", getGpu()._id));
        }

        // Perform an allreduce operation on the CPU buffer using MPI.
        MPI_Allreduce(MPI_IN_PLACE, pCPUBuffer.data(), pCPUBuffer.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // Copy the updated CPU buffer back to the P2P send buffer.
        status = cudaMemcpy(pSendBuffer, pCPUBuffer.data(), pCPUBuffer.size() * sizeof(float), cudaMemcpyDefault);

        // Check for cudaMemcpy errors.
        if (status != cudaSuccess) {
            throw std::runtime_error(std::format("Layer::Reduce: cudaMemcpy upload failed {}", getGpu()._id));
        }

        // Calculate the minimum and maximum X spans for the current processor.
        MinMaxSpan minMaxSpan = calcMinXMaxXSpan(getGpu()._id, getGpu()._numprocs, stride);

        if (updateCount > 0)
        {
            // Perform a 2D buffer addition operation on pBuffer.
            kAddBuffers2D(pBuffer, localStride, pSendBuffer + minMaxSpan.minX, stride, minMaxSpan.span, batch);
        }
        else
        {
            // Perform a 2D buffer copy operation on pBuffer.
            kCopy2D(pBuffer, localStride, pSendBuffer + minMaxSpan.minX, stride, minMaxSpan.span, batch);
        }
    }
}


/// <summary>
/// Copies data from the source buffer to the destination buffer using a 2D copy operation.
/// </summary>
/// <param name="dest">Pointer to the destination buffer.</param>
/// <param name="destStride">Stride of the destination buffer.</param>
/// <param name="src">Pointer to the source buffer.</param>
/// <param name="srcStride">Stride of the source buffer.</param>
/// <param name="span">Number of elements to copy in each dimension.</param>
/// <param name="batch">Number of batches to copy.</param>
void CopyData(float* dest, uint32_t destStride, float* src, uint32_t srcStride, uint32_t span, uint32_t batch) {
    // Synchronize CUDA device operations
    cudaDeviceSynchronize();

    // Synchronize MPI processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Call the kernel to perform the 2D copy
    kCopy2D(dest, destStride, src, srcStride, span, batch);
}

/// <summary>
/// Gathers data from multiple GPU devices to a single GPU device using a gather operation.
/// </summary>
/// <param name="batch">Number of batches in the data.</param>
/// <param name="stride">Stride of the data in the local GPU.</param>
/// <param name="pBuffer">Pointer to the local data buffer.</param>
/// <param name="localStride">Stride of the local data buffer.</param>
void Layer::Gather(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride)
{
    // If there is only one GPU, no gathering is needed
    if (getGpu()._numprocs <= 1)
    {
        return;
    }

    const uint32_t stages = getGpu()._numprocs - 1;
    const uint64_t myPos = getGpu()._id;
    float* pSendBuffer = getGpu()._pNetwork->GetP2PSendBuffer();

    // Calculate the initial range for data to be gathered by this GPU
    const uint32_t minXInitial = (stride * myPos) / getGpu()._numprocs;
    const uint32_t maxXInitial = (stride * (myPos + 1)) / getGpu()._numprocs;
    uint32_t minX = minXInitial;
    uint32_t maxX = maxXInitial;
    uint32_t span = maxX - minX;

    try
    {
        if (getGpu()._bP2P)
        {
            float* pPeerBuffer = getGpu()._pNetwork->GetPeerBackBuffer();

            // Copy data from the local buffer to the send buffer
            CopyData(pSendBuffer + minX, stride, pBuffer, localStride, span, batch);

            for (uint32_t i = 0; i < stages; i++)
            {
                // Copy data between peer GPUs
                CopyData(pPeerBuffer + minX, stride, pSendBuffer + minX, stride, span, batch);

                // Calculate the range for the next stage
                const uint64_t nextPos = (myPos + 1) % getGpu()._numprocs;
                const uint32_t nextMinX = (stride * nextPos) / getGpu()._numprocs;
                const uint32_t nextMaxX = (stride * (nextPos + 1)) / getGpu()._numprocs;
                span = nextMaxX - nextMinX;
                minX = nextMinX;
                maxX = nextMaxX;
            }

            minX = minXInitial;
            maxX = maxXInitial;
        }
        else
        {
            float* pCPUBuffer = getGpu()._pNetwork->GetP2PCPUBuffer();

            // Copy data from GPU to CPU buffer
            cudaError_t status = cudaMemcpy2D(pCPUBuffer + minX, stride * sizeof(float), pBuffer, localStride * sizeof(float), localStride * sizeof(float), batch, cudaMemcpyDefault);
            if (status != cudaSuccess) {
                throw std::runtime_error("Layer::Gather: cudaMemcpy download failed");
            }

            // Prepare sendCounts and displacements for MPI_Allgatherv
            std::vector<int> sendCounts(getGpu()._numprocs, 0);
            std::vector<int> displacements(getGpu()._numprocs, 0);

            for (uint32_t i = 0; i < getGpu()._numprocs; i++)
            {
                const uint32_t iMinX = (stride * i) / getGpu()._numprocs;
                const uint32_t iMaxX = (stride * (i + 1)) / getGpu()._numprocs;
                const uint32_t iSpan = iMaxX - iMinX;

                sendCounts[i] = iSpan * batch;
                displacements[i] = iMinX * batch;
            }

            // Gather data from all GPUs to the send buffer
            MPI_Allgatherv(pCPUBuffer, sendCounts[getGpu()._id], MPI_FLOAT, pSendBuffer, sendCounts.data(), displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
        }
    }
    catch (const std::runtime_error& e)
    {
        throw;
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
    {Layer::Attributes::BatchNormal, "BatchNormalization"}
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
                std::cerr << "NcException Layer::Layer: No " << attributeName << " supplied in NetCDF input file " << fname << " " << __FILE__ << " " << __LINE__ << '\n';
            }
        }
        catch (const netCDF::exceptions::NcException& e) {
            // Handle NetCDF exceptions
            std::cerr << "NcException Layer::Layer: " << e.what() << '\n';
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
                        std::cerr << "NcException Layer::Layer: No " << attributeName << " attributes supplied in NetCDF input file " << fname << " " << __FILE__ << " " << __LINE__ << '\n';
                    }
                }
            }
            else {
                // Handle the case when the count attribute is missing
                std::cerr << "NcException Layer::Layer: No " << attributeName << " supplied in NetCDF input file " << fname << " " << __FILE__ << " " << __LINE__ << '\n';
            }
        }
        catch (const netCDF::exceptions::NcException& e) {
            // Handle NetCDF exceptions
            std::cerr << "NcException Layer::Layer: " << e.what() << '\n';
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
            << std::setw(20) << "batchNormalization:" << std::boolalpha << ((d._attributes & Layer::Attributes::BatchNormal) != 0) << '\n';

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

// Function to broadcast a LayerDescriptor object over MPI
uint32_t MPI_Bcast_LayerDescriptor(LayerDescriptor& d)
{
    // Broadcast the name of the LayerDescriptor object
    MPI_Bcast_string(d._name);

    // Broadcast various attributes of the LayerDescriptor object
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

    // Broadcast the name of the 'dataSet' attribute
    MPI_Bcast_string(d._dataSet);

    // Broadcast the size of the '_vSource' vector and resize it accordingly
    size_t size = d._vSource.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSource.resize(size);

    // Broadcast the elements of the '_vSource' vector
    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSource[i]);

    // Broadcast the size of the '_vSkip' vector and resize it accordingly
    size = d._vSkip.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSkip.resize(size);

    // Broadcast the elements of the '_vSkip' vector
    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSkip[i]);

    // Return 0 to indicate success
    return 0;
}

bool Layer::WriteNetCDF(netCDF::NcFile& nc, uint32_t index)
{
    // Check if the GPU ID is not equal to 0
    if (auto& gpu = getGpu(); gpu._id != 0)
    {
        // If not, return false
        return false;
    }

    // Create a string for attribute naming
    std::string lstring = "layer" + std::to_string(index) + "_";

    // Set various attributes using the nc object
    nc.putAtt(lstring + "name", _name);
    nc.putAtt(lstring + "kind", netCDF::ncUint, _kind);
    nc.putAtt(lstring + "type", netCDF::ncUint, _type);
    nc.putAtt(lstring + "poolingfunction", netCDF::ncUint, _poolingFunction);
    nc.putAtt(lstring + "dataSet", _dataSet);
    nc.putAtt(lstring + "Nx", netCDF::ncUint, _Nx);
    nc.putAtt(lstring + "Ny", netCDF::ncUint, _Ny);
    nc.putAtt(lstring + "Nz", netCDF::ncUint, _Nz);
    nc.putAtt(lstring + "Nw", netCDF::ncUint, _Nw);
    nc.putAtt(lstring + "dimensions", netCDF::ncUint, _dimensions);
    nc.putAtt(lstring + "kernelX", netCDF::ncUint, _kernelX);
    nc.putAtt(lstring + "kernelY", netCDF::ncUint, _kernelY);
    nc.putAtt(lstring + "kernelZ", netCDF::ncUint, _kernelZ);
    nc.putAtt(lstring + "kernelDimensions", netCDF::ncUint, _kernelDimensions);
    nc.putAtt(lstring + "kernelStrideX", netCDF::ncUint, _kernelStrideX);
    nc.putAtt(lstring + "kernelStrideY", netCDF::ncUint, _kernelStrideY);
    nc.putAtt(lstring + "kernelStrideZ", netCDF::ncUint, _kernelStrideZ);
    nc.putAtt(lstring + "kernelPaddingX", netCDF::ncUint, _kernelPaddingX);
    nc.putAtt(lstring + "kernelPaddingY", netCDF::ncUint, _kernelPaddingY);
    nc.putAtt(lstring + "kernelPaddingZ", netCDF::ncUint, _kernelPaddingZ);
    nc.putAtt(lstring + "pDropout", netCDF::ncFloat, _pDropout);
    nc.putAtt(lstring + "weightInit", netCDF::ncUint, _weightInit);
    nc.putAtt(lstring + "weightInitScale", netCDF::ncFloat, _weightInitScale);
    nc.putAtt(lstring + "biasInit", netCDF::ncFloat, _biasInit);
    nc.putAtt(lstring + "weightNorm", netCDF::ncFloat, _weightNorm);
    nc.putAtt(lstring + "deltaNorm", netCDF::ncFloat, _deltaNorm);
    nc.putAtt(lstring + "activation", netCDF::ncUint, _activation);
    nc.putAtt(lstring + "sparsenessPenalty_p", netCDF::ncFloat, _sparsenessPenalty_p);
    nc.putAtt(lstring + "sparsenessPenalty_beta", netCDF::ncFloat, _sparsenessPenalty_beta);
    nc.putAtt(lstring + "RELUSlope", netCDF::ncFloat, _RELUSlope);
    nc.putAtt(lstring + "ELUAlpha", netCDF::ncFloat, _ELUAlpha);
    nc.putAtt(lstring + "SELULambda", netCDF::ncFloat, _SELULambda);

    // Initialize and set the attributes variable based on certain conditions
    uint32_t attributes = 0;
    if (_bSparse)
        attributes |= Layer::Attributes::Sparse;
    if (_bDenoising)
        attributes |= Layer::Attributes::Denoising;
    if (_bBatchNormalization)
        attributes |= Layer::Attributes::BatchNormal;
    nc.putAtt(lstring + "attributes", netCDF::ncUint, attributes);

    // Set the "sources" attribute based on the size of _vSource vector
    nc.putAtt(lstring + "sources", netCDF::ncUint, static_cast<uint32_t>(_vSource.size()));

    // Loop through _vSource and set attributes
    for (size_t i = 0; i < _vSource.size(); i++)
    {
        std::string nstring = std::to_string(i);
        nc.putAtt(lstring + "source" + nstring, _vSource[i]);
    }

    // Set the "skips" attribute based on the size of _vSkip vector
    nc.putAtt(lstring + "skips", netCDF::ncUint, static_cast<uint32_t>(_vSkip.size()));

    // Loop through _vSkip and set attributes
    for (size_t i = 0; i < _vSkip.size(); i++)
    {
        std::string nstring = std::to_string(i);
        nc.putAtt(lstring + "skip" + nstring, _vSkip[i]);
    }

    // Check if BatchNormalization is enabled
    if (_bBatchNormalization)
    {
        // Create a vector to store data and calculate the number of bytes
        std::vector<float> bndata(_strideBN);
        size_t bytes = _strideBN * sizeof(float);

        // Add a dimension to the netCDF file
        netCDF::NcDim bnDim = nc.addDim(lstring + "bnDim", _strideBN);

        // Copy data to the vector from GPU memory
        cudaMemcpy(bndata.data(), _pbScaleBN->_pDevData, bytes, cudaMemcpyDeviceToHost);

        // Create a variable and store the data in the netCDF file
        netCDF::NcVar scaleVar = nc.addVar(lstring + "scaleBN", "float", bnDim.getName());
        scaleVar.putVar(bndata.data());

        cudaMemcpy(bndata.data(), _pbBiasBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
        netCDF::NcVar biasVar = nc.addVar(lstring + "biasBN", "float", bnDim.getName());
        biasVar.putVar(bndata.data());

        cudaMemcpy(bndata.data(), _pbRunningMeanBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
        netCDF::NcVar runningMeanVar = nc.addVar(lstring + "runningMeanBN", "float", bnDim.getName());
        runningMeanVar.putVar(bndata.data());

        cudaMemcpy(bndata.data(), _pbRunningVarianceBN->_pDevData, bytes, cudaMemcpyDeviceToHost);
        netCDF::NcVar runningVarianceVar = nc.addVar(lstring + "runningVarianceBN", "float", bnDim.getName());
        runningVarianceVar.putVar(bndata.data());
    }

    // Return true to indicate success
    return true;
}
