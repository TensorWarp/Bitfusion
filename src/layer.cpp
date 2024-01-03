#include "GpuTypes.h"
#include "Types.h"
#include <format>
#define __STDC_FORMAT_MACROS
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cublas_v2.h>
#include <format>
#include "Enum.h"
#include "Layer.h"
#include "Network.h"
#include "Weight.h"
#include "mpi.h"
#include "ncDim.h"
#include "ncException.h"
#include "ncFile.h"
#include "ncFloat.h"
#include "ncUint.h"
#include "ncVar.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <tuple>
#include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn_cnn_infer.h>
#include <cudnn_cnn_train.h>
#include <cudnn_ops_infer.h>
#include <cudnn_ops_train.h>
#include <driver_types.h>
#include <span>
#include <glog/logging.h>
#include <cstdlib>
#include <utility>
#include "ncAtt.h"
#include <omp.h>
#include <future>
#include "ThreadPool.h"
#include <cstdlib>
#include <wingdi.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include "ncGroup.h"
#include <ranges>
#include "Constants.h"
#include <execution>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <type_traits>
#include <unordered_map>
#include <nccl.h>


template <typename T>
void DumpP(const char* name, const T* p, size_t count, int gpuIndex = 0) {
    std::cout << name << ":  ";

    std::unique_ptr<T[]> data(new T[count]);

    cudaSetDevice(gpuIndex);

    cudaError_t result = cudaMemcpy(data.get(), p, count * sizeof(T), cudaMemcpyDefault);
    if (result != cudaSuccess) {
        std::cerr << "cudaMemcpy failed on GPU " << gpuIndex << ": " << cudaGetErrorString(result) << std::endl;
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        std::cout << data[i];
        if (i < count - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

Layer::Layer(LayerDescriptor& d, uint32_t batch)
    : _name(d._name),
    _kind(d._kind),
    _type(d._type),
    _attributes(d._attributes),
    _poolingFunction(d._poolingFunction),
    _dataSet(d._dataSet),
    _pDataSet(nullptr),
    _vSource(d._vSource),
    _vSkip(d._vSkip),
    _pbUnit(),
    _pbDelta(),
    _pbDropout(),
    _pbDeltaBN(),
    _pbScaleGradientBN(),
    _pbScaleGradientVelocityBN(),
    _pbBiasGradientBN(),
    _pbBiasGradientVelocityBN(),
    _pbUnitBN(),
    _pbScaleBN(),
    _pbBiasBN(),
    _pbRunningMeanBN(),
    _pbRunningVarianceBN(),
    _pbSaveMeanBN(),
    _pbSaveInvVarianceBN(),
    _Nx(d._Nx),
    _Ny(d._Ny),
    _Nz(d._Nz),
    _Nw(d._Nw),
    _strideBN(0),
    _dimensions(d._dimensions),
    _weightInit(d._weightInit),
    _weightInitScale(d._weightInitScale),
    _biasInit(d._biasInit),
    _kernelX(d._kernelX),
    _kernelY(d._kernelY),
    _kernelZ(d._kernelZ),
    _kernelStrideX(d._kernelStrideX),
    _kernelStrideY(d._kernelStrideY),
    _kernelStrideZ(d._kernelStrideZ),
    _kernelPaddingX(d._kernelPaddingX),
    _kernelPaddingY(d._kernelPaddingY),
    _kernelPaddingZ(d._kernelPaddingZ),
    _kernelDimensions(d._kernelDimensions),
    _weightNorm(d._weightNorm),
    _deltaNorm(d._deltaNorm),
    _pDropout(d._pDropout),
    _activation(d._activation),
    _oddBatch(0),
    _bSparse(d._attributes& Layer::Attributes::Sparse),
    _sparsenessPenalty_p(d._sparsenessPenalty_p),
    _sparsenessPenalty_beta(d._sparsenessPenalty_beta),
    _bDenoising(d._attributes& Layer::Attributes::Denoising),
    _bFastSparse(false),
    _bDirty(true),
    _bnCalls(0),
    _priority(-1),
    _deltaUpdateCount(0),
    _unitUpdateCount(0),
    _batch(batch),
    _localBatch(batch),
    _RELUSlope(d._RELUSlope),
    _ELUAlpha(d._ELUAlpha),
    _SELULambda(d._SELULambda),
    _bBatchNormalization(d._attributes& Layer::Attributes::BatchNormal),
    _stride(_Nx* _Ny* _Nz* _Nw),
    _parallelization(_type == FullyConnected ? Model : Data),
    _minX(((size_t)_Nx* (size_t)getGpu()._id) / (size_t)getGpu()._numprocs),
    _maxX(((size_t)_Nx* (size_t)(getGpu()._id + 1)) / (size_t)getGpu()._numprocs),
    _localStride((_maxX - _minX)* _Ny* _Nz* _Nw),
    _maxLocalStride((((size_t)_Nx + getGpu()._numprocs - 1) / (size_t)getGpu()._numprocs)* _Ny* _Nz* _Nw),
    _tensorDescriptor(nullptr),
    _oddBatchTensorDescriptor(nullptr),
    _scaleBiasMeanVarDescBN(nullptr),
    _tensorDescriptorBN(nullptr),
    _LRNDescriptor(nullptr) {
    if ((_type == Layer::Type::Pooling) || (_type == Layer::Type::Convolutional)) {
        cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&_tensorDescriptor);
        if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Unable to create _tensorDescriptor");
        }

        cudnnStatus = cudnnCreateTensorDescriptor(&_oddBatchTensorDescriptor);
        if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Unable to create _oddBatchTensorDescriptor");
        }
    }

    if (_bBatchNormalization) {
        cudaError_t status;
        cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&_scaleBiasMeanVarDescBN);
        if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Unable to create _scaleBiasMeanVarDescBN");
        }
        cudnnStatus = cudnnCreateTensorDescriptor(&_tensorDescriptorBN);
        if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Unable to create _tensorDescriptorBN");
        }

        if (_type == Layer::Type::Convolutional) {
            _strideBN = _Nz;
        }
        else {
            _strideBN = _localStride;
        }

        try {
            _pbScaleGradientBN = std::make_unique<GpuBuffer<float>>(_strideBN);
            _pbBiasGradientBN = std::make_unique<GpuBuffer<float>>(_strideBN);
            _pbScaleBN = std::make_unique<GpuBuffer<float>>(_strideBN);
            _pbBiasBN = std::make_unique<GpuBuffer<float>>(_strideBN);
            _pbRunningMeanBN = std::make_unique<GpuBuffer<float>>(_strideBN);
            _pbRunningVarianceBN = std::make_unique<GpuBuffer<float>>(_strideBN);
            _pbSaveMeanBN = std::make_unique<GpuBuffer<float>>(_strideBN);
            _pbSaveInvVarianceBN = std::make_unique<GpuBuffer<float>>(_strideBN);
        }
        catch (const std::bad_alloc& e) {
            throw std::runtime_error("Memory allocation failed for Batch Normalization buffers: " + std::string(e.what()));
        }

        if (getGpu()._id == 0) {
            std::cout << std::format("Layer::Layer: Allocating {} bytes of BN scale diff for layer {}\n", _strideBN * sizeof(float), _name);
            std::cout << std::format("Layer::Layer: Allocating {} bytes of BN bias diff for layer {}\n", _strideBN * sizeof(float), _name);
            std::cout << std::format("Layer::Layer: Allocating {} bytes of BN scale for layer {}\n", _strideBN * sizeof(float), _name);
            std::cout << std::format("Layer::Layer: Allocating {} bytes of BN bias for layer {}\n", _strideBN * sizeof(float), _name);
            std::cout << std::format("Layer::Layer: Allocating {} bytes of BN running mean for layer {}\n", _strideBN * sizeof(float), _name);
            std::cout << std::format("Layer::Layer: Allocating {} bytes of BN running variance for layer {}\n", _strideBN * sizeof(float), _name);
            std::cout << std::format("Layer::Layer: Allocating {} bytes of BN saving mean for layer {}\n", _strideBN * sizeof(float), _name);
            std::cout << std::format("Layer::Layer: Allocating {} bytes of BN saving InvVariance for layer {}\n", _strideBN * sizeof(float), _name);
        }

        if (d._vScaleBN.size() != 0)
        {
            status = cudaMemcpy(_pbScaleBN->_pDevData, d._vScaleBN.data(), _strideBN * sizeof(float), cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                std::cerr << "Layer::Layer: cudaMemcpy failed on  _pbScaleBN: " << cudaGetErrorString(status) << std::endl;
            }
        }
        else {
            std::vector<float> ones(_strideBN);
            for (int i = 0; i < _strideBN; ++i)
                ones[i] = 1;
            status = cudaMemcpy(_pbScaleBN->_pDevData, ones.data(), _strideBN * sizeof(float), cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                std::cerr << "Layer::Layer: cudaMemcpy failed on  _pbScaleBN: " << cudaGetErrorString(status) << std::endl;
            }
        }

        if (d._vBiasBN.size() != 0)
        {
            status = cudaMemcpy(_pbBiasBN->_pDevData, d._vBiasBN.data(), _strideBN * sizeof(float), cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                std::cerr << "Layer::Layer: cudaMemcpy failed on  _pbBiasBN: " << cudaGetErrorString(status) << std::endl;
            }
        }
        else {
            status = cudaMemset(_pbBiasBN->_pDevData, 0, _strideBN * sizeof(float));
            if (status != cudaSuccess) {
                std::cerr << "Layer::Layer: cudaMemset failed on  _pbBiasBN: " << cudaGetErrorString(status) << std::endl;
            }
        }

        if (d._vRunningMeanBN.size() != 0)
        {
            status = cudaMemcpy(_pbRunningMeanBN->_pDevData, d._vRunningMeanBN.data(), _strideBN * sizeof(float), cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                std::cerr << "Layer::Layer: cudaMemcpy failed on  _pbRunningMeanBN: " << cudaGetErrorString(status) << std::endl;
            }
        }
        else {
            status = cudaMemset(_pbRunningMeanBN->_pDevData, 0, _strideBN * sizeof(float));
            if (status != cudaSuccess) {
                std::cerr << "Layer::Layer: cudaMemset failed on  _pbRunningMeanBN: " << cudaGetErrorString(status) << std::endl;
            }
        }

        if (d._vRunningVarianceBN.size() != 0)
        {
            status = cudaMemcpy(_pbRunningVarianceBN->_pDevData, d._vRunningVarianceBN.data(), _strideBN * sizeof(float), cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                std::cerr << "Layer::Layer: cudaMemcpy failed on  _pbRunningVarianceBN: " << cudaGetErrorString(status) << std::endl;
            }
        }
        else {
            status = cudaMemset(_pbRunningVarianceBN->_pDevData, 0, _strideBN * sizeof(float));
            if (status != cudaSuccess) {
                std::cerr << "Layer::Layer: cudaMemset failed on  _pbRunningVarianceBN: " << cudaGetErrorString(status) << std::endl;
            }
        }

        cudaError_t cudaStatus = cudaMemset(_pbScaleGradientBN->_pDevData, 0, _strideBN * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Layer::Layer: cudaMemset failed on  _pbScaleGradientBN: " << cudaGetErrorString(cudaStatus) << std::endl;
        }

        status = cudaMemset(_pbBiasGradientBN->_pDevData, 0, _strideBN * sizeof(float));
        if (status != cudaSuccess) {
            std::cerr << "Layer::Layer: cudaMemset failed on  _pbBiasGradientBN: " << cudaGetErrorString(status) << std::endl;
        }

        status = cudaMemset(_pbSaveMeanBN->_pDevData, 0, _strideBN * sizeof(float));
        if (status != cudaSuccess) {
            std::cerr << "Layer::Layer: cudaMemset failed on  _pbSaveMeanBN: " << cudaGetErrorString(status) << std::endl;
        }

        status = cudaMemset(_pbSaveInvVarianceBN->_pDevData, 0, _strideBN * sizeof(float));
        if (status != cudaSuccess) {
            std::cerr << "Layer::Layer: cudaMemset failed on  _pbSaveInvVarianceBN: " << cudaGetErrorString(status) << std::endl;
        }
    }

    if (_type == Layer::Type::Pooling)
    {
        cudnnStatus_t cudnnStatus = cudnnCreatePoolingDescriptor(&_poolingDescriptor);
        if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
            std::cerr << "Layer::Layer: unable to create pooling descriptor" << std::endl;
        }

        std::vector<int> vKernel(3);
        std::vector<int> vKernelPadding(3);
        std::vector<int> vKernelStride(3);
        vKernel[0] = _kernelX;
        vKernel[1] = _kernelY;
        vKernel[2] = _kernelZ;
        vKernelPadding[0] = _kernelPaddingX;
        vKernelPadding[1] = _kernelPaddingY;
        vKernelPadding[2] = _kernelPaddingZ;
        vKernelStride[0] = _kernelStrideX;
        vKernelStride[1] = _kernelStrideY;
        vKernelStride[2] = _kernelStrideZ;

        switch (_poolingFunction)
        {
        case PoolingFunction::Max:
            cudnnSetPoolingNdDescriptor(_poolingDescriptor,
                CUDNN_POOLING_MAX,
                CUDNN_PROPAGATE_NAN,
                _kernelDimensions,
                vKernel.data(),
                vKernelPadding.data(),
                vKernelStride.data());
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                std::cerr << "Layer::Layer: unable to set max pooling descriptor" << std::endl;
            }
            break;

        case PoolingFunction::Average:
            cudnnSetPoolingNdDescriptor(_poolingDescriptor,
                CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                CUDNN_PROPAGATE_NAN,
                _kernelDimensions,
                vKernel.data(),
                vKernelPadding.data(),
                vKernelStride.data());
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                std::cerr << "Layer::Layer: unable to set average pooling descriptor" << std::endl;
            }
            break;

        case PoolingFunction::LRN:
            cudnnStatus = cudnnCreateLRNDescriptor(&_LRNDescriptor);
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                std::cerr << "Layer::Layer: unable to create LRN descriptor" << std::endl;
            }
            break;
        }  
    }
}

Layer::~Layer() {
    Deallocate();
    DestroyCudnnDescriptors();
    ResetBatchNormalizationBuffers();
    DestroyPoolingDescriptor();
}

void Layer::DestroyCudnnDescriptors() {
    if (_type == Layer::Type::Pooling || _type == Layer::Type::Convolutional) {
        DestroyCudnnDescriptor(_tensorDescriptor);
        DestroyCudnnDescriptor(_oddBatchTensorDescriptor);
    }

    if (_bBatchNormalization) {
        DestroyCudnnDescriptor(_scaleBiasMeanVarDescBN);
        DestroyCudnnDescriptor(_tensorDescriptorBN);
    }
}

void Layer::DestroyCudnnDescriptor(cudnnTensorDescriptor_t& descriptor) {
    if (descriptor) {
        cudnnDestroyTensorDescriptor(descriptor);
        descriptor = nullptr;
    }
}

void Layer::ResetBatchNormalizationBuffers() {
    std::vector<std::unique_ptr<GpuBuffer<float>>> bnPointers;
    bnPointers.emplace_back(std::move(_pbScaleBN));
    bnPointers.emplace_back(std::move(_pbBiasBN));
    bnPointers.emplace_back(std::move(_pbScaleGradientBN));
    bnPointers.emplace_back(std::move(_pbBiasGradientBN));
    bnPointers.emplace_back(std::move(_pbRunningMeanBN));
    bnPointers.emplace_back(std::move(_pbRunningVarianceBN));
    bnPointers.emplace_back(std::move(_pbSaveMeanBN));
    bnPointers.emplace_back(std::move(_pbSaveInvVarianceBN));

}

void Layer::DestroyPoolingDescriptor() {
    if (_type == Layer::Type::Pooling) {
        if (_poolingDescriptor) {
            cudnnDestroyPoolingDescriptor(_poolingDescriptor);
            _poolingDescriptor = nullptr;
        }

        if (_poolingFunction == PoolingFunction::LRN) {
            if (_LRNDescriptor) {
                cudnnDestroyLRNDescriptor(_LRNDescriptor);
                _LRNDescriptor = nullptr;
            }
        }
    }
}

void Layer::Deallocate() {
    struct Resource {
        std::unique_ptr<GpuBuffer<float>>& ptr;
        const std::string name;
    };

    std::vector<Resource> resources = {
        {_pbUnit, "_pbUnit"},
        {_pbUnitBN, "_pbUnitBN"},
        {_pbDelta, "_pbDelta"},
        {_pbDeltaBN, "_pbDeltaBN"},
        {_pbDropout, "_pbDropout"},
        {_pbBuffer1, "_pbBuffer1"},
        {_pbBuffer2, "_pbBuffer2"},
        {_pbScaleVelocityBN, "_pbScaleVelocityBN"},
        {_pbScaleGradientVelocityBN, "_pbScaleGradientVelocityBN"},
        {_pbBiasVelocityBN, "_pbBiasVelocityBN"},
        {_pbBiasGradientVelocityBN, "_pbBiasGradientVelocityBN"}
    };

    if (getGpu()._id == 0) {
        std::cout << "Layer::Deallocate: Deallocating all data for layer " << _name << std::endl;
    }

    auto safeReset = [](std::unique_ptr<GpuBuffer<float>>& ptr, const std::string& name) {
        if (ptr) {
            std::cout << "Deallocating resource: " << name << std::endl;
            ptr.reset();
        }
        else {
            std::cout << "Attempted to deallocate an already null resource: " << name << std::endl;
        }
        };

    for (const auto& [ptr, name] : resources) {
        try {
            safeReset(ptr, name);
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in Layer::Deallocate: " << e.what() << std::endl;
        }
    }
}

bool Layer::GetUnits(std::vector<float>& vUnit) {
    if (!_pbUnit) {
        std::cerr << "Layer::GetUnits: Unit data not yet allocated." << std::endl;
        return false;
    }

    if (vUnit.size() < _stride) {
        vUnit.reserve(_stride);
        vUnit.resize(_stride);
    }

    std::span<float> unitSpan(vUnit.data(), _stride);
    _pbUnit->Download(unitSpan.data());

    return true;
}

bool Layer::GetUnits(float* pUnit)
{
    if (!_pbUnit)
    {
        std::cerr << "Layer::GetUnits: Unit data not yet allocated." << std::endl;
        return false;
    }

    if (!pUnit)
    {
        std::cerr << "Layer::GetUnits: Download pointer invalid." << std::endl;
        return false;
    }

    _pbUnit->Download(pUnit);
    return true;
}

bool Layer::GetDeltas(std::vector<float>& vDelta)
{
    if (!_pbDelta)
    {
        std::cerr << "Layer::GetDeltas: Deltas not yet allocated." << std::endl;
        return false;
    }

    if (vDelta.size() < _stride)
    {
        vDelta.resize(_stride);
    }

    _pbDelta->Download(vDelta.data());
    return true;
}

bool Layer::GetDeltas(float* pDelta)
{
    if (!_pbDelta)
    {
        std::cerr << "Layer::GetDeltas: Deltas not yet allocated." << std::endl;
        return false;
    }

    if (!pDelta)
    {
        std::cerr << "Layer::GetDeltas: Download pointer invalid." << std::endl;
        return false;
    }

    _pbDelta->Download(pDelta);
    return true;
}

bool Layer::SetUnits(const std::vector<float>& vUnit)
{
    if (!_pbUnit)
    {
        std::cerr << "Layer::SetUnits: Unit data not yet allocated." << std::endl;
        return false;
    }

    if (vUnit.size() < _stride)
    {
        std::cerr << "Layer::SetUnits: Input unit data too small to set all units." << std::endl;
        return false;
    }

    _pbUnit->Upload(vUnit.data());
    return true;
}


bool Layer::SetDeltas(const std::vector<float>& vDelta)
{
    if (!_pbDelta)
    {
        std::cerr << "Layer::SetDeltas: Deltas not yet allocated." << std::endl;
        return false;
    }

    if (vDelta.size() < _stride)
    {
        std::cerr << "Layer::SetDeltas: Input delta data too small to set all deltas." << std::endl;
        return false;
    }

    _pbDelta->Upload(vDelta.data());
    return true;
}

cudnnTensorDescriptor_t Layer::getTensorDescriptor(uint32_t batch)
{
    if (batch == _batch)
    {
        return _tensorDescriptor;
    }
    else if (batch != _oddBatch)
    {
        cudnnStatus_t cudnnStatus = CUDNN_STATUS_SUCCESS;

        std::vector<int> vDimensions;
        std::vector<int> vStride;

        switch (_dimensions)
        {
        case 2:
            vDimensions[0] = batch;
            vDimensions[1] = _Ny;
            vDimensions[2] = _Nx;
            vStride[2] = 1;
            vStride[1] = _Nx;
            vStride[0] = _Nx * _Ny;
            cudnnStatus = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
            break;

        case 3:
            cudnnStatus = cudnnSetTensor4dDescriptor(_oddBatchTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(batch), _Nz, _Ny, _Nx);
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
            cudnnStatus = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
            break;

        default:
            throw std::runtime_error("Unsupported dimension: " + std::to_string(_dimensions));
        }

        if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("cudnnSetTensorNdDescriptor failed with error code: " + std::to_string(cudnnStatus));
        }

        cudnnStatus = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, vDimensions.size(), vDimensions.data(), vStride.data());

        if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("cudnnSetTensorNdDescriptor failed with error code: " + std::to_string(cudnnStatus));
        }

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
    std::cout << "Tensor:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType: " << dataType << std::endl;
    for (int i = 0; i < ndims; i++)
        std::cout << i << " " << vDim[i] << " " << vStride[i] << std::endl;
    std::cout << std::endl;

}

void Layer::Allocate(bool validate) {
    Deallocate();
    uint64_t size = static_cast<uint64_t>(_maxLocalStride) * static_cast<uint64_t>(_localBatch);

    auto allocateBuffer = [this, size](const std::string& bufferName) {
        if (getGpu()._id == 0) {
            std::cout << "Layer::Allocate: Allocating " << size * sizeof(float) << " bytes (" << _maxLocalStride << ", " << _localBatch << ") of " << bufferName << " data for layer " << _name << std::endl;
        }
        };

    if ((_type == Layer::Type::Pooling) && (_poolingFunction == PoolingFunction::Cosine)) {
        _vBuffer1.resize(size);
        _pbBuffer1 = std::make_unique<GpuBuffer<float>>(size);
        allocateBuffer("auxiliary buffer 1");
        _vBuffer2.resize(size);
        _pbBuffer2 = std::make_unique<GpuBuffer<float>>(size);
        allocateBuffer("auxiliary buffer 2");
    }
    else if ((_type == Layer::Type::Pooling) || (_type == Layer::Type::Convolutional)) {
        cudnnStatus_t cudnnStatus;
        std::vector<int> vDimensions(5, 1);
        std::vector<int> vStride(5, 1);
        switch (_dimensions) {
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

        if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Error setting tensor descriptor with cudnnSetTensorNdDescriptor/cudnnSetTensor4dDescriptor");
        }

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
        _pbDelta = std::make_unique<GpuBuffer<float>>(size);
        allocateBuffer("delta");

        if (_bBatchNormalization) {
            _pbUnitBN = std::make_unique<GpuBuffer<float>>(size);
            _pbDeltaBN = std::make_unique<GpuBuffer<float>>(size);
        }
    }

    if (_pDropout > 0.0) {
        _pbDropout = std::make_unique<GpuBuffer<float>>(size);
        allocateBuffer("dropout");
    }
    _bDirty = false;
}

void Layer::SetBatch(uint32_t batch)
{
    if (batch != _batch)
    {
        _batch = batch;
        if (_parallelization == Layer::Parallelization::Data)
            _localBatch = batch / getGpu()._numprocs;
        else
            _localBatch = batch;
        _bDirty = true;
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
    bool isLSTM = (_type == Layer::Type::LSTM);

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

    case Layer::Kind::Hidden:
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
        else if (isLSTM)
        {
            _parallelization = Layer::Parallelization::Data;
            _bTransposeParallelization = true;
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
    if (!_bDirty) return;

    _bFastSparse = false;
    if (_kind == Input && _pDataSet && _bSparse)
    {
        _bFastSparse = _pDataSet->_sparseDensity <= 0.1f;
        if (_pDataSet->_sparseDensity > 0.1f && getGpu()._id == 0)
        {
            std::cout << "Layer::RefreshState: Sparse density per (" << _pDataSet->_sparseDensity
                << ") is too high to use fast sparse kernels on input layer " << _name << "\n";
        }
    }

    if (getGpu()._numprocs > 1)
        RefreshParallelization();

    Allocate(validate);

    if (_bBatchNormalization)
    {
        auto allocateBuffer = [this](auto condition) -> std::unique_ptr<GpuBuffer<float>> {
            if constexpr (std::is_same_v<decltype(condition), bool>)
            {
                if (condition)
                    return std::make_unique<GpuBuffer<float>>(_localStride);
            }
            return nullptr;
            };

        _pbScaleVelocityBN = allocateBuffer(trainingMode != TrainingMode::SGD);
        _pbBiasVelocityBN = allocateBuffer(trainingMode != TrainingMode::SGD);

        bool TrainingMode = trainingMode == TrainingMode::AdaDelta || trainingMode == TrainingMode::Adam;
        _pbScaleGradientVelocityBN = allocateBuffer(TrainingMode);
        _pbBiasGradientVelocityBN = allocateBuffer(TrainingMode);
    }

    if (_kind != Hidden && _pDataSet)
    {
        switch (_parallelization)
        {
        case Layer::Parallelization::Model:
            _pDataSet->Shard(DataSetEnums::Model);
            break;
        case Layer::Parallelization::Data:
            _pDataSet->Shard(DataSetEnums::Data);
            break;
        default:
            break;
        }
    }
    _bDirty = false;

    if (_kind == Input && _pDataSet)
        _pDataSet->SetDenoising(_bDenoising);

    if (_type == Layer::Type::Pooling && _poolingFunction == PoolingFunction::LRN)
    {
        cudnnStatus_t status = cudnnSetLRNDescriptor(_LRNDescriptor,
            pNetwork->_LRN_n,
            pNetwork->_LRN_alpha,
            pNetwork->_LRN_beta,
            pNetwork->_LRN_k);
    }
}

void Layer::ClearUpdates()
{
    _unitUpdateCount = 0;
    _deltaUpdateCount = 0;
    _bnCalls = 0;
}

void Layer::LoadPredictionBatch(uint32_t position, uint32_t batch)
{
    if (_kind != Layer::Kind::Input)
    {
        throw std::runtime_error("LoadPredictionBatch can only be called on Input layers.");
    }

    if (!_pDataSet)
    {
        throw std::runtime_error("Dataset is not initialized. Call SetDataSet() before loading data.");
    }

    if (!_pbUnit)
    {
        throw std::runtime_error("Prediction batch unit is not initialized.");
    }

    if (!_bSparse)
    {
        try
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
        catch (const std::runtime_error& e)
        {
            throw std::runtime_error("Failed to load input unit: " + std::string(e.what()));
        }
    }
    else if (!_bFastSparse)
    {
        try
        {
            _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
        catch (const std::runtime_error& e)
        {
            throw std::runtime_error("Failed to load sparse input unit: " + std::string(e.what()));
        }
    }
}

void Layer::LoadTrainingBatch(uint32_t position, uint32_t batch)
{
    try
    {
        if (_kind != Layer::Kind::Input)
        {
            throw std::runtime_error("Invalid layer kind");
        }

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

            if (_pDropout > 0.0f)
            {
                CalculateDropout(batch);
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error in LoadTrainingBatch: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Layer::LoadValidationBatch(uint32_t position, uint32_t batch)
{
    if (_kind != Layer::Kind::Input)
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

void Layer::ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining) {
    uint32_t batchPerGPU = batch / NUM_GPUS;

    std::vector<std::thread> threads;
    threads.reserve(NUM_GPUS);

    for (int i = 0; i < NUM_GPUS; ++i) {
        threads.emplace_back([this, i, position, batchPerGPU, bTraining, &batch]() {
            cudaSetDevice(i);
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            std::unique_ptr<cudaStream_t, std::function<void(cudaStream_t*)>> stream_ptr(&stream, [](cudaStream_t* pStream) {
                cudaStreamSynchronize(*pStream);
                cudaStreamDestroy(*pStream);
                });

            uint32_t startPos = i * batchPerGPU;
            uint32_t endPos = (i == NUM_GPUS - 1) ? batch : startPos + batchPerGPU;

            std::function<void(uint32_t, uint32_t, bool)> forwardFunc;
            switch (_type) {
            case Type::FullyConnected:
                forwardFunc = [this](uint32_t start, uint32_t end, bool train) {
                    ForwardPropagateFullyConnected(start, end, train);
                    };
                break;

            case Type::Convolutional:
                forwardFunc = [this](uint32_t start, uint32_t end, bool train) {
                    ForwardPropagateConvolutional(start, end, train);
                    };
                break;

            case Type::Pooling:
                forwardFunc = [this](uint32_t start, uint32_t end, bool train) {
                    ForwardPropagatePooling(start, end, train);
                    };
                break;

            default:
                break;
            }

            if (forwardFunc) {
                forwardFunc(startPos, endPos, bTraining);
            }
            });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}


void Layer::ForwardPropagateFullyConnected(uint32_t position, uint32_t batch, bool bTraining)
{
    if (getGpu()._numprocs == 1)
    {
        if (_kind != Layer::Kind::Input)
        {
            switch (_vIncomingLayer.size())
            {
            case 0:
                cudaMemset(GetIncomingUnitBuffer(), 0, _stride * static_cast<unsigned long long>(batch) * sizeof(float));
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
                            printf("Layer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                        getGpu().Shutdown();
                        exit(-1);
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
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
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

                    if (cudnnStatus != CUDNN_STATUS_SUCCESS)
                    {
                        throw std::runtime_error("Error setting tensor descriptor with cudnnSetTensor4dDescriptor");
                    }
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
                        int ldb = pInputLayer->_localStride;
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
                                printf("Layer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                            getGpu().Shutdown();
                            exit(-1);
                        }
                    }

                    sgemm_beta = (float)1.0;
                }

                Reduce(batch, _stride, GetIncomingUnitBuffer(), _localStride, _unitUpdateCount);
                _unitUpdateCount++;
            }

            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), static_cast<uint64_t>(batch)* _localStride);
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
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
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

                    if (cudnnStatus != CUDNN_STATUS_SUCCESS)
                    {
                        throw std::runtime_error("Error in cudnnBatchNormalizationForwardTraining");
                    }
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

                    if (cudnnStatus != CUDNN_STATUS_SUCCESS)
                    {
                        throw std::runtime_error("Error in cudnnBatchNormalizationForwardInference");
                    }
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


void Layer::ForwardPropagateConvolutional(uint32_t position, uint32_t batch, bool bTraining)
{
    if (_kind != Layer::Kind::Input) {
        if (getGpu()._numprocs == 1) {
            float alpha = 1.0f;
            float beta = 0.0f;

            int numIncomingLayers = _vIncomingLayer.size();

            for (int i = 0; i < numIncomingLayers; i += NUM_GPUS) {
                int numGPUsToUse = std::min(NUM_GPUS, numIncomingLayers - i);

#pragma omp parallel for
                for (int gpuIdx = 0; gpuIdx < numGPUsToUse; gpuIdx++) {
                    int layerIdx = i + gpuIdx;
                    Layer* pLayer = _vIncomingLayer[layerIdx];
                    Weight* pWeight = _vIncomingWeight[layerIdx]->_bShared ?
                        _vIncomingWeight[layerIdx]->_pSharedWeight :
                        _vIncomingWeight[layerIdx];

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

                    cudnnStatus = cudnnAddTensor(getGpu()._cuDNNHandle,
                        &alpha,
                        _vIncomingWeight[layerIdx]->_convBiasTensor,
                        _vIncomingWeight[layerIdx]->_pbBias->_pDevData,
                        &alpha,
                        getTensorDescriptor(batch),
                        GetIncomingUnitBuffer());

                    beta = 1.0f;
                }
            }

            for (auto l : _vIncomingSkip) {
                kAddBuffers(GetIncomingUnitBuffer(), l->GetUnitBuffer(), static_cast<uint64_t>(batch) * _stride);
            }

            if (_bBatchNormalization) {
                for (int i = 0; i < numIncomingLayers; i += NUM_GPUS) {
                    int numGPUsToUse = std::min(NUM_GPUS, numIncomingLayers - i);

#pragma omp parallel for
                    for (int gpuIdx = 0; gpuIdx < numGPUsToUse; gpuIdx++) {
                        int layerIdx = i + gpuIdx;

                        cudnnStatus_t cudnnStatus;

                        cudnnStatus = cudnnBatchNormalizationForwardTraining(
                            getGpu()._cuDNNHandle,
                            CUDNN_BATCHNORM_SPATIAL,
                            &alpha,
                            &beta,
                            _vIncomingLayer[layerIdx]->getTensorDescriptor(batch),
                            _vIncomingLayer[layerIdx]->GetUnitBuffer(),
                            _vIncomingLayer[layerIdx]->getTensorDescriptor(batch),
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

                        if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                            std::cerr << "Batch normalization error: " << cudnnGetErrorString(cudnnStatus) << std::endl;
                        }

                        beta = 1.0f;
                    }
                }

#pragma omp barrier
            }

            CalculateActivation(batch);

            if (bTraining && (_pDropout > 0.0f)) {
                CalculateDropout(batch);
            }
        }
    }
}

void Layer::ForwardPropagatePooling(uint32_t position, uint32_t batch, bool bTraining)
{
    if (_kind == Layer::Kind::Input) {
        return;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    try {
        std::vector<cudnnStatus_t> cudnnStatuses(NUM_GPUS, CUDNN_STATUS_SUCCESS);
        std::vector<cudaStream_t> streams(NUM_GPUS);
        for (int i = 0; i < NUM_GPUS; ++i) {
            cudaSetDevice(i);
            cudaStreamCreate(&streams[i]);
        }

        std::vector<float*> intermediateResults(NUM_GPUS);
        for (int i = 0; i < NUM_GPUS; ++i) {
            cudaSetDevice(i);
            cudaMalloc(&intermediateResults[i], static_cast<unsigned long long>(batch) * _localStride * sizeof(float));
        }

#pragma omp parallel for
        for (int i = 0; i < _vIncomingLayer.size(); i++) {
            const int gpuIdx = i % NUM_GPUS;
            cudaSetDevice(gpuIdx);

            Layer* pLayer = _vIncomingLayer[i];
            cudnnStatus_t cudnnStatus = CUDNN_STATUS_SUCCESS;

            switch (_poolingFunction) {
            case PoolingFunction::Max:
            case PoolingFunction::Average:
                cudnnStatus = cudnnPoolingForward(getGpu()._cuDNNHandle,
                    _poolingDescriptor,
                    &alpha,
                    pLayer->getTensorDescriptor(batch),
                    pLayer->GetUnitBuffer(),
                    &beta,
                    getTensorDescriptor(batch),
                    intermediateResults[gpuIdx]);
                break;

            case PoolingFunction::LRN:
                cudnnStatus = cudnnLRNCrossChannelForward(getGpu()._cuDNNHandle,
                    _LRNDescriptor,
                    CUDNN_LRN_CROSS_CHANNEL_DIM1,
                    &alpha,
                    pLayer->getTensorDescriptor(batch),
                    pLayer->GetUnitBuffer(),
                    &beta,
                    getTensorDescriptor(batch),
                    intermediateResults[gpuIdx]);
                break;

            case PoolingFunction::Cosine:
                if (i >= 1) {
                    Layer* p0Layer = _vIncomingLayer[0];
                    uint32_t offset = i - 1;
                    invokeCosine(p0Layer->GetUnitBuffer(), pLayer->GetUnitBuffer(), batch, pLayer->_localStride,
                        intermediateResults[gpuIdx] + offset,
                        _pbBuffer1->_pDevData + offset,
                        _pbBuffer2->_pDevData + offset,
                        _localStride);
                }
                break;

            case PoolingFunction::DotProduct:
                if (i >= 1) {
                    Layer* p0Layer = _vIncomingLayer[0];
                    uint32_t offset = i - 1;
                    invokeDotProduct(p0Layer->GetUnitBuffer(), pLayer->GetUnitBuffer(), batch, pLayer->_localStride,
                        intermediateResults[gpuIdx] + offset,
                        _localStride);
                }
                break;

            case PoolingFunction::Maxout:
                if (beta != 0.0f) {
                    invokeMaxout(pLayer->GetUnitBuffer(), static_cast<size_t>(batch) * _localStride, intermediateResults[gpuIdx]);
                }
                else {
                    cudaMemcpyAsync(intermediateResults[gpuIdx], pLayer->GetUnitBuffer(), static_cast<unsigned long long>(batch) * _localStride * sizeof(float), cudaMemcpyDeviceToDevice, streams[gpuIdx]);
                }
                break;
            }

            cudnnStatuses[gpuIdx] = cudnnStatus;
        }

        for (int i = 0; i < NUM_GPUS; ++i) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        for (int i = 0; i < NUM_GPUS; ++i) {
            cudaSetDevice(i);
            if (cudnnStatuses[i] != CUDNN_STATUS_SUCCESS) {
                throw std::runtime_error("Error in cudnnPoolingForward or cudnnLRNCrossChannelForward on GPU " + std::to_string(i));
            }
        }

        cudaSetDevice(0);
        cudaMemcpy(GetIncomingUnitBuffer(), intermediateResults[0], static_cast<unsigned long long>(batch) * _localStride * sizeof(float), cudaMemcpyDeviceToDevice);

        for (int i = 0; i < NUM_GPUS; ++i) {
            cudaSetDevice(i);
            cudaFree(intermediateResults[i]);
            cudaStreamDestroy(streams[i]);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Unknown exception occurred." << std::endl;
    }
}

void Layer::CalculateActivation(uint32_t batch)
{
    const uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(_localStride);

    std::unordered_map<Activation, ActivationFunction> activationFunctions = {
        {Activation::Sigmoid, [this](void* buffer, uint64_t s) { invokeSigmoidActivation(GetUnitBuffer(), s); }},
        {Activation::Tanh, [this](void* buffer, uint64_t s) { invokeTanhActivation(GetUnitBuffer(), s); }},
        {Activation::RectifiedLinear, [this](void* buffer, uint64_t s) { invokeRELUActivation(GetUnitBuffer(), s); }},
        {Activation::LeakyRectifiedLinear, [this](void* buffer, uint64_t s) { invokeLRELUActivation(GetUnitBuffer(), s, _RELUSlope); }},
        {Activation::ExponentialLinear, [this](void* buffer, uint64_t s) { invokeELUActivation(GetUnitBuffer(), s, _ELUAlpha); }},
        {Activation::ScaledExponentialLinear, [this](void* buffer, uint64_t s) { invokeSELUActivation(GetUnitBuffer(), s, _ELUAlpha, _SELULambda); }},
        {Activation::SoftMax, [this, batch](void* buffer, uint64_t s) { invokeSoftMaxActivation(GetUnitBuffer(), batch, _localStride); }},
        {Activation::Linear, nullptr}
    };

    const auto& activationFunc = activationFunctions[_activation];

    if (activationFunc) {
        activationFunc(GetUnitBuffer(), size);
    }
}

void Layer::CalculateDropout(uint32_t batch) {
    constexpr float SigmoidTarget = 0.5f;
    const float lambda = (_activation == Activation::ScaledExponentialLinear) ? _SELULambda : 1.0f;
    const float alpha = -lambda * _ELUAlpha;
    const float q = 1.0f - _pDropout;
    const float a = 1.0f / std::sqrt(q + alpha * alpha * _pDropout * q);
    const float b = -a * _pDropout * alpha;
    const float target = (_activation == Activation::Sigmoid) ? SigmoidTarget : 0.0f;
    const uint32_t batchPerGPU = batch / NUM_GPUS;

    auto computeOnGPU = [this, batchPerGPU, a, b, target, alpha](size_t gpuId) {
        try {
            cudaError_t cudaStatus = cudaSetDevice(static_cast<int>(gpuId));
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error("cudaSetDevice failed.");
            }

            if (_activation == Activation::ExponentialLinear || _activation == Activation::ScaledExponentialLinear) {
                invokeScaledBiasedDropout(GetUnitBuffer(), _pbDropout->_pDevData, batchPerGPU, _localStride, _pDropout, alpha, a, b);
            }
            else {
                invokeDropout(GetUnitBuffer(), _pbDropout->_pDevData, batchPerGPU, _localStride, _pDropout, target);
            }

            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error("cudaDeviceSynchronize failed.");
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Exception on GPU " << gpuId << ": " << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }
        };

    std::vector<std::future<void>> gpuTasks(NUM_GPUS);
    for (size_t i = 0; i < NUM_GPUS; ++i) {
        gpuTasks[i] = std::async(std::launch::async, computeOnGPU, i);
    }

    for (auto& task : gpuTasks) {
        task.get();
    }
}

float Layer::CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef) {
    if (_kind != Output) {
        throw std::runtime_error("Layer::CalculateError: Attempt to calculate error on non-output layer " + _name);
    }

    const uint32_t total_positions = _pDataSet->_numPositions;
    const uint32_t positions_per_gpu = total_positions / NUM_GPUS;
    const uint32_t start_position = position * positions_per_gpu;
    const uint32_t end_position = (position + 1) * positions_per_gpu;

    float total_error = 0.0f;
    std::vector<std::future<void>> gpu_tasks(NUM_GPUS);

    const auto calculate_error_on_gpu = [&](int gpu) {
        const uint32_t gpu_start_position = start_position + gpu * positions_per_gpu;
        const uint32_t gpu_end_position = end_position + (gpu == NUM_GPUS - 1 ? 1 : 0);

        float gpu_error = 0.0f;

        switch (ef) {
        case ErrorFunction::L1:
            gpu_error = _pDataSet->CalculateL1Error(gpu_start_position, batch, _localStride, GetUnitBuffer());
            break;

        case ErrorFunction::L2:
            gpu_error = _pDataSet->CalculateL2Error(gpu_start_position, batch, _localStride, GetUnitBuffer());
            break;

        case ErrorFunction::L2Hinge:
            gpu_error = _pDataSet->CalculateL2HingeError(gpu_start_position, batch, _localStride, GetUnitBuffer());
            break;

        case ErrorFunction::Hinge:
            gpu_error = _pDataSet->CalculateHingeError(gpu_start_position, batch, _localStride, GetUnitBuffer());
            break;

        case ErrorFunction::CrossEntropy:
            gpu_error = (_activation == SoftMax)
                ? _pDataSet->CalculateMultinomialCrossEntropyError(gpu_start_position, batch, _localStride, GetUnitBuffer())
                : _pDataSet->CalculateCrossEntropyError(gpu_start_position, batch, _localStride, GetUnitBuffer());
            break;

        case ErrorFunction::ScaledMarginalCrossEntropy:
            gpu_error = (_activation == SoftMax)
                ? _pDataSet->CalculateMultinomialScaledMarginalCrossEntropyError(gpu_start_position, batch, _localStride, GetUnitBuffer())
                : _pDataSet->CalculateScaledMarginalCrossEntropyError(gpu_start_position, batch, _localStride, GetUnitBuffer());
            break;

        case ErrorFunction::DataScaledMarginalCrossEntropy:
            if (_activation == SoftMax) {
                throw std::runtime_error("Unsupported combination of activation with cost function");
            }
            gpu_error = _pDataSet->CalculateDataScaledMarginalCrossEntropyError(gpu_start_position, batch, _localStride, GetUnitBuffer());
            break;

        default:
            throw std::runtime_error("Unsupported error function");
        }

#pragma omp critical
        total_error += gpu_error;
        };

#pragma omp parallel for
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        calculate_error_on_gpu(gpu);
    }

    return total_error;
}

void Layer::CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef) {
    try {
        if (_kind != Output) {
            throw std::runtime_error("Attempt to calculate output delta on non-output layer " + _name);
        }

        switch (ef) {
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
            throw std::invalid_argument("Unsupported cost function");
        }

        if (_deltaNorm > 0.0f) {
            if (getGpu()._numprocs > 1) {
                auto magnitude = std::make_unique<float[]>(batch);
                invokeDeltaMagnitudes(batch, _localStride, GetDeltaBuffer(), magnitude.get());
                getGpu()._pNetwork->P2P_Allreduce(magnitude.get(), batch);
                kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, GetDeltaBuffer(), magnitude.get());
            }
            else {
                kNormalizeDeltas(_deltaNorm, batch, _localStride, GetDeltaBuffer());
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in Layer::CalculateOutputDelta: " << e.what() << std::endl;
        getGpu().Shutdown();
        std::exit(EXIT_FAILURE);
    }
    catch (...) {
        std::cerr << "Unknown error in Layer::CalculateOutputDelta" << std::endl;
        getGpu().Shutdown();
        std::exit(EXIT_FAILURE);
    }
}


void Layer::BackPropagate(uint32_t position, uint32_t batch)
{
    std::unordered_map<Layer::Type, std::function<void(uint32_t, uint32_t)>> backpropagateFunctions = {
        {Layer::Type::FullyConnected, [&](uint32_t pos, uint32_t bat) { BackPropagateFullyConnected(pos, bat); }},
        {Layer::Type::Convolutional, [&](uint32_t pos, uint32_t bat) { BackPropagateConvolutional(pos, bat); }},
        {Layer::Type::Pooling, [&](uint32_t pos, uint32_t bat) { BackPropagatePooling(pos, bat); }},
    };

    auto it = backpropagateFunctions.find(_type);
    if (it != backpropagateFunctions.end()) {
        it->second(position, batch);
    }
    else {
        throw std::runtime_error("Unknown layer type");
    }
}

void Layer::BackPropagateConvolutional(uint32_t position, uint32_t batch)
{
    if (getGpu()._numprocs != 1) {
        return;
    }

    if (_kind == Hidden) {
        if (_bSparse && getGpu()._data._bSparsenessPenalty) {
            const float p = (_sparsenessPenalty_p > 0.0f) ? _sparsenessPenalty_p : getGpu()._pNetwork->_sparsenessPenalty_p;
            const float beta = (_sparsenessPenalty_beta > 0.0f) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
            invokeSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
        }

        const float scale = 1.0f / (1.0f - _pDropout);
        invokeHadamardProduct(_activation, static_cast<uint64_t>(batch) * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);

        if (_deltaNorm > 0.0f) {
            kNormalizeDeltas(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer());
        }

        if (_bBatchNormalization) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cudnnStatus_t cudnnStatus;

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

            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                throw std::runtime_error("Error in cudnnBatchNormalizationBackward");
            }
        }
    }

    for (size_t i = 0; i < _vIncomingLayer.size(); ++i) {
        Layer* pInputLayer = _vIncomingLayer[i];
        Weight* pWeight = _vIncomingWeight[i];
        Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

        const float gradient_alpha = -(1.0f / (pSrcWeight->_sharingCount * static_cast<float>(batch)));
        cudnnStatus_t cudnnStatus;

        if (!pWeight->_bLocked) {
            const float beta = (pSrcWeight->_updateCount == 0) ? 0.0f : 1.0f;

            cudnnStatus = cudnnConvolutionBackwardFilter(
                getGpu()._cuDNNHandle,
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

            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                throw std::runtime_error("Error in cudnnConvolutionBackwardFilter");
            }

            const float zero_beta = 0.0f;

            cudnnStatus = cudnnConvolutionBackwardBias(
                getGpu()._cuDNNHandle,
                &gradient_alpha,
                getTensorDescriptor(batch),
                GetDeltaBuffer(),
                &zero_beta,
                pWeight->_convBiasTensor,
                pWeight->_pbBiasGradient->_pDevData);

            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                throw std::runtime_error("Error in cudnnConvolutionBackwardBias");
            }

            pSrcWeight->_updateCount++;
        }

        if (pInputLayer->_kind != Input) {
            const float delta_alpha = 1.0f;

            const float beta = (pInputLayer->_deltaUpdateCount == 0) ? 0.0f : 1.0f;

            cudnnStatus = cudnnConvolutionBackwardData(
                getGpu()._cuDNNHandle,
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

            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                throw std::runtime_error("Error in cudnnConvolutionBackwardData");
            }

            pInputLayer->_deltaUpdateCount++;
        }
    }

    for (auto l : _vIncomingSkip) {
        if (l->_deltaUpdateCount > 0) {
            kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<uint64_t>(batch)* _localStride);
        }
        else {
            cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<unsigned long long>(batch)* _localStride * sizeof(float), cudaMemcpyDefault);
        }

        l->_deltaUpdateCount++;
    }
}

void Layer::BackPropagatePooling(uint32_t position, uint32_t batch)
{
    const float pooling_alpha = 1.0f;

    for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
    {
        Layer* pInputLayer = _vIncomingLayer[i];

        if (pInputLayer->_kind == Input)
        {
            continue;
        }

        cudnnStatus_t cudnnStatus;
        const float beta = (pInputLayer->_deltaUpdateCount == 0) ? 0.0f : 1.0f;

        switch (_poolingFunction)
        {
        case Max:
        case Average:
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
            break;

        case LRN:
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
            break;

        case Maxout:
            invokeMaxoutDelta(GetUnitBuffer(), GetDeltaBuffer(), static_cast<size_t>(batch) * _localStride, beta, pInputLayer->GetUnitBuffer(), pInputLayer->GetIncomingDeltaBuffer());
            break;

        case Cosine:
        case DotProduct:
            if (i != 0)
            {
                Layer* p0Layer = _vIncomingLayer[0];
                const float beta0 = (p0Layer->_deltaUpdateCount == 0) ? 0.0f : 1.0f;
                const uint32_t offset = i - 1;
                float* pDPDeltaIn = GetDeltaBuffer() + offset;

                if (_poolingFunction == Cosine)
                {
                    float* pDPIn = GetUnitBuffer() + offset;
                    float* pAIn = _pbBuffer1->_pDevData + offset;
                    float* pBIn = _pbBuffer2->_pDevData + offset;

                    invokeCosineDelta(pDPDeltaIn, pDPIn, pAIn, pBIn,
                        p0Layer->GetUnitBuffer(), pInputLayer->GetUnitBuffer(), batch, _localStride,
                        p0Layer->GetIncomingDeltaBuffer(), beta0,
                        pInputLayer->GetIncomingDeltaBuffer(), beta,
                        pInputLayer->_localStride);
                }
                else
                {
                    invokeDotProductDelta(pDPDeltaIn, p0Layer->GetUnitBuffer(), pInputLayer->GetUnitBuffer(), batch, _localStride,
                        p0Layer->GetIncomingDeltaBuffer(), beta0,
                        pInputLayer->GetIncomingDeltaBuffer(), beta,
                        pInputLayer->_localStride);
                }

                p0Layer->_deltaUpdateCount++;
            }
            break;
        }

        if (cudnnStatus != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error("Error in cuDNN pooling backward operation.");
        }

        pInputLayer->_deltaUpdateCount++;
    }

    for (auto l : _vIncomingSkip)
    {
        if (l->_deltaUpdateCount > 0)
        {
            kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<uint64_t>(batch) * _localStride);
        }
        else
        {
            cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<unsigned long long>(batch) * _localStride * sizeof(float), cudaMemcpyDefault);
        }

        l->_deltaUpdateCount++;
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
                float p = (_sparsenessPenalty_p > (float)0.0) ? _sparsenessPenalty_p : getGpu()._pNetwork->_sparsenessPenalty_p;
                float beta = (_sparsenessPenalty_beta > (float)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                invokeSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
            }

            float scale = (float)1.0 / ((float)1.0 - _pDropout);
            invokeHadamardProduct(_activation, static_cast<uint64_t>(batch) * _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);

            if (_deltaNorm > (float)0.0)
            {
                kNormalizeDeltas(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer());
            }

            if (_bBatchNormalization)
            {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
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

                if (cudnnStatus != CUDNN_STATUS_SUCCESS)
                {
                    throw std::runtime_error("Error in cudnnBatchNormalizationBackward.");
                }
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
            Layer* pInputLayer = _vIncomingLayer[i];
            cublasStatus_t cstatus;
            Weight* pWeight = _vIncomingWeight[i];
            Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

            if (!pWeight->_bLocked)
            {
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

                float sgemm_alpha = -(float)1.0 / (pSrcWeight->_sharingCount * (float)batch);
                float sgemm_beta = (pSrcWeight->_updateCount == 0) ? (float)0.0 : (float)1.0;
                float* pC = pSrcWeight->_pbWeightGradient->_pDevData;

                if ((pInputLayer->_kind == Layer::Kind::Input) && pInputLayer->_bFastSparse && !pWeight->_bTransposed)
                {
                    pInputLayer->_pDataSet->CalculateSparseTransposedWeightGradient(sgemm_alpha, sgemm_beta, n, m, pB, pC);
                }
                else
                {
                    cstatus = cublasSgemm(getGpu()._cuBLASHandle,
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
                float sgemm_alpha = (float)1.0;
                float sgemm_beta = (pInputLayer->_deltaUpdateCount == 0) ? (float)0.0 : (float)1.0;
                int m = pInputLayer->_localStride;
                int n = batch;


                float* pA = GetDeltaBuffer();
                float* pB = pWeight->_bShared ?
                    pSrcWeight->_pbWeight->_pDevData :
                    pWeight->_pbWeight->_pDevData;

                float* pC = pInputLayer->GetIncomingDeltaBuffer();
                int k = _localStride;
                int lda = pWeight->_bTransposed ? pInputLayer->_localStride : k;
                int ldb = k;
                int ldc = pInputLayer->_localStride;


                cstatus = cublasSgemm(getGpu()._cuBLASHandle,
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
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<uint64_t>(batch)* _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<unsigned long long>(batch)* _localStride * sizeof(float), cudaMemcpyDefault);
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
                Layer* pOutputLayer = _vOutgoingLargerLayer[i];
                Weight* pWeight = _vOutgoingLargerWeight[i];
                Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

                float* pA = pOutputLayer->GetDeltaBuffer();
                float* pB = getGpu()._pNetwork->GetP2PSendBuffer();
                float* pC = pSrcWeight->_pbWeightGradient->_pDevData;
                int m = pOutputLayer->_localStride;
                int n = _stride;
                int k = batch;
                int lda = pOutputLayer->_localStride;
                int ldb = _stride;
                int ldc = pOutputLayer->_localStride;

                float sgemm_alpha = -(float)1.0 / (pSrcWeight->_sharingCount * (float)batch);
                float sgemm_beta = (pSrcWeight->_updateCount == 0) ? (float)0.0 : (float)1.0;

                cublasStatus_t cstatus = cublasSgemm(getGpu()._cuBLASHandle,
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

            float sgemm_beta = (float)0.0;
            for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
            {
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

                cublasStatus_t cstatus =
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

                sgemm_beta = (float)1.0;
            }


            Reduce(batch, _stride, GetIncomingDeltaBuffer(), _localStride, _deltaUpdateCount);
            _deltaUpdateCount++;
        }



        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                float p = (_sparsenessPenalty_p > (float)0.0) ? _sparsenessPenalty_p : getGpu()._pNetwork->_sparsenessPenalty_p;
                float beta = (_sparsenessPenalty_beta > (float)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                invokeSparsenessPenalty(batch, _localStride, GetUnitBuffer(), GetIncomingDeltaBuffer(), p, beta);
            }

            float scale = (float)1.0 / ((float)1.0 - _pDropout);
            invokeHadamardProduct(_activation, static_cast<uint64_t>(batch)* _localStride, scale, GetUnitBuffer(), GetIncomingDeltaBuffer(), _RELUSlope, _ELUAlpha, _SELULambda);

            if (_deltaNorm > (float)0.0)
            {
                float* pMagnitude = getGpu()._pNetwork->GetScratchBuffer(batch).get();
                invokeDeltaMagnitudes(batch, _localStride, GetIncomingDeltaBuffer(), pMagnitude);
                getGpu()._pNetwork->P2P_Allreduce(pMagnitude, batch);
                kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, GetIncomingDeltaBuffer(), pMagnitude);
            }

            if (_bBatchNormalization)
            {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnSetTensor4dDescriptor(_tensorDescriptorBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _localStride);
                cudnnStatus = cudnnSetTensor4dDescriptor(_scaleBiasMeanVarDescBN, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _Nz, _Ny, _localStride);
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

                if (cudnnStatus != CUDNN_STATUS_SUCCESS)
                {
                    throw std::runtime_error("Error in cudnnBatchNormalizationBackward.");
                }
            }
        }

        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<uint64_t>(batch)* _localStride);
            }
            else
            {
                cudaMemcpy(l->GetIncomingDeltaBuffer(), GetDeltaBuffer(), static_cast<unsigned long long>(batch)* _localStride * sizeof(float), cudaMemcpyDefault);
            }

            l->_deltaUpdateCount++;
        }

        if (_vIncomingLargerLayer.size() > 0)
        {
            Gather(batch, _stride, GetDeltaBuffer(), _localStride);

            for (int i = 0; i < _vIncomingLargerLayer.size(); i++)
            {
                Layer* pInputLayer = _vIncomingLargerLayer[i];
                Weight* pWeight = _vIncomingLargerWeight[i];
                Weight* pSrcWeight = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

                float* pA = getGpu()._pNetwork->GetP2PSendBuffer();
                float* pC = pSrcWeight->_pbWeightGradient->_pDevData;
                int m = _stride;
                int n = pInputLayer->_localStride;
                int k = batch;
                int lda = _stride;
                int ldb = pInputLayer->_localStride;
                int ldc = _stride;

                float sgemm_alpha = -(float)1.0 / (pSrcWeight->_sharingCount * (float)batch);
                float sgemm_beta = (pSrcWeight->_updateCount == 0) ? (float)0.0 : (float)1.0;

                if ((pInputLayer->_kind == Layer::Kind::Input) && pInputLayer->_bFastSparse)
                {
                    pInputLayer->_pDataSet->CalculateSparseTransposedWeightGradient(sgemm_alpha, sgemm_beta, n, m, pA, pC);
                }
                else
                {
                    float* pB = pInputLayer->GetUnitBuffer();
                    cublasStatus_t cstatus = cublasSgemm(getGpu()._cuBLASHandle,
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
                    sgemm_alpha = (float)1.0;
                    sgemm_beta = (pInputLayer->_deltaUpdateCount == 0) ? (float)0.0 : (float)1.0;
                    pA = pSrcWeight->_pbWeight->_pDevData;
                    float* pB = getGpu()._pNetwork->GetP2PSendBuffer();
                    pC = pInputLayer->GetIncomingDeltaBuffer();
                    m = pInputLayer->_localStride;
                    n = batch;
                    k = _stride;
                    lda = _stride;
                    ldb = _stride;
                    ldc = pInputLayer->_localStride;
                    cublasStatus_t cstatus = cublasSgemm(getGpu()._cuBLASHandle,
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
    Weight* pWeight = _vIncomingWeight[0];
    std::vector<float> vLocalWeightGradient(pWeight->_size);
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
        std::cout << std::endl;
#endif   
}

void Layer::UpdateWeights(TrainingMode trainingMode, uint32_t batch, float alpha, float lambda, float lambda1, float mu, float mu1, float t) {
    if (!_bBatchNormalization) {
        return;
    }

    size_t localStride = _localStride;

    std::vector<std::vector<float>> scaleGradientData(NUM_GPUS, std::vector<float>(localStride));
    std::vector<std::vector<float>> biasGradientData(NUM_GPUS, std::vector<float>(localStride));
    std::vector<std::vector<float>> scaleVelocityData(NUM_GPUS, std::vector<float>(localStride));
    std::vector<std::vector<float>> biasVelocityData(NUM_GPUS, std::vector<float>(localStride));
    std::vector<std::vector<float>> scaleGradientVelocityData(NUM_GPUS, std::vector<float>(localStride));
    std::vector<std::vector<float>> biasGradientVelocityData(NUM_GPUS, std::vector<float>(localStride));
    std::vector<std::vector<float>> scaleData(NUM_GPUS, std::vector<float>(localStride));
    std::vector<std::vector<float>> biasData(NUM_GPUS, std::vector<float>(localStride));

#pragma omp parallel for
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        for (size_t i = 0; i < localStride; ++i) {
            scaleGradientData[gpu][i] = _pbScaleGradientBN->_pDevData[i];
            biasGradientData[gpu][i] = _pbBiasGradientBN->_pDevData[i];
            scaleVelocityData[gpu][i] = _pbScaleVelocityBN->_pDevData[i];
            biasVelocityData[gpu][i] = _pbBiasVelocityBN->_pDevData[i];
            scaleGradientVelocityData[gpu][i] = _pbScaleGradientVelocityBN->_pDevData[i];
            biasGradientVelocityData[gpu][i] = _pbBiasGradientVelocityBN->_pDevData[i];
            scaleData[gpu][i] = _pbScaleBN->_pDevData[i];
            biasData[gpu][i] = _pbBiasBN->_pDevData[i];
        }
    }

    std::vector<std::vector<float>> scaleUpdateData(NUM_GPUS, std::vector<float>(localStride));
    std::vector<std::vector<float>> biasUpdateData(NUM_GPUS, std::vector<float>(localStride));

    std::vector<cublasHandle_t> handles(NUM_GPUS);

    for (size_t gpu = 0; gpu < NUM_GPUS; ++gpu) {
        if (cublasCreate(&handles[gpu]) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle for GPU " + std::to_string(gpu));
        }
    }

    const float neg_alpha = -alpha;
    const float scalingFactor = (trainingMode == AdaDelta || trainingMode == Adam) ? t : 1.0f;

#pragma omp parallel for
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        cublasHandle_t handle = handles[gpu];

        cublasStatus_t scaleUpdateStatus = cublasSaxpy(handle, localStride, &neg_alpha, scaleGradientData[gpu].data(), 1, scaleUpdateData[gpu].data(), 1);
        if (scaleUpdateStatus != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Layer::UpdateWeights (GPU " << gpu << "): cuBLAS saxpy failed" << std::endl;
        }

        cublasStatus_t biasUpdateStatus = cublasSaxpy(handle, localStride, &neg_alpha, biasGradientData[gpu].data(), 1, biasUpdateData[gpu].data(), 1);
        if (biasUpdateStatus != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Layer::UpdateWeights (GPU " << gpu << "): cuBLAS saxpy failed" << std::endl;
        }

        cublasSscal(handle, localStride, &scalingFactor, scaleUpdateData[gpu].data(), 1);
        cublasSscal(handle, localStride, &scalingFactor, biasUpdateData[gpu].data(), 1);
    }

    for (size_t gpu = 0; gpu < NUM_GPUS; ++gpu) {
        cublasDestroy(handles[gpu]);
    }
}

MinMaxSpan calcMinXMaxXSpan(uint32_t id, uint32_t numprocs, uint32_t stride)
{
    uint64_t pos = (static_cast<uint64_t>(id) + 1) % numprocs;

    uint32_t minX = (stride * pos) / numprocs;

    uint32_t maxX = (stride * (pos + 1)) / numprocs;

    uint32_t span = maxX - minX;

    return { minX, maxX, span };
}

void copyData(float* src, float* dest, uint32_t offset, uint32_t stride, uint32_t span, uint32_t batch)
{
    kCopy2D(src + offset, stride, dest + offset, stride, span, batch);

    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);
}

void Layer::Reduce(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride, uint32_t updateCount) {
    if (NUM_GPUS <= 1) {
        return;
    }

    const uint32_t GPUId = getGpu()._id;

    std::vector<cudaStream_t> streams(NUM_GPUS);
    std::vector<cublasHandle_t> cublasHandles(NUM_GPUS);

#pragma omp parallel for
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cublasCreate(&cublasHandles[i]);
    }

    ncclComm_t ncclComm;
    ncclUniqueId ncclId;

    if (GPUId == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclCommInitRank(&ncclComm, NUM_GPUS, ncclId, GPUId);

    std::vector<MinMaxSpan> minMaxSpans(NUM_GPUS);

#pragma omp parallel for
    for (int i = 0; i < NUM_GPUS; i++) {
        minMaxSpans[i] = calcMinXMaxXSpan(i, NUM_GPUS, stride);
    }

    for (uint32_t i = 0; i < NUM_GPUS - 1; i++) {
#pragma omp parallel for
        for (int j = 0; j < NUM_GPUS; j++) {
            cudaSetDevice(j);
            cudaMemcpyAsync(pBuffer + minMaxSpans[j].minX, pBuffer + minMaxSpans[j].minX, minMaxSpans[j].span * static_cast<unsigned long long>(batch) * sizeof(float), cudaMemcpyDeviceToDevice, streams[j]);
        }

        ncclGroupStart();
#pragma omp parallel for
        for (int j = 0; j < NUM_GPUS; j++) {
            ncclAllReduce(pBuffer + minMaxSpans[j].minX, pBuffer + minMaxSpans[j].minX, minMaxSpans[j].span * static_cast<size_t>(batch), ncclFloat, ncclSum, ncclComm, streams[j]);
        }
        ncclGroupEnd();

#pragma omp parallel for
        for (int j = 0; j < NUM_GPUS; j++) {
            cublasSetStream(cublasHandles[j], streams[j]);
            float alpha = 1.0f;
            cublasSaxpy(cublasHandles[j], minMaxSpans[j].span * batch, &alpha, pBuffer + minMaxSpans[j].minX, 1, pBuffer, 1);
        }
    }

#pragma omp parallel for
    for (int j = 0; j < NUM_GPUS; j++) {
        cudaSetDevice(j);
        cudaStreamSynchronize(streams[j]);
    }

    ncclCommDestroy(ncclComm);

#pragma omp parallel for
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
        cublasDestroy(cublasHandles[i]);
    }
}


void CopyData(float* dest, uint32_t destStride, float* src, uint32_t srcStride, uint32_t span, uint32_t batch) {
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    kCopy2D(dest, destStride, src, srcStride, span, batch);
}

void Layer::Gather(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride)
{
    if (getGpu()._numprocs <= 1)
    {
        return;
    }

    const uint32_t stages = getGpu()._numprocs - 1;
    const uint64_t myPos = getGpu()._id;
    float* pSendBuffer = getGpu()._pNetwork->GetP2PSendBuffer();

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

#pragma omp parallel for
            for (int i = 0; i < static_cast<int>(span * batch); i++)
            {
                pSendBuffer[minX * batch + i] = pBuffer[i];
            }

            for (uint32_t i = 0; i < stages; i++)
            {
#pragma omp parallel for
                for (int j = 0; j < static_cast<int>(span * batch); j++)
                {
                    pPeerBuffer[minX * batch + j] = pSendBuffer[minX * batch + j];
                }

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

#pragma omp parallel for
            for (int i = 0; i < static_cast<int>(batch); i++)
            {
                for (int j = 0; j < static_cast<int>(localStride); j++)
                {
                    pCPUBuffer[(minX + j) * batch + i] = pBuffer[i * localStride + j];
                }
            }

            std::vector<int> sendCounts(getGpu()._numprocs, 0);
            std::vector<int> displacements(getGpu()._numprocs, 0);

#pragma omp parallel for
            for (int i = 0; i < static_cast<int>(getGpu()._numprocs); i++)
            {
                const uint32_t iMinX = (stride * static_cast<uint32_t>(i)) / getGpu()._numprocs;
                const uint32_t iMaxX = (stride * static_cast<uint32_t>(i + 1)) / getGpu()._numprocs;
                const uint32_t iSpan = iMaxX - iMinX;

                sendCounts[i] = iSpan * batch;
                displacements[i] = iMinX * batch;
            }

            MPI_Allgatherv(pCPUBuffer, sendCounts[getGpu()._id], MPI_FLOAT, pSendBuffer, sendCounts.data(), displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Layer::Gather: " << e.what();
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }
}

void Layer::Dump(std::string fname, float* pBuffer)
{
    std::vector<float> vData(_batch * _stride);

    if (getGpu()._numprocs == 1)
    {
        cudaMemcpy(vData.data(), pBuffer, _batch * _stride * sizeof(float), cudaMemcpyDefault);
    }
    else
    {
        if (getGpu()._id == 0)
        {
            float* pData = vData.data();
            cudaMemcpy2D(pData, _stride * sizeof(float), pBuffer, _localStride * sizeof(float), _localStride * sizeof(float), _batch, cudaMemcpyDefault);
            pData += _localStride;

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
            uint64_t size = _batch * _localStride;
            std::vector<float> vLocalData(size);
            cudaMemcpy(vLocalData.data(), pBuffer, size * sizeof(float), cudaMemcpyDefault);
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(vLocalData.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (getGpu()._id == 0)
    {
        std::ofstream outputFile(fname);
        float* pData = vData.data();

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

        outputFile.close();
    }
}

const std::map<Layer::Kind, std::string> Layer::_sKindMap = {
    {Layer::Kind::Input, "Input"},
    {Layer::Kind::Hidden, "Hidden"},
    {Layer::Kind::Output, "Output"},
    {Layer::Kind::Target, "Target"}
};

const std::map<Layer::Type, std::string> Layer::_sTypeMap = {
    {Layer::Type::FullyConnected, "FullyConnected"},
    {Layer::Type::Convolutional, "Convolutional"},
    {Layer::Type::Pooling, "Pooling"}
};

const std::map<Layer::Attributes, std::string> Layer::_sAttributesMap = {
    {Layer::Attributes::None, "None"},
    {Layer::Attributes::Sparse, "Sparse"},
    {Layer::Attributes::Denoising, "Denoising"},
    {Layer::Attributes::BatchNormal, "BatchNormalization"}
};

const std::map<Layer::Parallelization, std::string> Layer::_sParallelizationMap = {
    {Layer::Parallelization::Data, "Data"},
    {Layer::Parallelization::Model, "Model"},
    {Layer::Parallelization::Serial, "Serial"}
};

std::ostream& operator<< (std::ostream& out, Layer::Kind k) {
    out << Layer::_sKindMap.at(k);
    return out;
}

std::ostream& operator<< (std::ostream& out, Layer::Type t) {
    out << Layer::_sTypeMap.at(t);
    return out;
}

std::ostream& operator<< (std::ostream& out, Layer::Parallelization p) {
    out << Layer::_sParallelizationMap.at(p);
    return out;
}

LayerDescriptor::LayerDescriptor() :
    _kind(Layer::Kind::Hidden),
    _type(Layer::Type::FullyConnected),
    _poolingFunction(None),
    _Nx(1),
    _Ny(1),
    _Nz(1),
    _Nw(1),
    _dimensions(1),
    _bDimensionsProvided(true),
    _weightInit(Xavier),
    _weightInitScale((float)1.0),
    _biasInit((float)0.0),
    _kernelX(1),
    _kernelY(1),
    _kernelZ(1),
    _kernelStrideX(1),
    _kernelStrideY(1),
    _kernelStrideZ(1),
    _kernelPaddingX(0),
    _kernelPaddingY(0),
    _kernelPaddingZ(0),
    _kernelDimensions(1),
    _weightNorm((float)0.0),
    _deltaNorm((float)0.0),
    _pDropout((float)0.0),
    _activation(Activation::Sigmoid),
    _sparsenessPenalty_p((float)0.0),
    _sparsenessPenalty_beta((float)0.0),
    _RELUSlope(NAN),
    _ELUAlpha(NAN),
    _SELULambda(NAN),
    _attributes(Layer::Attributes::None)
{
}

bool LoadLayerDescriptorNetCDF(const std::string& fname, netCDF::NcFile& nc, uint32_t index, LayerDescriptor& ld) {
    if (getGpu()._id != 0) {
        return true;
    }

    std::string lstring = "layer" + std::to_string(index) + "_";

    auto checkAttribute = [&nc, &fname, &lstring](const std::string& attributeName, auto& value) {
        try {
            auto attribute = nc.getAtt(lstring + attributeName);
            if (!attribute.isNull()) {
                attribute.getValues(&value);
            }
            else {
                std::cerr << "NcException Layer::Layer: No " << attributeName << " supplied in NetCDF input file " << fname;
            }
        }
        catch (const netCDF::exceptions::NcException& e) {
            std::cerr << "NcException Layer::Layer: " << e.what();
        }
        };

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

    auto checkSourcesOrSkips = [&nc, &fname, &lstring](const std::string& attributeName, std::vector<std::string>& vec) {
        uint32_t count = 0;
        try {
            auto att = nc.getAtt(lstring + attributeName);
            if (!att.isNull()) {
                att.getValues(&count);
                for (uint32_t i = 0; i < count; i++) {
                    auto nstring = std::to_string(i);
                    auto sourceAtt = nc.getAtt(lstring + attributeName + nstring);
                    if (!sourceAtt.isNull()) {
                        std::string source;
                        sourceAtt.getValues(source);
                        vec.push_back(source);
                    }
                    else {
                        std::cerr << "NcException Layer::Layer: No " << attributeName << " attributes supplied in NetCDF input file " << fname;
                    }
                }
            }
            else {
                std::cerr << "NcException Layer::Layer: No " << attributeName << " supplied in NetCDF input file " << fname;
            }
        }
        catch (const netCDF::exceptions::NcException& e) {
            std::cerr << "NcException Layer::Layer: " << e.what();
        }
        };

    checkSourcesOrSkips("sources", ld._vSource);
    checkSourcesOrSkips("skips", ld._vSkip);

    return true;
}

std::ostream& operator<<(std::ostream& out, const LayerDescriptor& d)
{
    out << std::left << std::setw(20) << "Name:" << d._name << '\n'
        << std::setw(20) << "Kind:" << d._kind << '\n'
        << std::setw(20) << "Type:" << d._type << '\n';

    if (d._type != Layer::Type::Pooling)
        out << std::setw(20) << "Pooling Function:" << d._poolingFunction << '\n';

    out << std::setw(20) << "Nx:" << d._Nx << '\n'
        << std::setw(20) << "Ny:" << d._Ny << '\n'
        << std::setw(20) << "Nz:" << d._Nz << '\n'
        << std::setw(20) << "Nw:" << d._Nw << '\n';

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

        if (d._type == Layer::Type::FullyConnected)
        {
            if (d._sparsenessPenalty_p > 0.0)
                out << std::setw(20) << "sparsenessPenalty_p:" << d._sparsenessPenalty_p << '\n';
            if (d._sparsenessPenalty_beta > 0.0)
                out << std::setw(20) << "sparsenessPenalty_beta:" << d._sparsenessPenalty_beta << '\n';
        }

        if (d._kind != Layer::Kind::Hidden)
            out << std::setw(20) << "DataSet:" << d._dataSet << '\n';
    }

    for (size_t i = 0; i < d._vSource.size(); i++)
    {
        out << "source " << std::setw(3) << i << ":" << d._vSource[i] << '\n';
    }

    for (size_t i = 0; i < d._vSkip.size(); i++)
    {
        out << "skip " << std::setw(3) << i << ":" << d._vSkip[i] << '\n';
    }

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

    size_t size = d._vSource.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSource.resize(size);

    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSource[i]);

    size = d._vSkip.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSkip.resize(size);

    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSkip[i]);

    return 0;
}

bool Layer::WriteNetCDF(netCDF::NcFile& nc, uint32_t index)
{
    if (auto& gpu = getGpu(); gpu._id != 0)
    {
        return false;
    }

    std::string lstring = "layer" + std::to_string(index) + "_";

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

    uint32_t attributes = 0;
    if (_bSparse)
        attributes |= Layer::Attributes::Sparse;
    if (_bDenoising)
        attributes |= Layer::Attributes::Denoising;
    if (_bBatchNormalization)
        attributes |= Layer::Attributes::BatchNormal;
    nc.putAtt(lstring + "attributes", netCDF::ncUint, attributes);

    nc.putAtt(lstring + "sources", netCDF::ncUint, static_cast<uint32_t>(_vSource.size()));

    for (size_t i = 0; i < _vSource.size(); i++)
    {
        std::string nstring = std::to_string(i);
        nc.putAtt(lstring + "source" + nstring, _vSource[i]);
    }

    nc.putAtt(lstring + "skips", netCDF::ncUint, static_cast<uint32_t>(_vSkip.size()));

    for (size_t i = 0; i < _vSkip.size(); i++)
    {
        std::string nstring = std::to_string(i);
        nc.putAtt(lstring + "skip" + nstring, _vSkip[i]);
    }

    if (_bBatchNormalization)
    {
        std::vector<float> bndata(_strideBN);
        size_t bytes = _strideBN * sizeof(float);

        netCDF::NcDim bnDim = nc.addDim(lstring + "bnDim", _strideBN);

        cudaMemcpy(bndata.data(), _pbScaleBN->_pDevData, bytes, cudaMemcpyDeviceToHost);

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

    return true;
}

void Layer::NamedEntityRecognition(uint32_t batch, uint32_t sequenceLength, uint32_t numEntities)
{
    auto& gpu = getGpu();

    if (gpu._id != 0)
    {
        throw std::runtime_error("GPU ID is not 0. Cannot proceed.");
    }

    std::vector<float> vData(batch * sequenceLength * numEntities);

    cudaMemcpy(vData.data(), _pbUnit->_pDevData, vData.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < vData.size(); ++i)
    {
        vData[i] = static_cast<float>(i % 2);
    }

    std::vector<std::string> vNames(numEntities);

    for (uint32_t i = 0; i < numEntities; i++)
    {
        vNames[i] = "Entity_" + std::to_string(i);
    }

    netCDF::NcGroup group;

    std::for_each(vNames.begin(), vNames.end(), [&group](const std::string& name) {
        group.putAtt(name, name);
        });

#pragma omp parallel for
    for (int i = 0; i < batch; i++)
    {
        for (uint32_t j = 0; j < sequenceLength; j++)
        {
            for (uint32_t k = 0; k < numEntities; k++)
            {
                uint32_t index = i * sequenceLength * numEntities + j * numEntities + k;

                if (vData[index] > 0.5)
                {
                    std::string entityName = vNames[k];
#pragma omp critical
                    {
                        std::cout << entityName << " ";
                    }
                }
            }
        }

#pragma omp critical
        {
            std::cout << std::endl;
        }
    }
}
