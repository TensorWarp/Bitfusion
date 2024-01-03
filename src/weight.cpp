#include "gpuTypes.h"
#include "types.h"
#include "kernels.cuh"
#define __STDC_FORMAT_MACROS
#include <future>
#include "threadPool.h"
#include <optional>
#include <cublas_v2.h>
#include <format>
#include <iostream>
#include <filesystem>
#include "Layer.h"
#include "Network.h"
#include "Weight.h"
#include "mpi.h"
#include "ncChar.h"
#include "ncDim.h"
#include "ncException.h"
#include "ncFile.h"
#include "ncFloat.h"
#include "ncGroupAtt.h"
#include "ncInt.h"
#include "ncVar.h"
#include <assert.h>
#include <cinttypes>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <cublas_api.h>
#include <cuda_runtime_api.h>
#include <cudnn_cnn_infer.h>
#include <cudnn_cnn_train.h>
#include <cudnn_ops_infer.h>
#include <curand.h>
#include <driver_types.h>
#include <fstream>
#include <filesystem>
#include <format>
#include <cstdlib>
#include <time.h>
#include <array>
#include <span>
#include <utility>
#include "enum.h"

namespace fs = std::filesystem;

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
{}

static [[maybe_unused]] void DumpTensor(cudnnTensorDescriptor_t t)
{
    cudnnDataType_t dataType;
    int ndims;
    std::vector<int> vDim(16);
    std::vector<int> vStride(16);
    cudnnStatus_t cudnnStatus = cudnnGetTensorNdDescriptor(t, 8, &dataType, &ndims, vDim.data(), vStride.data());

    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cudnnGetTensorNdDescriptor error";
        return;
    }

    std::cout << "Tensor:   " << ndims << " dimensions";
    std::cout << "DataType: " << dataType;

    for (int i = 0; i < ndims; i++)
        std::cout << i << " " << vDim[i] << " " << vStride[i];

    std::cout << "";
}

void DumpFilter(cudnnFilterDescriptor_t f) {
    cudnnDataType_t dataType;
    cudnnTensorFormat_t format;
    int ndims;
    std::vector<int> vDim(16);
    cudnnStatus_t cudnnStatus = cudnnGetFilterNdDescriptor(f, 5, &dataType, &format, &ndims, vDim.data());

    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cudnnGetFilterNdDescriptor error";
        return;
    }

    std::cout << "Filter:   " << ndims << " dimensions";
    std::cout << "DataType: " << static_cast<int>(dataType);
    std::cout << "Format:   " << static_cast<int>(format);

    for (int i = 0; i < ndims; i++) {
        std::cout << i << " " << vDim[i];
    }

    std::cout << "";
}

static [[maybe_unused]] void DumpConvolution(cudnnConvolutionDescriptor_t c)
{
    cudnnDataType_t dataType;
    cudnnConvolutionMode_t mode;
    int ndims;
    std::vector<int> vPad(16);
    std::vector<int> vStride(16);
    std::vector<int> vUpscale(16);
    cudnnStatus_t cudnnStatus = cudnnGetConvolutionNdDescriptor(c, 5, &ndims, vPad.data(), vStride.data(), vUpscale.data(), &mode, &dataType);

    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cudnnGetConvolutionNdDescriptor error";
        return;
    }

    std::cout << "Convolution:   " << ndims << " dimensions";
    std::cout << "DataType:      " << dataType;
    std::cout << "Mode:          " << mode;

    for (int i = 0; i < ndims; i++)
        std::cout << i << " " << vPad[i] << " " << vStride[i] << " " << vUpscale[i];

    std::cout << "";
}

template <typename T>
std::optional<T> GetAttribute(netCDF::NcFile& nc, const std::string& attrName) {
    try {
        netCDF::NcGroupAtt attribute = nc.getAtt(attrName);
        if (!attribute.isNull()) {
            T value{};
            attribute.getValues(&value);
            return std::make_optional(value);
        }
    }
    catch (const netCDF::exceptions::NcException& e) {
        std::cerr << "Exception: " << e.what();
    }

    return std::nullopt;
}

bool LoadWeightDescriptorNetCDF(const std::string& fname, netCDF::NcFile& nc, uint32_t index, WeightDescriptor& wd) {
    if (getGpu()._id != 0) {
        std::cerr << "Exception: Missing required weight attributes.";
        return false;
    }

    const std::string wstring = "weight" + std::to_string(index) + "_";

    auto getAttribute = [&](const std::string& attrName) -> std::optional<std::string> {
        auto attrOpt = GetAttribute<std::string>(nc, attrName);
        if (!attrOpt) {
            std::cerr << "Exception: Missing " << attrName << " attribute.";
        }
        return attrOpt;
        };

    std::optional<std::string> inputLayerOpt = getAttribute(wstring + "inputLayer");
    std::optional<std::string> outputLayerOpt = getAttribute(wstring + "outputLayer");
    std::optional<float> normOpt = GetAttribute<float>(nc, wstring + "norm");
    std::optional<int> bSharedOpt = GetAttribute<int>(nc, wstring + "bShared");
    std::optional<int> bLockedOpt = GetAttribute<int>(nc, wstring + "bLocked");
    std::optional<int> widthOpt = GetAttribute<int>(nc, wstring + "width");
    std::optional<int> heightOpt = GetAttribute<int>(nc, wstring + "height");
    std::optional<int> lengthOpt = GetAttribute<int>(nc, wstring + "length");
    std::optional<int> depthOpt = GetAttribute<int>(nc, wstring + "depth");
    std::optional<int> breadthOpt = GetAttribute<int>(nc, wstring + "breadth");

    if (!inputLayerOpt || !outputLayerOpt || !normOpt || !bSharedOpt || !bLockedOpt || !widthOpt ||
        !heightOpt || !lengthOpt || !depthOpt || !breadthOpt) {
        return false;
    }

    wd._inputLayer = *inputLayerOpt;
    wd._outputLayer = *outputLayerOpt;
    wd._norm = *normOpt;
    wd._bShared = (*bSharedOpt != 0);
    wd._bLocked = *bLockedOpt;
    wd._width = *widthOpt;
    wd._height = *heightOpt;
    wd._length = *lengthOpt;
    wd._depth = *depthOpt;
    wd._breadth = *breadthOpt;

    if (wd._bShared) {
        std::optional<std::string> sourceInputLayerOpt = getAttribute(wstring + "sourceInputLayer");
        std::optional<std::string> sourceOutputLayerOpt = getAttribute(wstring + "sourceOutputLayer");
        std::optional<int> bTransposedOpt = GetAttribute<int>(nc, wstring + "bTransposed");

        if (!sourceInputLayerOpt || !sourceOutputLayerOpt || !bTransposedOpt) {
            std::cerr << "Exception: Missing shared weight attributes.";
            return false;
        }

        wd._sourceInputLayer = *sourceInputLayerOpt;
        wd._sourceOutputLayer = *sourceOutputLayerOpt;
        wd._bTransposed = (*bTransposedOpt != 0);
    }

    netCDF::NcDim biasDim = nc.getDim(wstring + "biasDim");
    netCDF::NcVar biasVar = nc.getVar(wstring + "bias");
    wd._vBias.resize(biasDim.getSize());
    biasVar.getVar(wd._vBias.data());

    if (!wd._bShared) {
        netCDF::NcDim weightDim = nc.getDim(wstring + "weightDim");
        netCDF::NcVar weightVar = nc.getVar(wstring + "weights");
        wd._vWeight.resize(weightDim.getSize());
        weightVar.getVar(wd._vWeight.data());
    }

    return true;
}

template <typename T>
void safe_MPI_Bcast(T* data, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    MPI_Bcast(data, count, datatype, root, comm);
}

template <typename T>
void safe_MPI_Bcast(std::vector<T>& data, int root, MPI_Comm comm) {
    uint64_t size = data.size();

    safe_MPI_Bcast(&size, 1, MPI_UINT64_T, root, comm);

    data.resize(size);

    safe_MPI_Bcast(data.data(), size, MPI_FLOAT, root, comm);
}

int MPI_Bcast_WeightDescriptor(std::shared_ptr<WeightDescriptor> d) {
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

        safe_MPI_Bcast(d->_vWeight, 0, MPI_COMM_WORLD);
        safe_MPI_Bcast(d->_vBias, 0, MPI_COMM_WORLD);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in MPI_Bcast_WeightDescriptor: " << e.what() << '\n';
        return 1;
    }
}

std::ostream& operator<<(std::ostream& o, const WeightDescriptor& d)
{
    if (getGpu()._id != 0) return o;

    auto p = [&](const std::string& label, auto& field) {
        o << std::left << std::setw(20) << label << field << '\n';
        };

    o << std::boolalpha;

    p("Input Layer:", d._inputLayer);
    p("Output Layer:", d._outputLayer);
    p("Width:", d._width);
    p("Height:", d._height);
    p("Length:", d._length);
    p("Depth:", d._depth);
    p("Breadth:", d._breadth);

    if (d._bShared) {
        p("Source Input Layer:", d._sourceInputLayer);
        p("Source Output Layer:", d._sourceOutputLayer);
    }

    p("Shared:", d._bShared);
    p("Transposed:", d._bTransposed);
    p("Locked:", d._bLocked);
    p("Norm:", d._norm);

    return o;
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

    std::cout << "Weight::initializeLayers: Registered weight between layers " << _inputLayer._name << " and " << _outputLayer._name;
}

void Weight::initializeConvolution()
{
    _transform = Convolution;

    auto checkCudnnError = [](cudnnStatus_t status, const std::string& message) {
        if (status != CUDNN_STATUS_SUCCESS) {
            std::ostringstream errMsg;
            errMsg << "Weight::initializeConvolution: " << message;
            throw std::runtime_error(errMsg.str());
        }
        };

    checkCudnnError(cudnnCreateTensorDescriptor(&_convBiasTensor), "Unable to create tensor descriptor");
    checkCudnnError(cudnnCreateFilterDescriptor(&_convFilterDesc), "Unable to create filter descriptor");
    checkCudnnError(cudnnCreateConvolutionDescriptor(&_convDesc), "Unable to create convolution descriptor");

    std::vector<int> vFilterDim(5, 1);

    switch (_outputLayer._dimensions) {
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

    checkCudnnError(cudnnSetFilterNdDescriptor(_convFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, _outputLayer._dimensions + 1, vFilterDim.data()), "Unable to set filter descriptor");

    std::tie(_width, _height, _length, _depth, _breadth) = std::make_tuple(vFilterDim[0], vFilterDim[1], vFilterDim[2], vFilterDim[3], vFilterDim[4]);

    std::vector<int> vConvPad(3, 0);
    std::vector<int> vConvStride(3, 1);
    std::vector<int> vConvUpscale(3, 1);

    switch (_outputLayer._dimensions) {
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

    checkCudnnError(cudnnSetConvolutionNdDescriptor(_convDesc, _outputLayer._kernelDimensions, vConvPad.data(), vConvStride.data(), vConvUpscale.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "cudnnSetConvolutionNdDescriptor failed.");

    std::vector<int> vBiasDim(5, 1);
    std::vector<int> vBiasStride(5, 1);

    vBiasDim[1] = vFilterDim[0];

    checkCudnnError(cudnnSetTensorNdDescriptor(_convBiasTensor, CUDNN_DATA_FLOAT, _outputLayer._dimensions + 1, vBiasDim.data(), vBiasStride.data()), "Unable to set bias tensor descriptor");

    _size = static_cast<uint64_t>(vFilterDim[0]) * static_cast<uint64_t>(vFilterDim[1]) * static_cast<uint64_t>(_outputLayer._kernelX) * static_cast<uint64_t>(_outputLayer._kernelY) * static_cast<uint64_t>(_outputLayer._kernelZ);
    _biasSize = vFilterDim[0];
    _localSize = _size;
    _localBiasSize = _biasSize;

    if (getGpu()._id == 0) {
        std::cout << "Weight::initializeConvolution: Allocating " << _localSize * sizeof(float)
            << " bytes (" << vFilterDim[0] << " x " << vFilterDim[1] << " x " << _outputLayer._kernelX;
        if (_outputLayer._dimensions >= 3)
            std::cout << " x " << _outputLayer._kernelY;
        if (_outputLayer._dimensions >= 4)
            std::cout << " x " << _outputLayer._kernelZ;
        std::cout << ") for convolutional weights between layers " << _inputLayer._name << " and " << _outputLayer._name << std::endl;
    }
}

void Weight::initializeLinear()
{
    _transform = Linear;

    uint32_t outgoingSize = _outputLayer._stride * 3;
    uint32_t incomingSize = _inputLayer._stride * 2;

    if (outgoingSize > incomingSize)
    {
        _inputLayer._vOutgoingLargerLayer.push_back(&_outputLayer);
        _inputLayer._vOutgoingLargerWeight.push_back(this);
        _width = _outputLayer._localStride;
        _height = _inputLayer._stride;
    }
    else
    {
        _outputLayer._vIncomingLargerLayer.push_back(&_inputLayer);
        _outputLayer._vIncomingLargerWeight.push_back(this);
        _width = _outputLayer._stride;
        _height = _inputLayer._localStride;
    }

    _localSize = _width * _height * 4 * 2 * 3;
    _localBiasSize = _outputLayer._localStride;
    _size = static_cast<uint64_t>(_outputLayer._stride) * _inputLayer._stride * 4 * 2 * 3;
    _biasSize = _outputLayer._stride;

    std::cout << "Weight::initializeLinear: Allocating " << _localSize * sizeof(float)
        << " bytes (" << _width << ", " << _height << ") for fully connected weights between layers "
        << _inputLayer._name << " and " << _outputLayer._name;
}

void Weight::setWeightValues(const std::vector<std::vector<float>>& values)
{
    if (values.size() != _width || values[0].size() != _height)
    {
        std::cerr << "Error: Invalid weight matrix dimensions." << std::endl;
        return;
    }

    _weightMatrix = values;
}

void Weight::randomizeWeightMatrix()
{
    _weightMatrix.resize(_width, std::vector<float>(_height));

    for (uint32_t i = 0; i < _width; ++i)
    {
        for (uint32_t j = 0; j < _height; ++j)
        {
            _weightMatrix[i][j] = static_cast<float>(rand()) / (RAND_MAX)-0.5f;
        }
    }
}

Weight::~Weight()
{}

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

void Weight::Randomize() const
{
    try
    {
        if (!_bShared)
        {
            const int numStreams = 2;

            cublasHandle_t handles[numStreams]{};
            curandGenerator_t generators[numStreams]{};

            for (int i = 0; i < numStreams; ++i)
            {
                cublasStatus_t cublasStatus = cublasCreate(&handles[i]);
                if (cublasStatus != CUBLAS_STATUS_SUCCESS)
                {
                    throw std::runtime_error("cuBLAS initialization failed.");
                }

                curandStatus_t curandStatus = curandCreateGenerator(&generators[i], CURAND_RNG_PSEUDO_DEFAULT);
                if (curandStatus != CURAND_STATUS_SUCCESS)
                {
                    throw std::runtime_error("cuRAND generator initialization failed.");
                }

                curandStatus = curandSetPseudoRandomGeneratorSeed(generators[i], time(NULL) + i);
                if (curandStatus != CURAND_STATUS_SUCCESS)
                {
                    throw std::runtime_error("cuRAND seed setting failed.");
                }
            }

            cudaStream_t streams[numStreams]{};
            for (int i = 0; i < numStreams; ++i)
            {
                cudaStreamCreate(&streams[i]);
            }

            std::thread weightThread([&]() {
                int streamIndex = 0;
                cublasSetStream(handles[streamIndex], streams[streamIndex]);
                curandSetStream(generators[streamIndex], streams[streamIndex]);

                if (_outputLayer._weightInit == CaffeXavier)
                {
                    std::cout << "Initializing weights using CaffeXavier method.";
                    curandGenerateUniform(generators[streamIndex], _pbWeight->_pDevData, _localSize);
                    float scale = _outputLayer._weightInitScale * 2.0f * sqrtf(3.0f / _outputLayer._stride);
                    float bias = 0.5f * scale;
                    kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
                }
                else if (_outputLayer._weightInit == Xavier)
                {
                    std::cout << "Initializing weights using Xavier method.";
                    curandGenerateUniform(generators[streamIndex], _pbWeight->_pDevData, _localSize);
                    float scale = _outputLayer._weightInitScale * sqrtf(6.0f / (_outputLayer._stride + _inputLayer._stride));
                    float bias = 0.5f * scale;
                    kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
                }
                else if (_outputLayer._weightInit == Uniform)
                {
                    std::cout << "Initializing weights using Uniform method.";
                    curandGenerateUniform(generators[streamIndex], _pbWeight->_pDevData, _localSize);
                    float scale = 2.0f * _outputLayer._weightInitScale;
                    float bias = 0.5f * scale;
                    kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
                }
                else if (_outputLayer._weightInit == Gaussian)
                {
                    std::cout << "Initializing weights using Gaussian method.";
                    curandGenerateNormal(generators[streamIndex], _pbWeight->_pDevData, _localSize, 0.0f, _outputLayer._weightInitScale);
                }
                else if (_outputLayer._weightInit == UnitBall)
                {
                    std::cout << "Initializing weights using UnitBall method.";
                    curandGenerateUniform(generators[streamIndex], _pbWeight->_pDevData, _localSize);
                    float scale = _outputLayer._weightInitScale;
                    kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, 0.0f);
                }
                else if (_outputLayer._weightInit == SELU)
                {
                    std::cout << "Initializing weights using SELU method.";
                    curandGenerateNormal(generators[streamIndex], _pbWeight->_pDevData, _localSize, 0.0f, 1.0f / _inputLayer._stride);
                }
                else if (_outputLayer._weightInit == Constant)
                {
                    std::cout << "Initializing weights using Constant method.";
                    cudaMemset(_pbWeight->_pDevData, 0, _localSize * sizeof(float));
                    float scale = _outputLayer._weightInitScale;
                    kScaleAndBias(_pbWeight->_pDevData, _localSize, (float)0.0, scale);
                }
                });

            std::thread biasThread([&]() {
                int streamIndex = 1;
                cublasSetStream(handles[streamIndex], streams[streamIndex]);
                curandSetStream(generators[streamIndex], streams[streamIndex]);

                if (static_cast<int>(_outputLayer._weightInit) == static_cast<int>(CaffeXavier))
                {
                    std::cout << "Initializing biases using CaffeXavier method.";
                    curandGenerateUniform(generators[streamIndex], _pbBias->_pDevData, _localBiasSize);
                    float scale = _outputLayer._biasInit * 2.0f * sqrtf(3.0f / _outputLayer._stride);
                    float bias = 0.5f * scale;
                    kScaleAndBias(_pbBias->_pDevData, _localBiasSize, scale, bias);
                }
                else if (static_cast<int>(_outputLayer._biasInit) == static_cast<int>(Xavier))
                {
                    std::cout << "Initializing biases using Xavier method.";
                    curandGenerateUniform(generators[streamIndex], _pbBias->_pDevData, _localBiasSize);
                    float scale = _outputLayer._biasInit * sqrtf(6.0f / (_outputLayer._stride + _inputLayer._stride));
                    float bias = 0.5f * scale;
                    kScaleAndBias(_pbBias->_pDevData, _localBiasSize, scale, bias);
                }
                else if (static_cast<int>(_outputLayer._biasInit) == static_cast<int>(Uniform))
                {
                    std::cout << "Initializing biases using Uniform method.";
                    curandGenerateUniform(generators[streamIndex], _pbBias->_pDevData, _localBiasSize);
                    float scale = 2.0f * _outputLayer._biasInit;
                    float bias = 0.5f * scale;
                    kScaleAndBias(_pbBias->_pDevData, _localBiasSize, scale, bias);
                }
                else if (static_cast<int>(_outputLayer._biasInit) == static_cast<int>(Gaussian))
                {
                    std::cout << "Initializing biases using Gaussian method.";
                    curandGenerateNormal(generators[streamIndex], _pbBias->_pDevData, _localBiasSize, 0.0f, _outputLayer._biasInit);
                }
                else if (static_cast<int>(_outputLayer._biasInit) == static_cast<int>(UnitBall))
                {
                    std::cout << "Initializing biases using UnitBall method.";
                    curandGenerateUniform(generators[streamIndex], _pbBias->_pDevData, _localBiasSize);
                    float scale = _outputLayer._biasInit;
                    kScaleAndBias(_pbBias->_pDevData, _localBiasSize, scale, 0.0f);
                }
                else if (static_cast<int>(_outputLayer._biasInit) == static_cast<int>(SELU))
                {
                    std::cout << "Initializing biases using SELU method.";
                    curandGenerateNormal(generators[streamIndex], _pbBias->_pDevData, _localBiasSize, 0.0f, 1.0f / _inputLayer._stride);
                }
                else if (static_cast<int>(_outputLayer._biasInit) == static_cast<int>(Constant))
                {
                    std::cout << "Initializing biases using Constant method.";
                    cudaMemset(_pbBias->_pDevData, 0, _localBiasSize * sizeof(float));
                }
                });

            weightThread.join();
            biasThread.join();

            for (int i = 0; i < numStreams; ++i)
            {
                cublasDestroy(handles[i]);
                curandDestroyGenerator(generators[i]);
                cudaStreamDestroy(streams[i]);
            }
        }

        std::cout << "Weight and bias initialization completed.";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

void Weight::Lock()
{
    _bLocked = true;
}

void Weight::Unlock()
{
    _bLocked = false;
}

template<typename T>
concept GpuBufferType = std::is_same_v<T, GpuBuffer<float>>;

template<typename BufferType>
void ResetBufferIfNeeded(std::unique_ptr<BufferType>& buffer, size_t size)
{
    if (!buffer)
        buffer = std::make_unique<BufferType>(size);
}

template <TrainingMode Mode, typename BufferType>
void ResetBufferBasedOnMode(std::unique_ptr<BufferType>& buffer, size_t size)
{
    if constexpr (Mode != TrainingMode::SGD)
    {
        ResetBufferIfNeeded(buffer, size);
        std::cout << "Buffer reset.";
    }
    else
    {
        buffer.reset();
        std::cout << "Buffer reset for SGD mode.";
    }
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
        
        ResetBufferBasedOnMode<TrainingMode::AdaDelta, GpuBuffer<float>>(_pbWeightGradientVelocity, _localSize);
        ResetBufferBasedOnMode<TrainingMode::AdaDelta, GpuBuffer<float>>(_pbBiasGradientVelocity, _localBiasSize);
        ResetBufferBasedOnMode<TrainingMode::Adam, GpuBuffer<float>>(_pbWeightGradientVelocity, _localSize);
        ResetBufferBasedOnMode<TrainingMode::Adam, GpuBuffer<float>>(_pbBiasGradientVelocity, _localBiasSize);
        ResetBufferBasedOnMode<TrainingMode::SGD, GpuBuffer<float>>(_pbWeightVelocity, _localSize);
        ResetBufferBasedOnMode<TrainingMode::SGD, GpuBuffer<float>>(_pbBiasVelocity, _localBiasSize);

        };

    auto cudnnProcessing = [this, pNetwork]() {
        
        if (_outputLayer._type == Layer::Type::Convolutional)
        {
            std::cout << "Getting algorithm between " << _inputLayer._name << " and " << _outputLayer._name << "\n";
            size_t workspaceSize;
            cudnnConvolutionFwdAlgoPerf_t convolutionAlgo;
            auto cudnnStatus = cudnnGetConvolutionForwardAlgorithm_v7(getGpu()._cuDNNHandle, _inputLayer._tensorDescriptor, _convFilterDesc, _convDesc, _outputLayer._tensorDescriptor, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, 0, &convolutionAlgo);
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                std::cerr << "Weight::Refresh: cudnnGetConvolutionForwardAlgorithm failed." << std::endl;
                exit(EXIT_FAILURE);
            }
            auto _convFWAlgo = convolutionAlgo.algo;
            cudnnStatus = cudnnGetConvolutionForwardWorkspaceSize(getGpu()._cuDNNHandle, _inputLayer._tensorDescriptor, _convFilterDesc, _convDesc, _outputLayer._tensorDescriptor, _convFWAlgo, &workspaceSize);
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                std::cerr << "Weight::Refresh: cudnnGetConvolutionForwardWorkspaceSize failed." << std::endl;
                exit(EXIT_FAILURE);
            }
            pNetwork->SetCUDNNWorkspace(workspaceSize);
            cudnnConvolutionBwdFilterAlgoPerf_t _convBWWeightAlgoPerf;
            cudnnStatus = cudnnGetConvolutionBackwardFilterAlgorithm_v7(getGpu()._cuDNNHandle, _inputLayer._tensorDescriptor, _outputLayer._tensorDescriptor, _convDesc, _convFilterDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, 0, &_convBWWeightAlgoPerf);
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                std::cerr << "Weight::Refresh: cudnnGetConvolutionBackwardFilterAlgorithm failed." << std::endl;
                exit(EXIT_FAILURE);
            }
            auto _convBWWeightAlgo = _convBWWeightAlgoPerf.algo;
            cudnnStatus = cudnnGetConvolutionBackwardFilterWorkspaceSize(getGpu()._cuDNNHandle, _inputLayer._tensorDescriptor, _outputLayer._tensorDescriptor, _convDesc, _convFilterDesc, _convBWWeightAlgo, &workspaceSize);
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                std::cerr << "Weight::Refresh: cudnnGetConvolutionBackwardFilterWorkspaceSize failed." << std::endl;
                exit(EXIT_FAILURE);
            }
            pNetwork->SetCUDNNWorkspace(workspaceSize);
            cudnnConvolutionBwdDataAlgoPerf_t perfData;
            cudnnStatus = cudnnGetConvolutionBackwardDataAlgorithm_v7(getGpu()._cuDNNHandle, _convFilterDesc, _outputLayer._tensorDescriptor, _convDesc, _inputLayer._tensorDescriptor, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, 0, &perfData);
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                std::cerr << "Weight::Refresh: cudnnGetConvolutionBackwardDataAlgorithm failed." << std::endl;
                exit(EXIT_FAILURE);
            }
            cudnnStatus = cudnnGetConvolutionBackwardDataWorkspaceSize(getGpu()._cuDNNHandle, _convFilterDesc, _outputLayer._tensorDescriptor, _convDesc, _inputLayer._tensorDescriptor, _convBWDeltaAlgo, &workspaceSize);
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                std::cerr << "Weight::Refresh: cudnnGetConvolutionBackwardDataWorkspaceSize failed." << std::endl;
                exit(EXIT_FAILURE);
            }
            pNetwork->SetCUDNNWorkspace(workspaceSize);
            std::vector<int> vOutputDim(8, 1);
            cudnnStatus = cudnnGetConvolutionNdForwardOutputDim(_convDesc, _inputLayer._tensorDescriptor, _convFilterDesc, _outputLayer._dimensions + 1, vOutputDim.data());
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                std::cerr << "Weight::Refresh: cudnnGetConvolutionNdForwardOutputDim failed." << std::endl;
                exit(EXIT_FAILURE);
            }
            size_t dim = std::accumulate(vOutputDim.begin(), vOutputDim.end(), 1, std::multiplies<int>());
            if (dim != static_cast<unsigned long long>(_outputLayer._maxLocalStride) * _outputLayer._localBatch)
            {
                if (getGpu()._id == 0)
                    std::cerr << "Output layer " << _outputLayer._name << ": has incorrectly calculated dimensions for cuDNN." << std::endl;
                getGpu().Shutdown();
                exit(EXIT_FAILURE);
            }
        }
    };

    auto monitorGPUMemory = [&]() {
        std::cout << "Monitoring GPU memory...";
        bool memoryOk = true;
        if (!memoryOk) {
            std::cerr << "GPU memory is not OK.";
            getGpu().Shutdown();
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
        std::cout << "Waiting for GPU readiness...";
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
        return invokeRegularizationError(lambda, lambda1, _pbWeight->_pDevData, _localSize);
}

void Weight::UpdateWeights(TrainingMode trainingMode)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    if (_bLocked)
        return; 

    if (!_bShared)
    {
        switch (trainingMode)
        {
        case SGD:
        {
            const float alpha = 0.01f;

            float* weightGradientDev = _pbWeightGradient->_pDevData;
            float* weightDev = _pbWeight->_pDevData;

            int localSize = _localSize;

            float* deltaWeightDev = nullptr;
            cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

            cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, deltaWeightDev, 1);

            cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

            cudaFree(deltaWeightDev);
        }
        break;

        case Momentum:
        {
            float alpha = 0.01f;

            float mu = 0.9f;

            float* weightVelocityDev = _pbWeightVelocity->_pDevData;
            float* weightGradientDev = _pbWeightGradient->_pDevData;
            float* weightDev = _pbWeight->_pDevData;

            int localSize = _localSize;

            float* deltaWeightDev = nullptr;

            cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

            cublasSscal(handle, localSize, &mu, weightVelocityDev, 1);

            cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, weightVelocityDev, 1);

            cublasSaxpy(handle, localSize, &alpha, weightVelocityDev, 1, deltaWeightDev, 1);

            cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

            cudaFree(deltaWeightDev);
            }
            break;
                        
            case AdaGrad:
            {
                const float alpha = 0.01f;

                float* weightVelocityDev = _pbWeightVelocity->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightDev = _pbWeight->_pDevData;

                int localSize = _localSize;

                float* deltaWeightDev = nullptr;
                cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

                cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, weightVelocityDev, 1);

                cublasSaxpy(handle, localSize, &alpha, weightVelocityDev, 1, deltaWeightDev, 1);

                cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

                cudaFree(deltaWeightDev);
            }
            break;
                        
            case Nesterov:
            {
                const float alpha = 0.01f;
                const float mu = 0.9f;

                float* weightVelocityDev = _pbWeightVelocity->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightDev = _pbWeight->_pDevData;

                int localSize = _localSize;

                float* deltaWeightDev = nullptr;
                cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

                cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, weightVelocityDev, 1);

                cublasSscal(handle, localSize, &mu, weightVelocityDev, 1);
                cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, weightVelocityDev, 1);

                cublasSaxpy(handle, localSize, &alpha, weightVelocityDev, 1, deltaWeightDev, 1);

                cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

                cudaFree(deltaWeightDev);
            }
            break;
                        
            case RMSProp:
            {
                const float alpha = 0.01f;
                const float mu = 0.9f;
                const float epsilon = 1e-7f;


                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightDev = _pbWeight->_pDevData;

                int localSize = _localSize;

                float* deltaWeightDev = nullptr;
                cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

                float squaredGradient = 0.0f;
                cublasSdot(handle, localSize, weightGradientDev, 1, weightGradientDev, 1, &squaredGradient);

                float squaredGradientAvg = 0.0f;
                squaredGradientAvg = mu * squaredGradientAvg + (1.0f - mu) * squaredGradient;

                float scalingFactor = alpha / (sqrtf(squaredGradientAvg) + epsilon);

                cublasSscal(handle, localSize, &scalingFactor, weightGradientDev, 1);

                cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, deltaWeightDev, 1);

                cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

                cudaFree(deltaWeightDev);
            }
            break;

            case AdaDelta:
            {
                const float mu = 0.9f;
                const float epsilon = 1e-7f;
                const float alpha = 0.01f;

                float* weightVelocityDev = _pbWeightVelocity->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightGradientVelocityDev = _pbWeightGradientVelocity->_pDevData;
                float* weightDev = _pbWeight->_pDevData;

                int localSize = _localSize;

                float* deltaWeightDev = nullptr;
                cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

                float squaredGradient = 0.0f;
                cublasSdot(handle, localSize, weightGradientDev, 1, weightGradientDev, 1, &squaredGradient);

                float squaredGradientAvg = 0.0f;
                squaredGradientAvg = mu * squaredGradientAvg + (1.0f - mu) * squaredGradient;

                float scalingFactor = (weightGradientVelocityDev && squaredGradientAvg)
                    ? sqrtf(*weightGradientVelocityDev) / (sqrtf(squaredGradientAvg) + epsilon)
                    : 0.0f;

                cublasSaxpy(handle, localSize, &scalingFactor, weightGradientDev, 1, weightVelocityDev, 1);

                cublasSaxpy(handle, localSize, &alpha, weightVelocityDev, 1, deltaWeightDev, 1);

                cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

                cublasSaxpy(handle, localSize, &mu, &squaredGradientAvg, 1, weightGradientVelocityDev, 1);

                cudaFree(deltaWeightDev);
            }
            break;

            case Adam:
            {
                const float alpha = 0.001f;
                const float mu = 0.9f;
                const float mu1 = 0.999f;
                const float epsilon = 1e-7f;
                const int t = 1;

                float* weightVelocityDev = _pbWeightVelocity->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightGradientVelocityDev = _pbWeightGradientVelocity->_pDevData;
                float* weightDev = _pbWeight->_pDevData;

                int localSize = _localSize;

                float* deltaWeightDev = nullptr;
                cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

                cublasSscal(handle, localSize, &mu, weightGradientDev, 1);
                cublasSscal(handle, localSize, &mu1, weightGradientVelocityDev, 1);

                float biasCorrectedMu = 1.0f / (1.0f - powf(mu, t));
                float biasCorrectedMu1 = 1.0f / (1.0f - powf(mu1, t));
                cublasSaxpy(handle, localSize, &biasCorrectedMu, weightGradientDev, 1, weightVelocityDev, 1);

                float scalingFactor = sqrtf(biasCorrectedMu1) / epsilon;

                cublasSaxpy(handle, localSize, &alpha, weightVelocityDev, 1, deltaWeightDev, 1);
                cublasSscal(handle, localSize, &scalingFactor, deltaWeightDev, 1);

                cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

                cudaFree(deltaWeightDev);
            }
            break;

            case LAMB:
            {
                const float alpha = 0.001f;
                const float beta1 = 0.9f;
                const float beta2 = 0.999f;
                const float weight_decay = 0.01f;

                float* weightVelocityDev = _pbWeightVelocity->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightDev = _pbWeight->_pDevData;

                int localSize = _localSize;

                float* deltaWeightDev = nullptr;
                float* updateDev = nullptr;
                cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));
                cudaMalloc((void**)&updateDev, localSize * sizeof(float));

                cublasSaxpy(handle, localSize, &weight_decay, weightDev, 1, weightGradientDev, 1);

                cublasSscal(handle, localSize, &beta1, weightVelocityDev, 1);
                cublasSscal(handle, localSize, &beta2, weightGradientDev, 1);

                cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, deltaWeightDev, 1);
                cublasSaxpy(handle, localSize, &weight_decay, weightDev, 1, updateDev, 1);
                cublasSaxpy(handle, localSize, &alpha, updateDev, 1, deltaWeightDev, 1);

                cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

                cudaFree(deltaWeightDev);
                cudaFree(updateDev);
            }
            break;
        }
    }

    if (_transform == Linear)
    {
        switch (trainingMode)
        {
        case SGD:
        {
            const float alpha = 0.01f;
            const int batch = 32;

            float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
            float* biasDev = _pbBias->_pDevData;

            int localBiasSize = _localBiasSize;

            float scaleFactor = alpha / batch;

            cublasSscal(handle, localBiasSize, &scaleFactor, deltaBiasDev, 1);
            cublasSaxpy(handle, localBiasSize, &alpha, deltaBiasDev, 1, biasDev, 1);
        }
        break;

        case Momentum:
        {
            const float alpha = 0.01f;
            const float mu = 0.9f;
            const int batch = 32;

            float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
            float* biasVelocityDev = _pbBiasVelocity->_pDevData;
            float* biasDev = _pbBias->_pDevData;

            int localBiasSize = _localBiasSize;

            float scaleFactor = alpha / batch;

            cublasSscal(handle, localBiasSize, &mu, biasVelocityDev, 1);

            cublasSaxpy(handle, localBiasSize, &scaleFactor, deltaBiasDev, 1, biasVelocityDev, 1);

            cublasSaxpy(handle, localBiasSize, &alpha, biasVelocityDev, 1, biasDev, 1);
        }
        break;

        case AdaGrad:
        {
            constexpr float alpha = 0.01f;
            constexpr int batch = 32;

            auto* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
            auto* biasVelocityDev = _pbBiasVelocity->_pDevData;
            auto* biasDev = _pbBias->_pDevData;

            const int localBiasSize = _localBiasSize;

            const float scaleFactor = alpha / batch;

            cublasSaxpy(handle, localBiasSize, &scaleFactor, deltaBiasDev, 1, biasVelocityDev, 1);
            cublasSaxpy(handle, localBiasSize, &alpha, biasVelocityDev, 1, biasDev, 1);
        }
        break;

        case Nesterov:
        {
            const float alpha = 0.01f;
            const float mu = 0.9f;
            const int batch = 32;

            float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
            float* biasVelocityDev = _pbBiasVelocity->_pDevData;
            float* biasDev = _pbBias->_pDevData;

            int localBiasSize = _localBiasSize;

            float scaleFactor = alpha / batch;

            cublasSaxpy(handle, localBiasSize, &mu, biasVelocityDev, 1, deltaBiasDev, 1);
            cublasSaxpy(handle, localBiasSize, &scaleFactor, deltaBiasDev, 1, biasVelocityDev, 1);

            cublasSaxpy(handle, localBiasSize, &alpha, biasVelocityDev, 1, biasDev, 1);
        }
        break;

        case RMSProp:
        {
            const float mu = 0.9f;
            const float epsilon = 1e-7f;

            float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
            float* biasVelocityDev = _pbBiasVelocity->_pDevData;
            float* biasDev = _pbBias->_pDevData;

            int localBiasSize = _localBiasSize;

            float squaredGradient;
            cublasSdot(handle, localBiasSize, deltaBiasDev, 1, deltaBiasDev, 1, &squaredGradient);

            cublasSscal(handle, localBiasSize, &mu, biasVelocityDev, 1);
            float tempMuDifference = 1.0f - mu;
            cublasSaxpy(handle, localBiasSize, &tempMuDifference, &squaredGradient, 1, biasVelocityDev, 1);

            float scalingFactor = (biasVelocityDev) ? sqrtf(*biasVelocityDev) / epsilon : 0.0f;

            cublasSaxpy(handle, localBiasSize, &scalingFactor, deltaBiasDev, 1, biasDev, 1);
        }
        break;

        case AdaDelta:
        {
            const float mu = 0.9f;
            const float epsilon = 1e-7f;

            float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
            float* biasVelocityDev = _pbBiasVelocity->_pDevData;
            float* biasGradientVelocityDev = _pbBiasGradientVelocity->_pDevData;
            float* biasDev = _pbBias->_pDevData;

            int localBiasSize = _localBiasSize;

            float squaredGradient;
            cublasSdot(handle, localBiasSize, deltaBiasDev, 1, deltaBiasDev, 1, &squaredGradient);

            cublasSscal(handle, localBiasSize, &mu, biasGradientVelocityDev, 1);
            float tempMuDiff = 1.0f - mu;
            cublasSaxpy(handle, localBiasSize, &tempMuDiff, &squaredGradient, 1, biasGradientVelocityDev, 1);

            float scalingFactor = (biasVelocityDev && biasGradientVelocityDev)
                ? sqrtf(*biasVelocityDev) / (sqrtf(*biasGradientVelocityDev) + epsilon)
                : 0.0f;

            cublasSaxpy(handle, localBiasSize, &scalingFactor, deltaBiasDev, 1, biasVelocityDev, 1);

            cublasSaxpy(handle, localBiasSize, &mu, deltaBiasDev, 1, biasDev, 1);
        }
        break;

        case Adam:
        {
            const float alpha = 0.01f;
            const float mu = 0.9f;
            const float mu1 = 0.999f;
            const int t = 1;
            const int batch = 32;
            const float epsilon = 1e-7f;

            float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
            float* biasVelocityDev = _pbBiasVelocity->_pDevData;
            float* biasGradientVelocityDev = _pbBiasGradientVelocity->_pDevData;
            float* biasDev = _pbBias->_pDevData;

            int localBiasSize = _localBiasSize;

            float scaleFactor = alpha / batch;

            cublasSaxpy(handle, localBiasSize, &scaleFactor, deltaBiasDev, 1, biasVelocityDev, 1);

            cublasSscal(handle, localBiasSize, &mu, biasVelocityDev, 1);
            float tempMuSubtraction = 1.0f - mu;
            cublasSaxpy(handle, localBiasSize, &tempMuSubtraction, deltaBiasDev, 1, biasVelocityDev, 1);

            cublasSscal(handle, localBiasSize, &mu1, biasGradientVelocityDev, 1);
            float tempMu1Subtraction = 1.0f - mu1;
            cublasSaxpy(handle, localBiasSize, &tempMu1Subtraction, deltaBiasDev, 1, biasGradientVelocityDev, 1);

            float biasCorrectedMu = 1.0f / (1.0f - powf(mu, t));
            float biasCorrectedMu1 = 1.0f / (1.0f - powf(mu1, t));

            cublasSscal(handle, localBiasSize, &biasCorrectedMu, biasVelocityDev, 1);
            cublasSscal(handle, localBiasSize, &biasCorrectedMu1, biasGradientVelocityDev, 1);

            float scalingFactor = (biasGradientVelocityDev) ? sqrtf(*biasGradientVelocityDev) / epsilon : 0.0f;

            cublasSaxpy(handle, localBiasSize, &scalingFactor, biasVelocityDev, 1, biasDev, 1);
        }
        break;

        case LAMB:
        {
            const float learningRate = 0.01f;
            const float beta1 = 0.9f;
            const float beta2 = 0.999f;
            const float epsilon = 1e-6f;
            const float weightDecay = 0.01f;

            float* weightDev = _pbWeight->_pDevData;
            float* gradientDev = _pbGradient->_pDevData;
            float* mDev = _pM->_pDevData;
            float* vDev = _pV->_pDevData;

            int localWeightSize = _localWeightSize;

            float weightDecayFactor = 1.0f - learningRate * weightDecay;

            cublasSscal(handle, localWeightSize, &beta1, mDev, 1);
            cublasSaxpy(handle, localWeightSize, &beta1, gradientDev, 1, mDev, 1);

            cublasSscal(handle, localWeightSize, &beta2, vDev, 1);
            cublasSaxpy(handle, localWeightSize, &beta2, gradientDev, 1, vDev, 1);

            int t = 1;
            t++;
            float biasCorrection1 = 1.0f / (1.0f - powf(beta1, t));
            float biasCorrection2 = 1.0f / (1.0f - powf(beta2, t));

            cublasSscal(handle, localWeightSize, &biasCorrection1, mDev, 1);
            cublasSscal(handle, localWeightSize, &biasCorrection2, vDev, 1);

            cublasSaxpy(handle, localWeightSize, &weightDecayFactor, weightDev, 1, gradientDev, 1);

            float scalingFactor = sqrtf(vDev[0]) + epsilon;

            cublasSscal(handle, localWeightSize, &learningRate, gradientDev, 1);
            cublasSaxpy(handle, localWeightSize, &scalingFactor, gradientDev, 1, weightDev, 1);
        }
        break;

            case RAdam:
            {
                const float learningRate = 0.01f;
                const float beta1 = 0.9f;
                const float beta2 = 0.999f;
                const float epsilon = 1e-6f;
                const float weightDecay = 0.01f;

                float* weightDev = _pbWeight->_pDevData;
                float* gradientDev = _pbGradient->_pDevData;
                float* mDev = _pM->_pDevData;
                float* vDev = _pV->_pDevData;

                int localWeightSize = _localWeightSize;

                float weightDecayFactor = 1.0f - learningRate * weightDecay;

                cublasSscal(handle, localWeightSize, &beta1, mDev, 1);
                cublasSaxpy(handle, localWeightSize, &beta1, gradientDev, 1, mDev, 1);
                cublasSscal(handle, localWeightSize, &beta2, vDev, 1);
                cublasSaxpy(handle, localWeightSize, &beta2, gradientDev, 1, vDev, 1);

                int t = 1;

                float biasCorrection1 = 1.0f - powf(beta1, t);
                float biasCorrection2 = 1.0f - powf(beta2, t);

                cublasSscal(handle, localWeightSize, &biasCorrection1, mDev, 1);
                cublasSscal(handle, localWeightSize, &biasCorrection2, vDev, 1);
                cublasSaxpy(handle, localWeightSize, &weightDecayFactor, weightDev, 1, gradientDev, 1);

                float scalingFactor = sqrtf(vDev[0]) + epsilon;

                cublasSscal(handle, localWeightSize, &learningRate, gradientDev, 1);
                cublasSaxpy(handle, localWeightSize, &scalingFactor, weightDev, 1, gradientDev, 1);

                break;
            }

            case Lookahead:
            {
                const float alpha = 0.01f;
                const float beta = 0.9f;

                float* weightDev = _pbWeight->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightDev2 = _pbWeight->_pDevData;
                float* weightGradientDev2 = _pbWeightGradient->_pDevData;

                int localWeightSize = _localWeightSize;

                cublasSaxpy(handle, localWeightSize, &alpha, weightGradientDev, 1, weightDev, 1);

                cublasSaxpy(handle, localWeightSize, &alpha, weightGradientDev, 1, weightDev2, 1);
                cublasSscal(handle, localWeightSize, &beta, weightDev2, 1);
                cublasSaxpy(handle, localWeightSize, &beta, weightDev, 1, weightDev2, 1);

                break;
            }

            case SGDW:
            {
                const float alpha = 0.01f;
                const float mu = 0.9f;
                const float weightDecay = 0.01f;

                float* weightDev = _pbWeight->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightVelocityDev = _pbWeightVelocity->_pDevData;

                int localWeightSize = _localWeightSize;

                float weightDecayFactor = 1.0f - alpha * weightDecay;

                cublasSaxpy(handle, localWeightSize, &alpha, weightGradientDev, 1, weightDev, 1);

                cublasSscal(handle, localWeightSize, &mu, weightVelocityDev, 1);
                cublasSaxpy(handle, localWeightSize, &alpha, weightGradientDev, 1, weightVelocityDev, 1);

                cublasSscal(handle, localWeightSize, &weightDecayFactor, weightDev, 1);
            }
            break;

            case SGDP:
            {
                const float alpha = 0.01f;
                const float mu = 0.9f;
                const float weightDecay = 0.01f;

                float* weightDev = _pbWeight->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightVelocityDev = _pbWeightVelocity->_pDevData;
                float* weightDev2 = _pbWeight->_pDevData;
                float* weightGradientDev2 = _pbWeightGradient->_pDevData;

                int localWeightSize = _localWeightSize;

                float weightDecayFactor = 1.0f - alpha * weightDecay;

                cublasSaxpy(handle, localWeightSize, &alpha, weightGradientDev, 1, weightDev, 1);
                cublasSscal(handle, localWeightSize, &mu, weightVelocityDev, 1);
                cublasSaxpy(handle, localWeightSize, &alpha, weightGradientDev, 1, weightVelocityDev, 1);
                cublasSaxpy(handle, localWeightSize, &weightDecayFactor, weightDev2, 1, weightDev, 1);
                cublasSaxpy(handle, localWeightSize, &mu, weightVelocityDev, 1, weightDev, 1);

                break;
            }

            case SWA:
            {
                const float alpha = 0.01f;
                const float beta = 0.9f;

                float* weightDev = _pbWeight->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightDev2 = _pbWeight->_pDevData;
                float* weightGradientDev2 = _pbWeightGradient->_pDevData;

                int localWeightSize = _localWeightSize;

                cublasSaxpy(handle, localWeightSize, &alpha, weightGradientDev, 1, weightDev, 1);

                cublasSaxpy(handle, localWeightSize, &alpha, weightGradientDev, 1, weightDev2, 1);
                cublasSscal(handle, localWeightSize, &beta, weightDev2, 1);
                cublasSaxpy(handle, localWeightSize, &beta, weightDev, 1, weightDev2, 1);

                break;
            }

            case Yogi:
            {
                const float alpha = 0.01f;
                const float beta1 = 0.9f;
                const float beta2 = 0.999f;
                const float epsilon = 1e-7f;
                const float weightDecay = 0.01f;

                float* weightDev = _pbWeight->_pDevData;
                float* gradientDev = _pbGradient->_pDevData;
                float* mDev = _pM->_pDevData;
                float* vDev = _pV->_pDevData;

                int localWeightSize = _localWeightSize;

                float weightDecayFactor = 1.0f - alpha * weightDecay;

                cublasSscal(handle, localWeightSize, &beta1, mDev, 1);
                cublasSaxpy(handle, localWeightSize, &beta1, gradientDev, 1, mDev, 1);

                cublasSscal(handle, localWeightSize, &beta2, vDev, 1);
                cublasSaxpy(handle, localWeightSize, &beta2, gradientDev, 1, vDev, 1);

                int t = 1;
                t++;
                float biasCorrection1 = 1.0f / (1.0f - powf(beta1, t));
                float biasCorrection2 = 1.0f / (1.0f - powf(beta2, t));

                cublasSscal(handle, localWeightSize, &biasCorrection1, mDev, 1);
                cublasSscal(handle, localWeightSize, &biasCorrection2, vDev, 1);

                cublasSaxpy(handle, localWeightSize, &weightDecayFactor, weightDev, 1, gradientDev, 1);

                float scalingFactor = sqrtf(vDev[0]) / (sqrtf(mDev[0]) + epsilon);

                cublasSscal(handle, localWeightSize, &alpha, gradientDev, 1);
                cublasSaxpy(handle, localWeightSize, &scalingFactor, gradientDev, 1, weightDev, 1);

                break;
            }

            case Ranger:
            {
                const float alpha = 0.01f;
                const float beta1 = 0.9f;
                const float beta2 = 0.999f;
                const float epsilon = 1e-7f;
                const float weightDecay = 0.01f;

                float* weightDev = _pbWeight->_pDevData;
                float* gradientDev = _pbGradient->_pDevData;
                float* mDev = _pM->_pDevData;
                float* vDev = _pV->_pDevData;

                int localWeightSize = _localWeightSize;

                float weightDecayFactor = 1.0f - alpha * weightDecay;

                cublasSscal(handle, localWeightSize, &beta1, mDev, 1);
                cublasSaxpy(handle, localWeightSize, &beta1, gradientDev, 1, mDev, 1);

                cublasSscal(handle, localWeightSize, &beta2, vDev, 1);
                cublasSaxpy(handle, localWeightSize, &beta2, gradientDev, 1, vDev, 1);

                int t = 1;
                t++;
                float biasCorrection1 = 1.0f / (1.0f - powf(beta1, t));
                float biasCorrection2 = 1.0f / (1.0f - powf(beta2, t));

                cublasSscal(handle, localWeightSize, &biasCorrection1, mDev, 1);
                cublasSscal(handle, localWeightSize, &biasCorrection2, vDev, 1);

                cublasSaxpy(handle, localWeightSize, &weightDecayFactor, weightDev, 1, gradientDev, 1);

                float scalingFactor = sqrtf(vDev[0]) / (sqrtf(mDev[0]) + epsilon);

                cublasSscal(handle, localWeightSize, &alpha, gradientDev, 1);
                cublasSaxpy(handle, localWeightSize, &scalingFactor, gradientDev, 1, weightDev, 1);

                break;
            }
        }
    }
    else
    {
        switch (trainingMode)
        {
            case SGD:
            {
                const float alpha = 0.01f;

                int localBiasSize = _localBiasSize;

                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                cublasSaxpy(handle, localBiasSize, &alpha, biasGradientDev, 1, biasDev, 1);
            }
            break;

            case Momentum:
            {
                const float alpha = 0.01f;
                const float mu = 0.9f;

                int localBiasSize = _localBiasSize;

                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                cublasSaxpy(handle, localBiasSize, &mu, biasVelocityDev, 1, biasGradientDev, 1);

                cublasSaxpy(handle, localBiasSize, &alpha, biasGradientDev, 1, biasDev, 1);
            }
            break;
                    
            case AdaGrad:
            {
                const float alpha = 0.01f;

                int localBiasSize = _localBiasSize;

                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                float oneValue = 1.0f;
                cublasSaxpy(handle, localBiasSize, &oneValue, biasGradientDev, 1, biasVelocityDev, 1);

                float negAlpha = -alpha;
                cublasSaxpy(handle, localBiasSize, &negAlpha, biasVelocityDev, 1, biasDev, 1);
            }
            break;
                        
            case Nesterov:
            {
                const float alpha = 0.01f;
                const float mu = 0.9f;

                int localBiasSize = _localBiasSize;

                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                float muMinusOne = mu - 1.0f;
                cublasSaxpy(handle, localBiasSize, &mu, biasVelocityDev, 1, biasGradientDev, 1);
                cublasSaxpy(handle, localBiasSize, &muMinusOne, biasGradientDev, 1, biasVelocityDev, 1);

                float negatedAlpha = -alpha;
                cublasSaxpy(handle, localBiasSize, &negatedAlpha, biasVelocityDev, 1, biasDev, 1);
            }
            break;
                        
            case RMSProp:
            {
                const float alpha = 0.01f;
                const float mu = 0.9f;

                int localBiasSize = _localBiasSize;

                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                float squaredGradient;
                cublasSdot(handle, localBiasSize, biasGradientDev, 1, biasGradientDev, 1, &squaredGradient);

                cublasSscal(handle, localBiasSize, &mu, biasVelocityDev, 1);
                float tempValue3 = 1.0f - mu;
                cublasSaxpy(handle, localBiasSize, &tempValue3, &squaredGradient, 1, biasVelocityDev, 1);

                float negativeAlphaValue = -alpha;
                cublasSaxpy(handle, localBiasSize, &negativeAlphaValue, biasGradientDev, 1, biasDev, 1);
            }
            break;

            case AdaDelta:
            {
                const float mu = 0.9f;
                const float epsilon = 1e-7f;

                int localBiasSize = _localBiasSize;

                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasGradientVelocityDev = _pbBiasGradientVelocity->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                float squaredGradient;
                cublasSdot(handle, localBiasSize, biasGradientDev, 1, biasGradientDev, 1, &squaredGradient);

                cublasSscal(handle, localBiasSize, &mu, biasGradientVelocityDev, 1);
                float tempValue2 = 1.0f - mu;
                cublasSaxpy(handle, localBiasSize, &tempValue2, &squaredGradient, 1, biasGradientVelocityDev, 1);

                float scalingFactor = 0.0f;
                if (biasVelocityDev && biasGradientVelocityDev) {
                    scalingFactor = sqrtf(*biasVelocityDev) / (sqrtf(*biasGradientVelocityDev) + epsilon);
                }

                cublasSaxpy(handle, localBiasSize, &scalingFactor, biasGradientDev, 1, biasDev, 1);
            }
            break;

            case Adam:
            {
                const float alpha = 0.01f;
                const float mu = 0.9f;
                const float mu1 = 0.999f;
                int t = 1;

                int localBiasSize = _localBiasSize;

                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasGradientVelocityDev = _pbBiasGradientVelocity->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                float* squaredGradientDev = new float[localBiasSize];

                cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, localBiasSize, 1, biasGradientDev, 1, biasGradientDev, 1, squaredGradientDev, 1);

                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                cublasSaxpy(handle, localBiasSize, &mu, biasVelocityDev, 1, biasGradientDev, 1);
                cublasSscal(handle, localBiasSize, &mu, biasVelocityDev, 1);
                float tempValue = 1.0f - mu;
                cublasSaxpy(handle, localBiasSize, &tempValue, biasGradientDev, 1, biasVelocityDev, 1);

                cublasSscal(handle, localBiasSize, &mu1, biasGradientVelocityDev, 1);
                float value = 1.0f - mu1;
                cublasSaxpy(handle, localBiasSize, &value, squaredGradientDev, 1, biasGradientVelocityDev, 1);

                float biasCorrectedMu = 1.0f / (1.0f - powf(mu, t));
                float biasCorrectedMu1 = 1.0f / (1.0f - powf(mu1, t));

                cublasSscal(handle, localBiasSize, &biasCorrectedMu, biasVelocityDev, 1);
                cublasSscal(handle, localBiasSize, &biasCorrectedMu1, biasGradientVelocityDev, 1);

                float negativeAlpha = -alpha;
                cublasSaxpy(handle, localBiasSize, &negativeAlpha, biasGradientDev, 1, biasDev, 1);

                delete[] squaredGradientDev;
            }
            break;

            case LAMB:
            {
                const float learningRate = 0.01f;
                const float beta1 = 0.9f;
                const float beta2 = 0.999f;

                int localWeightSize = _localWeightSize;

                float* weightDev = _pbWeight->_pDevData;
                float* gradientDev = _pbGradient->_pDevData;
                float* weightVelocityDev = _pbWeightVelocity->_pDevData;
                float* weightVelocityHatDev = _pbWeightVelocityHat->_pDevData;
                
                float l2NormGradients = 0.0f;

                for (int i = 0; i < localWeightSize; i++) {
                    l2NormGradients += gradientDev[i] * gradientDev[i];
                }

                l2NormGradients = sqrtf(l2NormGradients);

                cublasSscal(handle, localWeightSize, &beta1, weightVelocityDev, 1);
                cublasSscal(handle, localWeightSize, &beta2, weightVelocityHatDev, 1);
                cublasSaxpy(handle, localWeightSize, &beta1, gradientDev, 1, weightVelocityDev, 1);
                cublasSaxpy(handle, localWeightSize, &beta2, gradientDev, 1, weightVelocityHatDev, 1);
                
                int t = 1;
                t++;

                float biasCorrectedBeta1 = 1.0f / (1.0f - powf(beta1, t));
                float biasCorrectedBeta2 = 1.0f / (1.0f - powf(beta2, t));

                cublasSscal(handle, localWeightSize, &biasCorrectedBeta1, weightVelocityDev, 1);
                cublasSscal(handle, localWeightSize, &biasCorrectedBeta2, weightVelocityHatDev, 1);

                cublasSaxpy(handle, localWeightSize, &learningRate, weightVelocityHatDev, 1, weightDev, 1);
            }
            break;
        }       
    }
#if 0
        if (_width < 1024)
        {
            _pbBias->Download(&_vBias[0]);
            for (int i = 0; i < _width; i++)
                std::cout << ("%3d %16.8f\n", i, _vBias[i]);
        }
#endif
          
    if ((_norm > (float)0.0) && (!_bShared))
    {
        if (getGpu()._numprocs == 1)
            kNormalizeWeights(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData);
        else
        {
            std::shared_ptr<float[]> pScratchBuffer = getGpu()._pNetwork->GetScratchBuffer(_outputLayer._stride);
            float* pMagnitude = pScratchBuffer.get();
            invokeWeightMagnitudes(_outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, _outputLayer._stride);
            kNormalizeWeightMagnitudes(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);       
        }
    }
}

bool Weight::WriteNetCDF(netCDF::NcFile& nc, uint32_t index, float* pWeight, float* pBias)
{
    if (getGpu()._id != 0)
    {
        std::cout << "GPU ID is not 0. Skipping WriteNetCDF for weight index " << index;
        return true;
    }

    auto baseName = "weight" + std::to_string(index) + "_";

    const auto putAttString = [&nc, &baseName](const std::string& name, const std::string& value) {
        nc.putAtt(baseName + name, netCDF::ncChar, value.size(), value.c_str());
        };

    const auto putAttInt = [&nc, &baseName](const std::string& name, int value) {
        nc.putAtt(baseName + name, netCDF::ncInt, value);
        };

    const auto putAttFloat = [&nc, &baseName](const std::string& name, float value) {
        nc.putAtt(baseName + name, netCDF::ncFloat, value);
        };

    try {
        putAttString("inputLayer", _inputLayer._name);
        putAttString("outputLayer", _outputLayer._name);
        putAttInt("width", _width);
        putAttInt("height", _height);
        putAttInt("length", _length);
        putAttInt("depth", _depth);
        putAttInt("breadth", _breadth);
        putAttFloat("bShared", _bShared);
        putAttFloat("bLocked", _bLocked);
        putAttFloat("norm", _norm);

        auto biasDim = nc.addDim(baseName + "biasDim", _biasSize);
        auto biasVar = nc.addVar(baseName + "bias", "float", biasDim.getName());

        if (!pBias) pBias = _vBias.data();

        biasVar.putVar(pBias);

        if (_bShared)
        {
            putAttFloat("bTransposed", _bTransposed);
            putAttString("sourceInputLayer", _pSharedWeight->_inputLayer._name);
            putAttString("sourceOutputLayer", _pSharedWeight->_outputLayer._name);
        }
        else
        {
            auto weightDim = nc.addDim(baseName + "weightDim", _size);
            auto weightVar = nc.addVar(baseName + "weights", "float", weightDim.getName());

            if (!pWeight) pWeight = _vWeight.data();

            weightVar.putVar(pWeight);
        }

        std::cout << "Successfully wrote weight data to NetCDF for weight index " << index;
    }
    catch (const std::exception& e) {
        std::cerr << "Error writing NetCDF for weight index " << index << ": " << e.what();
        return false;
    }

    return true;
}

bool Weight::CopyWeights(const Weight* pSrcWeight) {
    auto& gpu = getGpu();

    Weight* pDstWeight = _bShared ? _pSharedWeight : this;

    try {
        if (!pSrcWeight) {
            throw std::invalid_argument("Invalid weight pointer.");
        }

        pSrcWeight = _bShared ? pSrcWeight->_pSharedWeight : pSrcWeight;

        if ((_width != pSrcWeight->_width) || (_height != pSrcWeight->_height) || (_length != pSrcWeight->_length)) {
            throw std::runtime_error("Mismatched weight dimensions (" +
                std::to_string(_width) + " x " + std::to_string(_height) + " x " + std::to_string(_length) +
                ") versus (" +
                std::to_string(pSrcWeight->_width) + " x " + std::to_string(pSrcWeight->_height) + " x " +
                std::to_string(pSrcWeight->_length) + ").");
        }

        pDstWeight->_vWeight = pSrcWeight->_vWeight;
        _vBias = pSrcWeight->_vBias;

        if (_pbWeight) {
            _pbWeight->Upload(pDstWeight->_vWeight.data());
        }

        if (_pbBias) {
            _pbBias->Upload(_vBias.data());
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Weight::CopyWeights: " << e.what();
        return false;
    }

    return true;
}

bool Weight::SetWeights(const std::vector<float>& vWeight) {
    const auto& gpuInfo = getGpu();

    Weight* pWeight = _bShared ? _pSharedWeight : this;

    if (gpuInfo._numprocs == 1) {
        if (vWeight.size() < pWeight->_vWeight.size()) {
            if (gpuInfo._id == 0) {
                std::cerr << "Weight::SetWeights: Input vector smaller than weight vector.";
            }
            return false;
        }

        if (vWeight.size() > pWeight->_vWeight.size()) {
            std::copy_n(vWeight.begin(), pWeight->_vWeight.size(), pWeight->_vWeight.begin());
        }
        else {
            pWeight->_vWeight = vWeight;
        }

        try {
            if (pWeight->_pbWeight) {
                pWeight->_pbWeight->Upload(pWeight->_vWeight.data());
            }
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error uploading weights to GPU: " << e.what();
            return false;
        }
    }

    const int numWeights = pWeight->_vWeight.size();
    const int chunkSize = numWeights / gpuInfo._numprocs;
    const int startIdx = gpuInfo._id * chunkSize;
    const int endIdx = (gpuInfo._id == gpuInfo._numprocs - 1) ? numWeights : (gpuInfo._id + 1) * chunkSize;

    const std::vector<float> gpuWeights(vWeight.begin() + startIdx, vWeight.begin() + endIdx);

    try {
        if (pWeight->_pbWeight) {
            pWeight->_pbWeight->Upload(gpuWeights.data());
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error uploading weights to GPU: " << e.what();
        return false;
    }
}

bool Weight::SetBiases(const std::vector<float>& vBias)
{
    if (vBias.size() < _vBias.size())
    {
        if (getGpu()._id == 0)
        {
            std::cerr << "Weight::SetBiases: Input vector smaller than bias vector.";
        }
        return false;
    }

    assert(_pbBias != nullptr);

    std::copy_n(vBias.begin(), _vBias.size(), _vBias.begin());

    try
    {
        _pbBias->Upload(_vBias.data());
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error uploading bias data to GPU: " << e.what();
        return false;
    }

    return true;
}

bool Weight::GetWeights(std::vector<float>& vWeight)
{
    bool bValid = true;

    if (vWeight.size() < _vWeight.size())
    {
        vWeight.resize(_vWeight.size());
    }

    if (_pbWeight != nullptr)
    {
        try
        {
            _pbWeight->Download(vWeight.data());
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error downloading weight data from GPU: " << e.what();
            bValid = false;
        }
    }
    else
    {
        vWeight = _vWeight;
    }
    return bValid;
}

void Weight::ApplyAdaptiveLearningRate(float learningRateDecay, float mu)
{
    if (_pbWeightVelocity && _pbWeight && _pbWeightGradient)
    {
        float* pWeightVelocityDev = _pbWeightVelocity->_pDevData;
        float* pWeightDev = _pbWeight->_pDevData;
        float* pWeightGradientDev = _pbWeightGradient->_pDevData;
        int localWeightSize = _localSize;
        cublasHandle_t cublasHandle;

        try
        {
            cublasCreate(&cublasHandle);

            cublasSscal(cublasHandle, localWeightSize, &learningRateDecay, pWeightGradientDev, 1);

            cublasSaxpy(cublasHandle, localWeightSize, &mu, pWeightVelocityDev, 1, pWeightGradientDev, 1);

            cublasSaxpy(cublasHandle, localWeightSize, &learningRateDecay, pWeightGradientDev, 1, pWeightDev, 1);

            std::cout << "Applied adaptive learning rate with decay: " << learningRateDecay;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error in ApplyAdaptiveLearningRate: " << e.what();
        }

        cublasDestroy(cublasHandle);
    }
}

void Weight::ApplyGradientNoise(float noiseMagnitude)
{
    if (_pbWeightGradient)
    {
        float* pWeightGradientDev = _pbWeightGradient->_pDevData;
        int localWeightSize = _localSize;

        try
        {
            std::unique_ptr<curandGenerator_t, void(*)(curandGenerator_t*)> curandGeneratorPtr(
                new curandGenerator_t(),
                [](curandGenerator_t* generator) {
                    curandDestroyGenerator(*generator);
                }
            );
            curandCreateGenerator(curandGeneratorPtr.get(), CURAND_RNG_PSEUDO_DEFAULT);

            curandGenerateNormal(*curandGeneratorPtr, pWeightGradientDev, localWeightSize, 0.0f, noiseMagnitude);

            std::cout << "Applied gradient noise with magnitude: " << noiseMagnitude;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error in ApplyGradientNoise: " << e.what();
        }
    }
}

void Weight::AdjustLearningRate(float newLearningRate)
{
    if (_pbWeightGradient)
    {
        float* pWeightGradientDev = _pbWeightGradient->_pDevData;
        int localWeightSize = _localSize;

        try
        {
            std::unique_ptr<cublasHandle_t, void(*)(cublasHandle_t*)> cublasHandlePtr(
                new cublasHandle_t(),
                [](cublasHandle_t* handle) {
                    cublasDestroy(*handle);
                    delete handle;
                }
            );
            cublasCreate(cublasHandlePtr.get());

            cublasSscal(*cublasHandlePtr, localWeightSize, &newLearningRate, pWeightGradientDev, 1);

            std::cout << "Adjusted learning rate to: " << newLearningRate;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error in AdjustLearningRate: " << e.what();
        }
    }
}

void Weight::ScaleWeights(float scaleFactor, int startIdx, int endIdx) {
    try {
        if (!_pbWeight || !_pbWeight->_pDevData) {
            std::cerr << "Invalid GpuBuffer";
            return;
        }

        if (startIdx < 0 || startIdx >= _localSize || endIdx <= startIdx || endIdx > _localSize) {
            std::cerr << "Invalid index range";
            return;
        }

        std::unique_ptr<cublasHandle_t, void(*)(cublasHandle_t*)> cublasHandlePtr(
            new cublasHandle_t(),
            [](cublasHandle_t* handle) {
                cublasDestroy(*handle);
                delete handle;
            }
        );
        cublasCreate(cublasHandlePtr.get());

        int numElements = endIdx - startIdx;

        cublasSscal(*cublasHandlePtr, numElements, &scaleFactor, _pbWeight->_pDevData + startIdx, 1);

        std::cout << "Scaled weights from index " << startIdx << " to " << endIdx
            << " by factor: " << scaleFactor;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in ScaleWeights: " << e.what();
    }
}

void Weight::NormalizeWeights() {
    try {
        if (getGpu()._numprocs == 1) {
            kNormalizeWeights(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData);
        }
        else {
            std::shared_ptr<float[]> pScratchBuffer = getGpu()._pNetwork->GetScratchBuffer(_outputLayer._stride);
            float* pMagnitude = pScratchBuffer.get();
            invokeWeightMagnitudes(_outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, _outputLayer._stride);
            kNormalizeWeightMagnitudes(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);
        }
        std::cout << "Weights normalized.";
    }
    catch (const std::exception& e) {
        std::cerr << "Error in NormalizeWeights: " << e.what();
    }
}

void Weight::WeightQuantization(float quantizationLevels) {
    try {
        if (_pbWeight && _pbWeightGradient) {
            float* pWeightDev = _pbWeight->_pDevData;
            float* pWeightGradientDev = _pbWeightGradient->_pDevData;
            int localWeightSize = _localSize;
            cublasHandle_t cublasHandle;

            cublasStatus_t cublasStatus = cublasCreate(&cublasHandle);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS initialization error.";
                throw std::runtime_error("cuBLAS initialization failed");
            }

            cublasStatus = cublasSscal(cublasHandle, localWeightSize, &quantizationLevels, pWeightGradientDev, 1);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS Sscal error for gradient update.";
                throw std::runtime_error("cuBLAS Sscal failed for gradient update");
            }

            cublasStatus = cublasSaxpy(cublasHandle, localWeightSize, &quantizationLevels, pWeightGradientDev, 1, pWeightDev, 1);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS Saxpy error for weight update.";
                throw std::runtime_error("cuBLAS Saxpy failed for weight update");
            }

            std::cout << "Applied weight quantization with levels: " << quantizationLevels;
        }
        else {
            std::cerr << "Invalid input data or buffers.";
            throw std::invalid_argument("Invalid input data or buffers");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in WeightQuantization: " << e.what();
    }
}

void Weight::ApplyGradientClipping(float gradientClipValue) {
    try {
        if (_pbWeightGradient) {
            float* pWeightGradientDev = _pbWeightGradient->_pDevData;
            int localWeightSize = _localSize;
            cublasHandle_t cublasHandle;

            cublasStatus_t cublasStatus = cublasCreate(&cublasHandle);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS initialization error.";
                throw std::runtime_error("cuBLAS initialization failed");
            }

            for (int i = 0; i < localWeightSize; ++i) {
                pWeightGradientDev[i] = std::min(pWeightGradientDev[i], gradientClipValue);
                pWeightGradientDev[i] = std::max(pWeightGradientDev[i], -gradientClipValue);
            }

            std::cout << "Applied gradient clipping with value: " << gradientClipValue;
        }
        else {
            std::cerr << "Invalid input data or buffers.";
            throw std::invalid_argument("Invalid input data or buffers");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in ApplyGradientClipping: " << e.what();
    }
}

void Weight::ApplyMomentum(float momentumCoefficient) {
    try {
        if (_pbWeightGradient && _pbWeightVelocity) {
            float* pWeightVelocityDev = _pbWeightVelocity->_pDevData;
            float* pWeightGradientDev = _pbWeightGradient->_pDevData;
            int localWeightSize = _localSize;
            cublasHandle_t cublasHandle;

            cublasStatus_t cublasStatus = cublasCreate(&cublasHandle);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS initialization error.";
                throw std::runtime_error("cuBLAS initialization failed");
            }

            cublasStatus = cublasSscal(cublasHandle, localWeightSize, &momentumCoefficient, pWeightGradientDev, 1);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS Sscal error for gradient update.";
                throw std::runtime_error("cuBLAS Sscal failed for gradient update");
            }

            cublasStatus = cublasSaxpy(cublasHandle, localWeightSize, &momentumCoefficient, pWeightGradientDev, 1, pWeightVelocityDev, 1);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS Saxpy error for velocity update.";
                throw std::runtime_error("cuBLAS Saxpy failed for velocity update");
            }

            std::cout << "Applied momentum with coefficient: " << momentumCoefficient;
        }
        else {
            std::cerr << "Invalid input data or buffers.";
            throw std::invalid_argument("Invalid input data or buffers");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in ApplyMomentum: " << e.what();
    }
}

void Weight::ApplyWeightSmoothing(float smoothingFactor) {
    try {
        if (_pbWeight && _pbWeightGradient) {
            float* pWeightDev = _pbWeight->_pDevData;
            float* pWeightGradientDev = _pbWeightGradient->_pDevData;
            int localWeightSize = _localSize;
            cublasHandle_t cublasHandle;

            cublasStatus_t cublasStatus = cublasCreate(&cublasHandle);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS initialization error.";
                throw std::runtime_error("cuBLAS initialization failed");
            }

            cublasStatus = cublasSscal(cublasHandle, localWeightSize, &smoothingFactor, pWeightGradientDev, 1);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS Sscal error for gradient update.";
                throw std::runtime_error("cuBLAS Sscal failed for gradient update");
            }

            cublasStatus = cublasSaxpy(cublasHandle, localWeightSize, &smoothingFactor, pWeightGradientDev, 1, pWeightDev, 1);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS Saxpy error for weight update.";
                throw std::runtime_error("cuBLAS Saxpy failed for weight update");
            }

            std::cout << "Applied weight smoothing with factor: " << smoothingFactor;
        }
        else {
            std::cerr << "Invalid input data or buffers.";
            throw std::invalid_argument("Invalid input data or buffers");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in ApplyWeightSmoothing: " << e.what();
    }
}

void Weight::ApplyWeightPerturbation(float perturbationStrength) {
    try {
        if (_pbWeight) {
            float* pWeightDev = _pbWeight->_pDevData;
            int localWeightSize = _localSize;
            cublasHandle_t cublasHandle;

            cublasStatus_t cublasStatus = cublasCreate(&cublasHandle);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS initialization error.";
                throw std::runtime_error("cuBLAS initialization failed");
            }

            cublasStatus = cublasSscal(cublasHandle, localWeightSize, &perturbationStrength, pWeightDev, 1);

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS Sscal error.";
                throw std::runtime_error("cuBLAS Sscal failed");
            }

            std::cout << "Applied weight perturbation with strength: " << perturbationStrength;
        }
        else {
            std::cerr << "Invalid input data or buffers.";
            throw std::invalid_argument("Invalid input data or buffers");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in ApplyWeightPerturbation: " << e.what();
    }
}

void Weight::RegularizeWeights(float regularizationStrength, DataSetEnums::RegularizationType regularizationType)
{
    if (!_pbWeight) {
        std::cerr << "Error: No GPU buffers available";
        return;
    }

    int localWeightSize = _localSize;

    cublasHandle_t cublasHandle;
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error: cuBLAS initialization failed";
        return;
    }

    try {
        if (regularizationType == DataSetEnums::RegularizationType::L1) {
            cublasSaxpy(cublasHandle, localWeightSize, &regularizationStrength, _pbWeightGradient->_pDevData, 1, _pbWeight->_pDevData, 1);
        }
        else if (regularizationType == DataSetEnums::RegularizationType::L2) {
            cublasSscal(cublasHandle, localWeightSize, &regularizationStrength, _pbWeight->_pDevData, 1);
        }
        else {
            std::cerr << "Error: Unsupported regularization type";
        }

        std::cout << "Applied weight regularization with strength: " << regularizationStrength;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what();
    }

    cublasDestroy(cublasHandle);
}

void Weight::QuantizeWeights(int numBits)
{
    if (_bShared) {
        std::cerr << "Quantization of shared weights is not supported.";
        throw std::runtime_error("Quantization of shared weights is not supported.");
    }

    if (numBits < 1 || numBits > 32) {
        std::cerr << "Invalid number of bits: " << numBits;
        throw std::runtime_error("Invalid number of bits.");
    }

    if (_bLocked) {
        std::cerr << "Weight is locked.";
        throw std::runtime_error("Weight is locked.");
    }

    if (_bQuantized) {
        std::cout << "Weight is already quantized.";
        return;
    }

    _minValue = std::numeric_limits<float>::max();
    _maxValue = std::numeric_limits<float>::min();

    for (uint32_t i = 0; i < _size; ++i)
    {
        if (_data[i] < _minValue)
        {
            _minValue = _data[i];
        }
        if (_data[i] > _maxValue)
        {
            _maxValue = _data[i];
        }
    }

    float range = _maxValue - _minValue;
    float stepSize = range / (std::pow(2.0f, numBits) - 1);

    for (uint32_t i = 0; i < _size; ++i)
    {
        int quantizedValue = static_cast<int>((_data[i] - _minValue) / stepSize);

        _data[i] = _minValue + quantizedValue * stepSize;
    }

    _bQuantized = true;

    std::cout << "Quantization done with range [" << _minValue << ", " << _maxValue << "]";
}

void Weight::ApplyQuantizationErrorMinimization(float targetQuantizationError, float learningRate, int numWeights) {
    if (!_pbWeightGradient) {
        return;
    }

    int localWeightSize = _localSize;
    std::vector<float*> weightGradients;
    weightGradients.resize(numWeights);

    for (int i = 0; i < numWeights; ++i) {
        weightGradients[i] = _pbWeightGradient->_pDevData + i * localWeightSize;
    }

    try {
        if (!cublasHandle) {
            cublasHandle = std::make_shared<cublasHandle_t>();
            if (cublasCreate(cublasHandle.get()) != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "Failed to create cublas handle.";
                throw std::runtime_error("Failed to create cublas handle.");
            }
        }

        std::vector<std::jthread> threads;
        threads.reserve(numWeights);

        for (int i = 0; i < numWeights; ++i) {
            threads.emplace_back([this, i, &targetQuantizationError, &learningRate, &weightGradients, localWeightSize]() {
                cublasStatus_t cublasResult = cublasSscal(*cublasHandle, localWeightSize, &learningRate,
                weightGradients[i], 1);

            if (cublasResult != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "Failed to apply scaling operation with cublasSscal for Weight " << i;
                throw std::runtime_error("Failed to apply scaling operation with cublasSscal.");
            }

            std::cout << "Applied quantization error minimization for Weight " << i << " with error: "
                << targetQuantizationError << " and learning rate: " << learningRate;
                });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what();
    }
}

void Weight::DequantizeWeights()
{
    if (!_bQuantized) {
        std::cerr << "Weight is not quantized.";
        throw std::runtime_error("Weight is not quantized.");
    }

    float range = _maxValue - _minValue;
    if (range == 0.0f) {
        std::cerr << "Dequantization error due to zero range.";
        throw std::runtime_error("Dequantization error due to zero range.");
    }

    try {
        std::span<float> dataView(_data, _size);
        for (auto& value : dataView) {
            value = (value - _minValue) / range;
        }

        _bQuantized = false;

        std::cout << "Dequantization done.";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what();
    }
}

bool Weight::SerializeWeights(const std::string& filename) {
    if (_bShared) {
        std::cerr << "Serialization of shared weights is not supported.";
    }
    else if (_bLocked) {
        std::cerr << "Weight is locked.";
    }
    else if (_bQuantized) {
        std::cerr << "Weight is quantized.";
    }
    else if (_bSerialized) {
        std::cerr << "Weight is already serialized.";
    }
    else if (!_data) {
        std::cerr << "Weight data is uninitialized.";
    }
    else {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << std::format("Failed to open '{}' for writing.", filename);
            return false;
        }

        file.write(reinterpret_cast<const char*>(&_size), sizeof(_size));

        const char* rawData = reinterpret_cast<const char*>(_data);
        size_t byteCount = _size * sizeof(float);
        file.write(rawData, byteCount);

        _bSerialized = true;

        std::cout << "Serialization done.";
        return true;
    }

    return false;
}

void Weight::ApplyRegularization(float lambda)
{
    if (!_pbWeight || !_pbWeight->_pDevData) {
        std::cerr << "Invalid input data or buffers.";
        throw std::invalid_argument("Invalid input data or buffers");
    }

    cudaStream_t stream;
    cudaError_t cudaStatus = cudaStreamCreate(&stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA stream creation error: " << cudaGetErrorString(cudaStatus);
        throw std::runtime_error("CUDA stream creation failed");
    }

    auto streamGuard = [&]() {
        cudaStreamDestroy(stream);
        };

    cublasHandle_t cublasHandle;
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        streamGuard();
        std::cerr << "cuBLAS initialization error.";
        throw std::runtime_error("cuBLAS initialization failed");
    }

    auto cublasGuard = [&]() {
        cublasDestroy(cublasHandle);
        };

    if (cublasSetStream(cublasHandle, stream) != CUBLAS_STATUS_SUCCESS) {
        streamGuard();
        cublasGuard();
        std::cerr << "cuBLAS stream setting error.";
        throw std::runtime_error("cuBLAS stream setting failed");
    }

    float* pLambdaDev;
    cudaStatus = cudaMalloc(&pLambdaDev, sizeof(float));
    if (cudaStatus != cudaSuccess) {
        streamGuard();
        cublasGuard();
        std::cerr << "CUDA memory allocation error: " << cudaGetErrorString(cudaStatus);
        throw std::runtime_error("CUDA memory allocation failed");
    }

    auto lambdaGuard = [&]() {
        cudaFree(pLambdaDev);
        };

    cudaStatus = cudaMemcpyAsync(pLambdaDev, &lambda, sizeof(float), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) {
        lambdaGuard();
        streamGuard();
        cublasGuard();
        std::cerr << "CUDA memory copy error: " << cudaGetErrorString(cudaStatus);
        throw std::runtime_error("CUDA memory copy failed");
    }

    float alpha = 1.0;
    float beta = 0.0;
    int localWeightSize = _localSize;
    float* pWeightDev = _pbWeight->_pDevData;

    if (cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, localWeightSize, localWeightSize, localWeightSize, &alpha, pWeightDev, localWeightSize, pLambdaDev, localWeightSize, &beta, pWeightDev, localWeightSize) != CUBLAS_STATUS_SUCCESS) {
        lambdaGuard();
        streamGuard();
        cublasGuard();
        std::cerr << "cuBLAS Sgemm error.";
        throw std::runtime_error("cuBLAS Sgemm failed");
    }

    std::cout << "Applied L2 regularization with lambda: " << lambda;

    lambdaGuard();
    streamGuard();
    cublasGuard();
}

bool Weight::GetBiases(std::vector<float>& vBias)
{
    const auto numBiasElements = _vBias.size();

    if (getGpu()._id == 0) {
        vBias.resize(numBiasElements);
    }

    auto checkMpiError = [](int status, const char* operation) -> bool {
        if (status == MPI_SUCCESS) return true;
        std::cerr << operation << " failed: " << status;
        return false;
        };

    std::span<float> localBiasView(_vBias.data(), numBiasElements);
    std::span<float> gatheredBiasView(vBias.data(), numBiasElements);

    bool gatherSuccess = checkMpiError(
        MPI_Gather(localBiasView.data(), numBiasElements, MPI_FLOAT, gatheredBiasView.data(), numBiasElements, MPI_FLOAT, 0, MPI_COMM_WORLD),
        "MPI_Gather"
    );

    bool bcastSuccess = checkMpiError(
        MPI_Bcast(gatheredBiasView.data(), numBiasElements, MPI_FLOAT, 0, MPI_COMM_WORLD),
        "MPI_Bcast"
    );

    return gatherSuccess && bcastSuccess;
}

bool Weight::GetDimensions(std::vector<uint64_t>& dimensions) const
{
    constexpr uint64_t min_dimensionality = 1;
    constexpr uint64_t max_dimensionality = 5;

    if (_dimensionality < min_dimensionality || _dimensionality > max_dimensionality) {
        std::cerr << "Weight::GetDimensions: _dimensionality = " << _dimensionality;
        return false;
    }

    std::array<uint64_t, 5> allDimensions = { _width, _height, _length, _depth, _breadth };

    dimensions.clear();

    auto dimensionsView = std::span(allDimensions).last(_dimensionality);

    dimensions.insert(dimensions.end(), dimensionsView.begin(), dimensionsView.end());

    return true;
}

void Weight::copySingleProcessor(float* pBuffer) {
    _vWeight.resize(_localSize);
    cudaError_t cudaStatus = cudaMemcpy(_vWeight.data(), pBuffer, _localSize * sizeof(float), cudaMemcpyDefault);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus);
        throw std::runtime_error("cudaMemcpy failed");
    }
}

void Weight::copyMultipleProcessors(float* pBuffer) {
    if (getGpu()._id == 0) {
        _vWeight.resize(static_cast<size_t>(_outputLayer._stride) * _inputLayer._stride);
    }

    cudaError_t cudaStatus = cudaMemcpy(_vWeight.data(), pBuffer, _localSize * sizeof(float), cudaMemcpyDefault);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus);
        throw std::runtime_error("cudaMemcpy failed");
    }

    float* pWeight = _vWeight.data();
    uint32_t outgoingSize = _outputLayer._stride * 3;

    if (outgoingSize > _inputLayer._stride * 2) {
        processOutgoingBiggerThanIncoming(pWeight);
    }
    else {
        processIncomingBiggerThanOutgoing(pWeight);
    }
}

void Weight::processOutgoingBiggerThanIncoming(float* pWeight) {
    const size_t srcStrideBytes = _outputLayer._localStride * sizeof(float);
    const size_t destStrideBytes = _outputLayer._stride * sizeof(float);

    cudaMemcpy2D(pWeight, destStrideBytes, _vWeight.data(), srcStrideBytes, srcStrideBytes, _inputLayer._stride, cudaMemcpyDefault);

    pWeight += _outputLayer._localStride;

    for (uint32_t i = 1; i < static_cast<uint32_t>(getGpu()._numprocs); i++) {
        uint64_t size;
        MPI_Status status;
        MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
        std::vector<float> vTemp(size);
        MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);

        uint64_t lstride = size / _inputLayer._stride;
        float* pSrcWeight = vTemp.data();
        float* pDstWeight = pWeight;

        for (uint32_t j = 0; j < _inputLayer._stride; j++) {
            std::ranges::copy(pSrcWeight, pSrcWeight + lstride, pDstWeight);
            pSrcWeight += lstride;
            pDstWeight += destStrideBytes / sizeof(float);
        }
        pWeight += lstride;
    }

    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        std::cerr << "cudaMemcpy2D or MPI_Recv failed: " << cudaGetErrorString(cudaErr);
        throw std::runtime_error("cudaMemcpy2D or MPI_Recv failed");
    }
}

void Weight::processIncomingBiggerThanOutgoing(float* pWeight) {
    const size_t copySize = static_cast<size_t>(_outputLayer._stride) * _inputLayer._localStride * sizeof(float);

    cudaError_t cudaErr;
    cudaStream_t cudaStream;
    cudaStreamCreate(&cudaStream);

    cudaErr = cudaMemcpyAsync(pWeight, _vWeight.data(), copySize, cudaMemcpyDefault, cudaStream);
    if (cudaErr != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorString(cudaErr);
        throw std::runtime_error("cudaMemcpyAsync failed");
    }

    cudaStreamSynchronize(cudaStream);
    cudaStreamDestroy(cudaStream);

    pWeight += _outputLayer._stride * _inputLayer._localStride;

    for (uint32_t i = 1; i < static_cast<uint32_t>(getGpu()._numprocs); i++) {
        uint64_t size;
        MPI_Status status;

        MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);

        std::vector<float> recvBuffer(size);

        MPI_Recv(recvBuffer.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);

        int mpiErr;
        MPI_Get_count(&status, MPI_FLOAT, &mpiErr);
        if (mpiErr != MPI_SUCCESS) {
            std::cerr << "MPI_Recv failed with error code: " << mpiErr;
            throw std::runtime_error("MPI_Recv failed");
        }

        cudaStreamCreate(&cudaStream);

        cudaErr = cudaMemcpyAsync(pWeight, recvBuffer.data(), size * sizeof(float), cudaMemcpyDefault, cudaStream);
        if (cudaErr != cudaSuccess) {
            std::cerr << "cudaMemcpyAsync for received data failed: " << cudaGetErrorString(cudaErr);
            throw std::runtime_error("cudaMemcpyAsync for received data failed");
        }

        cudaStreamSynchronize(cudaStream);
        cudaStreamDestroy(cudaStream);

        pWeight += size;
    }
}

void Weight::writeToOutput(const std::vector<float>& data, const fs::path& outputPath) {
    try {
        std::ofstream outFile(outputPath);
        if (!outFile)
            throw std::runtime_error("Failed to open the file " + outputPath.string());

        const int precision = 9;
        outFile << std::fixed << std::setprecision(precision);

        for (uint32_t i = 0; i < _inputLayer._stride; ++i) {
            for (uint32_t j = 0; j < _outputLayer._stride; ++j) {
                outFile << std::format("{:.9f} ", data[static_cast<std::vector<float, std::allocator<float>>::size_type>(i) * _outputLayer._stride + j]);
            }
            outFile << '\n';
        }

        std::cout << "Data successfully written to " << outputPath.string();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what();
    }
}

void Weight::Dump(const std::filesystem::path& filename, float* pBuffer) {
    try {

        if (getGpu()._numprocs == 1) {
            copySingleProcessor(pBuffer);
        }
        else {
            copyMultipleProcessors(pBuffer);
        }

        if (getGpu()._id == 0) {
            writeToOutput(_vWeight, filename);
            std::cout << "Data dumped to " << filename << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in Weight::Dump: " << e.what();
    }
}
