#include "GpuTypes.h"
#include "Types.h"
#include "Kernels.cuh"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <future>
#include <execution>
#include "ThreadPool.h"
#include <optional>
#include <cublas_v2.h>
#include <format>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <span>
#include <fstream>
#include <filesystem>

/// <summary>
/// Initializes a new instance of the WeightDescriptor class with default values.
/// </summary>
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

/// <summary>
/// Dumps information about a CUDA cuDNN tensor descriptor.
/// </summary>
/// <param name="t">The cuDNN tensor descriptor to dump.</param>
static void DumpTensor(cudnnTensorDescriptor_t t)
{
    cudnnDataType_t dataType;
    int ndims;
    std::vector<int> vDim(16);
    std::vector<int> vStride(16);
    cudnnStatus_t cudnnStatus = cudnnGetTensorNdDescriptor(t, 8, &dataType, &ndims, vDim.data(), vStride.data());

    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnGetTensorNdDescriptor error");
    }

    // Output tensor descriptor information
    std::cout << "Tensor:   " << ndims << " dimensions" << '\n';
    std::cout << "DataType: " << dataType << '\n';

    // Output dimensions and strides information
    for (int i = 0; i < ndims; i++)
        std::cout << i << " " << vDim[i] << " " << vStride[i] << '\n';

    std::cout << '\n';
}

/// <summary>
/// Dumps information about a CUDA cuDNN filter descriptor.
/// </summary>
/// <param name="f">The cuDNN filter descriptor to dump.</param>
void DumpFilter(cudnnFilterDescriptor_t f) {
    cudnnDataType_t dataType;
    cudnnTensorFormat_t format;
    int ndims;
    std::vector<int> vDim(16);
    cudnnStatus_t cudnnStatus = cudnnGetFilterNdDescriptor(f, 5, &dataType, &format, &ndims, vDim.data());

    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnGetFilterNdDescriptor error");
    }

    // Output filter descriptor information
    std::cout << "Filter:   " << ndims << " dimensions" << '\n';
    std::cout << "DataType: " << static_cast<int>(dataType) << '\n';
    std::cout << "Format:   " << static_cast<int>(format) << '\n';

    // Output dimensions information
    for (int i = 0; i < ndims; i++) {
        std::cout << i << " " << vDim[i] << '\n';
    }

    std::cout << '\n';
}

/// <summary>
/// Dumps information about a CUDA cuDNN convolution descriptor.
/// </summary>
/// <param name="c">The cuDNN convolution descriptor to dump.</param>
static void DumpConvolution(cudnnConvolutionDescriptor_t c)
{
    cudnnDataType_t dataType;
    cudnnConvolutionMode_t mode;
    int ndims;
    std::vector<int> vPad(16);
    std::vector<int> vStride(16);
    std::vector<int> vUpscale(16);
    cudnnStatus_t cudnnStatus = cudnnGetConvolutionNdDescriptor(c, 5, &ndims, vPad.data(), vStride.data(), vUpscale.data(), &mode, &dataType);

    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnGetConvolutionNdDescriptor error");
    }

    // Output convolution descriptor information
    std::cout << "Convolution:   " << ndims << " dimensions" << std::endl;
    std::cout << "DataType:      " << dataType << std::endl;
    std::cout << "Mode:          " << mode << std::endl;

    // Output padding, stride, and upscale information for each dimension
    for (int i = 0; i < ndims; i++)
        std::cout << i << " " << vPad[i] << " " << vStride[i] << " " << vUpscale[i] << '\n';

    std::cout << '\n';
}

/// <summary>
/// Gets an attribute from a netCDF file by name.
/// </summary>
/// <param name="nc">The netCDF file.</param>
/// <param name="attrName">The name of the attribute to retrieve.</param>
/// <returns>
/// An optional containing the attribute's value as a string if the attribute exists, 
/// or std::nullopt if the attribute does not exist.
/// </returns>
std::optional<std::string> GetAttribute(netCDF::NcFile& nc, const std::string& attrName) {
    try {
        netCDF::NcGroupAtt attribute = nc.getAtt(attrName);
        if (!attribute.isNull()) {
            std::string value;
            attribute.getValues(value);
            return std::make_optional(value);
        }
    }
    catch (netCDF::exceptions::NcException& e) {
        std::cerr << "Exception: " << e.what() << '\n';
    }

    return std::nullopt;
}

/// <summary>
/// Loads a WeightDescriptor from a netCDF file for a specified index.
/// </summary>
/// <param name="fname">The name of the netCDF file.</param>
/// <param name="nc">The netCDF file.</param>
/// <param name="index">The index of the weight to load.</param>
/// <param name="wd">Reference to the WeightDescriptor to populate.</param>
/// <returns>True if the weight descriptor is successfully loaded, false otherwise.</returns>
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

/// <summary>
/// MPI broadcast function for safely broadcasting data with exception handling.
/// </summary>
/// <typeparam name="T">The data type of the elements to be broadcasted.</typeparam>
/// <param name="data">Pointer to the data to be broadcasted.</param>
/// <param name="count">The number of elements to be broadcasted.</param>
/// <param name="datatype">The MPI data type of the elements.</param>
/// <param name="root">The rank of the root process broadcasting the data.</param>
/// <param name="comm">The MPI communicator.</param>
template <typename T>
void safe_MPI_Bcast(T* data, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    if (MPI_Bcast(data, count, datatype, root, comm) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Bcast failed");
    }
}

/// <summary>
/// MPI broadcast function for broadcasting a WeightDescriptor object.
/// </summary>
/// <param name="d">A shared pointer to the WeightDescriptor object to be broadcasted.</param>
/// <returns>0 if the broadcast is successful, 1 in case of error.</returns>
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
        std::cerr << "Error in MPI_Bcast_WeightDescriptor: " << e.what() << '\n';
        return 1;
    }
}

/// <summary>
/// Overloaded output stream operator for printing a WeightDescriptor object.
/// </summary>
/// <param name="out">The output stream to write to.</param>
/// <param name="d">The WeightDescriptor object to print.</param>
/// <returns>A reference to the output stream.</returns>
std::ostream& operator<< (std::ostream& out, WeightDescriptor& d)
{
    if (getGpu()._id == 0)
    {
        out << "Input Layer:        " << d._inputLayer << '\n';
        out << "Output Layer:       " << d._outputLayer << '\n';
        out << "Width               " << d._width << '\n';
        out << "Height              " << d._height << '\n';
        out << "Length              " << d._length << '\n';
        out << "Depth               " << d._depth << '\n';
        out << "Breadth             " << d._breadth << '\n';
        out << "bShared:            " << std::boolalpha << d._bShared << '\n';
        out << "bTransposed:        " << std::boolalpha << d._bTransposed << '\n';
        if (d._bShared)
        {
            out << "sourceInputLayer:   " << d._sourceInputLayer << '\n';
            out << "sourceOutputLayer:  " << d._sourceOutputLayer << '\n';
        }
        out << "bLocked:            " << std::boolalpha << d._bLocked << '\n';
        out << "norm:               " << d._norm << '\n';
    }
    return out;
}

/// <summary>
/// Weight constructor that initializes the weight parameters.
/// </summary>
/// <param name="inputLayer">The input layer connected to this weight.</param>
/// <param name="outputLayer">The output layer connected to this weight.</param>
/// <param name="bShared">A boolean indicating whether this weight is shared.</param>
/// <param name="bTransposed">A boolean indicating whether this weight is transposed.</param>
/// <param name="bLocked">A boolean indicating whether this weight is locked.</param>
/// <param name="norm">The normalization factor for the weight.</param>
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
    // Initialize the layers connected by this weight.
    initializeLayers();

    if (_outputLayer._type == Layer::Type::Convolutional)
    {
        _transform = Convolution;
        initializeConvolution(); // Initialize convolutional weight parameters.
    }
    else
    {
        _transform = Linear;
        initializeLinear(); // Initialize linear weight parameters.
    }
}

/// <summary>
/// Initializes the connections between input and output layers.
/// </summary>
void Weight::initializeLayers()
{
    // Register this weight with the input and output layers.
    _inputLayer._vOutgoingLayer.push_back(&_outputLayer);
    _outputLayer._vIncomingLayer.push_back(&_inputLayer);
    _inputLayer._vOutgoingWeight.push_back(this);
    _outputLayer._vIncomingWeight.push_back(this);
}

/// <summary>
/// Initializes the weight transformation as Convolution.
/// Creates necessary cuDNN descriptors for convolutional operations.
/// Determines dimensions and sizes of the weight tensor.
/// </summary>
void Weight::initializeConvolution()
{
    // Set the transformation type to Convolution.
    _transform = Convolution;

    // Create cuDNN descriptors for convolution.
    cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&_convBiasTensor);
    CUDNNERROR(cudnnStatus, "Weight::initializeConvolution: Unable to create tensor descriptor");
    cudnnStatus = cudnnCreateFilterDescriptor(&_convFilterDesc);
    CUDNNERROR(cudnnStatus, "Weight::initializeConvolution: Unable to create filter descriptor");
    cudnnStatus = cudnnCreateConvolutionDescriptor(&_convDesc);
    CUDNNERROR(cudnnStatus, "Weight::initializeConvolution: Unable to create convolution descriptor");

    // Define a vector to store filter dimensions.
    std::vector<int> vFilterDim(5, 1);

    // Determine filter dimensions based on layer dimensions.
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

    // Set the cuDNN filter descriptor.
    cudnnStatus = cudnnSetFilterNdDescriptor(_convFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, _outputLayer._dimensions + 1, vFilterDim.data());
    CUDNNERROR(cudnnStatus, "Weight::initializeConvolution: Unable to set filter descriptor");

    // Set width, height, length, depth, and breadth based on filter dimensions.
    _width = vFilterDim[0];
    _height = vFilterDim[1];
    _length = vFilterDim[2];
    _depth = vFilterDim[3];
    _breadth = vFilterDim[4];

    // Define vectors for convolution padding, stride, and upscale.
    std::vector<int> vConvPad(3, 0);
    std::vector<int> vConvStride(3, 1);
    std::vector<int> vConvUpscale(3, 1);

    // Set convolution parameters based on layer dimensions.
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

    // Set the cuDNN convolution descriptor.
    cudnnStatus = cudnnSetConvolutionNdDescriptor(_convDesc, _outputLayer._kernelDimensions, vConvPad.data(), vConvStride.data(), vConvUpscale.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    CUDNNERROR(cudnnStatus, "Weight::initializeConvolution: cudnnSetConvolutionNdDescriptor failed.");

    // Define vectors for bias tensor dimensions and stride.
    std::vector<int> vBiasDim(5, 1);
    std::vector<int> vBiasStride(5, 1);

    // Set bias tensor dimensions based on filter dimensions.
    vBiasDim[1] = vFilterDim[0];

    // Set the cuDNN bias tensor descriptor.
    cudnnStatus = cudnnSetTensorNdDescriptor(_convBiasTensor, CUDNN_DATA_FLOAT, _outputLayer._dimensions + 1, vBiasDim.data(), vBiasStride.data());
    CUDNNERROR(cudnnStatus, "Weight::initializeConvolution: Unable to set bias tensor descriptor");

    // Calculate sizes for weight and bias.
    _size = vFilterDim[0] * vFilterDim[1] * _outputLayer._kernelX * _outputLayer._kernelY * _outputLayer._kernelZ;
    _biasSize = vFilterDim[0];
    _localSize = _size;
    _localBiasSize = _biasSize;

    // Print allocation information for debugging.
    if (getGpu()._id == 0)
    {
        std::cout << std::format("Weight::initializeConvolution: Allocating %" PRIu64 " bytes (%d x %d x %u", _localSize * sizeof(float), vFilterDim[0], vFilterDim[1], _outputLayer._kernelX);
        if (_outputLayer._dimensions >= 3)
            std::cout << std::format(" x %u", _outputLayer._kernelY);
        if (_outputLayer._dimensions >= 4)
            std::cout << std::format(" x %u", _outputLayer._kernelZ);
        std::cout << std::format(") for convolutional weights between layers %s and %s\n", _inputLayer._name.c_str(), _outputLayer._name.c_str());
    }
}

/// <summary>
/// Initializes the weight transformation as Linear.
/// Determines the dimensions and sizes of the weight matrix based on layer sizes.
/// </summary>
void Weight::initializeLinear()
{
    // Set the transformation type to Linear.
    _transform = Linear;

    // Calculate outgoing and incoming sizes based on layer strides.
    uint32_t outgoingSize = _outputLayer._stride * 3;
    uint32_t incomingSize = _inputLayer._stride * 2;

    // Check which layer has the larger outgoing size and adjust dimensions accordingly.
    if (outgoingSize > incomingSize)
    {
        // Update relationships between layers and weights.
        _inputLayer._vOutgoingLargerLayer.push_back(&_outputLayer);
        _inputLayer._vOutgoingLargerWeight.push_back(this);
        _width = _outputLayer._localStride;
        _height = _inputLayer._stride;
    }
    else
    {
        // Update relationships between layers and weights.
        _outputLayer._vIncomingLargerLayer.push_back(&_inputLayer);
        _outputLayer._vIncomingLargerWeight.push_back(this);
        _width = _outputLayer._stride;
        _height = _inputLayer._localStride;
    }

    // Calculate local sizes and total sizes for weight and bias.
    _localSize = _width * _height * 4 * 2 * 3;
    _localBiasSize = _outputLayer._localStride;
    _size = static_cast<uint64_t>(_outputLayer._stride) * _inputLayer._stride * 4 * 2 * 3;
    _biasSize = _outputLayer._stride;

    // Print allocation information for debugging.
    if (getGpu()._id == 0)
        std::cout << "Weight::initializeLinear: Allocating " << _localSize * sizeof(float) << " bytes (" << _width << ", " << _height << ") for fully connected weights between layers " << _inputLayer._name << " and " << _outputLayer._name << std::endl;
}

/// <summary>
/// Sets weight values from a given matrix.
/// </summary>
/// <param name="values">A 2D vector containing weight values.</param>
void Weight::setWeightValues(const std::vector<std::vector<float>>& values)
{
    // Check if the dimensions of the input matrix match weight dimensions.
    if (values.size() != _width || values[0].size() != _height)
    {
        std::cerr << "Error: Invalid weight matrix dimensions." << std::endl;
        return;
    }

    // Assign the input matrix as weight values.
    _weightMatrix = values;
}

/// <summary>
/// Randomizes the weight matrix with values between -0.5 and 0.5.
/// </summary>
void Weight::randomizeWeightMatrix()
{
    // Resize the weight matrix to match weight dimensions.
    _weightMatrix.resize(_width, std::vector<float>(_height));

    // Fill the weight matrix with random values between -0.5 and 0.5.
    for (uint32_t i = 0; i < _width; ++i)
    {
        for (uint32_t j = 0; j < _height; ++j)
        {
            _weightMatrix[i][j] = static_cast<float>(rand()) / (RAND_MAX)-0.5f;
        }
    }
}

/// <summary>
/// Destructor for the Weight class.
/// </summary>
Weight::~Weight()
{}

/// <summary>
/// Clears velocity values for weights and biases.
/// </summary>
void Weight::ClearVelocity()
{
    // Clear the velocity values for weights and biases.
    cudaMemset(_pbWeightVelocity->_pDevData, 0, _localSize * sizeof(float));
    cudaMemset(_pbBiasVelocity->_pDevData, 0, _localBiasSize * sizeof(float));

    // Optionally clear velocity values for weight and bias gradient.
    if (_pbWeightGradientVelocity)
        cudaMemset(_pbWeightGradientVelocity->_pDevData, 0, _localSize * sizeof(float));
    if (_pbBiasGradientVelocity)
        cudaMemset(_pbBiasGradientVelocity->_pDevData, 0, _localBiasSize * sizeof(float));
}

/// <summary>
/// Clears gradient values for weights.
/// </summary>
void Weight::ClearGradient()
{
    // Clear the gradient values for weights.
    cudaMemset(_pbWeightGradient->_pDevData, 0, _localSize * sizeof(float));
}

/// <summary>
/// Randomly initializes the weight values and optionally bias values.
/// </summary>
void Weight::Randomize()
{
    if (!_bShared)
    {
        float scale, bias;

        // Depending on the weight initialization method specified, generate random weights.
        switch (_outputLayer._weightInit)
        {
        case CaffeXavier:
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale = _outputLayer._weightInitScale * 2.0f * sqrtf(3.0f / _outputLayer._stride);
            bias = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
            break;

        case Xavier:
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale = _outputLayer._weightInitScale * sqrtf(6.0f / (_outputLayer._stride + _inputLayer._stride));
            bias = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
            break;

        case Uniform:
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale = 2.0f * _outputLayer._weightInitScale;
            bias = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, bias);
            break;

        case Gaussian:
            curandGenerateNormal(getGpu()._RNG, _pbWeight->_pDevData, _localSize, 0.0f, _outputLayer._weightInitScale);
            break;

        case UnitBall:
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _localSize);
            scale = _outputLayer._weightInitScale;
            kScaleAndBias(_pbWeight->_pDevData, _localSize, scale, 0.0f);
            break;

        case SELU:
            curandGenerateNormal(getGpu()._RNG, _pbWeight->_pDevData, _localSize, 0.0f, 1.0f / _inputLayer._stride);
            break;

        case Constant:
            // Initialize weights with a constant value and bias with a negative biasInit value.
            cudaMemset(_pbWeight->_pDevData, 0, _localSize * sizeof(float));
            kScaleAndBias(_pbWeight->_pDevData, _localSize, (float)0.0, _outputLayer._weightInitScale);
            break;
        };
    }

    // Initialize bias values with zeros and optionally apply a negative biasInit value.
    cudaMemset(_pbBias->_pDevData, 0, _localBiasSize * sizeof(float));
    kScaleAndBias(_pbBias->_pDevData, _localBiasSize, (float)0.0, -_outputLayer._biasInit);
}

/// <summary>
/// Locks the weight to prevent further modifications.
/// </summary>
void Weight::Lock()
{
    _bLocked = true;
}

/// <summary>
/// Unlocks the weight, allowing modifications.
/// </summary>
void Weight::Unlock()
{
    _bLocked = false;
}

/// <summary>
/// Concept check for GPU buffer types.
/// </summary>
template<typename T>
concept GpuBufferType = std::is_same_v<T, GpuBuffer<float>>;

/// <summary>
/// Resets a buffer if needed, allocating memory if the buffer is null.
/// </summary>
/// <typeparam name="BufferType">Type of the buffer (e.g., GpuBuffer).</typeparam>
/// <param name="buffer">Reference to the buffer.</param>
/// <param name="size">Size of the buffer.</param>
template<typename BufferType>
void ResetBufferIfNeeded(std::unique_ptr<BufferType>& buffer, size_t size)
{
    if (!buffer)
        buffer = std::make_unique<BufferType>(size);
}

/// <summary>
/// Resets a buffer based on the training mode, deallocating it for SGD mode.
/// </summary>
/// <typeparam name="Mode">Training mode (e.g., AdaDelta, SGD).</typeparam>
/// <typeparam name="BufferType">Type of the buffer (e.g., GpuBuffer).</typeparam>
/// <param name="buffer">Reference to the buffer.</param>
/// <param name="size">Size of the buffer.</param>
template<TrainingMode Mode, typename BufferType>
void ResetBufferBasedOnMode(std::unique_ptr<BufferType>& buffer, size_t size)
{
    if constexpr (Mode != TrainingMode::SGD)
        ResetBufferIfNeeded(buffer, size);
    else
        buffer.reset();
}

/// <summary>
/// Refreshes the state of the weight, which includes performing various tasks such as buffer resets,
/// GPU status checks, cuDNN processing, and GPU memory monitoring.
/// </summary>
/// <param name="pNetwork">Pointer to the neural network object.</param>
/// <param name="mode">The training mode (e.g., SGD).</param>
void Weight::RefreshState(Network* pNetwork, TrainingMode mode)
{
    // Define the maximum number of threads for the thread pool.
    const size_t MAX_THREADS = 4;
    ThreadPool pool(MAX_THREADS);

    // Create synchronization objects for GPU status checking.
    std::mutex cv_m;
    std::condition_variable cv;
    bool gpu_ready = false;
    
    // Lambda function to check GPU status and signal readiness.
    auto checkGPUStatus = [&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        {
            std::lock_guard<std::mutex> lock(cv_m);
            gpu_ready = true;
        }
        cv.notify_one();
        };

    // Lambda function to refresh buffers based on the training mode.
    auto refreshBuffers = [this, &mode]() {
        
        // Reset various GPU buffers based on the training mode (e.g., AdaDelta, Adam, SGD).
        ResetBufferBasedOnMode<TrainingMode::AdaDelta, GpuBuffer<float>>(_pbWeightGradientVelocity, _localSize);
        ResetBufferBasedOnMode<TrainingMode::AdaDelta, GpuBuffer<float>>(_pbBiasGradientVelocity, _localBiasSize);
        ResetBufferBasedOnMode<TrainingMode::Adam, GpuBuffer<float>>(_pbWeightGradientVelocity, _localSize);
        ResetBufferBasedOnMode<TrainingMode::Adam, GpuBuffer<float>>(_pbBiasGradientVelocity, _localBiasSize);
        ResetBufferBasedOnMode<TrainingMode::SGD, GpuBuffer<float>>(_pbWeightVelocity, _localSize);
        ResetBufferBasedOnMode<TrainingMode::SGD, GpuBuffer<float>>(_pbBiasVelocity, _localBiasSize);
        };

    // Lambda function for cuDNN processing.
    auto cudnnProcessing = [this, pNetwork]() {
        
        // Perform cuDNN processing for convolutional layers.
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
                    std::cout << "Output layer " << _outputLayer._name << ": has incorrectly calculated dimensions for cuDNN.\n";
                getGpu().Shutdown();
        }
        }
};

    // Lambda function to monitor GPU memory.
    auto monitorGPUMemory = [&]() {
        bool memoryOk = true;
        if (!memoryOk) {
        }
        };

    // Enqueue tasks to the thread pool based on the training mode.
    if (mode == TrainingMode::SGD) {
        pool.enqueue(checkGPUStatus);
        pool.enqueue(refreshBuffers);
    }
    else {
        pool.enqueue(refreshBuffers);
        pool.enqueue(cudnnProcessing);
        }
    pool.enqueue(monitorGPUMemory);

    // Wait for GPU readiness in the case of SGD mode.
    if (mode == TrainingMode::SGD) {
        {
            std::unique_lock<std::mutex> lock(cv_m);
            cv.wait(lock, [&]() { return gpu_ready; });
        }
    }

    // Use asynchronous tasks to execute functions.
    std::async(std::launch::async, refreshBuffers).wait();
    std::async(std::launch::async, cudnnProcessing).wait();
    std::async(std::launch::async, checkGPUStatus).wait();
}

/// <summary>
/// Calculate the regularization error for the weight.
/// </summary>
/// <param name="lambda">The regularization parameter (L2).</param>
/// <param name="lambda1">The regularization parameter (L1).</param>
/// <returns>The regularization error value.</returns>
float Weight::CalculateRegularizationError(float lambda, float lambda1)
{
    // Check if the weight is shared.
    if (_bShared)
        return 0; // If shared, regularization error is zero.
    else
        // Calculate the regularization error using the kCalculateRegularizationError function.
        return kCalculateRegularizationError(lambda, lambda1, _pbWeight->_pDevData, _localSize);
}

/// <summary>
/// Update the weights of a neural network layer.
/// </summary>
void Weight::UpdateWeights(TrainingMode trainingMode)
{
    cublasStatus_t cstatus;
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
            // Hyperparameters
            const float alpha = 0.01f;

            // Pointers to data arrays
            float* weightGradientDev = _pbWeightGradient->_pDevData;
            float* weightDev = _pbWeight->_pDevData;

            // Local size
            int localSize = _localSize;

            // Allocate memory for deltaWeightDev
            float* deltaWeightDev;
            cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

            // SGD update step
            cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, deltaWeightDev, 1);

            // Update weights
            cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

            // Free allocated memory
            cudaFree(deltaWeightDev);
        }
        break;

        case Momentum:
        {
            // Set the learning rate alpha to 0.01
            float alpha = 0.01f;

            // Set the momentum parameter mu to 0.9
            float mu = 0.9f;

            // Get pointers to device data for weight velocity, weight gradient, and weights.
            float* weightVelocityDev = _pbWeightVelocity->_pDevData;
            float* weightGradientDev = _pbWeightGradient->_pDevData;
            float* weightDev = _pbWeight->_pDevData;

            // Get the localSize value, which is not defined in the provided code.
            int localSize = _localSize;

            // Declare a pointer to store delta weights on the device.
            float* deltaWeightDev;

            // Allocate memory on the GPU for deltaWeightDev.
            cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

            // Scale the weight velocity by mu using cublasSscal.
            cublasSscal(handle, localSize, &mu, weightVelocityDev, 1);

            // Update weight velocity using the weight gradient and learning rate alpha.
            cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, weightVelocityDev, 1);

            // Update delta weights by adding alpha times weight velocity.
            cublasSaxpy(handle, localSize, &alpha, weightVelocityDev, 1, deltaWeightDev, 1);

            // Update weights by adding alpha times delta weights.
            cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

            // Free the memory allocated for deltaWeightDev on the GPU.
            cudaFree(deltaWeightDev);
            }
            break;
                        
            case AdaGrad:
            {
                // Hyperparameters
                const float alpha = 0.01f;

                // Pointers to data arrays
                float* weightVelocityDev = _pbWeightVelocity->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightDev = _pbWeight->_pDevData;

                // Local size
                int localSize = _localSize;

                // Allocate memory for deltaWeightDev
                float* deltaWeightDev;
                cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

                // AdaGrad update step
                cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, weightVelocityDev, 1);

                // Update delta weights
                cublasSaxpy(handle, localSize, &alpha, weightVelocityDev, 1, deltaWeightDev, 1);

                // Update weights
                cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

                // Free allocated memory
                cudaFree(deltaWeightDev);
            }
            break;
                        
            case Nesterov:
            {
                // Hyperparameters
                const float alpha = 0.01f;
                const float mu = 0.9f;

                // Pointers to data arrays
                float* weightVelocityDev = _pbWeightVelocity->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightDev = _pbWeight->_pDevData;

                // Local size
                int localSize = _localSize;

                // Allocate memory for deltaWeightDev
                float* deltaWeightDev;
                cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

                // Nesterov update step
                cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, weightVelocityDev, 1);

                cublasSscal(handle, localSize, &mu, weightVelocityDev, 1);
                cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, weightVelocityDev, 1);

                // Update delta weights
                cublasSaxpy(handle, localSize, &alpha, weightVelocityDev, 1, deltaWeightDev, 1);

                // Update weights
                cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

                // Free allocated memory
                cudaFree(deltaWeightDev);
            }
            break;
                        
            case RMSProp:
            {
                // Hyperparameters
                const float alpha = 0.01f;
                const float mu = 0.9f;
                const float epsilon = 1e-7f;

                // Pointers to data arrays

                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightDev = _pbWeight->_pDevData;

                // Local size
                int localSize = _localSize;

                // Allocate memory for deltaWeightDev
                float* deltaWeightDev;
                cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

                // Compute the squared gradient
                float squaredGradient = 0.0f;
                cublasSdot(handle, localSize, weightGradientDev, 1, weightGradientDev, 1, &squaredGradient);

                // Exponentially weighted moving average for squared gradient
                float squaredGradientAvg = 0.0f;
                squaredGradientAvg = mu * squaredGradientAvg + (1.0f - mu) * squaredGradient;

                // Calculate the scaling factor
                float scalingFactor = alpha / (sqrtf(squaredGradientAvg) + epsilon);

                // Scale the weight gradient
                cublasSscal(handle, localSize, &scalingFactor, weightGradientDev, 1);

                // Update delta weights
                cublasSaxpy(handle, localSize, &alpha, weightGradientDev, 1, deltaWeightDev, 1);

                // Update weights
                cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

                // Free allocated memory
                cudaFree(deltaWeightDev);
            }
            break;

            case AdaDelta:
            {
                // Hyperparameters
                const float mu = 0.9f;
                const float epsilon = 1e-7f;
                const float alpha = 0.01f;

                // Pointers to data arrays
                float* weightVelocityDev = _pbWeightVelocity->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightGradientVelocityDev = _pbWeightGradientVelocity->_pDevData;
                float* weightDev = _pbWeight->_pDevData;

                // Local size
                int localSize = _localSize;

                // Allocate memory for deltaWeightDev
                float* deltaWeightDev;
                cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

                // Compute the squared gradient
                float squaredGradient = 0.0f;
                cublasSdot(handle, localSize, weightGradientDev, 1, weightGradientDev, 1, &squaredGradient);

                // Exponentially weighted moving average for squared gradient
                float squaredGradientAvg = 0.0f;
                squaredGradientAvg = mu * squaredGradientAvg + (1.0f - mu) * squaredGradient;

                // Calculate the scaling factor
                float scalingFactor = (weightGradientVelocityDev && squaredGradientAvg)
                    ? sqrtf(*weightGradientVelocityDev) / (sqrtf(squaredGradientAvg) + epsilon)
                    : 0.0f;

                // Update weight velocity
                cublasSaxpy(handle, localSize, &scalingFactor, weightGradientDev, 1, weightVelocityDev, 1);

                // Update delta weights
                cublasSaxpy(handle, localSize, &alpha, weightVelocityDev, 1, deltaWeightDev, 1);

                // Update weights
                cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

                // Update weight gradient velocity with squared gradient average
                cublasSaxpy(handle, localSize, &mu, &squaredGradientAvg, 1, weightGradientVelocityDev, 1);

                // Free allocated memory
                cudaFree(deltaWeightDev);
            }
            break;

            case Adam:
            {
                // Hyperparameters
                const float alpha = 0.001f;
                const float mu = 0.9f;
                const float mu1 = 0.999f;
                const float epsilon = 1e-7f;
                const int t = 1;

                // Pointers to data arrays
                float* weightVelocityDev = _pbWeightVelocity->_pDevData;
                float* weightGradientDev = _pbWeightGradient->_pDevData;
                float* weightGradientVelocityDev = _pbWeightGradientVelocity->_pDevData;
                float* weightDev = _pbWeight->_pDevData;

                // Local size
                int localSize = _localSize;

                // Allocate memory for deltaWeightDev
                float* deltaWeightDev;
                cudaMalloc((void**)&deltaWeightDev, localSize * sizeof(float));

                // Exponentially weighted moving averages
                cublasSscal(handle, localSize, &mu, weightGradientDev, 1);
                cublasSscal(handle, localSize, &mu1, weightGradientVelocityDev, 1);

                // Bias-corrected moving averages
                float biasCorrectedMu = 1.0f / (1.0f - powf(mu, t));
                float biasCorrectedMu1 = 1.0f / (1.0f - powf(mu1, t));
                cublasSaxpy(handle, localSize, &biasCorrectedMu, weightGradientDev, 1, weightVelocityDev, 1);

                // Calculate the scaling factor
                float scalingFactor = sqrtf(biasCorrectedMu1) / epsilon;

                // Update delta weights
                cublasSaxpy(handle, localSize, &alpha, weightVelocityDev, 1, deltaWeightDev, 1);
                cublasSscal(handle, localSize, &scalingFactor, deltaWeightDev, 1);

                // Update weights
                cublasSaxpy(handle, localSize, &alpha, deltaWeightDev, 1, weightDev, 1);

                // Free allocated memory
                cudaFree(deltaWeightDev);
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
                // Hyperparameters
                const float alpha = 0.01f;
                const int batch = 32;

                // Pointers to data arrays
                float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Calculate the scaling factor
                float scaleFactor = alpha / batch;

                // Update bias values with the scaled gradient
                cublasSscal(handle, localBiasSize, &scaleFactor, deltaBiasDev, 1);
                cublasSaxpy(handle, localBiasSize, &alpha, deltaBiasDev, 1, biasDev, 1);
            }
            break;

            case Momentum:
            {
                // Hyperparameters
                const float alpha = 0.01f;
                const float mu = 0.9f;
                const int batch = 32;

                // Pointers to data arrays
                float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Calculate the scaling factor
                float scaleFactor = alpha / batch;

                // Apply momentum term
                cublasSscal(handle, localBiasSize, &mu, biasVelocityDev, 1);

                // Update bias velocity with scaled delta bias
                cublasSaxpy(handle, localBiasSize, &scaleFactor, deltaBiasDev, 1, biasVelocityDev, 1);

                // Update bias values with the corrected gradient
                cublasSaxpy(handle, localBiasSize, &alpha, biasVelocityDev, 1, biasDev, 1);
            }
            break;
                    
            case AdaGrad:
            {
                // Hyperparameters
                const float alpha = 0.01f;
                const int batch = 32;

                // Pointers to data arrays
                float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Calculate the scaling factor
                float scaleFactor = alpha / batch;

                // Update bias velocity with scaled delta bias
                cublasSaxpy(handle, localBiasSize, &scaleFactor, deltaBiasDev, 1, biasVelocityDev, 1);

                // Update bias values with the scaled gradient
                cublasSaxpy(handle, localBiasSize, &alpha, biasVelocityDev, 1, biasDev, 1);
            }
            break;
                    
            case Nesterov:
            {
                // Hyperparameters
                const float alpha = 0.01f;
                const float mu = 0.9f;
                const int batch = 32;

                // Pointers to data arrays
                float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Calculate the scaling factor
                float scaleFactor = alpha / batch;

                // Nesterov momentum update
                cublasSaxpy(handle, localBiasSize, &mu, biasVelocityDev, 1, deltaBiasDev, 1);
                cublasSaxpy(handle, localBiasSize, &scaleFactor, deltaBiasDev, 1, biasVelocityDev, 1);

                // Update bias values with the corrected gradient
                cublasSaxpy(handle, localBiasSize, &alpha, biasVelocityDev, 1, biasDev, 1);
            }
            break;
                    
            case RMSProp:
            {
                // Hyperparameters
                const float mu = 0.9f;
                const float epsilon = 1e-7f;

                // Pointers to data arrays
                float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Compute squared gradient
                float squaredGradient;
                cublasSdot(handle, localBiasSize, deltaBiasDev, 1, deltaBiasDev, 1, &squaredGradient);

                // Exponentially weighted moving averages for squared gradient
                cublasSscal(handle, localBiasSize, &mu, biasVelocityDev, 1);
                float tempMuDifference = 1.0f - mu;
                cublasSaxpy(handle, localBiasSize, &tempMuDifference, &squaredGradient, 1, biasVelocityDev, 1);

                // Calculate the scaling factor with epsilon
                float scalingFactor = (biasVelocityDev) ? sqrtf(*biasVelocityDev) / epsilon : 0.0f;

                // Update bias values
                cublasSaxpy(handle, localBiasSize, &scalingFactor, deltaBiasDev, 1, biasDev, 1);
            }
            break;
                
            case AdaDelta:
            {
                // Hyperparameters
                const float mu = 0.9f;
                const float epsilon = 1e-7f;

                // Pointers to data arrays
                float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientVelocityDev = _pbBiasGradientVelocity->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Compute squared gradient
                float squaredGradient;
                cublasSdot(handle, localBiasSize, deltaBiasDev, 1, deltaBiasDev, 1, &squaredGradient);

                // Exponentially weighted moving averages for squared gradient
                cublasSscal(handle, localBiasSize, &mu, biasGradientVelocityDev, 1);
                float tempMuDiff = 1.0f - mu;
                cublasSaxpy(handle, localBiasSize, &tempMuDiff, &squaredGradient, 1, biasGradientVelocityDev, 1);

                // Calculate the scaling factor
                float scalingFactor = (biasVelocityDev && biasGradientVelocityDev)
                    ? sqrtf(*biasVelocityDev) / (sqrtf(*biasGradientVelocityDev) + epsilon)
                    : 0.0f;

                // Update bias velocity
                cublasSaxpy(handle, localBiasSize, &scalingFactor, deltaBiasDev, 1, biasVelocityDev, 1);

                // Update bias values
                cublasSaxpy(handle, localBiasSize, &mu, deltaBiasDev, 1, biasDev, 1);
            }
            break;

            case Adam:
            {
                // Hyperparameters
                const float alpha = 0.01f;
                const float mu = 0.9f;
                const float mu1 = 0.999f;
                const int t = 1;
                const int batch = 32;
                const float epsilon = 1e-7f;

                // Pointers to data arrays
                float* deltaBiasDev = _outputLayer._pbDelta->_pDevData;
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientVelocityDev = _pbBiasGradientVelocity->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Calculate the scaling factor
                float scaleFactor = alpha / batch;

                // Update bias velocity with scaled delta bias
                cublasSaxpy(handle, localBiasSize, &scaleFactor, deltaBiasDev, 1, biasVelocityDev, 1);

                // Exponentially weighted moving averages
                cublasSscal(handle, localBiasSize, &mu, biasVelocityDev, 1);
                float tempMuSubtraction = 1.0f - mu;
                cublasSaxpy(handle, localBiasSize, &tempMuSubtraction, deltaBiasDev, 1, biasVelocityDev, 1);

                cublasSscal(handle, localBiasSize, &mu1, biasGradientVelocityDev, 1);
                float tempMu1Subtraction = 1.0f - mu1;
                cublasSaxpy(handle, localBiasSize, &tempMu1Subtraction, deltaBiasDev, 1, biasGradientVelocityDev, 1);

                // Bias-corrected moving averages
                float biasCorrectedMu = 1.0f / (1.0f - powf(mu, t));
                float biasCorrectedMu1 = 1.0f / (1.0f - powf(mu1, t));

                cublasSscal(handle, localBiasSize, &biasCorrectedMu, biasVelocityDev, 1);
                cublasSscal(handle, localBiasSize, &biasCorrectedMu1, biasGradientVelocityDev, 1);

                // Calculate the scaling factor with epsilon
                float scalingFactor = (biasGradientVelocityDev) ? sqrtf(*biasGradientVelocityDev) / epsilon : 0.0f;

                // Update bias values
                cublasSaxpy(handle, localBiasSize, &scalingFactor, biasVelocityDev, 1, biasDev, 1);
            }
            break;
        }
    }
    else
    {
        switch (trainingMode)
        {
            case SGD:
            {
                // Hyperparameter: learning rate
                const float alpha = 0.01f;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Pointers to data arrays
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Update bias gradient with learning rate alpha
                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                // Update bias values with the scaled gradient
                cublasSaxpy(handle, localBiasSize, &alpha, biasGradientDev, 1, biasDev, 1);
            }
            break;

            case Momentum:
            {
                // Hyperparameters
                const float alpha = 0.01f;
                const float mu = 0.9f;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Pointers to data arrays
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Update bias gradient with learning rate alpha
                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                // Update bias velocity with momentum term mu
                cublasSaxpy(handle, localBiasSize, &mu, biasVelocityDev, 1, biasGradientDev, 1);

                // Update bias values with the corrected gradient
                cublasSaxpy(handle, localBiasSize, &alpha, biasGradientDev, 1, biasDev, 1);
            }
            break;
                    
            case AdaGrad:
            {
                // Hyperparameters
                const float alpha = 0.01f;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Pointers to data arrays
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Update bias gradient with learning rate alpha
                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                // Accumulate squared gradients
                float oneValue = 1.0f;
                cublasSaxpy(handle, localBiasSize, &oneValue, biasGradientDev, 1, biasVelocityDev, 1);

                // Update bias values
                float negAlpha = -alpha;
                cublasSaxpy(handle, localBiasSize, &negAlpha, biasVelocityDev, 1, biasDev, 1);
            }
            break;
                        
            case Nesterov:
            {
                // Hyperparameters
                const float alpha = 0.01f;
                const float mu = 0.9f;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Pointers to data arrays
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Update bias gradient with learning rate alpha
                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                // Calculate Nesterov momentum update
                float muMinusOne = mu - 1.0f;
                cublasSaxpy(handle, localBiasSize, &mu, biasVelocityDev, 1, biasGradientDev, 1);
                cublasSaxpy(handle, localBiasSize, &muMinusOne, biasGradientDev, 1, biasVelocityDev, 1);

                // Update bias values
                float negatedAlpha = -alpha;
                cublasSaxpy(handle, localBiasSize, &negatedAlpha, biasVelocityDev, 1, biasDev, 1);
            }
            break;
                        
            case RMSProp:
            {
                // Hyperparameters
                const float alpha = 0.01f;
                const float mu = 0.9f;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Pointers to data arrays
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Update bias gradient with learning rate alpha
                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                // Compute squared gradient
                float squaredGradient;
                cublasSdot(handle, localBiasSize, biasGradientDev, 1, biasGradientDev, 1, &squaredGradient);

                // Update bias velocity with decay factor mu
                cublasSscal(handle, localBiasSize, &mu, biasVelocityDev, 1);
                float tempValue3 = 1.0f - mu;
                cublasSaxpy(handle, localBiasSize, &tempValue3, &squaredGradient, 1, biasVelocityDev, 1);

                // Update bias values
                float negativeAlphaValue = -alpha;
                cublasSaxpy(handle, localBiasSize, &negativeAlphaValue, biasGradientDev, 1, biasDev, 1);
            }
            break;

            case AdaDelta:
            {
                // Hyperparameters
                const float mu = 0.9f;
                const float epsilon = 1e-7f;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Pointers to data arrays
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasGradientVelocityDev = _pbBiasGradientVelocity->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Compute squared gradient
                float squaredGradient;
                cublasSdot(handle, localBiasSize, biasGradientDev, 1, biasGradientDev, 1, &squaredGradient);

                // Update bias gradient velocity with decay factor mu
                cublasSscal(handle, localBiasSize, &mu, biasGradientVelocityDev, 1);
                float tempValue2 = 1.0f - mu;
                cublasSaxpy(handle, localBiasSize, &tempValue2, &squaredGradient, 1, biasGradientVelocityDev, 1);

                // Compute scaling factor
                float scalingFactor = 0.0f;
                if (biasVelocityDev && biasGradientVelocityDev) {
                    scalingFactor = sqrtf(*biasVelocityDev) / (sqrtf(*biasGradientVelocityDev) + epsilon);
                }

                // Update bias values
                cublasSaxpy(handle, localBiasSize, &scalingFactor, biasGradientDev, 1, biasDev, 1);
            }
            break;

            case Adam:
            {
                // Hyperparameters
                const float alpha = 0.01f;
                const float mu = 0.9f;
                const float mu1 = 0.999f;
                int t = 1;

                // Local bias size
                int localBiasSize = _localBiasSize;

                // Pointers to data arrays
                float* biasVelocityDev = _pbBiasVelocity->_pDevData;
                float* biasGradientDev = _pbBiasGradient->_pDevData;
                float* biasGradientVelocityDev = _pbBiasGradientVelocity->_pDevData;
                float* biasDev = _pbBias->_pDevData;

                // Allocate memory for squared gradient
                float* squaredGradientDev = new float[localBiasSize];

                // Compute squared gradient
                cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, localBiasSize, 1, biasGradientDev, 1, biasGradientDev, 1, squaredGradientDev, 1);

                // Update bias gradient with learning rate alpha
                cublasSscal(handle, localBiasSize, &alpha, biasGradientDev, 1);

                // Update bias velocity with momentum mu
                cublasSaxpy(handle, localBiasSize, &mu, biasVelocityDev, 1, biasGradientDev, 1);
                cublasSscal(handle, localBiasSize, &mu, biasVelocityDev, 1);
                float tempValue = 1.0f - mu;
                cublasSaxpy(handle, localBiasSize, &tempValue, biasGradientDev, 1, biasVelocityDev, 1);

                // Update bias gradient velocity with decay factor mu1
                cublasSscal(handle, localBiasSize, &mu1, biasGradientVelocityDev, 1);
                float value = 1.0f - mu1;
                cublasSaxpy(handle, localBiasSize, &value, squaredGradientDev, 1, biasGradientVelocityDev, 1);

                // Compute bias correction terms
                float biasCorrectedMu = 1.0f / (1.0f - powf(mu, t));
                float biasCorrectedMu1 = 1.0f / (1.0f - powf(mu1, t));

                // Scale bias velocity and bias gradient velocity
                cublasSscal(handle, localBiasSize, &biasCorrectedMu, biasVelocityDev, 1);
                cublasSscal(handle, localBiasSize, &biasCorrectedMu1, biasGradientVelocityDev, 1);

                // Update bias values
                float negativeAlpha = -alpha;
                cublasSaxpy(handle, localBiasSize, &negativeAlpha, biasGradientDev, 1, biasDev, 1);

                // Clean up allocated memory
                delete[] squaredGradientDev;
            }
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
            float* pMagnitude = getGpu()._pNetwork->GetScratchBuffer(_outputLayer._stride);
            kCalculateWeightMagnitudes(_outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, _outputLayer._stride);
            kNormalizeWeightMagnitudes(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);       
        }
    }
}

/// <summary>
/// Writes weight-related data to a NetCDF file.
/// </summary>
/// <param name="nc">A reference to the NetCDF file to write to.</param>
/// <param name="index">The index of the weight.</param>
/// <param name="pWeight">A pointer to the weight data.</param>
/// <param name="pBias">A pointer to the bias data.</param>
/// <returns>Returns true if the operation is successful, otherwise false.</returns>
bool Weight::WriteNetCDF(netCDF::NcFile& nc, uint32_t index, float* pWeight, float* pBias)
{
    // Check if the GPU ID is not 0; if it's not, return true (indicating success without doing anything).
    if (getGpu()._id != 0)
        return true;

    // Generate a base name for the weight attributes based on the provided index.
    auto baseName = "weight" + std::to_string(index) + "_";

    // Define a lambda function to add a NetCDF attribute with a string value.
    const auto putAttString = [&nc, &baseName](const std::string& name, const std::string& value) {
        nc.putAtt(baseName + name, netCDF::ncChar, value.size(), value.c_str());
        };

    // Define a lambda function to add a NetCDF attribute with an integer value.
    const auto putAttInt = [&nc, &baseName](const std::string& name, int value) {
        nc.putAtt(baseName + name, netCDF::ncInt, value);
        };

    // Define a lambda function to add a NetCDF attribute with a float value.
    const auto putAttFloat = [&nc, &baseName](const std::string& name, float value) {
        nc.putAtt(baseName + name, netCDF::ncFloat, value);
        };

    // Add various attributes related to the weight, such as layer names, dimensions, and properties.
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

    // Define dimensions and variables for bias and weight data in the NetCDF file.
    auto biasDim = nc.addDim(baseName + "biasDim", _biasSize);
    auto biasVar = nc.addVar(baseName + "bias", "float", biasDim.getName());

    // If pBias is not provided, use the internal bias data from the class.
    if (!pBias) pBias = _vBias.data();

    // Write bias data to the NetCDF variable.
    biasVar.putVar(pBias);

    // Check if the weight is shared or not.
    if (_bShared)
    {
        // Add attributes specific to shared weights and their source layers.
        putAttFloat("bTransposed", _bTransposed);
        putAttString("sourceInputLayer", _pSharedWeight->_inputLayer._name);
        putAttString("sourceOutputLayer", _pSharedWeight->_outputLayer._name);
    }
    else
    {
        // Define dimensions and variables for weight data in the NetCDF file.
        auto weightDim = nc.addDim(baseName + "weightDim", _size);
        auto weightVar = nc.addVar(baseName + "weights", "float", weightDim.getName());

        // If pWeight is not provided, use the internal weight data from the class.
        if (!pWeight) pWeight = _vWeight.data();

        // Write weight data to the NetCDF variable.
        weightVar.putVar(pWeight);
    }

    // Return true to indicate successful writing of the weight data to the NetCDF file.
    return true;
}

/// <summary>
/// Copies weights and biases from a source Weight object to this Weight object.
/// </summary>
/// <param name="pSrcWeight">The source Weight object to copy from.</param>
/// <returns>True if the copy operation is successful, otherwise false.</returns>
bool Weight::CopyWeights(const Weight* pSrcWeight) {
    // Get the GPU reference
    auto& gpu = getGpu();

    // Determine the destination weight based on whether it's shared
    Weight* pDstWeight = _bShared ? _pSharedWeight : this;

    try {
        // Check if the source weight pointer is valid
        if (!pSrcWeight) {
            throw std::invalid_argument("Invalid weight pointer.");
        }

        // If the destination weight is shared, use the shared weight of the source
        pSrcWeight = _bShared ? pSrcWeight->_pSharedWeight : pSrcWeight;

        // Check if the dimensions of the source and destination weights match
        if ((_width != pSrcWeight->_width) || (_height != pSrcWeight->_height) || (_length != pSrcWeight->_length)) {
            throw std::runtime_error(std::format("Mismatched weight dimensions ({0} x {1} x {2}) versus ({3} x {4} x {5}).",
                _width, _height, _length, pSrcWeight->_width, pSrcWeight->_height, pSrcWeight->_length));
        }

        // Copy the weight and bias values from the source to the destination
        pDstWeight->_vWeight = pSrcWeight->_vWeight;
        _vBias = pSrcWeight->_vBias;

        // Upload the weight data if a buffer is available
        if (_pbWeight) {
            _pbWeight->Upload(_vWeight.data());
        }

        // Upload the bias data if a buffer is available
        if (_pbBias) {
            _pbBias->Upload(_vBias.data());
        }
    }
    catch (const std::exception& e) {
        // Handle exception
        if (gpu._id == 0) {
            std::cerr << e.what() << std::endl;
        }
        return false;
    }

    // Return true if the copying process was successful
    return true;
}

/// <summary>
/// Sets weights for a Weight object from an input vector.
/// </summary>
/// <param name="vWeight">The input vector containing weights to set.</param>
/// <returns>
///   <c>true</c> if the weights were set successfully; otherwise, <c>false</c>.
/// </returns>
bool Weight::SetWeights(const std::vector<float>& vWeight) {
    // Get GPU information for the current context.
    const auto& gpuInfo = getGpu();

    // Determine the Weight object to operate on based on sharing.
    Weight* pWeight = _bShared ? _pSharedWeight : this;

    // If there is only one GPU process:
    if (gpuInfo._numprocs == 1) {
        // Check if the input vector is smaller than the weight vector.
        if (vWeight.size() < pWeight->_vWeight.size()) {
            // Print an error message if the input vector is smaller.
            if (gpuInfo._id == 0) {
                std::cout << "Weight::SetWeights: Input vector smaller than weight vector." << std::endl;
            }
            return false; // Return false to indicate failure.
        }

        // If the input vector is larger or equal in size to the weight vector:
        if (vWeight.size() > pWeight->_vWeight.size()) {
            // Copy the first part of the input vector to the weight vector.
            std::copy_n(vWeight.begin(), pWeight->_vWeight.size(), pWeight->_vWeight.begin());
        }
        else {
            // Set the weight vector to be the same as the input vector.
            pWeight->_vWeight = vWeight;
        }

        // If there is a GPU buffer for weights, upload the data.
        if (pWeight->_pbWeight) {
            pWeight->_pbWeight->Upload(_vWeight.data());
        }
        return true; // Return true to indicate success.
    }

    // If there are multiple GPU processes:
    const int numWeights = pWeight->_vWeight.size();
    const int chunkSize = numWeights / gpuInfo._numprocs;
    const int startIdx = gpuInfo._id * chunkSize;
    const int endIdx = (gpuInfo._id == gpuInfo._numprocs - 1) ? numWeights : (gpuInfo._id + 1) * chunkSize;

    // Extract the portion of the input vector relevant to this GPU process.
    const std::vector<float> gpuWeights(vWeight.begin() + startIdx, vWeight.begin() + endIdx);

    // If there is a GPU buffer for weights, upload the data.
    if (pWeight->_pbWeight) {
        pWeight->_pbWeight->Upload(gpuWeights.data());
    }

    return true; // Return true to indicate success.
}

/// <summary>
/// Sets the biases for the weight.
/// </summary>
/// <param name="vBias">The vector of bias values to set.</param>
/// <returns>True if the biases were set successfully, false otherwise.</returns>
bool Weight::SetBiases(const std::vector<float>& vBias)
{
    // Check if the input vector is smaller than the bias vector.
    if (vBias.size() < _vBias.size())
    {
        // Print an error message if GPU ID is 0 (conditional check).
        if (getGpu()._id == 0)
        {
            std::cout << "Weight::SetBiases: Input vector smaller than bias vector.\n";
        }
        return false; // Return false to indicate failure.
    }

    // Ensure that _pbBias is not nullptr using an assertion.
    assert(_pbBias != nullptr);

    // Copy exactly _vBias.size() elements from vBias to _vBias.
    std::copy_n(vBias.begin(), _vBias.size(), _vBias.begin());

    // Upload the updated _vBias data to _pbBias.
    _pbBias->Upload(_vBias.data());

    return true; // Return true to indicate success.
}

/// <summary>
/// Retrieves the weight values and stores them in a provided vector.
/// </summary>
/// <param name="vWeight">A vector to store the weight values.</param>
/// <returns>True if the operation was successful, false otherwise.</returns>
bool Weight::GetWeights(std::vector<float>& vWeight)
{
    bool bValid = true;

    // Ensure the provided vector has enough space for the weights.
    if (vWeight.size() < _vWeight.size())
    {
        vWeight.resize(_vWeight.size());
    }

    // Check if the weight data resides on the GPU (if _pbWeight is not NULL).
    if (_pbWeight != NULL)
    {
        // Download the weight data from the GPU to the provided vector.
        _pbWeight->Download(vWeight.data());
    }
    else
    {
        // Use the locally stored weight data if it's not on the GPU.
        vWeight = _vWeight;
    }
    return bValid;
}

/// <summary>
/// Applies adaptive learning rate and momentum to weight updates using cuBLAS.
/// </summary>
/// <param name="learningRateDecay">The learning rate decay factor.</param>
/// <param name="mu">The momentum term.</param>
void Weight::ApplyAdaptiveLearningRate(float learningRateDecay, float mu)
{
    // Check if the necessary GPU buffers are available.
    if (_pbWeightVelocity && _pbWeight && _pbWeightGradient)
    {
        float* pWeightVelocityDev = _pbWeightVelocity->_pDevData;
        float* pWeightDev = _pbWeight->_pDevData;
        float* pWeightGradientDev = _pbWeightGradient->_pDevData;
        int localWeightSize = _localSize;
        cublasHandle_t cublasHandle;

        // Initialize cuBLAS
        cublasCreate(&cublasHandle);

        try
        {
            // Update weight gradient with learning rate using cuBLAS
            cublasSscal(cublasHandle, localWeightSize, &learningRateDecay, pWeightGradientDev, 1);

            // Update weight velocity with momentum term mu using cuBLAS
            cublasSaxpy(cublasHandle, localWeightSize, &mu, pWeightVelocityDev, 1, pWeightGradientDev, 1);

            // Update weight values with the corrected gradient using cuBLAS
            cublasSaxpy(cublasHandle, localWeightSize, &learningRateDecay, pWeightGradientDev, 1, pWeightDev, 1);

            // Logging
            std::cout << "Applied adaptive learning rate with decay: " << learningRateDecay << std::endl;
        }
        catch (const std::exception& e)
        {
            // Handle exceptions and log errors.
            std::cerr << "Error: " << e.what() << std::endl;
        }

        // Clean up cuBLAS
        cublasDestroy(cublasHandle);
    }
}

/// <summary>
/// Adjusts the learning rate for this Weight object.
/// </summary>
/// <param name="newLearningRate">The new learning rate to be applied.</param>
void Weight::AdjustLearningRate(float newLearningRate)
{
    if (_pbWeightGradient)
    {
        float* pWeightGradientDev = _pbWeightGradient->_pDevData;
        int localWeightSize = _localSize;

        struct CublasHandleDeleter
        {
            void operator()(cublasHandle_t* handle)
            {
                cublasDestroy(*handle);
            }
        };
        std::unique_ptr<cublasHandle_t, CublasHandleDeleter> cublasHandlePtr;

        try
        {
            cublasHandle_t cublasHandle;
            cublasCreate(&cublasHandle);
            cublasHandlePtr.reset(new cublasHandle_t(cublasHandle));

            // Scale the weight gradient with the new learning rate.
            cublasSscal(*cublasHandlePtr, localWeightSize, &newLearningRate, pWeightGradientDev, 1);

            // Output a message to the console indicating the adjusted learning rate.
            std::cout << "Adjusted learning rate to: " << newLearningRate << std::endl;
        }
        catch (const std::exception& e)
        {
            // Output an error message to the console if an exception is caught.
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
}

/// <summary>
/// Quantizes the weights to a specified number of bits.
/// </summary>
/// <param name="numBits">The number of bits to use for quantization (1 to 32).</param>
/// <exception cref="std::runtime_error">Thrown when weight is shared, locked, or already quantized.</exception>
void Weight::QuantizeWeights(int numBits)
{
    // Check if the weights are shared; if so, quantization is not supported.
    if (_bShared) throw std::runtime_error("Quantization of shared weights is not supported.");

    // Check if the number of bits is within the valid range (1 to 32).
    if (numBits < 1 || numBits > 32) throw std::runtime_error("Invalid number of bits.");

    // Check if the weight is locked; quantization is not allowed for locked weights.
    if (_bLocked) throw std::runtime_error("Weight is locked.");

    // Check if the weight is already quantized; no need to re-quantize.
    if (_bQuantized) throw std::runtime_error("Weight is already quantized.");

    // Initialize the minimum and maximum values to extreme values for finding the range.
    _minValue = std::numeric_limits<float>::max();
    _maxValue = std::numeric_limits<float>::min();

    for (uint32_t i = 0; i < _size; ++i)
    {
        // Update _minValue and _maxValue based on the actual float values
        if (_data[i] < _minValue)
        {
            _minValue = _data[i];
        }
        if (_data[i] > _maxValue)
        {
            _maxValue = _data[i];
        }
    }

    // Calculate the range and step size for quantization.
    float range = _maxValue - _minValue;
    float stepSize = range / (std::pow(2.0f, numBits) - 1);

    // Quantize each weight value within the specified range.
    for (uint32_t i = 0; i < _size; ++i)
    {
        // Calculate the quantized value as an integer
        int quantizedValue = static_cast<int>((_data[i] - _minValue) / stepSize);

        // Map the quantized integer back to the floating-point range
        _data[i] = _minValue + quantizedValue * stepSize;
    }

    // Mark the weight as quantized.
    _bQuantized = true;

    // Print a message indicating the quantization range.
    std::cout << "Quantization done with range [" << _minValue << ", " << _maxValue << "]" << std::endl;
}

/// <summary>
/// Dequantizes the weights to their original range.
/// </summary>
/// <exception cref="std::runtime_error">Thrown when the weight is not quantized.</exception>
void Weight::DequantizeWeights()
{
    // Check if the weight is not quantized; dequantization is only applicable to quantized weights.
    if (!_bQuantized) throw std::runtime_error("Weight is not quantized.");

    // Calculate the range of quantization.
    float range = _maxValue - _minValue;


    for (uint32_t i = 0; i < _size; ++i)
    {
        _data[i] = (_data[i] - _minValue) / range;
    }

    // Mark the weight as dequantized.
    _bQuantized = false;

    // Print a message indicating the dequantization is done.
    std::cout << "Dequantization done." << std::endl;
}

/// <summary>
/// SerializeWeights method serializes weight data to a binary file.
/// </summary>
/// <param name="filename">The name of the file to write the serialized data to.</param>
/// <returns>True if serialization is successful, false otherwise.</returns>
bool Weight::SerializeWeights(const std::string& filename) {
    std::string error_msg;

    // Check conditions that may prevent serialization.
    if (_bShared) {
        error_msg = "Serialization of shared weights is not supported.";
    }
    else if (_bLocked) {
        error_msg = "Weight is locked.";
    }
    else if (_bQuantized) {
        error_msg = "Weight is quantized.";
    }
    else if (_bSerialized) {
        error_msg = "Weight is already serialized.";
    }
    else if (!_data) {
        error_msg = "Weight data is uninitialized.";
    }

    // Handle and log error messages if conditions prevent serialization.
    if (!error_msg.empty()) {
        std::cerr << error_msg << std::endl;
        return false;
    }

    // Open the file for writing in binary mode.
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return false;
    }

    // Write the size of the serialized data.
    file.write(reinterpret_cast<const char*>(&_size), sizeof(_size));

    // Serialize and write the weight data.
    const char* rawData = reinterpret_cast<const char*>(_data);
    size_t byteCount = _size * sizeof(float);
    file.write(rawData, byteCount);

    // Close the file and mark weight as serialized.
    file.close();
    _bSerialized = true;

    // Log successful serialization.
    std::cout << "Serialization done." << std::endl;
    return true;
}

/// <summary>
/// GetBiases method retrieves bias data.
/// </summary>
/// <param name="vBias">A vector to store the retrieved bias data.</param>
/// <returns>True if retrieval is successful, false otherwise.</returns>
bool Weight::GetBiases(std::vector<float>& vBias)
{
    bool bValid = true;
    int numBiasElements = _vBias.size();

    // Allocate memory for receiving bias data on GPU 0.
    if (getGpu()._id == 0)
    {
        vBias.resize(numBiasElements);
    }

    // Use MPI to gather bias data from all GPUs to GPU 0.
    MPI_Gather(_vBias.data(), numBiasElements, MPI_FLOAT, vBias.data(), numBiasElements, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Broadcast the retrieved bias data from GPU 0 to all other GPUs.
    MPI_Bcast(vBias.data(), numBiasElements, MPI_FLOAT, 0, MPI_COMM_WORLD);

    return bValid;
}

/// <summary>
/// Gets the dimensions of the weight.
/// </summary>
/// <param name="dimensions">A reference to a vector to store the dimensions.</param>
/// <returns>True if successful, false if _dimensionality is out of range.</returns>
bool Weight::GetDimensions(std::vector<uint64_t>& dimensions) const
{
    // Define constants for the minimum and maximum dimensionality values.
    constexpr uint64_t min_dimensionality = 2;
    constexpr uint64_t max_dimensionality = 5;

    // Check if the current _dimensionality is outside the valid range.
    if (_dimensionality < min_dimensionality || _dimensionality > max_dimensionality) {
        // Print an error message indicating the out-of-range _dimensionality value.
        std::cout << "Weight::GetDimensions: _dimensionality = " << _dimensionality << "\n";
        // Return false to indicate failure.
        return false;
    }

    // Use a switch statement to handle different cases based on _dimensionality.
    switch (_dimensionality) {
        // If _dimensionality is 5, append the _breadth to the dimensions vector and fall through to the next case.
    case 5: dimensions.push_back(_breadth); [[fallthrough]];
        // If _dimensionality is 4, append the _depth to the dimensions vector and fall through to the next case.
    case 4: dimensions.push_back(_depth); [[fallthrough]];
        // If _dimensionality is 3, append the _length to the dimensions vector and fall through to the next case.
    case 3: dimensions.push_back(_length); [[fallthrough]];
        // If _dimensionality is 2, append the _height to the dimensions vector and fall through to the next case.
    case 2: dimensions.push_back(_height); [[fallthrough]];
        // If _dimensionality is 1, append the _width to the dimensions vector.
    case 1: dimensions.push_back(_width);
    }
    // Return true to indicate success.
    return true;
}

/// <summary>
/// Copies data from a buffer to the _vWeight vector.
/// </summary>
/// <param name="pBuffer">Pointer to the source buffer.</param>
void Weight::copySingleProcessor(float* pBuffer) {
    _vWeight.resize(_localSize);
    cudaMemcpy(_vWeight.data(), pBuffer, _localSize * sizeof(float), cudaMemcpyDefault);
}

/// <summary>
/// Copies data from a buffer to _vWeight and processes it based on certain conditions.
/// </summary>
/// <param name="pBuffer">Pointer to the source buffer.</param>
void Weight::copyMultipleProcessors(float* pBuffer) {
    if (getGpu()._id == 0) {
        // If GPU ID is 0, resize the _vWeight vector.
        _vWeight.resize(static_cast<size_t>(_outputLayer._stride) * _inputLayer._stride);
    }
    uint32_t outgoingSize = _outputLayer._stride * 3;
    cudaMemcpy(_vWeight.data(), pBuffer, _localSize * sizeof(float), cudaMemcpyDefault);

    float* pWeight = _vWeight.data();
    if (outgoingSize > _inputLayer._stride * 2) {
        // If outgoingSize is greater than inputLayer._stride * 2, call processOutgoingBiggerThanIncoming.
        processOutgoingBiggerThanIncoming(pWeight);
    }
    else {
        // Otherwise, call processIncomingBiggerThanOutgoing.
        processIncomingBiggerThanOutgoing(pWeight);
    }
}

/// <summary>
/// Processes data when outgoingSize is greater than inputLayer._stride * 2.
/// </summary>
/// <param name="pWeight">Pointer to the data to be processed.</param>
void Weight::processOutgoingBiggerThanIncoming(float* pWeight) {
    cudaMemcpy2D(pWeight, _outputLayer._stride * sizeof(float), _vWeight.data(),
        _outputLayer._localStride * sizeof(float), _outputLayer._localStride * sizeof(float),
        _inputLayer._stride, cudaMemcpyDefault);

    pWeight += _outputLayer._localStride;

    for (uint32_t i = 1; i < static_cast<uint32_t>(getGpu()._numprocs); i++) {
        // Receive data from other processes using MPI and copy it to _vWeight.
        uint64_t size;
        MPI_Status status;
        MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
        std::vector<float> vTemp(size);
        MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);

        uint64_t lstride = size / _inputLayer._stride;
        float* pSrcWeight = vTemp.data();
        float* pDstWeight = pWeight;

        // Copy data from pSrcWeight to pDstWeight in a loop.
        for (uint32_t j = 0; j < _inputLayer._stride; j++) {
            memcpy(pDstWeight, pSrcWeight, lstride * sizeof(float));
            pSrcWeight += lstride;
            pDstWeight += _outputLayer._stride;
        }
        pWeight += lstride;
    }
}

/// <summary>
/// Processes data when outgoingSize is not greater than inputLayer._stride * 2.
/// </summary>
/// <param name="pWeight">Pointer to the data to be processed.</param>
void Weight::processIncomingBiggerThanOutgoing(float* pWeight) {
    cudaMemcpy(pWeight, _vWeight.data(),
        static_cast<unsigned long long>(_outputLayer._stride) * _inputLayer._localStride * sizeof(float),
        cudaMemcpyDefault);

    pWeight += _outputLayer._stride * _inputLayer._localStride;

    for (uint32_t i = 1; i < static_cast<uint32_t>(getGpu()._numprocs); i++) {
        // Receive data from other processes using MPI and copy it to pWeight.
        uint64_t size;
        MPI_Status status;
        MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(pWeight, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
        pWeight += size;
    }
}

/// <summary>
/// Writes data to a file.
/// </summary>
/// <param name="data">Vector of float data to be written.</param>
/// <param name="filename">Path to the output file.</param>
void Weight::writeToOutput(const std::vector<float>& data, const std::filesystem::path& outputPath) {
    try {

        std::ofstream outFile(outputPath);

        if (!outFile.is_open()) {
            throw std::runtime_error("Failed to open the file " + outputPath.string());
        }

        auto pData = data.begin();

        for (uint32_t i = 0; i < _inputLayer._stride; ++i) {
            for (uint32_t j = 0; j < _outputLayer._stride; ++j) {
                outFile << std::fixed << std::setprecision(9) << *pData << " ";
                ++pData;
            }
            outFile << "\n";
        }

        if (!outFile) {
            throw std::runtime_error("Error occurred while writing to the file " + outputPath.string());
        }
    }
    catch (const std::exception& e) {
        // Handle exceptions gracefully by printing an error message.
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

/// <summary>
/// Copies and processes data and then writes it to an output file.
/// </summary>
/// <param name="filename">Path to the output file.</param>
/// <param name="pBuffer">Pointer to the source buffer.</param>
void Weight::Dump(const std::filesystem::path& filename, float* pBuffer) {
    if (getGpu()._numprocs == 1) {
        // If there is only one GPU process, use copySingleProcessor to copy data.
        copySingleProcessor(pBuffer);
    }
    else {
        // Otherwise, use copyMultipleProcessors to copy and process data.
        copyMultipleProcessors(pBuffer);
    }

    if (getGpu()._id == 0) {
        // If the GPU ID is 0, write the data to the output file.
        writeToOutput(_vWeight, filename);
    }
}
