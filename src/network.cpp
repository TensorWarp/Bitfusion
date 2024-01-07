#include "gpuTypes.h"
#include "types.h"
#include "kernels.cuh"
#define __STDC_FORMAT_MACROS
#include <queue>
#include <set>
#include <chrono>
#include <memory>
#include "enum.h"
#include "gpuSort.h"
#include "layer.h"
#include "network.h"
#include "weight.h"
#include "reader.h"
#include "value.h"
#include "mpi.h"
#include "ncException.h"
#include "ncFile.h"
#include "ncFloat.h"
#include "ncGroupAtt.h"
#include "ncInt.h"
#include "ncUint.h"
#include <cctype>
#include <cinttypes>
#include <corecrt.h>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <format>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <map>
#include <new>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <driver_types.h>
#include <numeric>
#include <unordered_map>
#include <future>
#include "threadPool.h"
#include <execution>
#include "Constants.h"

NetworkDescriptor::NetworkDescriptor() :
    _kind(Network::Kind::FeedForward),

    _errorFunction(ErrorFunction::CrossEntropy),

    _bShuffleIndices(true),

    _maxout_k(2),

    _decay((float)0.0),

    _LRN_k(2),

    _LRN_n(5),

    _LRN_alpha((float)0.0001),

    _LRN_beta((float)0.75),

    _RELUSlope((float)1.0),

    _ELUAlpha((float)1),

    _SELULambda((float)1.050701),

    _bSparsenessPenalty(false),

    _sparsenessPenalty_p((float)0.0),

    _sparsenessPenalty_beta((float)0.0),

    _bDenoising(false),

    _denoising_p((float)0.0),

    _deltaBoost_one((float)1.0),

    _deltaBoost_zero((float)1.0),

    _SMCE_oneTarget((float)0.9),

    _SMCE_zeroTarget((float)0.1),

    _SMCE_oneScale((float)1.0),

    _SMCE_zeroScale((float)1.0),

    _name(""),

    _checkpoint_name("checkpoint"),

    _checkpoint_interval(0),

    _checkpoint_epochs(0),

    _bConvLayersCalculated(false)
{}

std::ostream& operator<< (std::ostream& out, NetworkDescriptor& d)
{
    out << "Name: " << d._name << '\n';
    out << "Kind: " << d._kind << '\n';
    out << "bShuffleIndices " << std::boolalpha << d._bShuffleIndices << '\n';
    out << "Error Function: " << d._errorFunction << '\n';
    out << "MaxOut_k: " << d._maxout_k << '\n';
    out << "LRN_k: " << d._LRN_k << '\n';
    out << "LRN_n: " << d._LRN_n << '\n';
    out << "LRN_beta: " << d._LRN_beta << '\n';
    out << "LRN_alpha: " << d._LRN_alpha << '\n';
    out << "bSparsenessPenalty: " << std::boolalpha << d._bSparsenessPenalty << '\n';
    out << "sparsenessPenalty_beta: " << d._sparsenessPenalty_beta << '\n';
    out << "sparsenessPenalty_p: " << d._sparsenessPenalty_p << '\n';
    out << "bDenoising: " << std::boolalpha << d._bDenoising << '\n';
    out << "denoising_p: " << d._denoising_p << '\n';
    out << "deltaBoost_one: " << d._deltaBoost_one << '\n';
    out << "deltaBoost_zero: " << d._deltaBoost_zero << '\n';
    out << "SMCE_oneTarget: " << d._SMCE_oneTarget << '\n';
    out << "SMCE_zeroTarget: " << d._SMCE_zeroTarget << '\n';
    out << "SMCE_oneScale: " << d._SMCE_oneScale << '\n';
    out << "SMCE_zeroScale: " << d._SMCE_zeroScale << '\n';
    out << "checkpoint_name: " << d._checkpoint_name << '\n';
    out << "checkpoint_interval: " << d._checkpoint_interval << '\n';

    out << std::endl << "Layers:" << '\n';
    for (uint32_t i = 0; i < d._vLayerDescriptor.size(); i++)
    {
        out << "Layer " << i << std::endl << d._vLayerDescriptor[i];
    }

    out << std::endl << "Weights:" << '\n';
    for (uint32_t i = 0; i < d._vWeightDescriptor.size(); i++)
    {
        out << "Weight " << i << std::endl << d._vWeightDescriptor[i];
    }
    return out;
}

bool ValidateNetworkDescriptor(NetworkDescriptor& d)
{
    return true;
}

std::tuple<float, uint32_t, float, float> Network::GetLRN() const
{
    return std::make_tuple(_LRN_k, _LRN_n, _LRN_alpha, _LRN_beta);
}

std::tuple<float> Network::GetDecay() const
{
    return std::make_tuple(_decay);
}

std::tuple<uint32_t> Network::GetMaxout() const
{
    return std::make_tuple(_maxout_k);
}

std::tuple<float, float> Network::GetSparsenessPenalty() const
{
    return std::make_tuple(_sparsenessPenalty_p, _sparsenessPenalty_beta);
}

std::tuple<float> Network::GetDenoising() const
{
    return std::make_tuple(_denoising_p);
}

std::tuple<float, float> Network::GetDeltaBoost() const
{
    return std::make_tuple(_deltaBoost_one, _deltaBoost_zero);
}

std::tuple<float, float, float, float> Network::GetSMCE() const
{
    return std::make_tuple(_SMCE_oneTarget, _SMCE_zeroTarget, _SMCE_oneScale, _SMCE_zeroScale);
}

std::tuple<bool> Network::GetShuffleIndices() const
{
    return std::make_tuple(_bShuffleIndices);
}

std::tuple<std::string, int32_t> Network::GetCheckPoint() const
{
    return make_tuple(_checkpoint_name, _checkpoint_interval);
}

Layer* Network::GetLayer(const std::string& layerName) const
{
    auto itr = _mLayer.find(layerName);

    if (itr != _mLayer.end())
    {
        Layer* layer = itr->second;
        if (layer)
        {
            std::cout << "Found layer '" << layerName << "'.\n";
            return layer;
        }
        else
        {
            throw std::runtime_error("Layer '" + layerName + "' is null.");
        }
    }

    if (const auto& gpu = getGpu(); gpu._id == 0)
    {
        throw std::runtime_error("Layer '" + layerName + "' not found.");
    }

    return nullptr;
}

std::vector<const Layer*>::iterator Network::GetLayers(Layer::Kind layerKind, std::vector<const Layer*>& layers) const
{
    try
    {
        int count = 0;
        for (auto& layerName : Network::GetLayers())
        {
            const Layer* layer = Network::GetLayer(layerName);
            if (layerKind == layer->_kind)
            {
                layers.insert(layers.end(), layer);
                ++count;

                std::cout << "Added layer '" << layerName << "' of kind " << layerKind << " to the list.";
            }
        }
        return layers.end() - count;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error in Network::GetLayers: " << e.what();
        throw;
    }
}

std::shared_ptr<float[]> Network::GetScratchBuffer(size_t size)
{
    if (size > _scratchBufferSize)
    {
        _pbScratchBuffer = std::make_shared<float[]>(size);
        _scratchBufferSize = size;
    }
    return _pbScratchBuffer;
}

void Network::SetCUDNNWorkspace(size_t size)
{
    _maxCUDNNWorkspaceSize = std::max(size, _maxCUDNNWorkspaceSize);
}

float* Network::GetP2PSendBuffer()
{
    if (_sendIndex < std::size(_pbP2PBuffer)) {
        return _pbP2PBuffer[_sendIndex]->_pDevData;
    }
    else {
        throw std::out_of_range("Send index is out of bounds.");
    }
}

float* Network::GetP2PReceiveBuffer()
{
    if (_receiveIndex < std::size(_pbP2PBuffer)) {
        return _pbP2PBuffer[_receiveIndex]->_pDevData;
    }
    else {
        throw std::out_of_range("Receive index is out of bounds.");
    }
}

auto Network::GetP2PCPUBuffer() -> float*
{
    return _pCPUBuffer.get();
}

float* Network::GetPeerBuffer() const
{
    if (_receiveIndex < std::size(_pPeerBuffer)) {
        return _pPeerBuffer[_receiveIndex];
    }
    else {
        throw std::out_of_range("Receive index is out of bounds.");
    }
}

float* Network::GetPeerBackBuffer() const
{
    if (_sendIndex < std::size(_pPeerBuffer)) {
        return _pPeerBuffer[_sendIndex];
    }
    else {
        throw std::out_of_range("Send index is out of bounds.");
    }
}

bool Network::SetLRN(float k, uint32_t n, float alpha, float beta)
{
    _LRN_k = k;
    _LRN_n = n;
    _LRN_alpha = alpha;
    _LRN_beta = beta;
    _bDirty = true;

    if (!getGpu()._id) {
        std::cout << std::format("Network::SetLRN: k={}, n={}, alpha={}, beta={}\n", k, n, alpha, beta);
    }

    return true;
}

bool Network::SetDecay(float decay)
{
    const auto& gpu = getGpu();

    if (gpu._id) {
        throw std::runtime_error("GPU is currently in use, cannot set decay.");
    }

    if (decay < 0.0f) {
        throw std::invalid_argument("Invalid decay rate provided. Decay rate must be non-negative.");
    }

    _decay = decay;

    std::cout << std::format("Decay rate set to {:.6f}.\n", decay);

    return true;
}

bool Network::SetMaxout(uint32_t k)
{
    _bDirty = _maxout_k != k;
    _maxout_k = k;

    if (!getGpu()._id) {
        std::cout << std::format("Network::SetMaxout: k set to {}.\n", k);
    }

    return true;
}

bool Network::SetSparsenessPenalty(float p, float beta)
{
    const auto& gpu = getGpu();

    if (gpu._id != 0) {
        return false;
    }

    if (p < 0.0f || p > 1.0f) {
        throw std::invalid_argument("Network::SetSparsenessPenalty: Target sparseness must be in [0, 1].");
    }

    _sparsenessPenalty_p = p;
    _sparsenessPenalty_beta = beta;
    _bSparsenessPenalty = beta > 0.0f;
    _bDirty = true;

    std::cout << std::format("Network::SetSparsenessPenalty: p set to {:.6f}, beta set to {:.6f}.\n", p, beta);

    return true;
}

bool Network::SetDenoising(float p)
{
    const int gpuId = getGpu()._id;
    const bool valid = (p >= 0.0f && p < 1.0f);

    if (valid) {
        _denoising_p = p;
        _bDenoising = (p > 0.0f);
        _bDirty = true;

        if (gpuId == 0) {
            std::cout << std::format("Network::SetDenoising: p set to {:.6f}.\n", p);
        }
    }
    else if (gpuId == 0) {
        std::cerr << "Network::SetDenoising: Denoising probability must be in [0, 1).\n";
    }

    return valid;
}

bool Network::SetDeltaBoost(float one, float zero)
{
    const int gpuId = getGpu()._id;
    const bool invalidOne = (one < 0.0f);
    const bool invalidZero = (zero < 0.0f);
    const bool valid = !(invalidOne || invalidZero);

    if (gpuId == 0) {
        std::string message;

        if (invalidOne) {
            message += std::format("Illegal value for one ({:.6f}). ", one);
        }
        else {
            message += std::format("one set to {:.6f}. ", one);
        }

        if (invalidZero) {
            message += std::format("Illegal value for zero ({:.6f}). ", zero);
        }
        else {
            message += std::format("zero set to {:.6f}. ", zero);
        }

        std::cout << "Network::SetDeltaBoost: " << message << std::endl;
    }

    return valid;
}

bool Network::SetSMCE(float oneTarget, float zeroTarget, float oneScale, float zeroScale)
{
    const float minVal = 0.0f, maxVal = 1.0f;
    auto isValid = [=](float v) { return minVal <= v && v <= maxVal; };

    const std::string names[] = { "oneTarget", "zeroTarget", "oneScale", "zeroScale" };
    const float* values[] = { &oneTarget, &zeroTarget, &oneScale, &zeroScale };

    try
    {
        auto gpuAvailable = std::async(std::launch::async, [this]() {
            return getGpu()._id == 0;
            });

        std::cout << "Network::SetSMCE:\n";

        std::vector<std::string> errorMessages;

        for (int i = 0; i < 4; ++i)
        {
            if (!isValid(*values[i]))
            {
                errorMessages.emplace_back(std::format("Invalid {} value: {:.2f}", names[i], *values[i]));
            }
        }

        bool isGpuAvailable = gpuAvailable.get();

        if (!isGpuAvailable)
        {
            throw std::runtime_error("GPU not available for Network::SetSMCE.");
        }

        if (!errorMessages.empty())
        {
            std::vector<std::future<void>> errorLogTasks;
            for (const auto& error : errorMessages)
            {
                errorLogTasks.emplace_back(std::async(std::launch::async, [&error]() {
                    std::cout << error << '\n';
                    }));
            }

            for (auto& task : errorLogTasks)
            {
                task.get();
            }

            std::string combinedErrorMessage = std::accumulate(
                errorMessages.begin(), errorMessages.end(), std::string("Parameter validation failed. Errors: "),
                [](const std::string& acc, const std::string& error) {
                    return acc + error + "; ";
                });

            throw std::invalid_argument(combinedErrorMessage);
        }

        _SMCE_oneTarget = oneTarget;
        _SMCE_zeroTarget = zeroTarget;
        _SMCE_oneScale = oneScale;
        _SMCE_zeroScale = zeroScale;
        _bDirty = true;

        std::cout << "Parameters set:\n";
        for (int i = 0; i < 4; ++i)
        {
            std::cout << names[i] << " = " << std::format("{:.2f}", *values[i]) << '\n';
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception in Network::SetSMCE: " << e.what() << '\n';
        return false;
    }
}

bool Network::SetCheckpoint(std::string_view name, int32_t interval) noexcept {
    _checkpoint_name = name;
    _checkpoint_interval = interval;

    if (!getGpu()._id) {
        std::cout << std::format("Network::SetCheckpoint: filename set to {}, interval set to {} epochs.\n", name, interval);
        return true;
    }

    return false;
}

Network::Network(NetworkDescriptor& d, uint32_t batch) :
    _name(d._name),

    _kind(d._kind),

    _mode(Prediction),

    _trainingMode(SGD),

    _batch(batch),

    _localBatch(batch),

    _position(0),

    _localPosition(0),

    _bShuffleIndices(d._bShuffleIndices),

    _shuffleIndices(0),

    _pShuffleIndex(nullptr),

    _pShuffleIndexSort(),

    _pbShuffleIndex(),

    _bExamplesFound(false),

    _bAllDataLoaded(true),

    _examples(0),

    _errorFunction(d._errorFunction),

    _decay(d._decay),

    _LRN_k(d._LRN_k),

    _LRN_n(d._LRN_n),

    _LRN_alpha(d._LRN_alpha),

    _LRN_beta(d._LRN_beta),

    _maxout_k(d._maxout_k),

    _bSparsenessPenalty(d._bSparsenessPenalty),

    _sparsenessPenalty_beta(d._sparsenessPenalty_beta),

    _sparsenessPenalty_p(d._sparsenessPenalty_p),

    _bDenoising(d._bDenoising),

    _denoising_p(d._denoising_p),

    _deltaBoost_one(d._deltaBoost_one),

    _deltaBoost_zero(d._deltaBoost_zero),

    _SMCE_oneTarget(d._SMCE_oneTarget),

    _SMCE_zeroTarget(d._SMCE_zeroTarget),

    _SMCE_oneScale(d._SMCE_oneScale),

    _SMCE_zeroScale(d._SMCE_zeroScale),

    _checkpoint_name(d._checkpoint_name),

    _checkpoint_interval(d._checkpoint_interval),

    _checkpoint_epochs(0),

    _epochs(0),

    _batches(0),

    _bClearVelocity(true),

    _bDirty(true),

    _maxStride(0),

    _scratchBufferSize(0),

    _pbScratchBuffer(),

    _pPeerBuffer{ nullptr, nullptr },

    _pbP2PBuffer(),

    _pCPUBuffer(),

    _sendIndex(0),

    _receiveIndex(1),

    _CUDNNWorkspaceSize(0),

    _maxCUDNNWorkspaceSize(0),

    _pbCUDNNWorkspace(),

    _verbose(false)
{
    InitializeLayers(d);
    
    ConnectLayers(d);
    
    InitializeWeights(d);
    
    CalculatePropagationOrder();
}


void Network::ConnectLayers(NetworkDescriptor& d)
{
    for (const auto& layer : _vLayer) try {
        ValidateLayerExistence(layer);
        for (const auto& skipLayerName : layer->_vSkip) ConnectSkipLayers(layer, _mLayer.at(skipLayerName));
        if (layer->_type == Layer::Type::Pooling)
            for (const auto& sourceLayerName : layer->_vSource)
                ConnectPoolingLayers(layer, _mLayer.at(sourceLayerName));
    }
    catch (const std::exception& e) { HandleException(e, e.what()); }
}

void Network::ValidateLayerExistence(const Layer* layer) const {
    if (!_mLayer.count(layer->_name)) {
        throw LayerNotFoundException(layer->_name);
    }
}

void Network::ValidateLayerDimensionMatch(const Layer* layer, const Layer* otherLayer, bool checkDimensionMatch) const {
    if (checkDimensionMatch && (otherLayer->_stride != layer->_vIncomingLayer[0]->_stride)) {
        throw DimensionMismatchException(layer->_name, otherLayer->_name);
    }
    else if (otherLayer->_stride != layer->_stride) {
        throw DimensionMismatchException(layer->_name, otherLayer->_name);
    }
}

void Network::ConnectSkipLayers(Layer* layer, Layer* skipLayer) {
    layer->_vIncomingSkip.push_back(skipLayer);
    skipLayer->_vOutgoingSkip.push_back(layer);
}

void Network::ConnectPoolingLayers(Layer* layer, Layer* sourceLayer) {
    layer->_vIncomingLayer.push_back(sourceLayer);
    sourceLayer->_vOutgoingLayer.push_back(layer);
}

void Network::HandleException(const std::exception& e, const std::string& errorMessage) {
    std::cerr << "Network::ConnectLayers: " << errorMessage << ": " << e.what() << '\n';
}

void Network::InitializeWeights(NetworkDescriptor& descriptor) {
    for (auto& weightDescriptor : descriptor._vWeightDescriptor) {
        auto* inputLayer = _mLayer[weightDescriptor._inputLayer];
        auto* outputLayer = _mLayer[weightDescriptor._outputLayer];
        auto* weight = new Weight(*inputLayer, *outputLayer, weightDescriptor._bShared, weightDescriptor._bTransposed, weightDescriptor._bLocked, weightDescriptor._norm);
        _vWeight.push_back(weight);

        if (weightDescriptor._vWeight.empty() || weightDescriptor._vBias.empty()) {
            weight->Randomize();
        }

        if (!weightDescriptor._vWeight.empty()) {
            if (getGpu()._numprocs > 1) {
                const float* src = weightDescriptor._vWeight.data();
                float* dst = weight->_vWeight.data();
                const size_t outSize = static_cast<size_t>(outputLayer->_stride) * 3;
                const size_t inSize = static_cast<size_t>(inputLayer->_stride) * 2;
                const size_t stride = outputLayer->_localStride * sizeof(float);
                if (outSize > inSize) {
                    for (size_t i = 0; i < inputLayer->_stride; i++, src += outputLayer->_stride, dst += outputLayer->_localStride) {
                        std::memcpy(dst, src, stride);
                    }
                }
                else {
                    std::memcpy(dst, weightDescriptor._vWeight.data() + inputLayer->_minX * outputLayer->_stride, static_cast<size_t>(inputLayer->_localStride) * outputLayer->_stride * sizeof(float));
                }
            }
            else {
                weight->_vWeight = weightDescriptor._vWeight;
            }
            weight->_pbWeight->Upload(weight->_vWeight.data());
        }

        if (!weightDescriptor._vBias.empty()) {
            if (getGpu()._numprocs > 1) {
                std::memcpy(weight->_vBias.data(), weightDescriptor._vBias.data() + outputLayer->_minX, outputLayer->_localStride * sizeof(float));
            }
            else {
                weight->_vBias = weightDescriptor._vBias;
            }
            weight->_pbBias->Upload(weight->_vBias.data());
        }
    }

    for (uint32_t i = 0; i < descriptor._vWeightDescriptor.size(); i++) {
        auto& weightDescriptor = descriptor._vWeightDescriptor[i];
        if (weightDescriptor._bShared) {
            auto* weight = _vWeight[i];
            const auto& sourceInputLayer = weightDescriptor._sourceInputLayer;
            const auto& sourceOutputLayer = weightDescriptor._sourceOutputLayer;
            bool found = false;

            for (int j = 0; j < _vWeight.size(); j++) {
                auto* other = _vWeight[j];
                if (!(other->_bShared) && (other->_inputLayer._name == sourceInputLayer) && (other->_outputLayer._name == sourceOutputLayer)) {
                    if (weightDescriptor._bTransposed) {
                        if (weightDescriptor._length > 1) {
                            throw std::runtime_error("Can't transpose 3D weight matrix for shared weights between layers " +
                                weight->_inputLayer._name + " and " + weight->_outputLayer._name);
                        }
                        if ((weight->_width != other->_height) || (weight->_height != other->_width)) {
                            throw std::runtime_error("Transposed dimensions for shared weights between layers " +
                                weight->_inputLayer._name + " and " + weight->_outputLayer._name + " do not match");
                        }
                    }
                    else if ((weight->_width != other->_width) || (weight->_height != other->_height) || (weight->_length != other->_length)) {
                        throw std::runtime_error("Dimensions for shared weights between layers " +
                            weight->_inputLayer._name + " and " + weight->_outputLayer._name + " do not match");
                    }
                    weight->_pSharedWeight = other;
                    if (other->_sharingCount == 1) {
                        _vSharedWeight.push_back(other);
                    }
                    other->_sharingCount++;
                    found = true;
                    break;
                }
            }

            if (!found) {
                throw std::runtime_error("Unable to locate shared weights for connection between layers " +
                    weight->_inputLayer._name + " and " + weight->_outputLayer._name);
            }
        }
    }
}

void Network::Randomize()
{
    for (auto pw : _vWeight)
        pw->Randomize();
}

void Network::SetBatch(uint32_t batch) {
    if (batch % getGpu()._numprocs && !(getGpu()._id == 0 && std::cerr << "Network::SetBatch: Batch size must be a multiple of process count.")) {
        return;
    }

    if (batch != _batch) {
        _batch = batch;
        for (auto& pL : _vLayer) {
            pL->SetBatch(batch);
        }
        _bDirty = true;
        if (getGpu()._id == 0) {
            std::cout << "Network::SetBatch: Batch size set to " << _batch << ".";
        }
    }
}

uint32_t Network::GetBatch() const
{
    return _batch;
}

uint32_t Network::GetExamples() const
{
    return _examples;
}

void Network::SetShuffleIndices(bool bShuffleIndices) {
    if (_bShuffleIndices != bShuffleIndices) {
        _bShuffleIndices = bShuffleIndices;
        _bDirty = true;
    }

    if (getGpu()._id == 0) {
        std::cout << "Network::SetShuffleIndices: Index shuffling is now " << (_bShuffleIndices ? "on" : "off");
    }
}

uint32_t Network::GetPosition() const
{
    return _position;
}

void Network::SetPosition(uint32_t position) {
    if (_bExamplesFound) {
        if (position < _examples) {
            _position = position;
        }
        else if (getGpu()._id == 0) {
            std::cerr << "Network::SetPosition: Invalid position setting: " << position << ", maximum " << _examples;
        }
    }
    else if (getGpu()._id == 0) {
        std::cerr << "Network::SetPosition: Illegal attempt to set position without examples count information.";
    }
}

bool Network::LockWeights(const std::string& inputLayer, const std::string& outputLayer) {
    Layer* pInputLayer = _mLayer[inputLayer];
    Layer* pOutputLayer = _mLayer[outputLayer];

    if (pInputLayer == nullptr) {
        if (getGpu()._id == 0) {
            std::cerr << "Network::LockWeights: Unable to find input layer '" << inputLayer << "'.";
        }
        return false;
    }

    if (pOutputLayer == nullptr) {
        if (getGpu()._id == 0) {
            std::cerr << "Network::LockWeights: Unable to find output layer '" << outputLayer << "'.";
        }
        return false;
    }

    for (uint32_t i = 0; i < _vWeight.size(); i++) {
        if ((_vWeight[i]->_inputLayer._name == pInputLayer->_name) && (_vWeight[i]->_outputLayer._name == pOutputLayer->_name)) {
            _vWeight[i]->Lock();
            return true;
        }
    }

    if (getGpu()._id == 0) {
        std::cerr << "Network::LockWeights: Unable to find weight matrix between input layer '" << inputLayer
            << "' and output layer '" << outputLayer << "'.";
    }
    return false;
}

bool Network::UnlockWeights(const std::string& inputLayer, const std::string& outputLayer) {
    Layer* pInputLayer = _mLayer[inputLayer];
    Layer* pOutputLayer = _mLayer[outputLayer];

    if (pInputLayer == nullptr) {
        if (getGpu()._id == 0) {
            std::cerr << "Network::UnlockWeights: Unable to find input layer " << inputLayer << ".";
        }
        return false;
    }

    if (pOutputLayer == nullptr) {
        if (getGpu()._id == 0) {
            std::cerr << "Network::UnlockWeights: Unable to find input layer " << outputLayer << ".";
        }
        return false;
    }

    for (uint32_t i = 0; i < _vWeight.size(); i++) {
        if ((_vWeight[i]->_inputLayer._name == pInputLayer->_name) && (_vWeight[i]->_outputLayer._name == pOutputLayer->_name)) {
            _vWeight[i]->Unlock();
            return true;
        }
    }

    if (getGpu()._id == 0) {
        std::cerr << "Network::UnlockWeights: Unable to find weight matrix between input layer " << inputLayer
            << " and output layer " << outputLayer << ".";
    }
    return false;
}

void Network::SetTrainingMode(TrainingMode mode) {
    if (_trainingMode != mode) {
        _trainingMode = mode;
        _bDirty = true;
    }

    if (getGpu()._id == 0) {
        std::cout << "Network::SetTrainingMode: Optimizer is now " << _trainingMode;
    }
}

void Network::RefreshShuffleBuffers() {

    if (!(_bAllDataLoaded && _bShuffleIndices && _mode == Training && _shuffleIndices != _examples)) {
        throw std::runtime_error("RefreshShuffleBuffers: Invalid conditions, cannot refresh shuffle buffers.");
    }

    const bool gpuZero = (getGpu()._id == 0);
    _shuffleIndices = _examples;

    MPI_Comm mpiComm = MPI_COMM_WORLD;

    if (gpuZero) {
        _pShuffleIndexSort = std::make_unique<GpuSort<uint32_t, uint32_t>>(_shuffleIndices, mpiComm);
        _pShuffleIndex = _pShuffleIndexSort->GetValuePointer();

        const size_t vIndexSize = (static_cast<size_t>((_shuffleIndices + 511) >> 9)) << 9;
        std::vector<uint32_t> vIndex(vIndexSize, 0);

        std::iota(vIndex.begin(), vIndex.begin() + _examples, 0);
        _pShuffleIndexSort->GetValueBuffer()->Upload(vIndex.data());

        std::cout << "Network::RefreshShuffleBuffers: Shuffle buffers refreshed on GPU " << (gpuZero ? 0 : getGpu()._id) << ".";
    }
    else {
        _pbShuffleIndex = std::make_unique<GpuBuffer<uint32_t>>(_shuffleIndices, mpiComm);
        _pShuffleIndex = _pbShuffleIndex->_pDevData;

        std::cout << "Network::RefreshShuffleBuffers: Shuffle buffers refreshed on GPU " << getGpu()._id << ".";
    }
}

void Network::ShuffleIndices() {
    int gpuId = getGpu()._id;
    int numProcs = getGpu()._numprocs;

    try {
        if (gpuId == 0) {
            const uint32_t stride = ((_shuffleIndices + 511) >> 9) << 9;
            std::vector<uint32_t> vIndex(stride * 2);
            std::iota(vIndex.begin(), vIndex.begin() + _examples, 0);

            _pShuffleIndexSort->GetValueBuffer()->Upload(vIndex.data());

            auto start = std::chrono::high_resolution_clock::now();
            curandStatus_t curandStatus = curandGenerate(getGpu()._RNG, _pShuffleIndexSort->GetKeyPointer(), _shuffleIndices);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            if (curandStatus != CURAND_STATUS_SUCCESS) {
                throw std::runtime_error("curandGenerate failed on GPU 0.");
            }
            _pShuffleIndexSort->Sort();

            std::cout << "Network::ShuffleIndices: Shuffle complete on GPU 0." << std::endl;
            std::cout << "Shuffle time on GPU 0: " << duration.count() << " milliseconds" << std::endl;
        }

        if (numProcs > 1) {
            cudaError_t cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error("cudaDeviceSynchronize failed.");
            }

            MPI_Barrier(MPI_COMM_WORLD);

            P2P_Bcast(_pShuffleIndex, _examples * sizeof(uint32_t));

            std::cout << "Network::ShuffleIndices: Shuffle broadcasted among GPUs." << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in Network::ShuffleIndices: " << e.what() << std::endl;
    }
}

void Network::RefreshState() {
    try {
        auto input_future = std::async(std::launch::async, [this]() {
            return std::all_of(_vInputLayer.begin(), _vInputLayer.end(), [](const Layer* l) {
                return l->_pDataSet != nullptr;
                });
            });

        bool output_data_loaded = true;
        if (_mode != Prediction) {
            output_data_loaded = std::all_of(_vOutputLayer.begin(), _vOutputLayer.end(), [](const Layer* l) {
                return l->_pDataSet != nullptr;
                });
        }

        bool input_data_loaded = input_future.get();

        if (!input_data_loaded || !output_data_loaded) {
            throw std::runtime_error("Missing data sets.");
        }

        if (_bDirty) {
            ThreadPool thread_pool(std::thread::hardware_concurrency());

            for (auto& l : _vLayer) {
                if (l->_bDirty) {
                    thread_pool.enqueue([&]() {
                        l->RefreshState(this, _trainingMode, _mode == Validation);
                        });
                }
            }

            for (auto& w : _vWeight) {
                thread_pool.enqueue([&]() {
                    w->RefreshState(this, _trainingMode);
                    });
            }

            thread_pool.enqueue([&]() {
                RefreshShuffleBuffers();
                });

            thread_pool.enqueue([&]() {
                thread_pool.~ThreadPool();
                });
        }

        if (getGpu()._numprocs > 1) {
            DeallocatePeerBuffers();
            AllocatePeerBuffers();
        }

        if (_maxCUDNNWorkspaceSize > _CUDNNWorkspaceSize) {
            if (getGpu()._id == 0) {
                std::cout << "Network::RefreshState: Setting cuDNN workspace size to " << _maxCUDNNWorkspaceSize << " bytes.";
            }
            _CUDNNWorkspaceSize = _maxCUDNNWorkspaceSize;
            _pbCUDNNWorkspace.reset(new GpuBuffer<uint8_t>(_CUDNNWorkspaceSize));
        }

        if (_bDirty || (getGpu()._pNetwork != this)) {
            getGpu().SetNeuralNetwork(this);
        }

        _bDirty = false;
    }
    catch (const std::exception& e) {
        std::cerr << "Network::RefreshState: Error - " << e.what();
    }
}

void Network::ClearDataSets() {
    _examples = 0;
    _bExamplesFound = false;

    for (auto l : _vInputLayer) {
        if (l->_pDataSet != nullptr) {
            std::cout << "Network::ClearDataSets: Cleared data set for input layer " << l->_name;
            l->_pDataSet = nullptr;
        }
    }

    for (auto l : _vOutputLayer) {
        if (l->_pDataSet != nullptr) {
            std::cout << "Network::ClearDataSets: Cleared data set for output layer " << l->_name;
            l->_pDataSet = nullptr;
        }
    }
}

void Network::LoadDataSets(std::vector<DataSetBase*> datasets) {
    _bAllDataLoaded = false;

    for (auto l : _vInputLayer) {
        bool foundDataSet = false;
        for (auto d : datasets) {
            if (l->_dataSet.compare(d->_name) == 0) {
                foundDataSet = true;

                if (l->_dimensions != d->_dimensions) {
                    std::cerr << "Network::LoadDataSets: Dimensionality mismatch " << l->_dimensions << "D input layer '" << l->_name
                        << "' versus " << d->_dimensions << "D data set '" << d->_name << "'";
                }

                if ((l->_Nx < d->_width) || (l->_Ny < d->_height) || (l->_Nz < d->_length)) {
                    std::cerr << "Network::LoadDataSets: Data element mismatch (" << l->_Nx << ", " << l->_Ny << ", " << l->_Nz
                        << ") input layer '" << l->_name << "' versus (" << d->_width << ", " << d->_height << ", " << d->_length
                        << ") data set '" << d->_name << "'";
                    break;
                }

                if (!_bExamplesFound) {
                    _examples = d->_examples;
                    _bExamplesFound = true;
                }

                if (d->_examples != _examples) {
                    std::cerr << "Network::LoadDataSets: Mismatched examples count (" << _examples << ", " << d->_examples
                        << ") in dataset '" << d->_name << "'";
                    break;
                }

                l->_pDataSet = d;
                l->_bSparse = d->_attributes & DataSetEnums::Attributes::Sparse;
                l->_bDirty = true;
                std::cout << "Network::LoadDataSets: Found data set '" << d->_name << "' for input layer '" << l->_name << "'";
                break;
            }
        }

        if (!foundDataSet) {
            std::cerr << "Network::LoadDataSets: No matching data set found for input layer '" << l->_name << "'";
        }
    }

    for (auto l : _vOutputLayer) {
        bool foundDataSet = false;
        for (auto d : datasets) {
            if (l->_dataSet.compare(d->_name) == 0) {
                foundDataSet = true;

                if (l->_dimensions != d->_dimensions) {
                    std::cerr << "Network::LoadDataSets: Dimensionality mismatch " << l->_dimensions << "D output layer '" << l->_name
                        << "' versus " << d->_dimensions << "D data set '" << d->_name << "'";
                }

                if ((l->_Nx < d->_width) || (l->_Ny < d->_height) || (l->_Nz < d->_length)) {
                    std::cerr << "Network::LoadDataSets: Data element mismatch (" << l->_Nx << ", " << l->_Ny << ", " << l->_Nz
                        << ") output layer '" << l->_name << "' versus (" << d->_width << ", " << d->_height << ", " << d->_length
                        << ") data set '" << d->_name << "'";
                    break;
                }

                if (!_bExamplesFound) {
                    _examples = d->_examples;
                    _bExamplesFound = true;
                }

                if (d->_examples != _examples) {
                    std::cerr << "Network::LoadDataSets: Mismatched examples count (" << _examples << ", " << d->_examples
                        << ") in dataset '" << d->_name << "'";
                    break;
                }

                l->_pDataSet = d;
                l->_bDirty = true;
                std::cout << "Network::LoadDataSets: Found data set '" << d->_name << "' for output layer '" << l->_name << "'";
                break;
            }
        }

        if (!foundDataSet) {
            std::cerr << "Network::LoadDataSets: No matching data set found for output layer '" << l->_name << "'";
        }
    }

    _bDirty = true;
}

void Network::LoadBatch()
{
    if (_bDirty)
        RefreshState();

    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    for (auto l : _vInputLayer)
    {
        switch (_mode)
        {
        case Prediction:
            l->LoadPredictionBatch(_position, batch);
            break;

        case Training:
            l->LoadTrainingBatch(_position, batch);
            break;

        case Validation:
            l->LoadValidationBatch(_position, batch);
            break;

        default:
            std::cerr << "Unsupported mode in LoadBatch";
            exit(EXIT_FAILURE);
        }
    }
}

void Network::SaveWeights(const std::string& fname, const std::string& inputLayer, const std::string& outputLayer)
{
    if (getGpu()._id != 0)
    {
        return;
    }

    Layer* pInputLayer = _mLayer[inputLayer];
    Layer* pOutputLayer = _mLayer[outputLayer];

    if (pInputLayer == nullptr)
    {
        std::cerr << "Network::SaveWeights: Unable to find input layer " << inputLayer;
        return;
    }

    if (pOutputLayer == nullptr)
    {
        std::cerr << "Network::SaveWeights: Unable to find output layer " << outputLayer;
        return;
    }

    bool foundWeightMatrix = false;
    for (const auto& w : _vWeight)
    {
        if ((w->_inputLayer._name == pInputLayer->_name) && (w->_outputLayer._name == pOutputLayer->_name))
        {
            std::ofstream outFile(fname);
            if (!outFile)
            {
                std::cerr << "Network::SaveWeights: Failed to open output file " << fname;
                return;
            }

            w->_pbWeight->Download(w->_vWeight.data());
            w->_pbBias->Download(w->_vBias.data());

            outFile << w->_width << "," << w->_height << "\n";
            for (int j = 0; j < w->_height; j++)
            {
                for (int k = 0; k < w->_width; k++)
                {
                    outFile << std::fixed << std::setprecision(8) << w->_vWeight[j * w->_width + k];
                    if (k != w->_width - 1)
                        outFile << ",";
                    else
                        outFile << "\n";
                }
            }
            for (int k = 0; k < w->_width; k++)
            {
                outFile << std::fixed << std::setprecision(8) << w->_vBias[k];
                if (k != w->_width - 1)
                    outFile << ",";
                else
                    outFile << "\n";
            }

            foundWeightMatrix = true;
            break;
        }
    }

    if (!foundWeightMatrix)
    {
        std::cerr << "Network::SaveWeights: Unable to find weight matrix between input layer " << inputLayer << " and output layer " << outputLayer;
        return;
    }
}

void Network::SaveLayer(const std::string& fname, const std::string& layer)
{
    bool bResult = true;
    if (getGpu()._id == 0)
    {
        Layer* pLayer = _mLayer[layer];
        if (pLayer == nullptr)
        {
            std::cerr << "Network::SaveLayer: Attempt to save nonexistent layer '" << layer << "'.";
            bResult = false;
            goto exit;
        }

        FILE* fp = nullptr;
        errno_t err = fopen_s(&fp, fname.c_str(), "w");
        if (err != 0 || fp == nullptr)
        {
            std::cerr << "Network::SaveLayer: Failed to open output file '" << fname << "'.";
            bResult = false;
            goto exit;
        }
        DumpLayer(fp, layer);
        fclose(fp);
    }

exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        std::cerr << "Network::SaveLayer: Error occurred while saving layer '" << layer << "'.";
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }
}

void Network::DumpLayer(FILE* fp, const std::string& layer)
{
    bool bResult = true;
    if (getGpu()._id != 0)
    {
        return;
    }

    Layer* pLayer = _mLayer[layer];
    if (pLayer == nullptr)
    {
        std::cerr << "Network::DumpLayer: Attempt to dump nonexistent layer '" << layer << "'.";
        MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (!bResult)
        {
            std::cerr << "Network::DumpLayer: Error occurred while dumping layer '" << layer << "'.";
            getGpu().Shutdown();
            exit(EXIT_FAILURE);
        }
        return;
    }

    const uint64_t batch = pLayer->_batch;
    const uint64_t position_limit = static_cast<uint64_t>(_examples) - static_cast<uint64_t>(_position);
    const uint64_t batch_size = std::min(batch, position_limit);
    const uint32_t stride = pLayer->_localStride;
    const uint64_t size = batch_size * static_cast<uint64_t>(stride);

    std::vector<float> vData(size);
    pLayer->_pbUnit->Download(vData.data());

    for (uint64_t j = 0; j < batch_size; j++)
    {
        for (uint32_t k = 0; k < stride; k++)
        {
            fprintf(fp, "%f", vData[j * stride + k]);
            if (k < (stride - 1))
            {
                fprintf(fp, ",");
            }
            else
            {
                fprintf(fp, "\n");
            }
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        std::cerr << "Network::DumpLayer: Error occurred while dumping layer '" << layer << "'.";
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }
}

void Network::SaveBatch(const std::string& fname)
{
    bool bResult = true;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        FILE* fp = nullptr;
        errno_t err = fopen_s(&fp, fname.c_str(), "w");
        if (err != 0 || fp == nullptr)
        {
            std::cerr << "Network::SaveBatch: Failed to open output file '" << fname << "'.";
            bResult = false;
        }
        else
        {
            DumpBatch(fp);
            fclose(fp);
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult)
    {
        std::cerr << "Network::SaveBatch: Error occurred while saving batch to file '" << fname << "'.";
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }
}

void Network::DumpBatch(FILE* fp)
{
    if (getGpu()._id != 0)
    {
        std::cout << "Only GPU 0 can dump batches.";
        return;
    }

    for (int i = 0; i < _vOutputLayer.size(); i++)
    {
        uint32_t stride = _vOutputLayer[i]->_localStride;
        uint32_t batch = _vOutputLayer[i]->_batch;

        if (batch + _position > _examples)
        {
            batch = _examples - _position;
        }

        uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
        std::vector<float> vData(size);
        _vOutputLayer[i]->_pbUnit->Download(vData.data());

        for (uint32_t j = 0; j < batch; j++)
        {
            for (uint32_t k = 0; k < stride; k++)
            {
                std::string formatted = (k < (stride - 1)) ?
                    std::to_string(vData[static_cast<std::vector<float, std::allocator<float>>::size_type>(j) * stride + k]) + ',' :
                    std::to_string(vData[static_cast<std::vector<float, std::allocator<float>>::size_type>(j) * stride + k]) + '\n';
                fwrite(formatted.c_str(), 1, formatted.size(), fp);
            }
        }
    }
}

void Network::PredictBatch(uint32_t layers)
{
    uint32_t maxLayers = _vLayer.size();
    if (layers > maxLayers)
    {
        std::cerr << "Attempt to predict more layers than present in neural network " << _name;
        return;
    }

    if (_mode != Prediction)
    {
        _mode = Prediction;
        _bDirty = true;
    }

    if (_bDirty)
    {
        RefreshState();

        if (!_bAllDataLoaded)
        {
            std::cerr << "Attempt to predict with neural network " << _name << " without providing data sets for all input and output layers.";
            getGpu().Shutdown();
            exit(EXIT_FAILURE);
        }
    }

    uint32_t batch = _batch;
    if (_position + batch > _examples)
    {
        batch = _examples - _position;
    }

    ClearUpdates();
    LoadBatch();

    for (auto l : _vFPOrder)
    {
        l->ForwardPropagate(_position, batch, false);
    }
}

void Network::PredictTrainingBatch(uint32_t layers)
{
    const uint32_t maxLayers = static_cast<uint32_t>(_vLayer.size());

    if (layers > maxLayers)
    {
        std::cerr << "Attempt to predict more layers than present in neural network " << _name;
        return;
    }

    if (_bDirty)
    {
        RefreshState();

        if (!_bAllDataLoaded)
        {
            std::cerr << "Attempt to predict with neural network " << _name << " without providing data sets for all input and output layers.";
            getGpu().Shutdown();
            exit(EXIT_FAILURE);
        }
    }

    uint32_t batch = _batch;
    if (_position + batch > _examples)
    {
        batch = _examples - _position;
    }

    LoadBatch();

    for (auto& layer : _vFPOrder)
    {
        layer->ForwardPropagate(_position, batch, true);
    }
}

void Network::PredictValidationBatch(uint32_t layers)
{
    const uint32_t maxLayers = static_cast<uint32_t>(_vLayer.size());

    if (layers > maxLayers)
    {
        std::cerr << "Attempt to predict more layers than present in neural network " << _name;
        return;
    }

    if (_mode != Validation)
    {
        _mode = Validation;
        _bDirty = true;
    }

    if (_bDirty)
    {
        RefreshState();

        if (!_bAllDataLoaded)
        {
            std::cerr << "Attempt to predict with neural network " << _name << " without providing data sets for all input and output layers.";
            getGpu().Shutdown();
            exit(EXIT_FAILURE);
        }
    }

    uint32_t batch = _batch;
    if (_position + batch > _examples)
    {
        batch = _examples - _position;
    }

    LoadBatch();

    ClearUpdates();
    for (auto& layer : _vFPOrder)
    {
        layer->ForwardPropagate(_position, batch, false);
    }
}

float Network::Train(uint32_t epochs, float initialAlpha, float lambda, float lambda1, float mu, float mu1)
{
    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    if (_mode != Training)
    {
        _mode = Training;
        _bDirty = true;
    }

    if (_bDirty)
    {
        RefreshState();

        if (!_bAllDataLoaded)
        {
            std::cerr << "Attempt to train neural network " << _name << " without providing data sets for all input and output layers.";
            getGpu().Shutdown();
            throw std::runtime_error("Data sets not provided for all layers.");
        }
    }

    if (_trainingMode != SGD && _bClearVelocity)
    {
        for (auto& weight : _vWeight)
            weight->ClearVelocity();
        _batches = 0;
    }

    float total_error_training = 0.0f;
    float total_error_regularization = 0.0f;
    float average_error_training = std::numeric_limits<float>::max();
    float average_error_regularization = 0.0f;
    float moving_average = 0.0f;
    uint32_t brake_steps = 0;
    uint32_t init_steps = 100;
    float alpha = initialAlpha;

    std::vector<float> error_history;
    error_history.reserve(epochs);

    for (uint32_t epoch = 0; epoch < epochs; epoch++)
    {
        auto start = std::chrono::steady_clock::now();
        total_error_training = 0.0f;
        total_error_regularization = 0.0f;

        if (_bDenoising)
        {
            for (auto& inputLayer : _vInputLayer)
            {
                if (inputLayer->_bDenoising)
                    inputLayer->GenerateDenoisingData();
            }
        }

        if (_bShuffleIndices)
        {
            ShuffleIndices();
        }

        uint32_t totalExamples = GetExamples();
        uint32_t batchSize = totalExamples / mpiSize;
        uint32_t startIndex = mpiRank * batchSize;
        uint32_t endIndex = (mpiRank == mpiSize - 1) ? totalExamples : startIndex + batchSize;

        for (uint32_t pos = startIndex; pos < endIndex; pos += GetBatch())
        {
            SetPosition(pos);
            ClearUpdates();
            PredictTrainingBatch();
            float error_training, error_regularization;
            std::tie(error_training, error_regularization) = CalculateError(lambda, lambda1);
            uint32_t minibatch = std::min(GetBatch(), endIndex - pos);
            total_error_training += error_training;
            total_error_regularization += error_regularization * minibatch;

            if (_verbose && mpiRank == 0)
            {
                std::cout << "Network::Train: Minibatch@" << pos << ", average error " << ((error_training / minibatch) + error_regularization)
                    << " (" << (error_training / minibatch) << " training, " << error_regularization << " regularization), alpha " << alpha;
            }

            float step_alpha = alpha;

            if (epoch >= 5)
            {
                step_alpha = alpha * 0.95f;
            }

            ApplyWeightDecayAndMomentum(lambda, mu);

            moving_average = 0.9f * moving_average + 0.1f * error_training;

            if (init_steps == 0)
            {
                if (error_training > 2.0f * moving_average)
                {
                    brake_steps = 25;
                    if (mpiRank == 0)
                        std::cout << "Network::Train: Detected network divergence, attempting recovery.";
                }
            }
            else
                init_steps--;

            if (brake_steps > 0)
            {
                step_alpha *= 0.1f;
                brake_steps--;
            }

            if (brake_steps < 24)
            {
                BackPropagate();
                _batches++;

                ApplyGradientClipping(5.0f);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        auto end = std::chrono::steady_clock::now();
        average_error_training = total_error_training / GetExamples();
        average_error_regularization = total_error_regularization / GetExamples();

        if (mpiRank == 0)
        {
            std::cout << "Network::Train: Epoch " << ++_epochs << ", average error " << (average_error_training + average_error_regularization)
                << ", average training error " << average_error_training << ", average regularization error " << average_error_regularization
                << ", elapsed time " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s";
        }

        error_history.push_back(average_error_training + average_error_regularization);

        if (_checkpoint_interval > 0 && epoch % _checkpoint_interval == 0)
        {
            std::string filename = _checkpoint_name + std::to_string(_epochs) + ".nc";
            if (mpiRank == 0)
            {
                std::cout << "Network::Train: saving checkpoint " << filename;
            }

            SaveNetCDF(filename);
            _checkpoint_epochs = 0;
        }
    }

    return average_error_training + average_error_regularization;
}

void Network::ApplyGradientClipping(float threshold)
{
    for (auto& weight : _vWeight)
    {
        weight->ApplyGradientClipping(threshold);
    }

    std::cout << "Applied gradient clipping with threshold = " << threshold;
}

void Network::ClearUpdates()
{
    for (auto w : _vWeight)
    {
        w->_updateCount = 0;
    }

    for (auto l : _vLayer)
    {
        l->ClearUpdates();
    }

    std::cout << "Cleared weight and layer updates.";
}

std::tuple<float, float> Network::CalculateError(float lambda, float lambda1)
{
    float error_training = 0.0f;
    float error_regularization = 0.0f;

    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    for (auto l : _vOutputLayer)
    {
        error_training += l->CalculateError(_position, batch, _errorFunction);
    }

    if ((lambda != 0.0f) || (lambda1 != 0.0f))
    {
        for (auto w : _vWeight)
        {
            error_regularization += w->CalculateRegularizationError(lambda, lambda1);
        }
    }

    if (getGpu()._numprocs > 1)
    {
        double derror_training = error_training;
        double derror_regularization = error_regularization;
        MPI_Allreduce(MPI_IN_PLACE, &derror_training, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &derror_regularization, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        error_training = static_cast<float>(derror_training);
        error_regularization = static_cast<float>(derror_regularization);
    }

    std::cout << "Calculated error_training: " << error_training << ", error_regularization: " << error_regularization;

    return std::make_tuple(error_training, error_regularization);
}

void Network::BackPropagate() {
    const uint32_t remainingExamples = _examples - _position;
    const uint32_t batchSize = std::min(_batch, remainingExamples);

    ThreadPool threadPool(std::thread::hardware_concurrency());

    std::for_each(std::execution::par_unseq, _vBPOrder.begin(), _vBPOrder.end(), [this, batchSize, &threadPool](Layer* l) {
        switch (l->_kind) {
            case Layer::Kind::Output: {
                l->CalculateOutputDelta(_position, batchSize, _errorFunction);
                [[fallthrough]];
            }
            case Layer::Kind::Hidden: {
                threadPool.Execute([l, this, batchSize]() {
                    try {
                        l->BackPropagate(_position, batchSize);
                        std::cout << "Backpropagated for layer with kind: " << static_cast<int>(l->_kind);
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error in backpropagation: " << e.what();
                    }
                });
                break;
            }
            default: {
                break;
            }
        }
    });
}

void Network::UpdateWeights(float alpha, float lambda, float lambda1, float mu, float mu1)
{
    const uint32_t batch = std::min(_batch, _examples - _position);

    for (auto i = static_cast<int64_t>(_vWeight.size()) - 1; i >= 0; --i)
    {
        _vWeight[i]->UpdateWeights(_trainingMode);
        std::cout << "Updated weights for weight #" << i;
    }

    for (const auto& l : _vLayer)
    {
        if (l->_bBatchNormalization)
        {
            l->UpdateWeights(_trainingMode, batch, alpha, lambda, lambda1, mu, mu1, _batches);
            std::cout << "Updated weights for layer with batch normalization.";
        }
    }
}

void Network::CalculateExamples(const std::string& layer, uint32_t k, GpuBuffer<float>* pbKey, GpuBuffer<unsigned int>* pbValue) {

    auto it = _mLayer.find(layer);
    if (it == _mLayer.end()) {
        throw std::invalid_argument("Unknown layer: " + layer);
    }

    if (k > 128 || k > (it->second->_Nx * it->second->_Ny * it->second->_Nz)) {
        throw std::invalid_argument("Invalid value of k: " + std::to_string(k));
    }

    const uint32_t numCores = std::thread::hardware_concurrency();
    const uint32_t batch = std::min(_batch, _examples - _position);

    if (batch == 0) {
        if (getGpu()._id == 0)
            std::cout << "Network::CalculateExamples: No examples to calculate.";
        return;
    }

    const uint32_t batchSizePerTask = batch / numCores;
    std::vector<std::thread> threads(numCores);
    std::vector<std::exception_ptr> exceptions(numCores);
    std::atomic<uint32_t> completedTasks(0);

    for (uint32_t i = 0; i < numCores; ++i) {
        const uint32_t startExample = i * batchSizePerTask;
        const uint32_t endExample = (i == numCores - 1) ? batch : (i + 1) * batchSizePerTask;

        threads[i] = std::thread([this, it, pbKey, pbValue, k, startExample, endExample, &exceptions, i, &completedTasks]() {
            try {
            }
            catch (...) {
                exceptions[i] = std::current_exception();
            }

            completedTasks.fetch_add(1, std::memory_order_relaxed);
        });
    }

    while (completedTasks.load(std::memory_order_relaxed) < numCores) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "Completed tasks: " << completedTasks.load(std::memory_order_relaxed) << " out of " << numCores << std::endl;
    }

    for (uint32_t i = 0; i < numCores; ++i) {
        threads[i].join();
        if (exceptions[i]) {
            try {
                std::rethrow_exception(exceptions[i]);
            }
            catch (const std::exception& e) {
                std::cerr << "Network::CalculateExamples: Error during calculation - " << e.what();
            }
        }
    }
}

bool Network::SaveNetCDF(const std::string& fname)
{
    bool bResult = true;

    std::vector<std::vector<float>> vvWeight;
    std::vector<std::vector<float>> vvBias;
    for (auto w : _vWeight)
    {
        std::vector<float> vWeight;
        std::vector<float> vBias;

        if (!w->_bShared)
        {
            w->_pbWeight->Download(w->_vWeight.data());

            if (getGpu()._numprocs == 1)
            {
                vWeight = w->_vWeight;
            }
            else
            {
                uint32_t outgoingSize = w->_outputLayer._stride * 3;
                uint32_t incomingSize = w->_inputLayer._stride * 2;
                if (getGpu()._id == 0)
                {
                    vWeight.resize(static_cast<std::vector<float, std::allocator<float>>::size_type>(w->_outputLayer._stride) * w->_inputLayer._stride);
                    float* pWeight = vWeight.data();
                    if (outgoingSize > incomingSize)
                    {
                        cudaMemcpy2D(pWeight, w->_outputLayer._stride * sizeof(float), w->_vWeight.data(), w->_outputLayer._localStride * sizeof(float), w->_outputLayer._localStride * sizeof(float), w->_inputLayer._stride, cudaMemcpyDefault);
                        pWeight += w->_outputLayer._localStride;
                        for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                        {
                            uint64_t size;
                            MPI_Status status;
                            MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                            std::vector<float> vTemp(size);
                            MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                            uint64_t lstride = size / w->_inputLayer._stride;
                            float* pSrcWeight = vTemp.data();
                            float* pDstWeight = pWeight;
                            for (uint32_t j = 0; j < w->_inputLayer._stride; j++)
                            {
                                memcpy(pDstWeight, pSrcWeight, lstride * sizeof(float));
                                pSrcWeight += lstride;
                                pDstWeight += w->_outputLayer._stride;
                            }
                            pWeight += lstride;
                        }
                    }
                    else
                    {
                        cudaMemcpy(pWeight, w->_vWeight.data(), static_cast<unsigned long long>(w->_outputLayer._stride) * w->_inputLayer._localStride * sizeof(float), cudaMemcpyDefault);
                        pWeight += w->_outputLayer._stride * w->_inputLayer._localStride;
                        for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                        {
                            uint64_t size;
                            MPI_Status status;
                            MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                            MPI_Recv(pWeight, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                            pWeight += size;
                        }
                    }
                }
                else
                {
                    uint64_t size = w->_vWeight.size();
                    MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
                    MPI_Send(w->_vWeight.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
                }
            }
        }

        w->_pbBias->Download(w->_vBias.data());
        if (getGpu()._id == 0)
        {
            vBias = w->_vBias;
            vBias.resize(w->_outputLayer._stride);
            uint64_t offset = w->_vBias.size();
            for (size_t i = 1; i < getGpu()._numprocs; i++)
            {
                uint64_t size;
                MPI_Status status;
                MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(vBias.data() + offset, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                offset += size;
            }
        }
        else
        {
            uint64_t size = w->_vBias.size();
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(w->_vBias.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }

        vvWeight.push_back(vWeight);
        vvBias.push_back(vBias);
    }

    if (getGpu()._id == 0)
    {
        try
        {
            netCDF::NcFile nc(fname, netCDF::NcFile::replace);

            nc.putAtt("version", netCDF::ncFloat, VERSION);
            nc.putAtt("name", _name);
            nc.putAtt("kind", netCDF::ncUint, _kind);
            nc.putAtt("errorFunction", netCDF::ncUint, _errorFunction);
            nc.putAtt("maxout_k", netCDF::ncInt, _maxout_k);
            nc.putAtt("decay", netCDF::ncFloat, _decay);
            nc.putAtt("LRN_k", netCDF::ncFloat, _LRN_k);
            nc.putAtt("LRN_n", netCDF::ncInt, _LRN_n);
            nc.putAtt("LRN_alpha", netCDF::ncFloat, _LRN_alpha);
            nc.putAtt("LRN_beta", netCDF::ncFloat, _LRN_beta);
            nc.putAtt("bSparsenessPenalty", netCDF::ncUint, (uint32_t)_bSparsenessPenalty);
            nc.putAtt("sparsenessPenalty_p", netCDF::ncFloat, _sparsenessPenalty_p);
            nc.putAtt("sparsenessPenalty_beta", netCDF::ncFloat, _sparsenessPenalty_beta);
            nc.putAtt("bDenoising", netCDF::ncUint, (uint32_t)_bDenoising);
            nc.putAtt("denoising_p", netCDF::ncFloat, _denoising_p);
            nc.putAtt("deltaBoost_one", netCDF::ncFloat, _deltaBoost_one);
            nc.putAtt("deltaBoost_zero", netCDF::ncFloat, _deltaBoost_zero);
            nc.putAtt("SMCE_oneScale", netCDF::ncFloat, _SMCE_oneScale);
            nc.putAtt("SMCE_zeroScale", netCDF::ncFloat, _SMCE_zeroScale);
            nc.putAtt("SMCE_oneTarget", netCDF::ncFloat, _SMCE_oneTarget);
            nc.putAtt("SMCE_zeroTarget", netCDF::ncFloat, _SMCE_zeroTarget);
            nc.putAtt("ShuffleIndices", netCDF::ncUint, (uint32_t)_bShuffleIndices);
            nc.putAtt("checkpoint_name", _checkpoint_name);
            nc.putAtt("checkpoint_interval", netCDF::ncInt, _checkpoint_interval);
            nc.putAtt("checkpoint_epochs", netCDF::ncInt, _checkpoint_epochs);

            nc.putAtt("layers", netCDF::ncUint, (uint32_t)_vLayer.size());
            for (uint32_t i = 0; i < _vLayer.size(); i++)
                _vLayer[i]->WriteNetCDF(nc, i);

            nc.putAtt("weights", netCDF::ncUint, (uint32_t)_vWeight.size());
            for (uint32_t i = 0; i < _vWeight.size(); i++)
                _vWeight[i]->WriteNetCDF(nc, i, vvWeight[i].data(), vvBias[i].data());
        }
        catch (netCDF::exceptions::NcException& e)
        {
            std::cerr << "Network::SaveNetCDF Error opening binary output file " << fname << " to save neural network " << _name << ".";
            bResult = false;
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    return bResult;
}

std::vector<std::string> Network::GetLayers() const
{
    std::vector<std::string> vResult;
    for (auto l : _vLayer)
    {
        vResult.push_back(l->_name);
    }

    std::cout << "Layers in the network:";
    for (const std::string& layerName : vResult)
    {
        std::cout << layerName;
    }

    return vResult;
}

const std::string& Network::GetName() const
{
    std::cout << "Network Name: " << _name;
    return _name;
}

float* Network::GetUnitBuffer(const std::string& layer)
{
    const auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        std::cerr << "Network::GetUnitBuffer: Unknown layer " << layer;
        return nullptr;
    }

    return itr->second->GetUnitBuffer();
}

float* Network::GetDeltaBuffer(const std::string& layer)
{
    auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        std::cerr << "Network::GetDeltaBuffer: Unknown layer " << layer << ".";
        return nullptr;
    }

    return itr->second->GetDeltaBuffer();
}

uint64_t Network::GetBufferSize(const std::string& layer) const
{
    auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        std::cerr << "Network::GetBufferSize: Unknown layer " << layer << ".";
        return 0;
    }

    return itr->second->GetBufferSize();
}

Weight* Network::GetWeight(const std::string& inputLayer, const std::string& outputLayer) const
{
    auto inputLayerItr = _mLayer.find(inputLayer);
    if (inputLayerItr == _mLayer.end())
    {
        std::cerr << "Network::GetWeight: Unknown input layer " << inputLayer;
        return nullptr;
    }

    const auto outputLayerItr = _mLayer.find(outputLayer);
    if (outputLayerItr == _mLayer.end())
    {
        std::cerr << "Network::GetWeight: Unknown output layer " << outputLayer;
        return nullptr;
    }

    const Layer* pInputLayer = inputLayerItr->second;
    const Layer* pOutputLayer = outputLayerItr->second;

    for (auto p : _vWeight)
    {
        if ((&(p->_inputLayer) == pInputLayer) && (&(p->_outputLayer) == pOutputLayer))
        {
            return p;
        }
    }

    std::cerr << "Network::GetWeight: No set of weights connecting layer " << inputLayer << " to layer " << outputLayer;
    return nullptr;
}

float* Network::GetWeightBuffer(const std::string& inputLayer, const std::string& outputLayer)
{
    const auto inputLayerItr = _mLayer.find(inputLayer);
    if (inputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            std::cout << std::format("Network::GetWeightBuffer: Unknown input layer '{}'.\n", inputLayer);
        }
        return nullptr;
    }

    const auto outputLayerItr = _mLayer.find(outputLayer);
    if (outputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            std::cout << std::format("Network::GetWeightBuffer: Unknown output layer '{}'.\n", outputLayer);
        }
        return nullptr;
    }

    const Layer* pInputLayer = inputLayerItr->second;
    const Layer* pOutputLayer = outputLayerItr->second;

    for (auto p : _vWeight)
    {
        if ((&(p->_inputLayer) == pInputLayer) && (&(p->_outputLayer) == pOutputLayer))
        {
            return p->_vWeight.data();
        }
    }

    if (getGpu()._id == 0)
    {
        std::cout << std::format("Network::GetWeightBuffer: No set of weights connecting layer '{}' to layer '{}'.\n", inputLayer, outputLayer);
    }

    return nullptr;
}

Network::~Network()
{
    DeallocatePeerBuffers();

    for (uint32_t i = 0; i < _vWeight.size(); i++)
        delete _vWeight[i];

    for (uint32_t i = 0; i < _vLayer.size(); i++)
        delete _vLayer[i];
}

uint32_t CalculateConvolutionDimensions(uint32_t width, uint32_t filter, uint32_t stride)
{
    if (width <= filter)
        return 1;
    else if (stride == 1)
        return width;
    else
        return (width - filter) / stride + 1;
}

void CalculateDerivedLayerDimensions(NetworkDescriptor& d) {
    std::map<LayerDescriptor*, bool> mbDimensionsCalculated;
    std::map<std::string, LayerDescriptor*> mLayer;

    for (auto& layer : d._vLayerDescriptor) {
        bool bFlag = layer._kind == Layer::Kind::Hidden &&
            (layer._type == Layer::Type::Pooling || layer._type == Layer::Type::Convolutional);
        mbDimensionsCalculated[&layer] = bFlag;
        mLayer[layer._name] = &layer;
    }

    bool bFinished;
    do {
        bFinished = true;

        for (auto& layer : d._vLayerDescriptor) {
            bool bPooling = layer._type == Layer::Type::Pooling;
            bool bLRN = bPooling && (layer._poolingFunction == PoolingFunction::LRN);
            bool bDotProduct = bPooling && (layer._poolingFunction == PoolingFunction::DotProduct || layer._poolingFunction == PoolingFunction::Cosine);

            if (!mbDimensionsCalculated[&layer]) {
                bool bAllInputsCalculated = true;
                for (auto& s : layer._vSource) {
                    LayerDescriptor* pS = mLayer[s];
                    bAllInputsCalculated &= mbDimensionsCalculated[pS];
                }

                if (!bAllInputsCalculated) {
                    bFinished = false;
                    continue;
                }

                uint32_t nx = 0, ny = 0, nz = 0, nw = 0;
                uint32_t oldNx = 0, oldNy = 0, oldNz = 0;
                bool bSized = false;

                for (auto& s : layer._vSource) {
                    LayerDescriptor* pS = mLayer[s];

                    if (bDotProduct) {
                        if ((oldNx != pS->_Nx) || (oldNy != pS->_Ny) || (oldNz != pS->_Nz)) {
                            std::cerr << "Network::CalculateDerivedLayerDimensions: Inconsistent incoming data size for dot product layer " << layer._name << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                    else {
                        if (!bLRN) {
                            nx = CalculateConvolutionDimensions(pS->_Nx, layer._kernelX, layer._kernelStrideX);
                            ny = CalculateConvolutionDimensions(pS->_Ny, layer._kernelY, layer._kernelStrideY);
                            nz = CalculateConvolutionDimensions(pS->_Nz, layer._kernelZ, layer._kernelStrideZ);
                            nw = pS->_Nw;
                            if (bPooling) {
                                layer._dimensions = pS->_dimensions;
                            }
                        }
                        else {
                            nx = pS->_Nx;
                            ny = pS->_Ny;
                            nz = pS->_Nz;
                            nw = pS->_Nw;
                            layer._dimensions = pS->_dimensions;
                        }

                        switch (layer._kernelDimensions) {
                        case 3:
                            if (pS->_Nz < layer._kernelZ) {
                                layer._kernelPaddingZ = (layer._kernelZ - pS->_Nz + 1) / 2;
                            }
                            else if (layer._kernelStrideZ == 1) {
                                layer._kernelPaddingZ = layer._kernelZ / 2;
                            }
                            [[fallthrough]];

                        case 2:
                            if (pS->_Ny < layer._kernelY) {
                                layer._kernelPaddingY = (layer._kernelY - pS->_Ny + 1) / 2;
                            }
                            else if (layer._kernelStrideY == 1) {
                                layer._kernelPaddingY = layer._kernelY / 2;
                            }
                            [[fallthrough]];

                        case 1:
                            if (pS->_Nx < layer._kernelX) {
                                layer._kernelPaddingX = (layer._kernelX - pS->_Nx + 1) / 2;
                            }
                            else if (layer._kernelStrideX == 1) {
                                layer._kernelPaddingX = layer._kernelX / 2;
                            }
                        }

                        if (bSized) {
                            if ((nx != oldNx) || (ny != oldNy) || (nz != oldNz)) {
                                std::cerr << "Network::CalculateDerivedLayerDimensions: Inconsistent incoming data size for convolution layer " << layer._name << std::endl;
                                exit(EXIT_FAILURE);
                            }
                        }
                        bSized = true;
                        oldNx = nx;
                        oldNy = ny;
                        oldNz = nz;
                        mbDimensionsCalculated[&layer] = true;
                    }
                }
                layer._Nx = nx;
                layer._Ny = ny;
                layer._Nz = nz;
                layer._Nw = nw;
                if (!bPooling) {
                    switch (layer._kernelDimensions) {
                    case 1:
                        layer._Ny = layer._Nx;
                        layer._dimensions = 2;
                        break;

                    case 2:
                        layer._Nz = layer._Nx;
                        layer._dimensions = 3;
                        break;

                    case 3:
                        layer._Nw = layer._Nx;
                        layer._dimensions = 4;
                        break;
                    }
                }
            }
        }
    } while (!bFinished);
}

void Network::CalculatePropagationOrder()
{
    struct CompareLayer {
        bool operator()(Layer* l1, Layer* l2)
        {
            return (l1->_priority < l2->_priority);
        }
    };

    for (auto p : _vLayer)
    {
        p->_priority = (p->_kind == Layer::Kind::Input) ? 0 : -1;
    }

    std::priority_queue<Layer*, std::vector<Layer*>, CompareLayer> pqueue;
    for (auto p : _vInputLayer)
    {
        pqueue.push(p);
    }

    while (!pqueue.empty())
    {
        Layer* pLayer = pqueue.top();
        pqueue.pop();

        int32_t priority = pLayer->_priority + 1;
        for (auto p : pLayer->_vOutgoingLayer)
        {
            if (p->_priority < priority)
            {
                p->_priority = priority;
                pqueue.push(p);
            }
        }

        for (auto p : pLayer->_vOutgoingSkip)
        {
            if (p->_priority < priority)
            {
                p->_priority = priority;
                pqueue.push(p);
            }
        }
    }

    _vFPOrder.resize(0);
    for (auto p : _vLayer)
    {
        _vFPOrder.push_back(p);
    }
    sort(_vFPOrder.begin(), _vFPOrder.end(), CompareLayer());

    for (auto p : _vLayer)
    {
        p->_priority = (p->_kind == Layer::Kind::Output) ? 0 : -1;
    }

    for (auto p : _vOutputLayer)
    {
        pqueue.push(p);
    }

    while (!pqueue.empty())
    {
        Layer* pLayer = pqueue.top();
        pqueue.pop();
        int32_t priority = pLayer->_priority + 1;
        for (auto p : pLayer->_vIncomingLayer)
        {
            if (p->_priority < priority)
            {
                p->_priority = priority;
                pqueue.push(p);
            }
        }

        for (auto p : pLayer->_vIncomingSkip)
        {
            if (p->_priority < priority)
            {
                p->_priority = priority;
                pqueue.push(p);
            }
        }
    }

    _vBPOrder.resize(0);
    for (auto p : _vLayer)
    {
        _vBPOrder.push_back(p);
    }

    sort(_vBPOrder.begin(), _vBPOrder.end(), CompareLayer());
}

bool Network::Validate()
{
    const float delta = 0.001f;
    const float epsilon = delta * 20.f;
    float network_lambda = 0.0f;
    float lambda1 = 0.0f;
    float alpha = 1.0f;
    float mu = 0.0f;
    float mu1 = 0.0f;
    float initialError;

    if (_mode != Validation)
    {
        _mode = Validation;
        _bDirty = true;
    }

    if (_bDirty)
    {
        RefreshState();
        if (!_bAllDataLoaded)
        {
            std::cerr << "Network::Validate: Attempt to validate neural network " << _name << " without providing data sets" << '\n';
            return false;
        }
    }

    if (_trainingMode != SGD && _bClearVelocity)
    {
#pragma omp parallel for
        for (int i = 0; i < _vWeight.size(); i++)
            _vWeight[i]->ClearVelocity();
    }

    if (_bShuffleIndices)
    {
        ShuffleIndices();
    }

    std::cout << "Validating network weights and biases with epsilon error threshold of " << epsilon << '\n';

    SetPosition(0);
    ClearUpdates();
    PredictValidationBatch();
    auto errorTuple = CalculateError(network_lambda, lambda1);
    float newInitialErrorTraining = std::get<0>(errorTuple);
    float newInitialErrorRegularization = std::get<1>(errorTuple);
    initialError = newInitialErrorTraining + newInitialErrorRegularization;

    std::cout << "initialErrorTraining " << newInitialErrorTraining << "; initialErrorRegularization " << newInitialErrorRegularization << '\n';

    BackPropagate();

    std::vector<std::vector<float>> vWeightGradient(_vWeight.size());
    std::vector<std::vector<float>> vBiasGradient(_vWeight.size());

#pragma omp parallel for
    for (int id = 0; id < _vWeight.size(); id++)
    {
        Weight* w = _vWeight[id];
        w->_pbWeight->Download(w->_vWeight.data());
        w->_pbBias->Download(w->_vBias.data());
        w->_pbWeightGradient->Download(vWeightGradient[id].data());
    }

    UpdateWeights(alpha, network_lambda, lambda1, mu, mu1);

#pragma omp parallel for
    for (int id = 0; id < _vWeight.size(); id++)
    {
        Weight* w = _vWeight[id];
        w->_pbWeight->Upload(w->_vWeight.data());

        std::vector<float> bias_g(w->_pbBias->_length);
        w->_pbBias->Download(bias_g.data());

        for (int b = 0; b < bias_g.size(); b++)
        {
            vBiasGradient[id][b] = bias_g[b] - w->_vBias[b];
        }

        w->_pbBias->Upload(w->_vBias.data());
    }

#pragma omp parallel for
    for (int id = 0; id < _vWeight.size(); id++)
    {
        Weight* w = _vWeight[id];

        std::cout << "Validating weights between layer " << w->_inputLayer._name << " and " << w->_outputLayer._name << '\n';

        for (size_t i = 0; i < w->_vWeight.size(); i++)
        {
            float oldWeight = w->_vWeight[i];
            w->_vWeight[i] += delta / (_batch * w->_sharingCount);
            w->_pbWeight->Upload(w->_vWeight.data());
            PredictValidationBatch();
            w->_vWeight[i] = oldWeight;

            errorTuple = CalculateError(network_lambda, lambda1);
            float errorTraining = std::get<0>(errorTuple);
            float errorRegularization = std::get<1>(errorTuple);
            float error = errorTraining + errorRegularization;
            float dEdW = (error - initialError) / delta;
            float weightGradient = vWeightGradient[id][i];

            std::cout << "errorTraining " << errorTraining << "; errorRegularization " << errorRegularization <<
                "; dEdW " << dEdW << "; weightGradient " << weightGradient << '\n';

            if (std::fabs(dEdW + weightGradient) > epsilon)
            {
                std::cerr << "Failed Weight " << i << " exceeds error threshold: " << dEdW << " vs " << weightGradient << '\n';
            }
        }

        w->_pbWeight->Upload(w->_vWeight.data());

        for (size_t i = 0; i < w->_vBias.size(); i++)
        {
            float oldBias = w->_vBias[i];
            w->_vBias[i] += delta / (_batch);
            w->_pbBias->Upload(w->_vBias.data());
            PredictValidationBatch();
            w->_vBias[i] = oldBias;

            errorTuple = CalculateError(network_lambda, lambda1);
            float errorTraining = std::get<0>(errorTuple);
            float errorRegularization = std::get<1>(errorTuple);
            float error = errorTraining + errorRegularization;
            float dEdb = (error - initialError) / delta;
            float biasGradient = vBiasGradient[id][i];

            std::cout << "errorTraining " << errorTraining << "; errorRegularization " << errorRegularization <<
                "; dEdb " << dEdb << "; biasGradient " << biasGradient << '\n';

            if (std::fabs(dEdb + biasGradient) > epsilon)
            {
                std::cerr << "Failed Bias " << i << " exceeds error threshold: " << dEdb << " vs " << biasGradient << '\n';
            }
        }

        w->_pbBias->Upload(w->_vBias.data());
    }

    return true;
}

void Network::DeallocatePeerBuffers()
{
    try
    {
        if (getGpu()._numprocs > 1)
        {
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            for (size_t i = 0; i < 2; i++)
            {
                if (_pPeerBuffer[i] != nullptr)
                {
                    cudaError_t status = cudaIpcCloseMemHandle(_pPeerBuffer[i]);
                    if (status != cudaSuccess)
                    {
                        throw std::runtime_error("Network::DeallocatePeerBuffers: Error closing IpcMemHandle");
                    }
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);

            for (size_t i = 0; i < 2; i++)
            {
                _pbP2PBuffer[i].reset();
            }

            _pCPUBuffer.reset();
        }
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "Caught an exception: " << e.what() << '\n';
    }
}

void Network::AllocatePeerBuffers()
{
    if (getGpu()._numprocs > 1)
    {
        _maxStride = 0;

        for (auto w : _vWeight)
        {
            uint32_t stride = (w->_outputLayer._stride * 2) > (w->_inputLayer._stride * 2) ? w->_inputLayer._stride : w->_outputLayer._stride;
            if (stride > _maxStride)
            {
                _maxStride = stride;
            }
        }

        uint64_t maxMemory = static_cast<uint64_t>(_maxStride) * static_cast<uint64_t>(_batch);
        if (maxMemory < _examples)
        {
            maxMemory = _examples;
        }

        for (size_t i = 0; i < 2; i++)
        {
            _pbP2PBuffer[i].reset(new GpuBuffer<float>(maxMemory));
        }

        if (getGpu()._bP2P)
        {
            cudaIpcMemHandle_t* pMemHandle = new cudaIpcMemHandle_t[2 * getGpu()._numprocs];
            size_t pos = static_cast<size_t>(getGpu()._id) * 2;

            cudaError_t status = cudaIpcGetMemHandle(&(pMemHandle[pos]), _pbP2PBuffer[0].get());
            if (status != cudaSuccess)
            {
                delete[] pMemHandle;
                throw std::runtime_error("Network::AllocatePeerBuffers: Error getting first P2P IPCMemHandle");
            }
            status = cudaIpcGetMemHandle(&(pMemHandle[pos + 1]), _pbP2PBuffer[1].get());
            if (status != cudaSuccess)
            {
                delete[] pMemHandle;
                throw std::runtime_error("Network::AllocatePeerBuffers: Error getting second P2P IPCMemHandle");
            }

            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pMemHandle, 2 * sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);

            unsigned int peer = 2 * ((getGpu()._id + getGpu()._numprocs - 1) % getGpu()._numprocs);
            status = cudaIpcOpenMemHandle((void**)&(_pPeerBuffer[0]), pMemHandle[peer], cudaIpcMemLazyEnablePeerAccess);
            if (status != cudaSuccess)
            {
                delete[] pMemHandle;
                throw std::runtime_error("Network::AllocatePeerBuffers: Unable to open first peer IPCMemHandle");
            }
            status = cudaIpcOpenMemHandle((void**)&(_pPeerBuffer[1]), pMemHandle[peer + 1], cudaIpcMemLazyEnablePeerAccess);
            delete[] pMemHandle;
            if (status != cudaSuccess)
            {
                throw std::runtime_error("Network::AllocatePeerBuffers: Unable to open second peer IPCMemHandle");
            }
        }
        else
        {
            std::vector<float> cpuBuffer(maxMemory);
        }
    }
}

void Network::SwapPeerBuffers()
{
    _sendIndex = 1 - _sendIndex;
    _receiveIndex = 1 - _receiveIndex;
}

std::pair<Network::Kind, std::string> Network::_sKindPair[] =
{
    std::pair<Network::Kind, std::string>(Network::Kind::FeedForward, "FeedForward"),
    std::pair<Network::Kind, std::string>(Network::Kind::AutoEncoder, "AutoEncoder")
};

std::map<Network::Kind, std::string> Network::_sKindMap =
std::map<Network::Kind, std::string>(_sKindPair, Network::_sKindPair + sizeof(Network::_sKindPair) / sizeof(Network::_sKindPair[0]));

std::ostream& operator<< (std::ostream& out, Network::Kind& k)
{
    out << Network::_sKindMap[k];
    return out;
}

void MPI_Bcast_NetworkDescriptor(NetworkDescriptor& d) {
    std::cout << "Broadcasting NetworkDescriptor...";

    uint32_t layers = d._vLayerDescriptor.size();
    uint32_t weights = d._vWeightDescriptor.size();

    std::cout << "Broadcasting sizes...";

    MPI_Bcast(&layers, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&weights, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    d._vLayerDescriptor.resize(layers);
    d._vWeightDescriptor.resize(weights);

    std::cout << "Broadcasting NetworkDescriptor struct...";

    MPI_Bcast_string(d._name);
    MPI_Bcast(&d._kind, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._errorFunction, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._maxout_k, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._LRN_k, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._LRN_n, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._LRN_alpha, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._LRN_beta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bSparsenessPenalty, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_beta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bDenoising, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._denoising_p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._deltaBoost_one, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._deltaBoost_zero, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._SMCE_oneScale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._SMCE_zeroScale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._SMCE_oneTarget, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._SMCE_zeroTarget, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._checkpoint_interval, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._checkpoint_epochs, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast_string(d._checkpoint_name);
    MPI_Bcast(&d._bShuffleIndices, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    std::cout << "Broadcasting vectors and containers...";

    MPI_Bcast(d._vLayerDescriptor.data(), layers, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(d._vWeightDescriptor.data(), weights, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::cout << "Broadcast completed.";
}

Network* LoadNeuralNetworkJSON(const std::string& fname, const uint32_t batch, const std::vector<DataSetBase*>& vDataSet)
{
    Network* pNetwork = NULL;
    NetworkDescriptor nd;
    Json::Value index;
    Json::Reader reader;
    bool bValid = true;
    std::string wfname;

    if (getGpu()._id == 0)
    {
        std::ifstream stream(fname, std::ifstream::binary);
        bool parsedSuccess = reader.parse(stream, index, false);

        if (!parsedSuccess)
        {
            std::cout << std::format("LoadNeuralNetworkJSON: Failed to parse JSON file: {}, error: {}\n", fname, reader.getFormattedErrorMessages());
            bValid = false;
        }
        else
        {
            std::set<std::string> sLayer;
            for (Json::ValueIterator itr = index.begin(); itr != index.end() ; itr++)
            {
                std::string name = itr.name();
                std::transform(name.begin(), name.end(), name.begin(), ::tolower);
                Json::Value key = itr.key();
                Json::Value value = *itr;
                std::string vstring = value.isString() ? value.asString() : "";
                std::transform(vstring.begin(), vstring.end(), vstring.begin(), ::tolower);

                if (name == "version") {
                    float version = value.asFloat();
                    if (version < 0.7) {
                        std::cout << std::format("LoadNeuralNetworkJSON: version {} (must be at least 0.7)\n", version);
                        bValid = false;
                    }
                }

                else if (name.compare("name") == 0)
                {
                    nd._name = value.asString();
                }

                else if (name.compare("kind") == 0)
                {
                    if (vstring.compare("feedforward") == 0)
                        nd._kind = Network::Kind::FeedForward;
                    else if (vstring.compare("autoencoder") == 0)
                        nd._kind = Network::Kind::AutoEncoder;
                    else
                    {
                        std::cout << std::format("LoadNeuralNetworkJSON: Invalid network kind: {}\n", value.asString());
                        bValid = false;
                        goto exit;
                    }
                }

                else if (name.compare("weightsdata") == 0)
                {
                    wfname = value.asString();
                }

                else if ((name.compare("lrn") == 0) || (name.compare("localresponsenormalization") == 0))
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        std::string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("k") == 0)
                            nd._LRN_k = pvalue.asFloat();
                        else if (pname.compare("n") == 0)
                            nd._LRN_n = pvalue.asInt();
                        else if (pname.compare("alpha") == 0)
                            nd._LRN_alpha = pvalue.asFloat();
                        else if (pname.compare("beta") == 0)
                            nd._LRN_beta = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            std::cout << std::format("LoadNeuralNetworkJSON: Invalid LocalResponseNormalization parameter: {}\n", name);
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                else if (name.compare("maxout") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        std::string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("k") == 0)
                            nd._maxout_k = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            std::cout << std::format("LoadNeuralNetworkJSON: Invalid MaxOut parameter: {}\n", name);
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                else if (name.compare("sparsenesspenalty") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        std::string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("p") == 0)
                            nd._sparsenessPenalty_p = pvalue.asFloat();
                        else if (pname.compare("beta") == 0)
                            nd._sparsenessPenalty_beta  = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            std::cout << std::format("LoadNeuralNetworkJSON: Invalid SparsenessPenalty parameter: {}\n", name);
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                else if (name.compare("denoising") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        std::string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("p") == 0)
                        {
                            nd._denoising_p = pvalue.asFloat();
                        }
                        else
                        {
                            name = pitr.name();
                            std::cout << std::format("LoadNeuralNetworkJSON: Invalid Denoising parameter: {}\n", name);
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                else if (name == "deltaboost") {
                    float deltaBoostOne = std::numeric_limits<float>::quiet_NaN();
                    float deltaBoostZero = std::numeric_limits<float>::quiet_NaN();

                    const std::vector<std::string> memberNames = value.getMemberNames();
                    for (const std::string& pname : memberNames) {
                        std::string pname_lower = pname;
                        std::transform(pname_lower.begin(), pname_lower.end(), pname_lower.begin(), ::tolower);
                        const Json::Value& pvalue = value[pname];

                        if (pname_lower == "one") {
                            deltaBoostOne = pvalue.asFloat();
                        }
                        else if (pname_lower == "zero") {
                            deltaBoostZero = pvalue.asFloat();
                        }
                        else {
                            std::cout << "LoadNeuralNetworkJSON: Invalid DeltaBoost parameter: " << pname << std::endl;
                            bValid = false;
                            break;
                        }
                    }

                    if (!std::isnan(deltaBoostOne) && !std::isnan(deltaBoostZero)) {
                        nd._deltaBoost_one = deltaBoostOne;
                        nd._deltaBoost_zero = deltaBoostZero;
                    }
                    else {
                        std::cout << "LoadNeuralNetworkJSON: DeltaBoost parameters 'one' and 'zero' are required." << std::endl;
                        bValid = false;
                    }
                }

                else if ((name.compare("scaledmarginalcrossentropy") == 0) ||
                         (name.compare("datascaledmarginalcrossentropy") == 0))
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        std::string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("onescale") == 0)
                            nd._SMCE_oneScale = pvalue.asFloat();
                        else if (pname.compare("zeroscale") == 0)
                            nd._SMCE_zeroScale = pvalue.asFloat();
                        else if (pname.compare("onetarget") == 0)
                            nd._SMCE_oneTarget = pvalue.asFloat();
                        else if (pname.compare("zerotarget") == 0)
                            nd._SMCE_zeroTarget = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            std::cout << std::format("LoadNeuralNetworkJSON: Invalid ScaledMarginalCrossentropy parameter: {}\n", name);
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                else if (name.compare("shuffleindices") == 0)
                {
                    nd._bShuffleIndices = value.asBool();
                    }

                else if ((name.compare("reluslope") == 0) || (name.compare("slope") == 0))
                {
                    nd._RELUSlope = value.asFloat();
                    }

                else if (name.compare("elualpha") == 0)
                {
                    nd._ELUAlpha = value.asFloat();
                    }

                else if (name.compare("selulambda") == 0)
                {
                    nd._SELULambda = value.asFloat();
                    }

                else if (name.compare("decay") == 0)
                {
                    nd._decay = value.asFloat();
                    }

                else if (name == "errorfunction")
                {
                    static const std::unordered_map<std::string, ErrorFunction> errorFunctionMap = {
                        {"l1", ErrorFunction::L1},
                        {"l2", ErrorFunction::L2},
                        {"l2hinge", ErrorFunction::L2Hinge},
                        {"hinge", ErrorFunction::Hinge},
                        {"crossentropy", ErrorFunction::CrossEntropy},
                        {"cross entropy", ErrorFunction::CrossEntropy},
                        {"scaledmarginalcrossentropy", ErrorFunction::ScaledMarginalCrossEntropy},
                        {"datascaledmarginalcrossentropy", ErrorFunction::DataScaledMarginalCrossEntropy}
                    };

                    auto it = errorFunctionMap.find(vstring);
                    if (it != errorFunctionMap.end())
                    {
                        nd._errorFunction = it->second;
                    }
                    else
                    {
                        std::cout << std::format("LoadNeuralNetworkJSON: Invalid error function: {}\n", value.asString());
                        bValid = false;
                    }
                }

                else if (name.compare("layers") == 0)
                {
                    uint32_t size = value.isArray() ? value.size() : 1;
                    for (uint32_t i = 0; i < size; i++)
                    {
                        std::vector<WeightDescriptor> vSharedWeight;
                        LayerDescriptor ldl;
                        bool bSource = false;
                        Json::Value layer = value.isArray() ? value[i] : value;
                        bool bAutoSize = false;

                        if (i == 0)
                            ldl._kind = Layer::Kind::Input;
                        else if (i == size - 1)
                            ldl._kind = Layer::Kind::Output;
                        else
                            ldl._kind = Layer::Kind::Hidden;
                        ldl._type = Layer::Type::FullyConnected;


                        for (Json::ValueIterator litr = layer.begin(); litr != layer.end() ; litr++)
                        {
                            std::string lname = litr.name();
                            std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);
                            Json::Value lkey = litr.key();
                            Json::Value lvalue = *litr;

                            if (lname.compare("kind") == 0)
                            {
                                std::string s = lvalue.asString();
                                std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                if (s.compare("input") == 0)
                                    ldl._kind = Layer::Kind::Input;
                                else if (s.compare("hidden") == 0)
                                    ldl._kind = Layer::Kind::Hidden;
                                else if (s.compare("target") == 0)
                                    ldl._kind = Layer::Kind::Target;
                                else if (s.compare("output") == 0)
                                    ldl._kind = Layer::Kind::Output;
                                else
                                {
                                    std::cout << std::format("LoadNeuralNetworkJSON: Invalid layer kind: {}\n", lvalue.asString());
                                    bValid = false;
                                    goto exit;
                                }
                            }

                            else if (lname.compare("type") == 0)
                            {
                                std::string s = lvalue.asString();
                                std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                if (s.compare("fullyconnected") == 0)
                                    ldl._type = Layer::Type::FullyConnected;
                                else if (s.compare("convolutional") == 0)
                                    ldl._type = Layer::Type::Convolutional;
                                else if (s.compare("pooling") == 0)
                                    ldl._type = Layer::Type::Pooling;
                                else
                                {
                                    std::cout << std::format("LoadNeuralNetworkJSON: Invalid layer type: {}\n", lvalue.asString());
                                    bValid = false;
                                    goto exit;
                                }
                            }
                        }

                        if ((ldl._type == Layer::Type::Pooling) || (ldl._type == Layer::Type::Convolutional))
                        {
                            ldl._bDimensionsProvided = false;
                        }

                        switch (ldl._kind)
                        {
                        case Layer::Kind::Input:
                            ldl._name = "Input" + std::to_string(nd._vLayerDescriptor.size());
                            break;

                        case Layer::Kind::Hidden:
                            ldl._name = "Hidden" + std::to_string(nd._vLayerDescriptor.size());
                            break;

                        case Layer::Kind::Output:
                            ldl._name = "Output" + std::to_string(nd._vLayerDescriptor.size());
                            break;

                        case Layer::Kind::Target:
                            ldl._name = "Target" + std::to_string(nd._vLayerDescriptor.size());
                            break;
                        }

                        for (Json::ValueIterator litr = layer.begin(); litr != layer.end() ; litr++)
                        {
                            std::string lname = litr.name();
                            std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);
                            Json::Value lkey = litr.key();
                            Json::Value lvalue = *litr;

                            if ((lname.compare("kind") == 0) || (lname.compare("type") == 0))
                            {
                                continue;
                            }

                            if (lname.compare("name") == 0)
                            {
                                ldl._name = lvalue.asString();
                                if (sLayer.find(ldl._name) != sLayer.end())
                                {
                                    std::cout << std::format("LoadNeuralNetworkJSON: Duplicate layer name detected: {}\n", ldl._name);
                                    bValid = false;
                                    goto exit;
                                }
                                sLayer.insert(ldl._name);
                                continue;
                            }

                            if (lname.compare("sparse") == 0)
                            {
                                if (lvalue.asBool())
                                    ldl._attributes|= Layer::Attributes::Sparse;
                                continue;
                            }
                            else if (lname.compare("n") == 0)
                            {
                                if (lvalue.isArray())
                                {
                                    if (lvalue.size() < 5)
                                    {
                                        ldl._dimensions = lvalue.size();
                                        switch (lvalue.size())
                                        {
                                            case 4:
                                                ldl._Nw = lvalue[3].asInt();
                                            case 3:
                                                ldl._Nz = lvalue[2].asInt();
                                            case 2:
                                                ldl._Ny = lvalue[1].asInt();
                                            case 1:
                                                ldl._Nx = lvalue[0].asInt();
                                        }

                                    }
                                    else
                                    {
                                        std::cout << std::format("LoadNeuralNetworkJSON: >4 dimensions detected in layer: {}\n", ldl._name);
                                        bValid = false;
                                        goto exit;
                                    }

                                }
                                else if (lvalue.isString())
                                {
                                    std::string nstring = lvalue.asString();
                                    std::transform(nstring.begin(), nstring.end(), nstring.begin(), ::tolower);
                                    if ((ldl._kind != Layer::Kind::Hidden) && (nstring.compare("auto") == 0))
                                        bAutoSize = true;
                                    else if (nstring.compare("auto") == 0)
                                    {
                                        std::cout << std::format("LoadNeuralNetworkJSON: Illegal attempt to use auto for hidden layer: {}\n", ldl._name);
                                        bValid = false;
                                        goto exit;
                                    }
                                }
                                else
                                {
                                    ldl._Nx = lvalue.asInt();
                                    ldl._dimensions = 1;
                                }
                                continue;
                            }
                            else if (lname.compare("pdropout") == 0)
                            {
                                ldl._pDropout = lvalue.asFloat();
                                continue;
                            }


                            if (ldl._kind != Layer::Kind::Input)
                            {
                                if (lname.compare("source") == 0)
                                {
                                    uint32_t size = lvalue.isArray() ? lvalue.size() : 1;

#if 0
                                    if ((ldl._type == Layer::Type::Pooling) && (size > 1))
                                    {
                                            std::cout << std::format("LoadNeuralNetworkJSON: Pooling layer {} has multiple sources\n", ldl._name);
                                            bValid = false;
                                            goto exit;
                                    }
#endif

                                    for (uint32_t j = 0; j < size; j++)
                                    {
                                        Json::Value src = lvalue.isArray() ? lvalue[j] : lvalue;
                                        ldl._vSource.push_back(src.asString());
                                        bSource = true;
                                    }
                                    continue;
                                }

                                else if ((lname.compare("kernel") == 0) || (lname.compare("kernelstride") == 0))
                                {
                                    uint32_t x = 1;
                                    uint32_t y = 1;
                                    uint32_t z = 1;
                                    uint32_t dimensions = 1;

                                    if (lvalue.isArray())
                                    {
                                        if (lvalue.size() < 4)
                                        {
                                            dimensions = lvalue.size();
                                            switch (lvalue.size())
                                            {
                                            case 3:
                                                z = lvalue[2].asInt();
                                            case 2:
                                                y = lvalue[1].asInt();
                                            case 1:
                                                x = lvalue[0].asInt();
                                            }
                                        }
                                        else
                                        {
                                            bValid = false;
                                            goto exit;
                                        }
                                    }
                                    else
                                    {
                                        x = lvalue.asInt();
                                    }

                                    if (lname.compare("kernel") == 0)
                                    {
                                        ldl._kernelX = x;
                                        ldl._kernelY = y;
                                        ldl._kernelZ = z;
                                        ldl._kernelDimensions = dimensions;
                                    }
                                    else
                                    {
                                        ldl._kernelStrideX = x;
                                        ldl._kernelStrideY = y;
                                        ldl._kernelStrideZ = z;
                                    }

                                    continue;
                                }
                            }

                            if (ldl._kind == Layer::Kind::Hidden)
                            {
                                if (lname.compare("batchnormalization") == 0)
                                {
                                    if (lvalue.asBool())
                                        ldl._attributes |= Layer::Attributes::BatchNormal;

                                    continue;
                                }
                                else if (lname.compare("sparsenesspenalty") == 0)
                                {
                                    for (Json::ValueIterator pitr = lvalue.begin(); pitr != lvalue.end(); pitr++)
                                    {
                                        std::string pname = pitr.name();
                                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);

                                        Json::Value pkey = pitr.key();
                                        Json::Value pvalue = *pitr;

                                        if (pname.compare("p") == 0)
                                            ldl._sparsenessPenalty_p = pvalue.asFloat();
                                        else if (pname.compare("beta") == 0)
                                            ldl._sparsenessPenalty_beta = pvalue.asFloat();
                                        else
                                        {
                                            std::cout << std::format("LoadNeuralNetworkJSON: Invalid sparseness penalty parameter for hidden layer {}\n", ldl._name);
                                            bValid = false;
                                            goto exit;
                                        }
                                    }
                                    continue;
                                }
                            }

                            if (ldl._kind == Layer::Kind::Output)
                            {
								if (lname.compare("softmax") == 0)
								{
									if (lvalue.asBool())
										ldl._attributes |= Activation::SoftMax;

									continue;
								}

                            }

                            if ((ldl._kind == Layer::Kind::Hidden) || (ldl._kind == Layer::Kind::Output))
                            {
                                if (ldl._type == Layer::Type::Pooling && lname == "function")
                                {
                                    std::string s = lvalue.asString();
                                    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });

                                    static const std::unordered_map<std::string, PoolingFunction> poolingFunctionMap = {
                                        {"max", PoolingFunction::Max},
                                        {"maxout", PoolingFunction::Maxout},
                                        {"dotproduct", PoolingFunction::DotProduct},
                                        {"cosine", PoolingFunction::Cosine},
                                        {"average", PoolingFunction::Average},
                                        {"lrn", PoolingFunction::LRN},
                                        {"localresponsenormalization", PoolingFunction::LRN}
                                    };

                                    auto it = poolingFunctionMap.find(s);
                                    if (it != poolingFunctionMap.end())
                                    {
                                        ldl._poolingFunction = it->second;
                                        continue;
                                    }
                                    else
                                    {
                                        std::cerr << std::format("LoadNeuralNetworkJSON: Invalid pooling function ({}) for pooling layer {}\n", lvalue.asString(), ldl._name);
                                        exit(EXIT_FAILURE);
                                    }
                                }

                                if (lname.compare("skip") == 0)
                                {
                                    uint32_t size = lvalue.isArray() ? lvalue.size() : 1;
                                    for (uint32_t j = 0; j < size; j++)
                                    {
                                        Json::Value src = lvalue.isArray() ? lvalue[j] : lvalue;
                                        ldl._vSkip.push_back(src.asString());
                                    }
                                    continue;
                                }

                                else if (lname == "activation")
                                {
                                    std::string s = lvalue.asString();
                                    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });

                                    static const std::unordered_map<std::string, Activation> activationMap = {
                                        {"sigmoid", Activation::Sigmoid},
                                        {"tanh", Activation::Tanh},
                                        {"linear", Activation::Linear},
                                        {"relu", Activation::RectifiedLinear},
                                        {"rectifiedlinear", Activation::RectifiedLinear},
                                        {"lrelu", Activation::LeakyRectifiedLinear},
                                        {"leakyrectifiedlinear", Activation::LeakyRectifiedLinear},
                                        {"elu", Activation::ExponentialLinear},
                                        {"exponentiallinear", Activation::ExponentialLinear},
                                        {"selu", Activation::ScaledExponentialLinear},
                                        {"scaledexponentiallinear", Activation::ScaledExponentialLinear},
                                        {"softplus", Activation::SoftPlus},
                                        {"softsign", Activation::SoftSign},
                                        {"softmax", Activation::SoftMax},
                                        {"relumax", Activation::RELUMax},
                                        {"linearmax", Activation::LinearMax}
                                    };

                                    auto it = activationMap.find(s);
                                    if (it != activationMap.end())
                                    {
                                        ldl._activation = it->second;
                                        continue;
                                    }
                                    else
                                    {
                                        std::cerr << std::format("LoadNeuralNetworkJSON: Invalid layer activation: {}\n", lvalue.asString());
                                        exit(EXIT_FAILURE);
                                    }
                                    continue;
                                }

                                else if ((lname.compare("reluslope") == 0) || (lname.compare("slope") == 0))
                                {
                                    ldl._RELUSlope = lvalue.asFloat();
                                    continue;
                                }
                                else if (lname.compare("elualpha") == 0)
                                {
                                    ldl._ELUAlpha = lvalue.asFloat();
                                    continue;
                                }
                                else if (lname.compare("selulambda") == 0)
                                {
                                    ldl._SELULambda = lvalue.asFloat();
                                    continue;
                                }

                                else if (lname.compare("weightnorm") == 0)
                                {
                                    ldl._weightNorm = lvalue.asFloat();
                                    continue;
                                }

                                else if (lname.compare("deltanorm") == 0)
                                {
                                    ldl._deltaNorm = lvalue.asFloat();
                                    continue;
                                }

                                else if (lname.compare("weightinit") == 0)
                                {
                                    for (int i = 0; i < lvalue.size(); i++)
                                    {
                                        for (Json::ValueIterator witr = lvalue.begin(); witr != lvalue.end() ; witr++)
                                        {
                                            std::string wname = witr.name();
                                            std::transform(wname.begin(), wname.end(), wname.begin(), ::tolower);
                                            Json::Value wkey = witr.key();
                                            Json::Value wvalue = *witr;

                                            if (wname == "scheme")
                                            {
                                                std::string scheme = wvalue.asString();
                                                std::transform(scheme.begin(), scheme.end(), scheme.begin(), [](unsigned char c) { return std::tolower(c); });

                                                static const std::unordered_map<std::string, WeightInitialization> weightInitMap = {
                                                    {"xavier", Xavier},
                                                    {"caffexavier", CaffeXavier},
                                                    {"gaussian", Gaussian},
                                                    {"uniform", Uniform},
                                                    {"unitball", UnitBall},
                                                    {"constant", Constant},
                                                    {"selu", SELU}
                                                };

                                                auto it = weightInitMap.find(scheme);
                                                if (it != weightInitMap.end())
                                                {
                                                    ldl._weightInit = it->second;
                                                }
                                                else
                                                {
                                                    std::cerr << "LoadNeuralNetworkJSON: Invalid weight initialization scheme: " << scheme << "\n";
                                                    exit(EXIT_FAILURE);
                                                }
                                            }
                                            else if (wname.compare("scale") == 0)
                                            {
                                               ldl._weightInitScale = wvalue.asFloat();
                                            }
                                            else if (wname.compare("bias") == 0)
                                            {
                                               ldl._biasInit = wvalue.asFloat();
                                            }
                                            else
                                            {
                                                std::cout << std::format("LoadNeuralNetworkJSON: Invalid weight initialization field: {}\n", wname);
                                                bValid = false;
                                                goto exit;
                                            }
                                        }
                                    }
                                    continue;
                                }

                                else if (lname.compare("sharedweights") == 0)
                                {
                                    uint32_t size = lvalue.isArray() ? lvalue.size() : 1;
                                    for (uint32_t i = 0; i < size; i++)
                                    {
                                        WeightDescriptor nd;
                                        Json::Value share = lvalue.isArray() ? lvalue[i] : lvalue;
                                        for (Json::ValueIterator sitr = share.begin(); sitr != share.end() ; sitr++)
                                        {
                                            std::string sname = sitr.name();
                                            std::transform(sname.begin(), sname.end(), sname.begin(), ::tolower);
                                            Json::Value skey = sitr.key();
                                            Json::Value svalue = *sitr;

                                            if (sname.compare("sourceinputlayer") == 0)
                                            {
                                                nd._sourceInputLayer = svalue.asString();
                                            }
                                            else if (sname.compare("sourceoutputlayer") == 0)
                                            {
                                                nd._sourceOutputLayer = svalue.asString();
                                            }
                                            else if (sname.compare("inputlayer") == 0)
                                            {
                                                nd._inputLayer = svalue.asString();
                                            }
                                            else if (sname.compare("transposed") == 0)
                                            {
                                                nd._bTransposed = svalue.asBool();
                                            }
                                            else
                                            {
                                                std::cout << std::format("LoadNeuralNetworkJSON: Invalid shared weight field: {}\n", sname);
                                                bValid = false;
                                                goto exit;
                                            }
                                        }
                                        nd._bShared = true;
                                        vSharedWeight.push_back(nd);
                                    }
                                    continue;
                                }
                            }


                            if ((ldl._kind == Layer::Kind::Input) || (ldl._kind == Layer::Kind::Output))
                            {
                                if (lname.compare("dataset") == 0)
                                {
                                    ldl._dataSet = lvalue.asString();
                                    continue;
                                }
                            }

                            std::cout << std::format("LoadNeuralNetworkJSON: Unknown neural network layer field: {}\n", lname);
                            bValid = false;
                            goto exit;
                        }

                        if (bAutoSize)
                        {
                            bool bFound = false;
                            for (auto p : vDataSet)
                            {
                                if (p->_name.compare(ldl._dataSet) == 0)
                                {
                                    ldl._Nx = p->_width;
                                    ldl._Ny = p->_height;
                                    ldl._Nz = p->_length;
                                    ldl._dimensions = p->_dimensions;
                                    bFound = true;
                                }
                            }
                            if (!bFound)
                            {
                                std::cout << std::format("LoadNeuralNetworkJSON: Unable to find data set {} to determine dimensions for layer: {}\n", ldl._dataSet, ldl._name);
                                bValid = false;
                                goto exit;
                            }
                        }

                        if (!bSource && (ldl._kind != Layer::Kind::Input))
                        {
                            ldl._vSource.push_back(nd._vLayerDescriptor.back()._name);
                        }

                        if ((ldl._type == Layer::Type::Pooling) &&
                            (ldl._poolingFunction == PoolingFunction::DotProduct) || (ldl._poolingFunction == PoolingFunction::Cosine))
                        {
                            if (ldl._vSource.size() < 2)
                            {
                                std::cout << std::format("LoadNeuralNetworkJSON: Dot product layer {} must have 2 or more sources\n", ldl._name);
                                bValid = false;
                                goto exit;
                            }
                            ldl._Nx = ldl._vSource.size() - 1;
                            ldl._Ny = 1;
                            ldl._Nz = 1;
                            ldl._dimensions = 1;
                        }

                        if (ldl._type != Layer::Type::Pooling)
                        {

                            uint32_t sharedWeightsFound         = 0;
                            for (uint32_t i = 0; i < ldl._vSource.size(); i++)
                            {
                                WeightDescriptor wd;
                                wd._inputLayer = ldl._vSource[i];
                                wd._outputLayer = ldl._name;
                                wd._norm = ldl._weightNorm;

                                for (uint32_t j = 0; j < vSharedWeight.size(); j++)
                                {
                                    if (vSharedWeight[j]._inputLayer == wd._inputLayer)
                                    {
                                        wd._bShared = true;
                                        wd._bTransposed = vSharedWeight[j]._bTransposed;
                                        wd._sourceInputLayer = vSharedWeight[j]._sourceInputLayer;
                                        wd._sourceOutputLayer = vSharedWeight[j]._sourceOutputLayer;
                                        sharedWeightsFound++;
                                        break;
                                    }
                                }
                                nd._vWeightDescriptor.push_back(wd);
                            }

                            if (sharedWeightsFound < vSharedWeight.size())
                            {
                                std::cout << ("LoadNeuralNetworkJSON: Unable to locate all shared weights\n");
                                bValid = false;
                                goto exit;
                            }
                        }

                        if (ldl._dimensions < ldl._kernelDimensions)
                        {
                            ldl._bDimensionsProvided = false;
                        }

                        nd._vLayerDescriptor.push_back(ldl);
                    }
                }

                else
                {
                    std::cout << std::format("LoadNeuralNetworkJSON: Unknown neural network field: {}\n", name);
                    bValid = false;
                    goto exit;
                }
            }
        }

        if (nd._sparsenessPenalty_beta > (float)0.0)
        {
            nd._bSparsenessPenalty = true;
        }

        if (nd._denoising_p > (float)0.0)
        {
            nd._bDenoising = true;

            for (size_t i = 0; i < nd._vLayerDescriptor.size(); i++)
            {
                if ((nd._vLayerDescriptor[i]._kind == Layer::Kind::Input) && ((nd._vLayerDescriptor[i]._attributes & Layer::Attributes::Sparse) != 0))
                {
                    nd._vLayerDescriptor[i]._attributes |= Layer::Attributes::Denoising;
                }
            }
        }
    }

    for (size_t i = 0; i <  nd._vLayerDescriptor.size(); i++)
    {
        if (isnan(nd._vLayerDescriptor[i]._RELUSlope))
            nd._vLayerDescriptor[i]._RELUSlope = nd._RELUSlope;
        if (isnan(nd._vLayerDescriptor[i]._ELUAlpha))
            nd._vLayerDescriptor[i]._ELUAlpha = nd._ELUAlpha;
        if (isnan(nd._vLayerDescriptor[i]._SELULambda))
            nd._vLayerDescriptor[i]._SELULambda = nd._SELULambda;
    }

    CalculateDerivedLayerDimensions(nd);

exit:
    MPI_Bcast(&bValid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bValid)
    {
        getGpu().Shutdown();
        std::exit(-1);
    }

    MPI_Bcast_NetworkDescriptor(nd);

    if (getGpu()._id == 0)
    {
        std::cout << "LoadNeuralNetworkJSON: Enumerating network:" << std::endl;
        std::cout << nd << std::endl;
    }

    pNetwork = new Network(nd, batch);
    return pNetwork;
}

Network* LoadNeuralNetworkNetCDF(const std::string& fname, const uint32_t batch) {
    Network* pNetwork = nullptr;
    NetworkDescriptor nd;
    bool bResult = true;
    uint32_t layers = 0;
    uint32_t weights = 0;

    std::cout << "Loading Neural Network from NetCDF file: " << fname;

    MPI_Bcast_string(nd._name);
    nd._bConvLayersCalculated = true;
    int MPI_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

    if (MPI_rank == 0) {
        try {
            netCDF::NcFile nc(fname, netCDF::NcFile::read);
            auto readAttribute = [&](const std::string& attrName, auto& target) {
                netCDF::NcGroupAtt attr = nc.getAtt(attrName);
                if (attr.isNull()) {
                    std::cerr << "Missing attribute: " << attrName << " in NetCDF input file " << fname;
                    throw std::runtime_error("Missing attribute: " + attrName);
                }

                if (attrName == "version") {
                    std::string versionStr;
                    attr.getValues(&versionStr);

                    try {
                        target = std::stof(versionStr);
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Failed to convert version attribute to float: " << e.what();
                        throw std::runtime_error("Failed to convert version attribute to float");
                    }
                }
                else {
                    attr.getValues(&target);
                }
                };

            std::vector<std::string> attrNames = {
                "version", "name", "kind", "errorFunction", "decay", "maxout_k", "LRN_k",
                "LRN_n", "LRN_alpha", "LRN_beta", "bSparsenessPenalty", "sparsenessPenalty_p",
                "sparsenessPenalty_beta", "bDenoising", "denoising_p", "deltaBoost_one",
                "deltaBoost_zero", "SMCE_oneScale", "SMCE_zeroScale", "SMCE_oneTarget",
                "SMCE_zeroTarget", "checkpoint_name", "checkpoint_interval", "checkpoint_epochs",
                "ShuffleIndices", "layers", "weights"
            };

            for (const std::string& attrName : attrNames) {
                readAttribute(attrName, nd._name);
            }

            for (uint32_t i = 0; i < layers; i++) {
                LayerDescriptor ld;
                if (!LoadLayerDescriptorNetCDF(fname, nc, i, ld)) {
                    std::cerr << "Error reading layer data in NetCDF input file " << fname;
                }
                nd._vLayerDescriptor.push_back(ld);
            }

            for (uint32_t i = 0; i < weights; i++) {
                WeightDescriptor wd;
                if (!LoadWeightDescriptorNetCDF(fname, nc, i, wd)) {
                    std::cerr << "Error reading weight data in NetCDF input file " << fname;
                }
                nd._vWeightDescriptor.push_back(wd);
            }
        }
        catch (netCDF::exceptions::NcException& e) {
            std::cerr << "Exception: " << e.what();
            bResult = false;
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!bResult) {
        std::cerr << "Failed to load neural network from NetCDF file.";
        getGpu().Shutdown();
        exit(EXIT_FAILURE);
    }

    std::cout << "Broadcasting NetworkDescriptor...";

    MPI_Bcast_NetworkDescriptor(nd);

    if (MPI_rank == 0) {
        std::cout << "Enumerating network:\n" << nd;
    }

    pNetwork = new Network(nd, batch);
    pNetwork->RefreshState();

    std::cout << "Neural Network loaded successfully.";
    return pNetwork;
}


bool Network::P2P_Bcast(void* pBuffer, size_t size) {
    auto& gpu = getGpu();

    if (gpu._numprocs <= 1) {
        return true;
    }

    if (!gpu._bP2P) {
        std::unique_ptr<char[]> pCPUBuffer(new char[size]);
        cudaMemcpy(pCPUBuffer.get(), pBuffer, size, cudaMemcpyDeviceToHost);
        MPI_Bcast(pCPUBuffer.get(), size, MPI_BYTE, 0, MPI_COMM_WORLD);
        cudaMemcpy(pBuffer, pCPUBuffer.get(), size, cudaMemcpyHostToDevice);
        return true;
    }

    if (gpu._numprocs != NUM_GPUS) {
        throw std::runtime_error("Network::P2P_Bcast: Number of GPUs does not match NUM_GPUS");
    }

    int cudaDeviceId;
    cudaGetDevice(&cudaDeviceId);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (gpu._id == 0) {
        cudaError_t status;
        status = cudaMemcpyAsync(GetP2PSendBuffer(), pBuffer, size, cudaMemcpyDeviceToDevice, stream);
        if (status != cudaSuccess) {
            throw std::runtime_error("Network::P2P_Bcast: Failure to copy to P2P send buffer on root GPU");
        }
    }

    for (int i = 1; i < NUM_GPUS; ++i) {
        if (gpu._id == i) {
            cudaError_t status;
            status = cudaMemcpyPeerAsync(pBuffer, cudaDeviceId, GetP2PSendBuffer(), i, size, stream);
            if (status != cudaSuccess) {
                throw std::runtime_error("Network::P2P_Bcast: Failure to copy from P2P send buffer");
            }
        }
    }

    cudaStreamSynchronize(stream);

    MPI_Barrier(MPI_COMM_WORLD);

    cudaStreamDestroy(stream);

    return true;
}

bool Network::P2P_Allreduce(float* pBuffer, size_t size) {
    const auto& gpu = getGpu();

    if (gpu._numprocs <= 1 || NUM_GPUS <= 1) {
        return true;
    }

    auto bufferCopy = [&](auto* destination, auto* source, size_t count, int gpuIndex) {
        cudaSetDevice(gpuIndex);
        auto status = cudaMemcpy(destination, source, count * sizeof(float), cudaMemcpyDefault);
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
        };

    auto synchronizeAndBarrier = [&]() {
        for (int i = 0; i < NUM_GPUS; ++i) {
            cudaSetDevice(i);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
            }
        }
        if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Barrier failed");
        }
        };

    try {
        if (!gpu._bP2P) {
            for (int i = 0; i < NUM_GPUS; ++i) {
                bufferCopy(_pCPUBuffer.get(), pBuffer, size, i);
            }
            MPI_Allreduce(MPI_IN_PLACE, _pCPUBuffer.get(), size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            for (int i = 0; i < NUM_GPUS; ++i) {
                bufferCopy(pBuffer, _pCPUBuffer.get(), size, i);
            }
            return true;
        }

        for (int stage = 0; stage < NUM_GPUS - 1; ++stage) {
            for (int srcGpu = 0; srcGpu < NUM_GPUS; ++srcGpu) {
                int dstGpu = (srcGpu + stage + 1) % NUM_GPUS;
                cudaSetDevice(srcGpu);
                bufferCopy(GetPeerBuffer(), pBuffer, size, srcGpu);
                cudaDeviceEnablePeerAccess(dstGpu, 0);
                cudaSetDevice(dstGpu);
                bufferCopy(GetP2PReceiveBuffer(), GetPeerBuffer(), size, dstGpu);
                synchronizeAndBarrier();
                kAddBuffers(GetP2PReceiveBuffer(), pBuffer, size);
            }
        }

    }
    catch (const std::runtime_error& e) {
        return false;
    }

    return true;
}