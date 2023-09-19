#include "GpuTypes.h"
#include "Types.h"
#include "Kernels.cuh"
#include "../src/utils/Utils.h"
#define __STDC_FORMAT_MACROS
#include <queue>
#include <set>
#include <chrono>
#include <memory>
#include "Enum.h"
#include "GpuSort.h"
#include "Layer.h"
#include "Network.h"
#include "Weight.h"
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
{
}

std::ostream& operator<< (std::ostream& out, NetworkDescriptor& d)
{
    out << "Name:                    " << d._name << '\n';
    out << "Kind:                    " << d._kind << '\n';
    out << "bShuffleIndices          " << std::boolalpha << d._bShuffleIndices << '\n';
    out << "Error Function:          " << d._errorFunction << '\n';
    out << "MaxOut_k:                " << d._maxout_k << '\n';
    out << "LRN_k:                   " << d._LRN_k << '\n';
    out << "LRN_n:                   " << d._LRN_n << '\n';
    out << "LRN_beta:                " << d._LRN_beta << '\n';
    out << "LRN_alpha:               " << d._LRN_alpha << '\n';
    out << "bSparsenessPenalty:      " << std::boolalpha << d._bSparsenessPenalty << '\n';
    out << "sparsenessPenalty_beta:  " << d._sparsenessPenalty_beta << '\n';
    out << "sparsenessPenalty_p:     " << d._sparsenessPenalty_p << '\n';
    out << "bDenoising:              " << std::boolalpha << d._bDenoising << '\n';
    out << "denoising_p:             " << d._denoising_p << '\n';
    out << "deltaBoost_one:          " << d._deltaBoost_one << '\n';
    out << "deltaBoost_zero:         " << d._deltaBoost_zero << '\n';
    out << "SMCE_oneTarget:          " << d._SMCE_oneTarget << '\n';
    out << "SMCE_zeroTarget:         " << d._SMCE_zeroTarget << '\n';
    out << "SMCE_oneScale:           " << d._SMCE_oneScale << '\n';
    out << "SMCE_zeroScale:          " << d._SMCE_zeroScale << '\n';
    out << "checkpoint_name:         " << d._checkpoint_name << '\n';
    out << "checkpoint_interval:     " << d._checkpoint_interval << '\n';

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

Layer* Network::GetLayer(const std::string& layer) const
{
    auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("Network::GetLayerDimensions: Unknown layer %s.\n", layer.c_str());
        }

        return NULL;
    }

    return itr->second;
}

std::vector<const Layer*>::iterator Network::GetLayers(Layer::Kind layerKind, std::vector<const Layer*>& layers) const
{
    int count = 0;
    for (auto& layerName : Network::GetLayers())
    {
        const Layer* layer = Network::GetLayer(layerName);
        if (layerKind == layer->_kind)
        {
            layers.insert(layers.end(), layer);
            ++count;
        }
    }
    return layers.end() - count;
}

float* Network::GetScratchBuffer(size_t size)
{
    if (size > _scratchBufferSize)
    {
        _pbScratchBuffer.reset(new GpuBuffer<float>(size));
        _scratchBufferSize = size;

    }
    return _pbScratchBuffer->_pDevData;
}

void Network::SetCUDNNWorkspace(size_t size)
{
    if (size > _maxCUDNNWorkspaceSize)
    {
        _maxCUDNNWorkspaceSize = size;
    }
}

float* Network::GetP2PSendBuffer()
{
    return _pbP2PBuffer[_sendIndex]->_pDevData;
}

float* Network::GetP2PReceiveBuffer()
{
    return _pbP2PBuffer[_receiveIndex]->_pDevData;
}

float* Network::GetP2PCPUBuffer()
{
    return _pCPUBuffer.get();
}

float* Network::GetPeerBuffer() const
{
    return _pPeerBuffer[_receiveIndex];
}

float* Network::GetPeerBackBuffer() const
{
    return _pPeerBuffer[_sendIndex];
}

bool Network::SetLRN(float k, uint32_t n, float alpha, float beta)
{
    _LRN_k = k;
    _LRN_n = n;
    _LRN_alpha = alpha;
    _LRN_beta = beta;
    _bDirty = true;

    if (getGpu()._id == 0)
        printf("Network::SetLRN: k set to %f, n set to %u, alpha set to %f, beta set to %f.\n", k, n, alpha, beta);

    return true;
}

bool Network::SetDecay(float decay)
{
    if (decay >= (float)0.0)
    {
        _decay = decay;
        if (getGpu()._id == 0)
            printf("Network::SetDecay: decay set to %f\n.", decay);
        return true;
    }
    else
    {
        if (getGpu()._id == 0)
            printf("Network::SetDecay: invalid decay rate (<0.0) %f\n.", decay);
        return false;
    }
}

bool Network::SetMaxout(uint32_t k)
{
    if (k != _maxout_k)
    {
        _maxout_k = k;
        _bDirty = true;
    }

    if (getGpu()._id == 0)
        printf("Network::SetMaxout: k set to %u\n.", k);

    return true;
}

bool Network::SetSparsenessPenalty(float p, float beta)
{
    if ((p < (float)0.0) || (p > (float)1.0))
    {
        if (getGpu()._id == 0)
            printf("Network::SetSparsenessPenalty: Target sparseness must be >=0 and <=1.\n");
        return false;
    }

    _sparsenessPenalty_p = p;
    _sparsenessPenalty_beta = beta;
    _bSparsenessPenalty = (_sparsenessPenalty_beta > (float)0.0);
    _bDirty = true;

    if (getGpu()._id == 0)
        printf("Network::SetSparsenessPenalty: p set to %f, beta set to %f.\n", p, beta);

    return true;
}

bool Network::SetDenoising(float p)
{
    if ((p < (float)0.0) || (p >= (float)1.0))
    {
        if (getGpu()._id == 0)
            printf("Network::SetDenoising: Denoising probability must be >=0 and <1.\n");
        return false;
    }

    if (_denoising_p != p)
    {
        _denoising_p = p;
        _bDenoising = (_denoising_p > (float)0.0);
        _bDirty = true;
    }

    if (getGpu()._id == 0)
        printf("Network::SetDenoising: p set to %f.\n", p);

    return true;
}

bool Network::SetDeltaBoost(float one, float zero)
{
    if (one < (float)0.0)
    {
        if (getGpu()._id == 0)
            printf("Network::SetDeltaBoost: Illegal value for one (%f).\n", one);
        return false;
    }
    else if (zero < (float)0.0)
    {
        if (getGpu()._id == 0)
            printf("Network::SetDeltaBoost: Illegal value for zero (%f).\n", zero);
        return false;
    }

    _deltaBoost_one = one;
    _deltaBoost_zero = zero;
    _bDirty = true;

    if (getGpu()._id == 0)
        printf("Network::SetDeltaBoost: one set to %f, zero set to %f.\n", one, zero);

    return true;
}
bool Network::SetSMCE(float oneTarget, float zeroTarget, float oneScale, float zeroScale)
{
    if ((oneTarget < (float)0.0) || (oneTarget > (float)1.0))
    {
        if (getGpu()._id == 0)
            printf("Network::SetSMCE: Illegal value for oneTarget (%f).\n", oneTarget);
        return false;
    }
    else if ((zeroTarget < (float)0.0) || (zeroTarget > (float)1.0))
    {
        if (getGpu()._id == 0)
            printf("Network::SetSMCE: Illegal value for zeroTarget (%f).\n", zeroTarget);
        return false;
    }
    else if (oneScale < (float)0.0)
    {
        if (getGpu()._id == 0)
            printf("Network::SetSMCE: Illegal value for oneScale (%f).\n", oneScale);
        return false;
    }
    else if (zeroScale < (float)0.0)
    {
        if (getGpu()._id == 0)
            printf("Network::SetSMCE: Illegal value for zeroScale (%f).\n", zeroScale);
        return false;
    }

    _SMCE_oneTarget = oneTarget;
    _SMCE_zeroTarget = zeroTarget;
    _SMCE_oneScale = oneScale;
    _SMCE_zeroScale = zeroScale;
    _bDirty = true;

    if (getGpu()._id == 0)
        printf("Network::SetSMCE: oneTarget set to %f, zeroTarget set to %f, oneScale set to %f, zeroScale set to %f.\n", oneTarget, zeroTarget, oneScale, zeroScale);

    return true;
}

bool Network::SetCheckpoint(std::string name, int32_t interval)
{
    _checkpoint_name = name;
    _checkpoint_interval = interval;

    if (getGpu()._id == 0) {
        printf("Network::SetCheckPoint: filename set to %s, interval set to %d epochs.\n", name.c_str(), interval);
    }
    return true;
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

void Network::InitializeLayers(NetworkDescriptor& d)
{
    for (auto& l : d._vLayerDescriptor)
    {
        _vLayer.push_back(new Layer(l, _batch));
        _mLayer[_vLayer.back()->_name] = _vLayer.back();

        if (_vLayer.back()->_kind == Layer::Kind::Input)
        {
            _vInputLayer.push_back(_vLayer.back());
        }
        else if (_vLayer.back()->_kind == Layer::Kind::Output)
        {
            _vOutputLayer.push_back(_vLayer.back());
        }
    }

    if (getGpu()._id == 0)
    {
        std::cout << "Network::Network: " << _vInputLayer.size() << " input layer" << (_vInputLayer.size() > 1 ? "s" : "") << '\n';
        std::cout << "Network::Network: " << _vOutputLayer.size() << " output layer" << (_vOutputLayer.size() > 1 ? "s" : "") << '\n';
    }
}

void Network::ConnectLayers(NetworkDescriptor& d)
{
    try {
        for (const auto& layer : _vLayer) {
            ValidateLayerExistence(layer);

            for (const auto& skipLayerName : layer->_vSkip) {
                Layer* skipLayer = _mLayer[skipLayerName];
                ValidateLayerDimensionMatch(layer, skipLayer);
                ConnectSkipLayers(layer, skipLayer);
            }

            if (layer->_type == Layer::Type::Pooling) {
                for (const auto& sourceLayerName : layer->_vSource) {
                    Layer* sourceLayer = _mLayer[sourceLayerName];
                    ValidateLayerDimensionMatch(layer, sourceLayer, true);
                    ConnectPoolingLayers(layer, sourceLayer);
                }
            }
        }
    }
    catch (const LayerNotFoundException& e) {
        HandleException(e, "Layer not found");
    }
    catch (const DimensionMismatchException& e) {
        HandleException(e, "Dimension mismatch");
    }
    catch (const std::exception& e) {
        HandleException(e, "Unknown error");
    }
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

void Network::InitializeWeights(NetworkDescriptor& d)
{
    auto isShared = [](const WeightDescriptor& wd) { return wd._bShared; };

    auto isNotShared = [](const WeightDescriptor& wd) { return !wd._bShared; };

    for (auto& wd : d._vWeightDescriptor)
    {
        Layer* pInputLayer = _mLayer[wd._inputLayer];
        Layer* pOutputLayer = _mLayer[wd._outputLayer];
        Weight* pWeight = new Weight(*pInputLayer, *pOutputLayer, wd._bShared, wd._bTransposed, wd._bLocked, wd._norm);
        _vWeight.push_back(pWeight);

        if (wd._vWeight.empty() || wd._vBias.empty())
        {
            pWeight->Randomize();
        }

        if (isNotShared(wd) && !wd._vWeight.empty())
        {
            if (getGpu()._numprocs > 1)
            {
                float* pDst = pWeight->_vWeight.data();
                const uint32_t outgoingSize = pOutputLayer->_stride * 3;
                const uint32_t incomingSize = pInputLayer->_stride * 2;

                if (outgoingSize > incomingSize)
                {
                    const float* pSrc = wd._vWeight.data() + pOutputLayer->_minX;
                    for (size_t i = 0; i < pInputLayer->_stride; i++)
                    {
                        std::memcpy(pDst, pSrc, pOutputLayer->_localStride * sizeof(float));
                        pSrc += pOutputLayer->_stride;
                        pDst += pOutputLayer->_localStride;
                    }
                }
                else
                {
                    const float* pSrc = wd._vWeight.data() + pInputLayer->_minX * pOutputLayer->_stride;
                    std::memcpy(pDst, pSrc, pInputLayer->_localStride * pOutputLayer->_stride * sizeof(float));
                }
            }
            else
            {
                pWeight->_vWeight = wd._vWeight;
            }
            pWeight->_pbWeight->Upload(pWeight->_vWeight.data());
        }

        if (!wd._vBias.empty())
        {
            if (getGpu()._numprocs > 1)
            {
                const float* pSrc = wd._vBias.data() + pOutputLayer->_minX;
                float* pDst = pWeight->_vBias.data();
                std::memcpy(pDst, pSrc, pOutputLayer->_localStride * sizeof(float));
            }
            else
            {
                pWeight->_vBias = wd._vBias;
            }
            pWeight->_pbBias->Upload(pWeight->_vBias.data());
        }
    }

    for (uint32_t i = 0; i < d._vWeightDescriptor.size(); i++)
    {
        WeightDescriptor& wd = d._vWeightDescriptor[i];
        if (wd._bShared)
        {
            Weight* pWeight = _vWeight[i];
            const std::string inputLayer = wd._sourceInputLayer;
            const std::string outputLayer = wd._sourceOutputLayer;
            bool bFound = false;
            for (int j = 0; j < _vWeight.size(); j++)
            {
                if (!(_vWeight[j]->_bShared) &&
                    (_vWeight[j]->_inputLayer._name == inputLayer) &&
                    (_vWeight[j]->_outputLayer._name == outputLayer))
                {
                    if (wd._bTransposed)
                    {
                        if (wd._length > 1)
                        {
                            if (getGpu()._id == 0)
                                std::cout << "Network::Network: Can't transpose 3D weight matrix for shared weights between layers "
                                << _vWeight[i]->_inputLayer._name.c_str() << " and " << _vWeight[i]->_outputLayer._name.c_str() << '\n';
                            getGpu().Shutdown();
                            std::exit(-1);
                        }

                        if ((_vWeight[i]->_width != _vWeight[j]->_height) || (_vWeight[i]->_height != _vWeight[j]->_width))
                        {
                            if (getGpu()._id == 0)
                                std::cout << "Network::Network: Transposed dimensions for shared weights between layers "
                                << _vWeight[i]->_inputLayer._name.c_str() << " and " << _vWeight[i]->_outputLayer._name.c_str() << " do not match" << '\n';
                            getGpu().Shutdown();
                            std::exit(-1);
                        }
                    }
                    else if ((_vWeight[i]->_width != _vWeight[j]->_width) ||
                        (_vWeight[i]->_height != _vWeight[j]->_height) ||
                        (_vWeight[i]->_length != _vWeight[j]->_length))
                    {
                        if (getGpu()._id == 0)
                            std::cout << "Network::Network: Dimensions for shared weights between layers "
                            << _vWeight[i]->_inputLayer._name.c_str() << " and " << _vWeight[i]->_outputLayer._name.c_str() << " do not match" << '\n';
                        getGpu().Shutdown();
                        std::exit(-1);
                    }

                    _vWeight[i]->_pSharedWeight = _vWeight[j];
                    if (_vWeight[j]->_sharingCount == 1)
                        _vSharedWeight.push_back(_vWeight[j]);
                    _vWeight[j]->_sharingCount++;
                    bFound = true;
                    break;
                }
            }

            if (!bFound)
            {
                if (getGpu()._id == 0)
                    std::cout << "Network::Network: Unable to locate shared weights for connection between layers "
                    << _vWeight[i]->_inputLayer._name.c_str() << " and " << _vWeight[i]->_outputLayer._name.c_str() << '\n';
                getGpu().Shutdown();
                std::exit(-1);
            }
        }
    }
}

void Network::Randomize()
{
    for (auto pw : _vWeight)
        pw->Randomize();
}

void Network::SetBatch(uint32_t batch)
{
    if (batch % getGpu()._numprocs)
    {
        if (getGpu()._id == 0)
            printf("Network::SetBatch: Batch size must be a multiple of process count.\n");
        return;
    }

    if (batch != _batch)
    {
        _batch = batch;
        for (auto pL : _vLayer)
        {
            pL->SetBatch(batch);
        }

        _bDirty = true;
        if (getGpu()._id == 0)
            printf("Network::SetBatch: Batch size set to %d.\n", _batch);
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

void Network::SetShuffleIndices(bool bShuffleIndices)
{
    if (_bShuffleIndices != bShuffleIndices)
    {
        _bShuffleIndices = bShuffleIndices;
        _bDirty = true;
    }

    if (getGpu()._id == 0)
        std::cout << std::format("Network::SetShuffleIndices: Index shuffling is now %s\n", (_bShuffleIndices ? "on" : "off"));
}

uint32_t Network::GetPosition() const
{
    return _position;
}

void Network::SetPosition(uint32_t position)
{
    if (_bExamplesFound)
    {
        if (position < _examples)
            _position = position;
        else if (getGpu()._id == 0)
            std::cout << std::format("Network::SetPosition: Invalid position setting: %u, maximum %u\n", position, _examples);
    }
    else if (getGpu()._id == 0)
    {
        std::cout << std::format("Network::SetPosition: Illegal attempt to set position without examples count information.\n");
    }
}

bool Network::LockWeights(const std::string& inputLayer, const std::string& outputLayer)
{
    Layer* pInputLayer = _mLayer[inputLayer];
    Layer* pOutputLayer = _mLayer[outputLayer];

    if (pInputLayer == nullptr)
    {
        if (getGpu()._id == 0)
            std::cout << std::format("Network::LockWeights: Unable to find input layer %s.\n", inputLayer.c_str());
        return false;
    }

    if (pOutputLayer == nullptr)
    {
        if (getGpu()._id == 0)
            std::cout << std::format("Network::LockWeights: Unable to find input layer %s.\n", outputLayer.c_str());
        return false;
    }

    for (uint32_t i = 0; i < _vWeight.size(); i++)
    {
        if ((_vWeight[i]->_inputLayer._name == pInputLayer->_name) && (_vWeight[i]->_outputLayer._name == pOutputLayer->_name))
        {
            _vWeight[i]->Lock();
            return true;
        }
    }

    if (getGpu()._id == 0)
        std::cout << std::format("Network::LockWeights: Unable to find weight matrix between input layer %s and outputlayer %s.\n", inputLayer.c_str(), outputLayer.c_str());
    return false;
}

bool Network::UnlockWeights(const std::string& inputLayer, const std::string& outputLayer)
{
    Layer* pInputLayer = _mLayer[inputLayer];
    Layer* pOutputLayer = _mLayer[outputLayer];

    if (pInputLayer == nullptr)
    {
        if (getGpu()._id == 0)
            std::cout << std::format("Network::UnlockWeights: Unable to find input layer %s.\n", inputLayer.c_str());
        return false;
    }

    if (pOutputLayer == nullptr)
    {
        if (getGpu()._id == 0)
            std::cout << std::format("Network::UnlockWeights: Unable to find input layer %s.\n", outputLayer.c_str());
        return false;
    }

    for (uint32_t i = 0; i < _vWeight.size(); i++)
    {
        if ((_vWeight[i]->_inputLayer._name == pInputLayer->_name) && (_vWeight[i]->_outputLayer._name == pOutputLayer->_name))
        {
            _vWeight[i]->Unlock();
            return true;
        }
    }

    if (getGpu()._id == 0)
        std::cout << std::format("Network::UnlockWeights: Unable to find weight matrix between input layer %s and outputlayer %s.\n", inputLayer.c_str(), outputLayer.c_str());
    return false;
}

void Network::SetTrainingMode(TrainingMode mode)
{
    if (_trainingMode != mode)
    {
        _trainingMode = mode;
        _bDirty = true;
    }

    if (getGpu()._id == 0)
        std::cout << "Network::SetTrainingMode: Optimizer is now " << _trainingMode << '\n';
}

void Network::RefreshShuffleBuffers()
{
    if (_bAllDataLoaded)
    {
        if (_bShuffleIndices && (_mode == Training))
        {
            if (_shuffleIndices != _examples)
            {
                if (getGpu()._id == 0)
                {
                    _pShuffleIndexSort.reset();
                }
                else
                {
                    _pbShuffleIndex.reset();
                }

                _shuffleIndices = _examples;

                if (getGpu()._id == 0)
                {
                    _pShuffleIndexSort.reset(new GpuSort<uint32_t, uint32_t>(_shuffleIndices));
                    _pShuffleIndex = _pShuffleIndexSort->GetValuePointer();

                    uint32_t stride = ((_shuffleIndices + 511) >> 9) << 9;
                    std::vector<uint32_t> vIndex(stride * 2);
                    for (uint32_t i = 0; i < _examples; i++)
                    {
                        vIndex[i] = i;
                    }
                    _pShuffleIndexSort->GetValueBuffer()->Upload(vIndex.data());
                }
                else
                {
                    _pbShuffleIndex.reset(new GpuBuffer<uint32_t>(_shuffleIndices));
                    _pShuffleIndex = _pbShuffleIndex->_pDevData;
                }
            }
        }
    }
}

void Network::ShuffleIndices()
{
    if (getGpu()._id == 0)
    {
        uint32_t stride = ((_shuffleIndices + 511) >> 9) << 9;
        std::vector<uint32_t> vIndex(stride * 2);
        for (uint32_t i = 0; i < _examples; i++)
        {
            vIndex[i] = i;
        }
        _pShuffleIndexSort->GetValueBuffer()->Upload(vIndex.data());

        curandGenerate(getGpu()._RNG, _pShuffleIndexSort->GetKeyPointer(), _shuffleIndices);

        _pShuffleIndexSort->Sort();
    }

    if (getGpu()._numprocs > 1)
    {
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        P2P_Bcast(_pShuffleIndex, _examples * sizeof(uint32_t));
    }
}

void Network::RefreshState()
{
    if (!_bAllDataLoaded)
    {
        _bAllDataLoaded = true;

        for (auto l : _vInputLayer)
        {
            if (l->_pDataSet == nullptr)
            {
                if (getGpu()._id == 0)
                    std::cout << "Network::RefreshState: Missing data set " << l->_dataSet << " for input layer " << l->_name << '\n';
                _bAllDataLoaded = false;
            }
        }

        if (_mode != Prediction)
        {
            for (auto l : _vOutputLayer)
            {
                if (l->_pDataSet == nullptr)
                {
                    if (getGpu()._id == 0)
                        std::cout << "Network::RefreshState: Missing data set " << l->_dataSet << " for output layer " << l->_name << '\n';
                    _bAllDataLoaded = false;
                }
            }
        }
    }

    if (_bDirty)
    {
        for (auto l : _vLayer)
        {
            if (l->_bDirty)
            {
                l->RefreshState(this, _trainingMode, _mode == Validation);
            }
        }

        for (auto w : _vWeight)
        {
            w->RefreshState(this, _trainingMode);
        }

        RefreshShuffleBuffers();
    }

    if (getGpu()._numprocs > 1)
    {
        DeallocatePeerBuffers();
        AllocatePeerBuffers();
    }

    if (_maxCUDNNWorkspaceSize > _CUDNNWorkspaceSize)
    {
        if (getGpu()._id == 0)
            std::cout << "Network::RefreshState: Setting cuDNN workspace size to " << _maxCUDNNWorkspaceSize << " bytes." << '\n';
        _CUDNNWorkspaceSize = _maxCUDNNWorkspaceSize;
        _pbCUDNNWorkspace.reset(new GpuBuffer<uint8_t>(_CUDNNWorkspaceSize));
    }

    if (_bDirty || (getGpu()._pNetwork != this))
    {
        getGpu().SetNeuralNetwork(this);
    }

    _bDirty = false;
}

void Network::ClearDataSets()
{
    _examples = 0;
    _bExamplesFound = false;
    for (auto l : _vInputLayer)
        l->_pDataSet = NULL;
    for (auto l : _vOutputLayer)
        l->_pDataSet = NULL;
}

void Network::LoadDataSets(std::vector<DataSetBase*>& vData)
{
    _bAllDataLoaded = false;
    for (auto l : _vInputLayer)
    {
        for (auto d : vData)
        {
            if (l->_dataSet.compare(d->_name) == 0)
            {
                if (l->_dimensions != d->_dimensions)
                {
                    if (getGpu()._id == 0)
                    {
                        std::cout << std::format("Network::LoadDataSets: Dimensionality mismatch %uD input layer %s versus %uD data set %s\n",
                            l->_dimensions, l->_name.c_str(), d->_dimensions, d->_name.c_str());
                    }
                }

                if ((l->_Nx < d->_width) ||
                    (l->_Ny < d->_height) ||
                    (l->_Nz < d->_length))
                {
                    if (getGpu()._id == 0)
                    {
                        std::cout << std::format("Network::LoadDataSets: Data element mismatch (%u, %u, %u) input layer %s versus (%u, %u, %u) data set %s\n",
                            l->_Nx, l->_Ny, l->_Nz, l->_name.c_str(),
                            d->_width, d->_height, d->_length, d->_name.c_str());
                    }
                    break;
                }

                if (!_bExamplesFound)
                {
                    _examples = d->_examples;
                    _bExamplesFound = true;
                }

                if (d->_examples != _examples)
                {
                    if (getGpu()._id == 0)
                        std::cout << std::format("Network::LoadDataSets: Mismatched examples count (%u vs %u) in dataset %s\n", _examples, d->_examples, d->_name.c_str());
                    break;
                }

                l->_pDataSet = d;
                l->_bSparse = d->_attributes & DataSetEnums::Attributes::Sparse;
                l->_bDirty = true;
                if (getGpu()._id == 0)
                {
                    std::cout << std::format("Network::LoadDataSets: Found data set %s for input layer %s\n", d->_name.c_str(), l->_name.c_str());
                }
                break;
            }
        }
    }

    for (auto l : _vOutputLayer)
    {
        for (auto d : vData)
        {
            if (l->_dataSet.compare(d->_name) == 0)
            {
                if (l->_dimensions != d->_dimensions)
                {
                    if (getGpu()._id == 0)
                    {
                        std::cout << std::format("Network::LoadDataSets: Dimensionality mismatch %uD output layer %s versus %uD data set %s\n",
                            l->_dimensions, l->_name.c_str(), d->_dimensions, d->_name.c_str());
                    }
                }

                if ((l->_Nx < d->_width) ||
                    (l->_Ny < d->_height) ||
                    (l->_Nz < d->_length))
                {
                    if (getGpu()._id == 0)
                    {
                        printf("Network::LoadDataSets: Data element mismatch (%u, %u, %u) output layer %s versus (%u, %u, %u) data set %s\n",
                            l->_Nx, l->_Ny, l->_Nz, l->_name.c_str(),
                            d->_width, d->_height, d->_length, d->_name.c_str());
                    }
                    break;
                }

                if (!_bExamplesFound)
                {
                    _examples = d->_examples;
                    _bExamplesFound = true;
                }

                if (d->_examples != _examples)
                {
                    if (getGpu()._id == 0)
                        std::cout << std::format("Network::LoadDataSets: Mismatched examples count (%u vs %u) in dataset %s\n", _examples, d->_examples, d->_name.c_str());
                    break;
                }

                l->_pDataSet = d;
                l->_bDirty = true;
                if (getGpu()._id == 0)
                    std::cout << std::format("Network::LoadDataSets: Found data set %s for output layer %s\n", d->_name.c_str(), l->_name.c_str());
                break;
            }
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
            std::cout << "unsupported mode in LoadBatch" << '\n';
            exit(1);
        }
    }
}

void Network::SaveWeights(const std::string& fname, const std::string& inputLayer, const std::string& outputLayer)
{
    bool bResult = true;
    if (getGpu()._id == 0)
    {
        Layer* pInputLayer = _mLayer[inputLayer];
        Layer* pOutputLayer = _mLayer[outputLayer];

        if (pInputLayer == nullptr)
        {
            std::cout << std::format("Network::SaveWeights: Unable to find input layer %s.\n", inputLayer.c_str());
            bResult = false;
            goto exit;
        }

        if (pOutputLayer == nullptr)
        {
            std::cout << std::format("Network::SaveWeights: Unable to find input layer %s.\n", outputLayer.c_str());
            bResult = false;
            goto exit;
        }

        for (auto w : _vWeight)
        {
            if ((w->_inputLayer._name == pInputLayer->_name) && (w->_outputLayer._name == pOutputLayer->_name))
            {
                FILE* fp;
                errno_t err = fopen_s(&fp, fname.c_str(), "w");
                if (err != 0 || fp == nullptr)
                {
                    printf("Network::SaveWeights: Failed to open output file %s.\n", fname.c_str());
                    bResult = false;
                    goto exit;
                }

                w->_pbWeight->Download(w->_vWeight.data());
                w->_pbBias->Download(w->_vBias.data());
                fprintf(fp, "%" PRIu64 ",%" PRIu64 "\n", w->_width, w->_height);
                for (int j = 0; j < w->_height; j++)
                {
                    for (int k = 0; k < w->_width; k++)
                    {
                        fprintf(fp, "%12.8f", w->_vWeight[j * w->_width + k]);
                        if (k != w->_width - 1)
                            fprintf(fp, ",");
                        else
                            fprintf(fp, "\n");
                    }
                }
                for (int k = 0; k < w->_width; k++)
                {
                    fprintf(fp, "%12.8f", w->_vBias[k]);
                    if (k != w->_width - 1)
                        fprintf(fp, ",");
                    else
                        fprintf(fp, "\n");
                }
                fclose(fp);
                bResult = true;
                goto exit;
            }
        }

        std::cout << std::format("Network::SaveWeights: Unable to find weight matrix between input layer %s and outputlayer %s.\n", inputLayer.c_str(), outputLayer.c_str());
        bResult = false;
    }

exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        std::exit(-1);
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
            if (getGpu()._id == 0)
                std::cout << std::format("Network::SaveLayer: Attempt to save nonexistent layer %s.\n", layer.c_str());
            bResult = false;
            goto exit;
        }

        FILE* fp = nullptr;
        errno_t err = fopen_s(&fp, fname.c_str(), "w");
        if (err != 0 || fp == nullptr)
        {
            if (getGpu()._id == 0)
                std::cout << std::format("Network::SaveLayer: Failed to open output file %s.\n", fname.c_str());
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
        getGpu().Shutdown();
        std::exit(-1);
    }
}

void Network::DumpLayer(FILE* fp, const std::string& layer)
{
    bool bResult = true;
    if (getGpu()._id == 0)
    {
        Layer* pLayer = _mLayer[layer];
        if (pLayer == NULL)
        {
            printf("Network::SaveLayer: Attempt to dump nonexistent layer %s.\n", layer.c_str());
            bResult = false;
            goto exit;
        }

        uint64_t batch = pLayer->_batch;
        if (batch + _position > _examples)
        {
            batch = _examples - _position;
        }
        uint32_t stride = pLayer->_localStride;
        uint64_t size = _batch * stride;
        std::vector<float> vData(size);
        pLayer->_pbUnit->Download(vData.data());
        for (uint32_t j = 0; j < batch; j++)
        {
            for (uint32_t k = 0; k < stride; k++)
            {
                fprintf(fp, "%f", vData[j * stride + k]);
                if (k < (stride - 1))
                    fprintf(fp, ",");
                else
                    fprintf(fp, "\n");
            }
        }
    }


exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }
}

void Network::SaveBatch(std::string fname)
{
    bool bResult = true;
    if (getGpu()._id == 0)
    {
        FILE* fp;
        errno_t err = fopen_s(&fp, fname.c_str(), "w");
        if (err != 0 || fp == nullptr)
        {
            if (getGpu()._id == 0)
                std::cout << std::format("Network::SaveBatch: Failed to open output file %s.\n", fname.c_str());
            bResult = false;
            goto exit;
        }

        DumpBatch(fp);
        fclose(fp);
    }

exit:
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        std::exit(-1);
    }
}

void Network::DumpBatch(FILE* fp)
{
    if (getGpu()._id == 0)
    {
        for (int i = 0; i < _vOutputLayer.size(); i++)
        {
            uint32_t stride = _vOutputLayer[i]->_localStride;
            uint32_t batch = _vOutputLayer[i]->_batch;
            if (batch + _position > _examples)
                batch = _examples - _position;
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            std::vector<float> vData(size);
            _vOutputLayer[i]->_pbUnit->Download(vData.data());
            for (uint32_t j = 0; j < batch; j++)
            {
                for (uint32_t k = 0; k < stride; k++)
                {
                    fprintf(fp, "%f", vData[j * stride + k]);
                    if (k < (stride - 1))
                        fprintf(fp, ",");
                    else
                        fprintf(fp, "\n");
                }
            }
        }
    }
}

void Network::PredictBatch(uint32_t layers)
{
    uint32_t maxLayers = _vLayer.size();
    if (layers > _vLayer.size())
    {
        if (getGpu()._id == 0)
            printf("Network::PredictBatch: Attempt to predict more layers than present in neural network %s\n", _name.c_str());
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
            if (getGpu()._id == 0)
            {
                std::cout << "Network::PredictBatch: Attempt to predict with neural network " << _name << " without providing data sets" << '\n';
                std::cout << "for all input and output layers." << '\n';
            }
            getGpu().Shutdown();
            exit(-1);
        }
    }

    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    ClearUpdates();
    LoadBatch();

    for (auto l : _vFPOrder)
    {
        l->ForwardPropagate(_position, batch, false);
    }
}

void Network::PredictTrainingBatch(uint32_t layers)
{
    uint32_t maxLayers = _vLayer.size();
    if (layers > _vLayer.size())
    {
        if (getGpu()._id == 0)
            printf("Network::PredictTrainingBatch: Attempt to predict more layers than present in neural network %s\n", _name.c_str());
        return;
    }

    if (_bDirty)
    {
        RefreshState();

        if (!_bAllDataLoaded)
        {
            if (getGpu()._id == 0)
            {
                std::cout << "Network::PredictTrainingBatch: Attempt to predict with neural network " << _name << " without providing data sets" << '\n';
                std::cout << "for all input and output layers." << '\n';
            }
            getGpu().Shutdown();
            exit(-1);
        }
    }

    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    LoadBatch();

    for (auto l : _vFPOrder)
    {
        l->ForwardPropagate(_position, batch, true);
    }
}

void Network::PredictValidationBatch(uint32_t layers)
{
    uint32_t maxLayers = _vLayer.size();
    if (layers > _vLayer.size())
    {
        if (getGpu()._id == 0)
            printf("Network::PredictValidationBatch: Attempt to predict more layers than present in neural network %s\n", _name.c_str());
        return;
    }

    if (_mode != Validation)
    {
        _mode = Prediction;
        _bDirty = true;
    }

    if (_bDirty)
    {
        RefreshState();

        if (!_bAllDataLoaded)
        {
            if (getGpu()._id == 0)
            {
                std::cout << "Network::PredictValidationBatch: Attempt to predict with neural network " << _name << " without providing data sets" << '\n';
                std::cout << "for all input and output layers." << '\n';
            }
            getGpu().Shutdown();
            exit(-1);
        }
    }

    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    LoadBatch();

    ClearUpdates();
    for (auto l : _vFPOrder)
    {
        l->ForwardPropagate(_position, batch, false);
    }
}

float Network::Train(uint32_t epochs, float alpha, float lambda, float lambda1, float mu, float mu1)
{
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
            if (getGpu()._id == 0)
            {
                std::cout << "Network::Train: Attempt to train neural network " << _name << " without providing data sets" << '\n';
                std::cout << "for all input and output layers." << '\n';
            }
            getGpu().Shutdown();
            exit(-1);
        }
    }

    if (_trainingMode != SGD && _bClearVelocity)
    {
        for (uint32_t i = 0; i < _vWeight.size(); i++)
            _vWeight[i]->ClearVelocity();
        _batches = 0;
    }

    float total_error_training = (float)0.0;
    float total_error_regularization = (float)0.0;
    float average_error_training = (float)FLT_MAX;
    float average_error_regularization = (float)0.0;
    float moving_average = (float)0.0;
    uint32_t brake_steps = 0;
    uint32_t init_steps = 100;

    for (uint32_t epoch = 0; epoch < epochs; epoch++)
    {
        auto const start = std::chrono::steady_clock::now();
        total_error_training = (float)0.0;
        total_error_regularization = (float)0.0;

        if (_bDenoising)
        {
            for (auto l : _vInputLayer)
            {
                if (l->_bDenoising)
                    l->GenerateDenoisingData();
            }
        }

        if (_bShuffleIndices)
        {
            ShuffleIndices();
        }

        for (uint32_t pos = 0; pos < GetExamples(); pos += GetBatch())
        {
            SetPosition(pos);
            ClearUpdates();
            PredictTrainingBatch();
            float error_training, error_regularization, error;
            std::tuple<float, float> errorTuple = CalculateError(lambda, lambda1);
            error_training = std::get<0>(errorTuple);
            error_regularization = std::get<1>(errorTuple);
            uint32_t minibatch = GetBatch();
            if (_examples - pos < minibatch)
                minibatch = _examples - pos;
            total_error_training += error_training;
            total_error_regularization += error_regularization * minibatch;
            if (_verbose && getGpu()._id == 0) {
                printf("Network::Train: Minibatch@%u, average error %f, (%f training, %f regularization), alpha %f\n", pos, error_training / minibatch + error_regularization, error_training / minibatch, error_regularization, alpha);
            }

            float step_alpha = (_decay <= 0.0) ? alpha : alpha * ((float)1.0 / ((float)1.0 + _decay * ((float)_batches)));
            moving_average = 0.9 * moving_average + 0.1 * error_training;
            if (init_steps == 0)
            {
                if (error_training > 2.0 * moving_average)
                {
                    brake_steps = 25;
                    if (getGpu()._id == 0)
                        printf("Network::Train: Detected network divergence, attempting recovery.\n");
                }
            }
            else
                init_steps--;

            if (brake_steps > 0)
            {
                step_alpha *= (float)0.1;
                brake_steps--;
            }

            if (brake_steps < 24)
            {
                BackPropagate();

                _batches++;

                UpdateWeights(step_alpha, lambda, lambda1, mu, mu1);
            }

#if 0
            static const int WSIZE = 32;
            if (getGpu()._id == 0)
            {
                vector<float> vGrad(WSIZE);
                for (auto w : _vWeight)
                {
                    cudaMemcpy(vGrad.data(), w->_pbWeight->_pDevData, WSIZE * sizeof(float), cudaMemcpyDefault);
                    printf("WG %s %s\n", w->_inputLayer._name.c_str(), w->_outputLayer._name.c_str());
                    for (int i = 0; i < WSIZE; i++)
                        printf("%10.6f ", vGrad[i]);
                    printf("\n");
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }
        auto const end = std::chrono::steady_clock::now();
        average_error_training = total_error_training / GetExamples();
        average_error_regularization = total_error_regularization / GetExamples();
        if (getGpu()._id == 0)
            printf("Network::Train: Epoch %d, average error %f, average training error %f, average regularization error %f, elapsed time %fs\n", ++_epochs,
                average_error_training + average_error_regularization,
                average_error_training, average_error_regularization,
                elapsed_seconds(start, end));

        if (_checkpoint_interval > 0)
        {
            _checkpoint_epochs++;
            if (_checkpoint_epochs >= _checkpoint_interval)
            {
                std::string filename = _checkpoint_name + std::to_string(_epochs) + ".nc";
                if (getGpu()._id == 0)
                    printf("Network::Train: saving checkpoint %s\n", filename.c_str());

                SaveNetCDF(filename);
                _checkpoint_epochs = 0;
            }
        }
    }

    return average_error_training + average_error_regularization;
}

void Network::ClearUpdates()
{
    for (auto w : _vWeight)
    {
        w->_updateCount = 0;
    }

    for (auto l : _vLayer)
        l->ClearUpdates();
}

std::tuple<float, float> Network::CalculateError(float lambda, float lambda1)
{
    float error_training = (float)0.0;
    float error_regularization = (float)0.0;

    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    for (auto l : _vOutputLayer)
    {
        error_training += l->CalculateError(_position, batch, _errorFunction);
    }

    if ((lambda != (float)0.0) || (lambda1 != (float)0.0))
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
        error_training = derror_training;
        error_regularization = derror_regularization;
    }
    return std::make_tuple(error_training, error_regularization);
}

void Network::BackPropagate()
{
    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    for (auto l : _vBPOrder)
    {
        switch (l->_kind)
        {
        case Layer::Kind::Output:
            l->CalculateOutputDelta(_position, batch, _errorFunction);
            l->BackPropagate(_position, batch);
            break;

        case Layer::Kind::Hidden:
            l->BackPropagate(_position, batch);
            break;
        }
    }
}

void Network::UpdateWeights(float alpha, float lambda, float lambda1, float mu, float mu1)
{
    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;

    for (int64_t i = _vWeight.size() - 1; i >= 0; i--)
    {
        _vWeight[i]->UpdateWeights(_trainingMode);
    }

    for (auto l : _vLayer)
    {
        if (l->_bBatchNormalization)
            l->UpdateWeights(_trainingMode, batch, alpha, lambda, lambda1, mu, mu1, _batches);
    }
}

void Network::CalculateTopK(const std::string& layer, uint32_t k, GpuBuffer<float>* pbKey, GpuBuffer<unsigned int>* pbValue)
{
    Layer* pLayer = _mLayer[layer];
    if (pLayer == NULL)
    {
        if (getGpu()._id == 0)
            printf("Network::CalculateTopK: Unknown layer %s.\n", layer.c_str());
        return;
    }
    else if (k > 128)
    {
        if (getGpu()._id == 0)
            printf("Network::CalculateTopK: Can only calculate 128 or fewer elements.\n");
        return;
    }
    else if (k > pLayer->_Nx * pLayer->_Ny * pLayer->_Nz)
    {
        if (getGpu()._id == 0)
            printf("Network::CalculateTopK: Layer has fewer elements than k (%u vs %u).\n", k, pLayer->_Nx * pLayer->_Ny * pLayer->_Nz);
        return;
    }

    uint32_t batch = _batch;
    if (_position + batch > _examples)
        batch = _examples - _position;
    CalculateOutput(pLayer->_pbUnit->_pDevData, pbKey->_pDevData, pbValue->_pDevData, batch, pLayer->_localStride, k);

    return;
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
            printf("Network::SaveNetCDF Error opening binary output file %s to save neural network %s.\n", fname.c_str(), _name.c_str());
            bResult = false;
        }
    }

    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
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

    return vResult;
}

const std::string& Network::GetName() const
{
    return _name;
}

float* Network::GetUnitBuffer(const std::string& layer)
{
    const auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("Network::GetUnitBuffer: Unknown layer %s.\n", layer.c_str());
        }

        return NULL;
    }

    return itr->second->GetUnitBuffer();
}

float* Network::GetDeltaBuffer(const std::string& layer)
{
    const auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("Network::GetDeltaBuffer: Unknown layer %s.\n", layer.c_str());
        }
        return NULL;
    }

    return itr->second->GetDeltaBuffer();
}

uint64_t Network::GetBufferSize(const std::string& layer) const
{
    const auto itr = _mLayer.find(layer);
    if (itr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("Network::GetDeltaBuffer: Unknown layer %s.\n", layer.c_str());
        }
        return 0;
    }

    return itr->second->GetBufferSize();
}

Weight* Network::GetWeight(const std::string& inputLayer, const std::string& outputLayer) const
{
    auto inputLayerItr = _mLayer.find(inputLayer);
    if (inputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("Network::GetWeight: Unknown input layer %s.\n", inputLayer.c_str());
        }
        return NULL;
    }

    const auto outputLayerItr = _mLayer.find(outputLayer);
    if (outputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("Network::GetWeight: Unknown output layer %s.\n", outputLayer.c_str());
        }
        return NULL;
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

    if (getGpu()._id == 0)
    {
        printf("Network::GetWeight: No set of weights connecting layer %s to layer %s.\n", inputLayer.c_str(), outputLayer.c_str());
    }

    return NULL;
}

float* Network::GetWeightBuffer(const std::string& inputLayer, const std::string& outputLayer)
{
    const auto inputLayerItr = _mLayer.find(inputLayer);
    if (inputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("Network::GetWeight: Unknown input layer %s.\n", inputLayer.c_str());
        }
        return NULL;
    }

    const auto outputLayerItr = _mLayer.find(outputLayer);
    if (outputLayerItr == _mLayer.end())
    {
        if (getGpu()._id == 0)
        {
            printf("Network::GetWeightBuffer: Unknown output layer %s.\n", outputLayer.c_str());
        }
        return NULL;
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
        printf("Network::GetWeightBuffer: No set of weights connecting layer %s to layer %s.\n", inputLayer.c_str(), outputLayer.c_str());
    }

    return NULL;
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

void CalculateDerivedLayerDimensions(NetworkDescriptor& d)
{
    std::map<LayerDescriptor*, bool> mbDimensionsCalculated;
    std::map<std::string, LayerDescriptor*> mLayer;

    for (size_t i = 0; i < d._vLayerDescriptor.size(); i++)
    {
        LayerDescriptor* pL = &(d._vLayerDescriptor[i]);
        bool bFlag = true;
        if ((pL->_kind == Layer::Kind::Hidden) &&
            ((pL->_type == Layer::Type::Pooling) || (pL->_type == Layer::Type::Convolutional)))
            bFlag = false;
        mbDimensionsCalculated[pL] = bFlag;
        mLayer[pL->_name] = pL;
    }

    bool bFinished;
    do {
        bFinished = true;

        for (size_t i = 0; i < d._vLayerDescriptor.size(); i++)
        {
            LayerDescriptor* pL = &(d._vLayerDescriptor[i]);
            bool bPooling = pL->_type == Layer::Type::Pooling;
            bool bLRN = bPooling && (pL->_poolingFunction == PoolingFunction::LRN);
            bool bDotProduct = bPooling && ((pL->_poolingFunction == PoolingFunction::DotProduct) || (pL->_poolingFunction == PoolingFunction::Cosine));

            if (!mbDimensionsCalculated[pL])
            {
                bool bAllInputsCalculated = true;
                for (auto& s : pL->_vSource)
                {
                    LayerDescriptor* pS = mLayer[s];
                    bAllInputsCalculated &= mbDimensionsCalculated[pS];
                }

                if (!bAllInputsCalculated)
                {
                    bFinished = false;
                    continue;
                }

                bool bSized = false;
                LayerDescriptor* pL0 = mLayer[pL->_vSource[0]];
                uint32_t N = pL->_Nx;
                uint32_t oldNx = bDotProduct ? pL0->_Nx : 1;
                uint32_t oldNy = bDotProduct ? pL0->_Ny : 1;
                uint32_t oldNz = bDotProduct ? pL0->_Nz : 1;
                uint32_t nx = bDotProduct ? pL->_vSource.size() - 1 : 1;
                uint32_t ny = 1;
                uint32_t nz = 1;
                uint32_t nw = 1;
                for (auto& s : pL->_vSource)
                {
                    LayerDescriptor* pS = mLayer[s];

                    if (bDotProduct)
                    {
                        if ((oldNx != pS->_Nx) || (oldNy != pS->_Ny) || (oldNz != pS->_Nz))
                        {
                            if (getGpu()._id == 0)
                                printf("Network::CalculateDerivedLayerDimensions: Inconsistent incoming data size for dot product layer %s\n", pL->_name.c_str());
                            getGpu().Shutdown();
                            exit(-1);
                        }
                    }
                    else
                    {
                        if (!bLRN)
                        {
                            nx = CalculateConvolutionDimensions(pS->_Nx, pL->_kernelX, pL->_kernelStrideX);
                            ny = CalculateConvolutionDimensions(pS->_Ny, pL->_kernelY, pL->_kernelStrideY);
                            nz = CalculateConvolutionDimensions(pS->_Nz, pL->_kernelZ, pL->_kernelStrideZ);
                            nw = pS->_Nw;
                            if (bPooling)
                                pL->_dimensions = pS->_dimensions;
                        }
                        else
                        {
                            nx = pS->_Nx;
                            ny = pS->_Ny;
                            nz = pS->_Nz;
                            nw = pS->_Nw;
                            pL->_dimensions = pS->_dimensions;
                        }

                        switch (pL->_kernelDimensions)
                        {
                        case 3:
                            if (pS->_Nz < pL->_kernelZ)
                            {
                                pL->_kernelPaddingZ = (pL->_kernelZ - pS->_Nz + 1) / 2;
                            }
                            else if (pL->_kernelStrideZ == 1)
                            {
                                pL->_kernelPaddingZ = pL->_kernelZ / 2;
                            }

                        case 2:
                            if (pS->_Ny < pL->_kernelY)
                            {
                                pL->_kernelPaddingY = (pL->_kernelY - pS->_Ny + 1) / 2;
                            }
                            else if (pL->_kernelStrideY == 1)
                            {
                                pL->_kernelPaddingY = pL->_kernelY / 2;
                            }

                        case 1:
                            if (pS->_Nx < pL->_kernelX)
                            {
                                pL->_kernelPaddingX = (pL->_kernelX - pS->_Nx + 1) / 2;
                            }
                            else if (pL->_kernelStrideX == 1)
                            {
                                pL->_kernelPaddingX = pL->_kernelX / 2;
                            }
                        }

                        if (bSized)
                        {
                            if ((nx != oldNx) || (ny != oldNy) || (nz != oldNz))
                            {
                                if (getGpu()._id == 0)
                                    printf("Network::CalculateDerivedLayerDimensions: Inconsistent incoming data size for convolution layer %s\n", pL->_name.c_str());
                                getGpu().Shutdown();
                                exit(-1);
                            }
                        }
                        bSized = true;
                        oldNx = nx;
                        oldNy = ny;
                        oldNz = nz;
                        mbDimensionsCalculated[pL] = true;
                    }
                }
                pL->_Nx = nx;
                pL->_Ny = ny;
                pL->_Nz = nz;
                pL->_Nw = nw;
                if (!bPooling)
                {
                    switch (pL->_kernelDimensions)
                    {
                    case 1:
                        pL->_Ny = N;
                        pL->_dimensions = 2;
                        break;

                    case 2:
                        pL->_Nz = N;
                        pL->_dimensions = 3;
                        break;

                    case 3:
                        pL->_Nw = N;
                        pL->_dimensions = 4;
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

/// <summary>
/// Validates the neural network.
/// </summary>
/// <returns>True if validation is successful; otherwise, false.</returns>
bool Network::Validate()
{
    // Declare variables at the beginning for better readability
    const float delta = 0.001f; // Small value for numerical differentiation.
    const float epsilon = delta * 20.f; // Error threshold for validation.
    float network_lambda = 0.0f;
    float lambda1 = 0.0f;
    float alpha = 1.0f;
    float mu = 0.0f;
    float mu1 = 0.0f;
    float initialError;

    // Ensure correct network mode
    if (_mode != Validation)
    {
        _mode = Validation;
        _bDirty = true;
    }

    // Check if the network is dirty and needs to be refreshed
    if (_bDirty)
    {
        RefreshState();
        if (!_bAllDataLoaded)
        {
            std::cerr << "Network::Validate: Attempt to validate neural network " << _name << " without providing data sets" << '\n';
            return false;
        }
    }

    // Clear velocity if not using Stochastic Gradient Descent (SGD)
    if (_trainingMode != SGD && _bClearVelocity)
    {
#pragma omp parallel for
        for (int i = 0; i < _vWeight.size(); i++)
            _vWeight[i]->ClearVelocity();
    }

    // Shuffle indices if required
    if (_bShuffleIndices)
    {
        ShuffleIndices();
    }

    std::cout << "Validating network weights and biases with epsilon error threshold of " << epsilon << '\n';

    // Calculate initial error
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

    // Parallelize weight-related operations
#pragma omp parallel for
    for (int id = 0; id < _vWeight.size(); id++)
    {
        Weight* w = _vWeight[id];
        w->_pbWeight->Download(w->_vWeight.data());
        w->_pbBias->Download(w->_vBias.data());
        w->_pbWeightGradient->Download(vWeightGradient[id].data());
    }

    UpdateWeights(alpha, network_lambda, lambda1, mu, mu1);

    // Parallelize weight and bias updates
#pragma omp parallel for
    for (int id = 0; id < _vWeight.size(); id++)
    {
        Weight* w = _vWeight[id];
        w->_pbWeight->Upload(w->_vWeight.data());

        std::vector<float> bias_g(w->_pbBias->_length);
        w->_pbBias->Download(bias_g.data());

        // Calculate bias gradients
        for (int b = 0; b < bias_g.size(); b++)
        {
            vBiasGradient[id][b] = bias_g[b] - w->_vBias[b];
        }

        w->_pbBias->Upload(w->_vBias.data());
    }

    // Validate weights and biases
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

            // Calculate error and gradients
            errorTuple = CalculateError(network_lambda, lambda1);
            float errorTraining = std::get<0>(errorTuple);
            float errorRegularization = std::get<1>(errorTuple);
            float error = errorTraining + errorRegularization;
            float dEdW = (error - initialError) / delta;
            float weightGradient = vWeightGradient[id][i];

            std::cout << "errorTraining " << errorTraining << "; errorRegularization " << errorRegularization <<
                "; dEdW " << dEdW << "; weightGradient " << weightGradient << '\n';

            // Check if weight gradient exceeds the error threshold.
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

            // Calculate error and gradients for bias
            errorTuple = CalculateError(network_lambda, lambda1);
            float errorTraining = std::get<0>(errorTuple);
            float errorRegularization = std::get<1>(errorTuple);
            float error = errorTraining + errorRegularization;
            float dEdb = (error - initialError) / delta;
            float biasGradient = vBiasGradient[id][i];

            std::cout << "errorTraining " << errorTraining << "; errorRegularization " << errorRegularization <<
                "; dEdb " << dEdb << "; biasGradient " << biasGradient << '\n';

            // Check if bias gradient exceeds the error threshold.
            if (std::fabs(dEdb + biasGradient) > epsilon)
            {
                std::cerr << "Failed Bias " << i << " exceeds error threshold: " << dEdb << " vs " << biasGradient << '\n';
            }
        }

        w->_pbBias->Upload(w->_vBias.data());
    }

    return true;
}

/// <summary>
/// Deallocates peer buffers used for interprocess communication in a network.
/// </summary>
void Network::DeallocatePeerBuffers()
{
    try
    {
        if (getGpu()._numprocs > 1)
        {
            // Synchronize CUDA devices and MPI processes
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            for (size_t i = 0; i < 2; i++)
            {
                if (_pPeerBuffer[i] != nullptr)
                {
                    // Close the CUDA IPC memory handle
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
                // Reset P2P buffer
                _pbP2PBuffer[i].reset();
            }

            // Reset CPU buffer
            _pCPUBuffer.reset();
        }
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "Caught an exception: " << e.what() << '\n';
    }
}

/// <summary>
/// Allocates peer buffers for network communication.
/// </summary>
void Network::AllocatePeerBuffers()
{
    if (getGpu()._numprocs > 1)
    {
        _maxStride = 0;

        // Calculate the maximum stride among weight layers.
        for (auto w : _vWeight)
        {
            uint32_t stride = (w->_outputLayer._stride * 2) > (w->_inputLayer._stride * 2) ? w->_inputLayer._stride : w->_outputLayer._stride;
            if (stride > _maxStride)
            {
                _maxStride = stride;
            }
        }

        // Calculate the maximum required memory.
        uint64_t maxMemory = _maxStride * _batch;
        if (maxMemory < _examples)
        {
            maxMemory = _examples;
        }

        // Allocate GPU buffers.
        for (size_t i = 0; i < 2; i++)
        {
            _pbP2PBuffer[i].reset(new GpuBuffer<float>(maxMemory));
        }

        if (getGpu()._bP2P)
        {
            cudaIpcMemHandle_t* pMemHandle = new cudaIpcMemHandle_t[2 * getGpu()._numprocs];
            size_t pos = static_cast<size_t>(getGpu()._id) * 2;

            // Get IPC memory handles.
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

            // Gather IPC memory handles across all GPUs.
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pMemHandle, 2 * sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);

            // Determine the peer GPU and open memory handles for peer access.
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
            // Allocate CPU buffer.
            std::vector<float> cpuBuffer(maxMemory);
        }
    }
}

/// <summary>
/// Swaps the peer buffers, effectively toggling between two buffer indices.
/// </summary>
void Network::SwapPeerBuffers()
{
    _sendIndex = 1 - _sendIndex;
    _receiveIndex = 1 - _receiveIndex;
}

/// <summary>
/// An array of pairs mapping Network::Kind enum values to their string representations.
/// </summary>
std::pair<Network::Kind, std::string> Network::_sKindPair[] =
{
    std::pair<Network::Kind, std::string>(Network::Kind::FeedForward, "FeedForward"),
    std::pair<Network::Kind, std::string>(Network::Kind::AutoEncoder, "AutoEncoder")
};

/// <summary>
/// A map that associates Network::Kind enum values with their string representations.
/// </summary>
std::map<Network::Kind, std::string> Network::_sKindMap =
std::map<Network::Kind, std::string>(_sKindPair, Network::_sKindPair + sizeof(Network::_sKindPair) / sizeof(Network::_sKindPair[0]));

/// <summary>
/// Overload of the << operator to allow outputting Network::Kind enum values as strings.
/// </summary>
/// <param name="out">The output stream.</param>
/// <param name="k">The Network::Kind enum value to be converted to a string.</param>
/// <returns>The output stream after the conversion.</returns>
std::ostream& operator<< (std::ostream& out, Network::Kind& k)
{
    out << Network::_sKindMap[k];
    return out;
}

/// <summary>
/// Broadcasts a NetworkDescriptor structure using MPI_Bcast to synchronize it across all processes.
/// </summary>
/// <param name="d">The NetworkDescriptor structure to broadcast.</param>
void MPI_Bcast_NetworkDescriptor(NetworkDescriptor& d) {
    // Define the size of vectors and containers to broadcast
    uint32_t layers = d._vLayerDescriptor.size();
    uint32_t weights = d._vWeightDescriptor.size();

    // Broadcast the sizes
    MPI_Bcast(&layers, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&weights, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Resize vectors and containers on all processes
    d._vLayerDescriptor.resize(layers);
    d._vWeightDescriptor.resize(weights);

    // Broadcast the NetworkDescriptor struct using MPI_Bcast
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

    // Broadcast the vectors and containers
    MPI_Bcast(d._vLayerDescriptor.data(), layers, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(d._vWeightDescriptor.data(), weights, MPI_UINT32_T, 0, MPI_COMM_WORLD);
}

Network* LoadNeuralNetworkJSON(const std::string& fname, const uint32_t batch, const std::vector<DataSetBase*>& vDataSet)
{
    Network* pNetwork = NULL;
    NetworkDescriptor nd;
    Json::Value index;
    Json::Reader reader;
    bool bValid = true;
    bool bWeightsSupplied = false;
    std::string wfname;

    if (getGpu()._id == 0)
    {
        std::ifstream stream(fname, std::ifstream::binary);
        bool parsedSuccess = reader.parse(stream, index, false);

        if (!parsedSuccess)
        {
            std::cout << ("LoadNeuralNetworkJSON: Failed to parse JSON file: %s, error: %s\n", fname.c_str(), reader.getFormattedErrorMessages().c_str());
            bValid = false;
        }
        else
        {
            float version = VERSION;
            std::set<std::string> sLayer;
            for (Json::ValueIterator itr = index.begin(); itr != index.end() ; itr++)
            {
                std::string name = itr.name();
                std::transform(name.begin(), name.end(), name.begin(), ::tolower);
                Json::Value key = itr.key();
                Json::Value value = *itr;
                std::string vstring = value.isString() ? value.asString() : "";
                std::transform(vstring.begin(), vstring.end(), vstring.begin(), ::tolower);

                if (name.compare("version") == 0)
                {
                    version = value.asFloat();
                    if (version < 0.6999)
                    {
                        std::cout << ("LoadNeuralNetworkJSON: version %f (must be at least 0.7)\n", version);
                        bValid = false;
                        goto exit;
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
                        printf("LoadNeuralNetworkJSON: Invalid network kind: %s\n", value.asString().c_str());
                        bValid = false;
                        goto exit;
                    }
                }

                else if (name.compare("weightsdata") == 0)
                {
                    bWeightsSupplied = true;
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
                            printf("LoadNeuralNetworkJSON: Invalid LocalResponseNormalization parameter: %s\n", name.c_str());
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
                            std::cout << ("LoadNeuralNetworkJSON: Invalid MaxOut parameter: %s\n", name.c_str());
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
                            std::cout << ("LoadNeuralNetworkJSON: Invalid SparsenessPenalty parameter: %s\n", name.c_str());
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
                            std::cout << ("LoadNeuralNetworkJSON: Invalid Denoising parameter: %s\n", name.c_str());
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                else if (name.compare("deltaboost") == 0)
                {
                    for (Json::ValueIterator pitr = value.begin(); pitr != value.end() ; pitr++)
                    {
                        std::string pname = pitr.name();
                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);
                        Json::Value pkey = pitr.key();
                        Json::Value pvalue = *pitr;
                        if (pname.compare("one") == 0)
                            nd._deltaBoost_one = pvalue.asFloat();
                        else if (pname.compare("zero") == 0)
                            nd._deltaBoost_zero = pvalue.asFloat();
                        else
                        {
                            name = pitr.name();
                            std::cout << ("LoadNeuralNetworkJSON: Invalid DeltaBoost parameter: %s\n", name.c_str());
                            bValid = false;
                            goto exit;
                        }
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
                            std::cout << ("LoadNeuralNetworkJSON: Invalid ScaledMarginalCrossentropy parameter: %s\n", name.c_str());
                            bValid = false;
                            goto exit;
                        }
                    }
                }

                // Check if the attribute name is "shuffleindices"
                else if (name.compare("shuffleindices") == 0)
                {
                    // If it is, set the shuffleIndices attribute in the neural network descriptor (nd) based on the boolean value
                    nd._bShuffleIndices = value.asBool();
                    }

                    // Check if the attribute name is "reluslope" or "slope"
                else if ((name.compare("reluslope") == 0) || (name.compare("slope") == 0))
                {
                    // If it is, set the RELUSlope attribute in the neural network descriptor (nd) based on the float value
                    nd._RELUSlope = value.asFloat();
                    }

                    // Check if the attribute name is "elualpha"
                else if (name.compare("elualpha") == 0)
                {
                    // If it is, set the ELUAlpha attribute in the neural network descriptor (nd) based on the float value
                    nd._ELUAlpha = value.asFloat();
                    }

                    // Check if the attribute name is "selulambda"
                else if (name.compare("selulambda") == 0)
                {
                    // If it is, set the SELULambda attribute in the neural network descriptor (nd) based on the float value
                    nd._SELULambda = value.asFloat();
                    }

                    // Check if the attribute name is "decay"
                else if (name.compare("decay") == 0)
                {
                    // If it is, set the decay attribute in the neural network descriptor (nd) based on the float value
                    nd._decay = value.asFloat();
                    }

                    // Check if the attribute name is "errorfunction"
                else if (name.compare("errorfunction") == 0)
                {
                    // Depending on the value (vstring), set the error function in the neural network descriptor (nd)
                    if (vstring.compare("l1") == 0)
                        nd._errorFunction = ErrorFunction::L1;
                    else if (vstring.compare("l2") == 0)
                        nd._errorFunction = ErrorFunction::L2;
                    else if (vstring.compare("l2hinge") == 0)
                        nd._errorFunction = ErrorFunction::L2Hinge;
                    else if (vstring.compare("hinge") == 0)
                        nd._errorFunction = ErrorFunction::Hinge;
                    else if ((vstring.compare("crossentropy") == 0) || (vstring.compare("cross entropy") == 0))
                        nd._errorFunction = ErrorFunction::CrossEntropy;
                    else if (vstring.compare("scaledmarginalcrossentropy") == 0)
                        nd._errorFunction = ErrorFunction::ScaledMarginalCrossEntropy;
                    else if (vstring.compare("datascaledmarginalcrossentropy") == 0)
                        nd._errorFunction = ErrorFunction::DataScaledMarginalCrossEntropy;
                    else
                    {
                        // Handle the case of an invalid error function and mark the loading as invalid
                        std::cout << ("LoadNeuralNetworkJSON: Invalid error function: %s\n", value.asString().c_str());
                        bValid = false;
                        goto exit; // Exit the loading process
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
                                    std::cout << ("LoadNeuralNetworkJSON: Invalid layer kind: %s\n", lvalue.asString().c_str());
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
                                    std::cout << ("LoadNeuralNetworkJSON: Invalid layer type: %s\n", lvalue.asString().c_str());
                                    bValid = false;
                                    goto exit;
                                }
                            }
                        }

                        // Check if the layer type is either Pooling or Convolutional
                        if ((ldl._type == Layer::Type::Pooling) || (ldl._type == Layer::Type::Convolutional))
                        {
                            // If the layer type is Pooling or Convolutional, set bDimensionsProvided to false
                            ldl._bDimensionsProvided = false;
                        }

                        // Use a switch statement to determine the kind of layer
                        switch (ldl._kind)
                        {
                        case Layer::Kind::Input:
                            // If the layer kind is Input, set the name as "Input" followed by the current layer count
                            ldl._name = "Input" + std::to_string(nd._vLayerDescriptor.size());
                            break;

                        case Layer::Kind::Hidden:
                            // If the layer kind is Hidden, set the name as "Hidden" followed by the current layer count
                            ldl._name = "Hidden" + std::to_string(nd._vLayerDescriptor.size());
                            break;

                        case Layer::Kind::Output:
                            // If the layer kind is Output, set the name as "Output" followed by the current layer count
                            ldl._name = "Output" + std::to_string(nd._vLayerDescriptor.size());
                            break;

                        case Layer::Kind::Target:
                            // If the layer kind is Target, set the name as "Target" followed by the current layer count
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
                                    std::cout << ("LoadNeuralNetworkJSON: Duplicate layer name detected: %s\n", ldl._name.c_str());
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
                                        std::cout << ("LoadNeuralNetworkJSON: >4 dimensions detected in layer: %s\n", ldl._name.c_str());
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
                                        std::cout << ("LoadNeuralNetworkJSON: Illegal attempt to use auto for hidden layer: %s\n", ldl._name.c_str());
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
                                            std::cout << ("LoadNeuralNetworkJSON: Pooling layer %s has multiple sources\n", ldl._name.c_str());
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

                                // Check if the attribute name is "kernel" or "kernelstride"
                                else if ((lname.compare("kernel") == 0) || (lname.compare("kernelstride") == 0))
                                {
                                    // Initialize default values for dimensions
                                    uint32_t x = 1;
                                    uint32_t y = 1;
                                    uint32_t z = 1;
                                    uint32_t dimensions = 1;

                                    // Check if the attribute value is an array
                                    if (lvalue.isArray())
                                    {
                                        // Check if the array size is less than 4
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
                                            // Handle the case where the array size is too large, marking loading as invalid
                                            bValid = false;
                                            goto exit;
                                        }
                                    }
                                    else
                                    {
                                        // If the value is not an array, assume it as the 'x' dimension
                                        x = lvalue.asInt();
                                    }

                                    // Check if the attribute name is "kernel"
                                    if (lname.compare("kernel") == 0)
                                    {
                                        // Set kernel dimensions and values accordingly
                                        ldl._kernelX = x;
                                        ldl._kernelY = y;
                                        ldl._kernelZ = z;
                                        ldl._kernelDimensions = dimensions;
                                    }
                                    else
                                    {
                                        // If the attribute name is "kernelstride," set kernel stride dimensions and values
                                        ldl._kernelStrideX = x;
                                        ldl._kernelStrideY = y;
                                        ldl._kernelStrideZ = z;
                                    }

                                    // Continue to the next iteration of the loop
                                    continue;
                                }
                            }

                            // Check if the layer kind is Hidden
                            if (ldl._kind == Layer::Kind::Hidden)
                            {
                                // Check if the layer name is "batchnormalization"
                                if (lname.compare("batchnormalization") == 0)
                                {
                                    // If it is, check the value (lvalue) as a boolean and set the BatchNormal attribute accordingly
                                    if (lvalue.asBool())
                                        ldl._attributes |= Layer::Attributes::BatchNormal;

                                    // Continue to the next iteration of the loop
                                    continue;
                                }
                                // If the layer name is not "batchnormalization," check if it is "sparsenesspenalty"
                                else if (lname.compare("sparsenesspenalty") == 0)
                                {
                                    // Iterate through the parameters (pitr) for sparseness penalty
                                    for (Json::ValueIterator pitr = lvalue.begin(); pitr != lvalue.end(); pitr++)
                                    {
                                        // Get the parameter name (pname) and convert it to lowercase for case-insensitive comparison
                                        std::string pname = pitr.name();
                                        std::transform(pname.begin(), pname.end(), pname.begin(), ::tolower);

                                        // Get the parameter key (pkey) and value (pvalue)
                                        Json::Value pkey = pitr.key();
                                        Json::Value pvalue = *pitr;

                                        // Check the parameter name and assign values accordingly
                                        if (pname.compare("p") == 0)
                                            ldl._sparsenessPenalty_p = pvalue.asFloat();
                                        else if (pname.compare("beta") == 0)
                                            ldl._sparsenessPenalty_beta = pvalue.asFloat();
                                        else
                                        {
                                            // Handle the case of an invalid sparseness penalty parameter
                                            std::cout << ("LoadNeuralNetworkJSON: Invalid sparseness penalty parameter for hidden layer %s\n", ldl._name.c_str());
                                            bValid = false;
                                            goto exit; // Exit the loop and indicate that the JSON loading is not valid
                                        }
                                    }
                                    // Continue to the next iteration of the loop
                                    continue;
                                }
                            }

                            // Check if the layer kind is Output
                            if (ldl._kind == Layer::Kind::Output)
                            {
                                // Check if the layer name is "softmax"
								if (lname.compare("softmax") == 0)
								{
									// If it is, check the value (lvalue) as a boolean and set the SoftMax attribute accordingly
									if (lvalue.asBool())
										ldl._attributes |= Activation::SoftMax;

									// Continue to the next iteration of the loop
									continue;
								}

                            }

                            if ((ldl._kind == Layer::Kind::Hidden) || (ldl._kind == Layer::Kind::Output))
                            {
                                if (ldl._type == Layer::Type::Pooling)
                                {
                                    if (lname.compare("function") == 0)
                                    {
                                        std::string s = lvalue.asString();
                                        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                        if (s.compare("max") == 0)
                                            ldl._poolingFunction = PoolingFunction::Max;
                                        else if (s.compare("maxout") == 0)
                                            ldl._poolingFunction = PoolingFunction::Maxout;
                                        else if (s.compare("dotproduct") == 0)
                                            ldl._poolingFunction = PoolingFunction::DotProduct;
                                        else if (s.compare("cosine") == 0)
                                            ldl._poolingFunction = PoolingFunction::Cosine;
                                        else if (s.compare("average") == 0)
                                            ldl._poolingFunction = PoolingFunction::Average;
                                        else if ((s.compare("lrn") == 0) || (s.compare("localresponsenormalization") == 0))
                                            ldl._poolingFunction = PoolingFunction::LRN;
                                        else
                                        {
                                            std::cout << ("LoadNeuralNetworkJSON: Invalid pooling function (%s) for pooling layer %s\n", lvalue.asString().c_str(), ldl._name.c_str());
                                            bValid = false;
                                            goto exit;
                                        }
                                        continue;
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

                                else if (lname.compare("activation") == 0)
                                {
                                    std::string s = lvalue.asString();
                                    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                                    if (s.compare("sigmoid") == 0)
                                        ldl._activation = Activation::Sigmoid;
                                    else if (s.compare("tanh") == 0)
                                        ldl._activation = Activation::Tanh;
                                    else if (s.compare("linear") == 0)
                                        ldl._activation = Activation::Linear;
                                    else if ((s.compare("relu") == 0) || (s.compare("rectifiedlinear") == 0))
                                        ldl._activation = Activation::RectifiedLinear;
                                    else if ((s.compare("lrelu") == 0) || (s.compare("leakyrectifiedlinear") == 0))
                                        ldl._activation = Activation::LeakyRectifiedLinear;
                                    else if ((s.compare("elu") == 0) || (s.compare("exponentiallinear") == 0))
                                        ldl._activation = Activation::ExponentialLinear;
                                    else if ((s.compare("selu") == 0) || (s.compare("scaledexponentiallinear") == 0))
                                        ldl._activation = Activation::ScaledExponentialLinear;
                                    else if (s.compare("softplus") == 0)
                                        ldl._activation = Activation::SoftPlus;
                                    else if (s.compare("softsign") == 0)
                                        ldl._activation = Activation::SoftSign;
                                    else if (s.compare("softmax") == 0)
                                        ldl._activation = Activation::SoftMax;
                                    else if (s.compare("relumax") == 0)
                                        ldl._activation = Activation::RELUMax;
                                    else if (s.compare("linearmax") == 0)
                                        ldl._activation = Activation::LinearMax;
                                    else
                                    {
                                        std::cout << ("LoadNeuralNetworkJSON: Invalid layer activation: %s\n", lvalue.asString().c_str());
                                        bValid = false;
                                        goto exit;
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

                                            if (wname.compare("scheme") == 0)
                                            {
                                                std::string scheme = wvalue.asString();
                                                std::transform(scheme.begin(), scheme.end(), scheme.begin(), ::tolower);
                                                if (scheme.compare("xavier") == 0)
                                                    ldl._weightInit = Xavier;
                                                else if (scheme.compare("caffexavier") == 0)
                                                    ldl._weightInit = CaffeXavier;
                                                else if (scheme.compare("gaussian") == 0)
                                                    ldl._weightInit = Gaussian;
                                                else if (scheme.compare("uniform") == 0)
                                                    ldl._weightInit = Uniform;
                                                else if (scheme.compare("unitball") == 0)
                                                    ldl._weightInit = UnitBall;
                                                else if (scheme.compare("constant") == 0)
                                                    ldl._weightInit = Constant;
                                                else if (scheme.compare("selu") == 0)
                                                    ldl._weightInit = SELU;
                                                else
                                                {
                                                    printf("LoadNeuralNetworkJSON: Invalid weight initialization scheme: %s\n", scheme.c_str());
                                                    bValid = false;
                                                    goto exit;
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
                                                std::cout << ("LoadNeuralNetworkJSON: Invalid weight initialization field: %s\n", wname.c_str());
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
                                                std::cout << ("LoadNeuralNetworkJSON: Invalid shared weight field: %s\n", sname.c_str());
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

                            std::cout << ("LoadNeuralNetworkJSON: Unknown neural network layer field: %s\n", lname.c_str());
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
                                std::cout << ("LoadNeuralNetworkJSON: Unable to find data set %s to determine dimensions for layer: %s\n", ldl._dataSet.c_str(), ldl._name.c_str());
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
                                std::cout << ("LoadNeuralNetworkJSON: Dot product layer %s must have 2 or more sources\n", ldl._name.c_str());
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
                    std::cout << ("LoadNeuralNetworkJSON: Unknown neural network field: %s\n", name.c_str());
                    bValid = false;
                    goto exit;
                }
            }
        }

        // Check if the sparseness penalty beta is greater than 0.0
        if (nd._sparsenessPenalty_beta > (float)0.0)
        {
            // If beta is greater than 0.0, enable the sparseness penalty flag
            nd._bSparsenessPenalty = true;
        }

        // Check if the denoising probability (p) is greater than 0.0
        if (nd._denoising_p > (float)0.0)
        {
            // If p is greater than 0.0, enable the denoising flag
            nd._bDenoising = true;

            // Iterate through the layer descriptors in the neural network descriptor
            for (size_t i = 0; i < nd._vLayerDescriptor.size(); i++)
            {
                // Check if the layer kind is Input and it has the Sparse attribute set
                if ((nd._vLayerDescriptor[i]._kind == Layer::Kind::Input) && ((nd._vLayerDescriptor[i]._attributes & Layer::Attributes::Sparse) != 0))
                {
                    // Set the Denoising attribute for Input layers with the Sparse attribute
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

/// <summary>
/// Loads a neural network from a NetCDF file.
/// </summary>
/// <param name="fname">The filename of the NetCDF file.</param>
/// <param name="batch">The batch size for the network.</param>
/// <returns>A pointer to the loaded neural network.</returns>
Network* LoadNeuralNetworkNetCDF(const std::string& fname, const uint32_t batch) {
    Network* pNetwork = nullptr;
    NetworkDescriptor nd;

    bool bResult = true;
    float version = 0.0;
    uint32_t layers = 0;
    uint32_t weights = 0;

    // Broadcast the network name via MPI communication.
    MPI_Bcast_string(nd._name);

    // Set the convolutional layers calculation flag to true.
    nd._bConvLayersCalculated = true;

    int MPI_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

    if (MPI_rank == 0) {
        try {
            netCDF::NcFile nc(fname, netCDF::NcFile::read);

            // Helper function to read attribute values or handle missing attributes.
            auto readAttribute = [&](const std::string& attrName, auto& target) {
                netCDF::NcGroupAtt attr = nc.getAtt(attrName);
                if (attr.isNull()) {
                    std::cerr << "NcException: NNetwork::NNetwork: No " << attrName << " supplied in NetCDF input file " << fname << std::endl;
                    throw std::runtime_error("Missing attribute: " + attrName);
                }
                attr.getValues(&target);
                };

            // Read various attributes from the NetCDF file.
            readAttribute("version", version);
            readAttribute("name", nd._name);
            readAttribute("kind", nd._kind);
            readAttribute("errorFunction", nd._errorFunction);
            readAttribute("decay", nd._decay);
            readAttribute("maxout_k", nd._maxout_k);
            readAttribute("LRN_k", nd._LRN_k);
            readAttribute("LRN_n", nd._LRN_n);
            readAttribute("LRN_alpha", nd._LRN_alpha);
            readAttribute("LRN_beta", nd._LRN_beta);
            readAttribute("bSparsenessPenalty", nd._bSparsenessPenalty);
            readAttribute("sparsenessPenalty_p", nd._sparsenessPenalty_p);
            readAttribute("sparsenessPenalty_beta", nd._sparsenessPenalty_beta);
            readAttribute("bDenoising", nd._bDenoising);
            readAttribute("denoising_p", nd._denoising_p);
            readAttribute("deltaBoost_one", nd._deltaBoost_one);
            readAttribute("deltaBoost_zero", nd._deltaBoost_zero);
            readAttribute("SMCE_oneScale", nd._SMCE_oneScale);
            readAttribute("SMCE_zeroScale", nd._SMCE_zeroScale);
            readAttribute("SMCE_oneTarget", nd._SMCE_oneTarget);
            readAttribute("SMCE_zeroTarget", nd._SMCE_zeroTarget);
            readAttribute("checkpoint_name", nd._checkpoint_name);
            readAttribute("checkpoint_interval", nd._checkpoint_interval);
            readAttribute("checkpoint_epochs", nd._checkpoint_epochs);
            readAttribute("ShuffleIndices", nd._bShuffleIndices);
            readAttribute("layers", layers);

            // Load layer descriptors from the NetCDF file.
            for (uint32_t i = 0; i < layers; i++) {
                LayerDescriptor ld;
                if (!LoadLayerDescriptorNetCDF(fname, nc, i, ld)) {
                    std::cerr << "NcException: NNetwork::NNetwork: Error reading layer data in NetCDF input file " << fname << std::endl;
                }
                nd._vLayerDescriptor.push_back(ld);
            }

            readAttribute("weights", weights);

            // Load weight descriptors from the NetCDF file.
            for (uint32_t i = 0; i < weights; i++) {
                WeightDescriptor wd;
                if (!LoadWeightDescriptorNetCDF(fname, nc, i, wd)) {
                    std::cerr << "NcException: NNetwork::NNetwork: Error reading weight data in NetCDF input file " << fname << std::endl;
                }
                nd._vWeightDescriptor.push_back(wd);
            }
        }
        catch (netCDF::exceptions::NcException& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            bResult = false;
        }
    }

    // Broadcast the result of loading the network.
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Shutdown GPU and exit if loading failed.
    if (!bResult) {
        getGpu().Shutdown();
        exit(-1);
    }

    // Broadcast the network descriptor via MPI communication.
    MPI_Bcast_NetworkDescriptor(nd);

    if (MPI_rank == 0) {
        std::cout << "LoadNeuralNetworkJSON: Enumerating network:" << std::endl;
        std::cout << nd << std::endl;
    }

    // Create a new network instance and refresh its state.
    pNetwork = new Network(nd, batch);
    pNetwork->RefreshState();
    return pNetwork;
}

bool Network::P2P_Bcast(void* pBuffer, size_t size)
{
    cudaError_t status;

    // Check if there are multiple GPUs in use.
    if (getGpu()._numprocs > 1)
    {
        // Check if P2P (Peer-to-Peer) communication is available.
        if (getGpu()._bP2P)
        {
            // Case for two GPUs.
            if (getGpu()._numprocs == 2)
            {
                // Only GPU 0 copies data to P2P backbuffer.
                if (getGpu()._id == 0)
                {
                    status = cudaMemcpy(GetPeerBackBuffer(), pBuffer, size, cudaMemcpyDefault);
                    if (status != cudaSuccess)
                    {
                        throw std::runtime_error("Network::P2P_Bcast: Failure to copy source data to P2P backbuffer");
                    }
                }

                // Synchronize CUDA devices and MPI barrier to ensure proper timing.
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
            }
            else
            {
                // For more than two GPUs.
                if (getGpu()._id == 0)
                {
                    status = cudaMemcpy(GetP2PSendBuffer(), pBuffer, size, cudaMemcpyDefault);
                    if (status != cudaSuccess)
                    {
                        throw std::runtime_error("Network::P2P_Bcast: Failure to copy source data to P2P backbuffer");
                    }
                }

                // Calculate the number of stages and distance for data exchange.
                uint32_t stages = 2 * getGpu()._numprocs - 2;
                uint32_t distance = (getGpu()._numprocs - getGpu()._id) % getGpu()._numprocs;
                uint64_t segment = 0;

                // Loop through stages for data exchange.
                for (uint32_t i = 0; i < stages; i++)
                {
                    // Check conditions for copying data to P2P backbuffer.
                    if ((getGpu()._id != 1) && (i >= distance) && (segment < getGpu()._numprocs))
                    {
                        size_t start = (size * segment) / getGpu()._numprocs;
                        size_t end = (size * (segment + 1)) / getGpu()._numprocs;

                        // Copy data from P2P sendbuffer to P2P backbuffer.
                        status = cudaMemcpy((char*)GetPeerBackBuffer() + start, (char*)GetP2PSendBuffer() + start, end - start, cudaMemcpyDefault);
                        if (status != cudaSuccess)
                        {
                            throw std::runtime_error("Network::P2P_Bcast: Failure to copy source data to P2P backbuffer");
                        }
                        segment++;
                    }

                    // Synchronize CUDA devices and MPI barrier.
                    cudaDeviceSynchronize();
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            }

            // GPUs other than 0 copy data from P2P sendbuffer.
            if (getGpu()._id > 0)
            {
                status = cudaMemcpy(pBuffer, GetP2PSendBuffer(), size, cudaMemcpyDefault);
                if (status != cudaSuccess)
                {
                    throw std::runtime_error("Network::P2P_Bcast: Failure to copy source data from P2P sendbuffer");
                }
            }
        }
        else
        {
            // P2P communication not available, use CPU buffer and MPI_Bcast for broadcasting.
            status = cudaMemcpy(_pCPUBuffer.get(), pBuffer, size, cudaMemcpyDefault);
            if (status != cudaSuccess)
            {
                throw std::runtime_error("Network::P2P_Bcast: Failure to copy source data to CPU buffer");
            }

            // Use MPI_Bcast to broadcast data from CPU buffer.
            MPI_Bcast(_pCPUBuffer.get(), size, MPI_BYTE, 0, MPI_COMM_WORLD);

            // Copy data from CPU buffer back to device.
            status = cudaMemcpy(pBuffer, _pCPUBuffer.get(), size, cudaMemcpyDefault);
            if (status != cudaSuccess)
            {
                throw std::runtime_error("Network::P2P_Bcast: Failure to copy data from CPU buffer to device");
            }
        }
    }

    // Return true to indicate success.
    return true;
}

bool Network::P2P_Allreduce(float* pBuffer, size_t size)
{
    // Define constants for float size, MPI data types, and MPI operations.
    constexpr int FLOAT_SIZE = sizeof(float);
    constexpr int MPI_FLOAT_TYPE = MPI_FLOAT;
    constexpr int MPI_SUM_OP = MPI_SUM;

    // Get information about the current GPU and its communication capabilities.
    int numprocs = getGpu()._numprocs;
    int gpuId = getGpu()._id;
    bool isP2P = getGpu()._bP2P;

    // If there is only one process, no communication is needed, so return true.
    if (numprocs <= 1)
    {
        return true;
    }

    // Check if P2P (peer-to-peer) communication is available.
    if (isP2P)
    {
        // Handle the case where there are only two processes.
        if (numprocs == 2)
        {
            // Copy data from pBuffer to the peer buffer.
            cudaMemcpy(GetPeerBuffer(), pBuffer, size * FLOAT_SIZE, cudaMemcpyDefault);
            
            // Synchronize GPU operations.
            cudaDeviceSynchronize();
            
            // Synchronize MPI processes.
            MPI_Barrier(MPI_COMM_WORLD);
            kAddBuffers(pBuffer, GetP2PReceiveBuffer(), size);
        }
        else
        {
            // For multiple processes, perform an all-reduce operation in stages.
            int stages = numprocs - 1;
            int start = (size * gpuId) / numprocs;
            int end = (size * (static_cast<unsigned long long>(gpuId) + 1)) / numprocs;
            int nextGpuId = (gpuId + 1) % numprocs;

            // Iterate through the stages of communication.
            for (int i = 0; i < stages; i++)
            {
                // Copy data from pBuffer (or the send buffer in subsequent iterations) to the peer buffer.
                cudaMemcpy(GetPeerBuffer(), (i == 0) ? (pBuffer + start) : GetP2PSendBuffer(),
                    (static_cast<size_t>(end) - start) * FLOAT_SIZE, cudaMemcpyDefault);
                
                // Synchronize GPU operations.
                cudaDeviceSynchronize();
                
                // Synchronize MPI processes.
                MPI_Barrier(MPI_COMM_WORLD);
                
                // Swap peer buffers for the next stage.
                SwapPeerBuffers();
                start = (size * nextGpuId) / numprocs;
                end = (size * (static_cast<unsigned long long>(nextGpuId) + 1)) / numprocs;
                kAddBuffers(GetP2PSendBuffer(), pBuffer + start, static_cast<uint64_t>(end) - start);
                nextGpuId = (nextGpuId + 1) % numprocs;
            }

            // Copy data from the send buffer back to pBuffer.
            cudaMemcpy(pBuffer + start, GetP2PSendBuffer(), (static_cast<size_t>(end) - start) * FLOAT_SIZE, cudaMemcpyDefault);

            // Perform additional stages to ensure complete communication.
            for (int i = 0; i < stages; i++)
            {
                // Copy data from the peer buffer to the send buffer.
                cudaMemcpy(GetPeerBuffer(), GetP2PSendBuffer(), (static_cast<size_t>(end) - start) * FLOAT_SIZE, cudaMemcpyDefault);
                
                // Synchronize GPU operations.
                cudaDeviceSynchronize();
                
                // Synchronize MPI processes.
                MPI_Barrier(MPI_COMM_WORLD);
                
                // Swap peer buffers for the next stage.
                SwapPeerBuffers();
                
                start = (size * nextGpuId) / numprocs;
                end = (size * (static_cast<unsigned long long>(nextGpuId) + 1)) / numprocs;

                // Copy data from the send buffer back to pBuffer.
                cudaMemcpy(pBuffer + start, GetP2PSendBuffer(), (static_cast<size_t>(end) - start) * FLOAT_SIZE, cudaMemcpyDefault);
                nextGpuId = (nextGpuId + 1) % numprocs;
            }
        }
    }
    else
    {
        // If P2P communication is not available, use non-P2P communication.

        // Copy data from pBuffer to the CPU buffer.
        cudaMemcpy(_pCPUBuffer.get(), pBuffer, size * FLOAT_SIZE, cudaMemcpyDefault);

        // Perform an all-reduce operation using MPI on the CPU buffer.
        MPI_Allreduce(MPI_IN_PLACE, _pCPUBuffer.get(), size, MPI_FLOAT_TYPE, MPI_SUM_OP, MPI_COMM_WORLD);

        // Copy the result from the CPU buffer back to pBuffer.
        cudaMemcpy(pBuffer, _pCPUBuffer.get(), size * FLOAT_SIZE, cudaMemcpyDefault);
    }

    // Return true to indicate successful communication.
    return true;
}
