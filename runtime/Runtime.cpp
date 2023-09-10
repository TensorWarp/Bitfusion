#include "Runtime.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <future>
#include <memory>
#include <format>
#include "../system/ThreadPool.h"

namespace
{
    constexpr int ARGC = 1;
    constexpr const char* ARGV = "bitfusion";
    constexpr unsigned long SEED = 12134ULL;
}

/// <summary>
/// Constructs a Runtime object.
/// </summary>
/// <param name="networkFilename">The filename of the network configuration.</param>
/// <param name="batchSize">The batch size for network execution.</param>
/// <param name="maxK">Optional. The maximum value of K for recommendations.</param>
Runtime::Runtime(const std::string& networkFilename, uint32_t batchSize, int maxK) :
    networkFilename(networkFilename),
    batchSize(batchSize),
    maxK(maxK)
{
    std::unique_ptr<Network> network(LoadNeuralNetworkNetCDF(networkFilename, batchSize));
    getGpu().Startup(ARGC, const_cast<char**>(&ARGV));
    getGpu().SetRandomSeed(SEED);
    getGpu().SetNeuralNetwork(network.get());

    std::vector<const Layer*> outputLayers;
    network->GetLayers(Layer::Kind::Output, outputLayers);

    for (const Layer* layer : outputLayers)
    {
        const std::string& layerName = layer->GetName();

        if (maxK != ALL)
        {
            if (layer->GetNumDimensions() > 1)
            {
                throw std::invalid_argument(std::format("Runtime::Runtime: Layer {} has more than one dimension, but maxK is not set to ALL", layerName));
            }
            size_t outputBufferLength = maxK * batchSize;
            printf("Runtime::Runtime: Allocating output score and index buffers, each of size %zu for output layer %s\n",
                outputBufferLength, layerName.c_str());

            std::unique_ptr<GpuBuffer<float>> outputScores(new GpuBuffer<float>(outputBufferLength, false, false));
            std::unique_ptr<GpuBuffer<uint32_t>> outputIndexes(new GpuBuffer<uint32_t>(outputBufferLength, false, false));

            dOutputScores[layerName] = outputScores.release();
            dOutputIndexes[layerName] = outputIndexes.release();
        }
    }
}

/// <summary>
/// Gets the output scores buffer for a specific layer.
/// </summary>
/// <param name="layerName">The name of the layer for which to retrieve the output scores buffer.</param>
/// <returns>A pointer to the output scores buffer.</returns>
GpuBuffer<float>* Runtime::getOutputScoresBuffer(const std::string& layerName)
{
    return dOutputScores.at(layerName);
}

/// <summary>
/// Gets the output indexes buffer for a specific layer.
/// </summary>
/// <param name="layerName">The name of the layer for which to retrieve the output indexes buffer.</param>
/// <returns>A pointer to the output indexes buffer.</returns>
GpuBuffer<uint32_t>* Runtime::getOutputIndexesBuffer(const std::string& layerName)
{
    return dOutputIndexes.at(layerName);
}

/// <summary>
/// Destructor for the Runtime class.
/// </summary>
Runtime::~Runtime()
{
    const std::string networkName = getNetwork()->GetName();

    for (const auto& [layerName, outputScores] : dOutputScores)
    {
        delete outputScores;
    }
    for (const auto& [layerName, outputIndexes] : dOutputIndexes)
    {
        delete outputIndexes;
    }

    dOutputScores.clear();
    dOutputIndexes.clear();

    std::unique_ptr<Network> networkPtr(getNetwork());

    getGpu().Shutdown();
    printf("Runtime::~Runtime: Destroyed context for network %s\n", networkName.c_str());
}

/// <summary>
/// Gets the associated Network object for this Runtime.
/// </summary>
/// <returns>A pointer to the Network object.</returns>
Network* Runtime::getNetwork() const
{
    return getGpu()._pNetwork;
}

/// <summary>
/// Initializes input layer data sets based on the provided dataset descriptors.
/// </summary>
/// <param name="datasetDescriptors">A vector of DataSetDescriptor objects describing the input datasets.</param>
void Runtime::initInputLayerDataSets(const std::vector<DataSetDescriptor>& datasetDescriptors)
{
    ThreadPool threadPool(4);

    std::vector<DataSetBase*> datasets(datasetDescriptors.size(), nullptr);

#pragma omp parallel for
    for (int i = 0; i < datasetDescriptors.size(); ++i)
    {
        const DataSetDescriptor& descriptor = datasetDescriptors[i];

        auto future = threadPool.enqueue([this, descriptor]() -> DataSetBase* {
            try {
                return createDataSet(descriptor);
            }
            catch (const std::exception& ex) {
                std::cerr << "Error creating dataset: " << ex.what() << std::endl;
                return nullptr;
            }
            });

        datasets[i] = future.get();
    }

    if (std::any_of(datasets.begin(), datasets.end(), [](DataSetBase* dataset) { return dataset == nullptr; })) {
        throw std::runtime_error("One or more datasets could not be created.");
    }

    Network* network = getNetwork();
    network->LoadDataSets(datasets);
    network->PredictBatch();
    network->SetPosition(0);
}