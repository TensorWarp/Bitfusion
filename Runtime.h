#pragma once

#include "GpuTypes.h"
#include "Types.h"
#include "Layer.h"

using namespace std;

/// <summary>
/// The Runtime class manages the execution of a neural network on the GPU.
/// </summary>
class Runtime {
private:
    static const int ALL = -1;

    const std::string networkFilename;
    const uint32_t batchSize;
    const uint32_t maxK;

    std::map<string, GpuBuffer<float>*> dOutputScores;
    std::map<string, GpuBuffer<uint32_t>*> dOutputIndexes;

public:
    /// <summary>
    /// Constructs a Runtime object.
    /// </summary>
    /// <param name="networkFilename">The filename of the network configuration.</param>
    /// <param name="batchSize">The batch size for network execution.</param>
    /// <param name="maxK">Optional. The maximum value of K for recommendations.</param>
    Runtime(const std::string& networkFilename, uint32_t batchSize, int maxK = ALL);

    /// <summary>
    /// Destructor for the Runtime class.
    /// </summary>
    ~Runtime();

    /// <summary>
    /// Gets a pointer to the Network object associated with this Runtime.
    /// </summary>
    /// <returns>A pointer to the Network object.</returns>
    Network* getNetwork() const;

    /// <summary>
    /// Initializes input layer data sets based on the provided dataset descriptors.
    /// </summary>
    /// <param name="datasetDescriptors">A vector of DataSetDescriptor objects describing the input datasets.</param>
    void initInputLayerDataSets(const std::vector<DataSetDescriptor>& datasetDescriptors);

    /// <summary>
    /// Gets the output scores buffer for a specific layer.
    /// </summary>
    /// <param name="layerName">The name of the layer for which to retrieve the output scores buffer.</param>
    /// <returns>A pointer to the output scores buffer.</returns>
    GpuBuffer<float>* getOutputScoresBuffer(const std::string& layerName);

    /// <summary>
    /// Gets the output indexes buffer for a specific layer.
    /// </summary>
    /// <param name="layerName">The name of the layer for which to retrieve the output indexes buffer.</param>
    /// <returns>A pointer to the output indexes buffer.</returns>
    GpuBuffer<uint32_t>* getOutputIndexesBuffer(const std::string& layerName);

    /// <summary>
    /// Converts a long pointer to a Runtime object pointer.
    /// </summary>
    /// <param name="ptr">The long pointer to convert.</param>
    /// <returns>A pointer to the Runtime object.</returns>
    static Runtime* fromPtr(long ptr) {
        Runtime* dc = (Runtime*)ptr;
        if (dc == nullptr) {
            throw std::runtime_error("Cannot convert nullptr to Runtime");
        }
        return dc;
    }
};
