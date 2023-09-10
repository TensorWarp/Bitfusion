#pragma once

#include <vector>
#include <string>
#include <memory>

#include "../system/GpuTypes.h"
#include "../system/Types.h"

class FilterConfig;
class Network;

/// <summary>
/// Class for generating recommendations.
/// </summary>
class RecsGenerator
{
    std::unique_ptr<GpuBuffer<float>> pbKey;
    std::unique_ptr<GpuBuffer<unsigned int>> pbUIValue;
    std::unique_ptr<GpuBuffer<float>> pFilteredOutput;
    std::vector<GpuBuffer<float>*>* vNodeFilters;
    std::string recsGenLayerLabel;
    std::string scorePrecision;

public:
    /// <summary>
    /// Default layer label for recommendation generation.
    /// </summary>
    static const std::string DEFAULT_LAYER_RECS_GEN_LABEL;

    /// <summary>
    /// Top K scalar value for recommendation generation.
    /// </summary>
    static const unsigned int TOPK_SCALAR;

    /// <summary>
    /// Default score precision for recommendation generation.
    /// </summary>
    static const std::string DEFAULT_SCORE_PRECISION;

    /// <summary>
    /// Constructor for RecsGenerator.
    /// </summary>
    /// <param name="xBatchSize">Batch size.</param>
    /// <param name="xK">Value of K.</param>
    /// <param name="xOutputBufferSize">Output buffer size.</param>
    /// <param name="layer">Layer label (optional, defaults to DEFAULT_LAYER_RECS_GEN_LABEL).</param>
    /// <param name="precision">Score precision (optional, defaults to DEFAULT_SCORE_PRECISION).</param>
    RecsGenerator(unsigned int xBatchSize,
        unsigned int xK,
        unsigned int xOutputBufferSize,
        const std::string& layer = DEFAULT_LAYER_RECS_GEN_LABEL,
        const std::string& precision = DEFAULT_SCORE_PRECISION);

    /// <summary>
    /// Generate recommendations based on the provided parameters.
    /// </summary>
    /// <param name="network">Pointer to the Network object.</param>
    /// <param name="topK">Top K recommendations to generate.</param>
    /// <param name="filters">Pointer to the FilterConfig object.</param>
    /// <param name="customerIndex">Vector of customer indices.</param>
    /// <param name="featureIndex">Vector of feature indices.</param>
    void generateRecs(Network* network,
        unsigned int topK,
        const FilterConfig* filters,
        const std::vector<std::string>& customerIndex,
        const std::vector<std::string>& featureIndex);
};
