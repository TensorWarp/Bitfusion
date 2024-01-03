
#pragma once

#include "common.h"

#include <optional>
#include <vector>

namespace bitfusion::runtime
{

class SamplingConfig
{
    using FloatType = float;

    template <typename T>
    using OptVec = std::optional<std::vector<T>>;

public:
    explicit SamplingConfig(SizeType beamWidth = 1)
        : beamWidth{beamWidth}
    {
    }

    SizeType beamWidth;

    OptVec<FloatType> temperature;
    OptVec<SizeType> minLength;
    OptVec<FloatType> repetitionPenalty;
    OptVec<FloatType> presencePenalty;

    OptVec<SizeType> topK;
    OptVec<FloatType> topP;
    OptVec<unsigned long long> randomSeed;
    OptVec<FloatType> topPDecay;
    OptVec<FloatType> topPMin;
    OptVec<SizeType> topPResetIds;

    OptVec<FloatType> beamSearchDiversityRate;
    OptVec<FloatType> lengthPenalty;

    OptVec<FloatType> draftAcceptanceThreshold;
};

}
