
#pragma once

#include "../common/cudaUtils.h"
#include "bufferManager.h"
#include "common.h"
#include "iTensor.h"

#include <utility>

namespace bitfusion::runtime
{
class DecodingOutput
{
public:
    using TensorPtr = ITensor::SharedPtr;

    class BeamHypotheses
    {
    public:
        TensorPtr outputIdsTgt;
        TensorPtr sequenceLengthsTgt;
        TensorPtr cumLogProbs;
        TensorPtr normedScores;
        TensorPtr logProbs;
        TensorPtr minNormedScores;
        TensorPtr numBeams;
        TensorPtr isDone;

        void empty(BufferManager& manager);

        void reshape(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength);

        void release();

        void init(BufferManager& manager, TokenIdType endId);

        BeamHypotheses slice(SizeType batchIndex, SizeType size) const;
    };

    static float constexpr kNegativeInfinity = -1e20f;

    explicit DecodingOutput(TensorPtr ids)
        : ids{std::move(ids)}
    {
        CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
    }

    TensorPtr ids;
    TensorPtr newTokensSteps;
    TensorPtr newTokens;
    std::vector<TensorPtr> newTokensVec;

    TensorPtr finished;
    TensorPtr finishedSum;

    TensorPtr logProbs;
    TensorPtr cumLogProbs;
    TensorPtr parentIds;
    TensorPtr lengths;
    TensorPtr cacheIndirection;

    BeamHypotheses beamHypotheses;
};

}
