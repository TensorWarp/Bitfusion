#pragma once

#include "../common/cudaUtils.h"
#include "bufferManager.h"
#include "common.h"
#include "iTensor.h"

#include <utility>

namespace bitfusion::runtime
{
    /// <summary>
    /// Class representing the output of a decoding operation.
    /// </summary>
    class DecodingOutput
    {
    public:
        using TensorPtr = ITensor::SharedPtr;

        /// <summary>
        /// Class representing beam hypotheses during decoding.
        /// </summary>
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

            /// <summary>
            /// Clear and release resources associated with the beam hypotheses.
            /// </summary>
            /// <param name="manager">Buffer manager for resource management.</param>
            void empty(BufferManager& manager);

            /// <summary>
            /// Reshape the beam hypotheses tensors.
            /// </summary>
            /// <param name="batchSize">Size of the batch.</param>
            /// <param name="beamWidth">Beam width.</param>
            /// <param name="maxSequenceLength">Maximum sequence length.</param>
            void reshape(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength);

            /// <summary>
            /// Release resources associated with the beam hypotheses.
            /// </summary>
            void release();

            /// <summary>
            /// Initialize the beam hypotheses.
            /// </summary>
            /// <param name="manager">Buffer manager for resource management.</param>
            /// <param name="endId">End identifier for decoding.</param>
            void init(BufferManager& manager, TokenIdType endId);

            /// <summary>
            /// Create a slice of the beam hypotheses.
            /// </summary>
            /// <param name="batchIndex">Batch index for the slice.</param>
            /// <param name="size">Size of the slice.</param>
            /// <returns>A new BeamHypotheses object representing the slice.</returns>
            BeamHypotheses slice(SizeType batchIndex, SizeType size) const;
        };

        static float constexpr kNegativeInfinity = -1e20f;

        /// <summary>
        /// Constructor for DecodingOutput.
        /// </summary>
        /// <param name="ids">Pointer to the output tensor for IDs.</param>
        explicit DecodingOutput(TensorPtr ids)
            : ids{ std::move(ids) }
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
