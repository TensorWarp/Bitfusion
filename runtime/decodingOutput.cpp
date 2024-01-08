#include "decodingOutput.h"
#include "runtimeKernels.h"

namespace bitfusion::runtime {

    /// <summary>
    /// Empty the tensor buffers for beam hypotheses.
    /// </summary>
    /// <param name="manager">Buffer manager for memory allocation.</param>
    void DecodingOutput::BeamHypotheses::empty(BufferManager& manager) {
        constexpr auto nvTokenIdType = DataType<TokenIdType>::value;
        constexpr auto nvSizeType = DataType<SizeType>::value;
        constexpr auto nvFloatType = DataType<float>::value;
        constexpr auto nvBoolType = DataType<bool>::value;

        outputIdsTgt = manager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
        sequenceLengthsTgt = manager.emptyTensor(MemoryType::kGPU, nvSizeType);
        cumLogProbs = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
        normedScores = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
        logProbs = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
        minNormedScores = manager.emptyTensor(MemoryType::kGPU, nvFloatType);
        numBeams = manager.emptyTensor(MemoryType::kGPU, nvSizeType);
        isDone = manager.emptyTensor(MemoryType::kGPU, nvBoolType);
    }

    /// <summary>
    /// Reshape the tensor buffers for beam hypotheses.
    /// </summary>
    /// <param name="batchSize">Batch size.</param>
    /// <param name="beamWidth">Beam width.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    void DecodingOutput::BeamHypotheses::reshape(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength) {
        outputIdsTgt->reshape(ITensor::makeShape({ batchSize, 2 * beamWidth, maxSequenceLength }));
        sequenceLengthsTgt->reshape(ITensor::makeShape({ batchSize, 2 * beamWidth }));
        cumLogProbs->reshape(ITensor::makeShape({ batchSize, 2 * beamWidth }));
        normedScores->reshape(ITensor::makeShape({ batchSize, 2 * beamWidth }));
        logProbs->reshape(ITensor::makeShape({ batchSize, 2 * beamWidth, maxSequenceLength }));
        minNormedScores->reshape(ITensor::makeShape({ batchSize }));
        numBeams->reshape(ITensor::makeShape({ batchSize }));
        isDone->reshape(ITensor::makeShape({ batchSize }));
    }

    /// <summary>
    /// Initialize beam hypotheses with a specified endId.
    /// </summary>
    /// <param name="manager">Buffer manager for memory allocation.</param>
    /// <param name="endId">TokenIdType representing the end of a sequence.</param>
    void DecodingOutput::BeamHypotheses::init(BufferManager& manager, TokenIdType endId)
    {
        kernels::invokeFill(*outputIdsTgt, endId, manager.getStream());
    }

    /// <summary>
    /// Create a slice of beam hypotheses for a specified batch index and size.
    /// </summary>
    /// <param name="batchIndex">Index of the batch.</param>
    /// <param name="size">Size of the slice.</param>
    /// <returns>A new BeamHypotheses instance representing the slice.</returns>
    DecodingOutput::BeamHypotheses DecodingOutput::BeamHypotheses::slice(SizeType batchIndex, SizeType size) const {
        DecodingOutput::BeamHypotheses bh{};
        bh.outputIdsTgt = ITensor::slice(outputIdsTgt, batchIndex, size);
        bh.sequenceLengthsTgt = ITensor::slice(sequenceLengthsTgt, batchIndex, size);
        bh.cumLogProbs = ITensor::slice(cumLogProbs, batchIndex, size);
        bh.normedScores = ITensor::slice(normedScores, batchIndex, size);
        bh.logProbs = ITensor::slice(logProbs, batchIndex, size);
        bh.minNormedScores = ITensor::slice(minNormedScores, batchIndex, size);
        bh.numBeams = ITensor::slice(numBeams, batchIndex, size);
        bh.isDone = ITensor::slice(isDone, batchIndex, size);
        return bh;
    }

    /// <summary>
    /// Release the tensor buffers for beam hypotheses.
    /// </summary>
    void DecodingOutput::BeamHypotheses::release() {
        outputIdsTgt->release();
        sequenceLengthsTgt->release();
        cumLogProbs->release();
        normedScores->release();
        logProbs->release();
        minNormedScores->release();
        numBeams->release();
        isDone->release();
    }
}
