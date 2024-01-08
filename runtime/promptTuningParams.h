#pragma once

#include "../common/cudaUtils.h"
#include "bufferManager.h"
#include "common.h"
#include "iTensor.h"
#include "buffers.h"

#include <optional>
#include <utility>

namespace bitfusion::runtime
{

    /// <summary>
    /// Template class for generic prompt tuning parameters.
    /// </summary>
    template <typename TTensor>
    class GenericPromptTuningParams
    {
    public:
        using TensorPtr = TTensor;
        using SizeType = bitfusion::runtime::SizeType;

        /// <summary>
        /// Constructor for GenericPromptTuningParams.
        /// </summary>
        /// <param name="embeddingTable">Pointer to the embedding table tensor.</param>
        /// <param name="tasks">Pointer to the tasks tensor.</param>
        /// <param name="vocabSize">Pointer to the vocabSize tensor.</param>
        explicit GenericPromptTuningParams(
            TensorPtr embeddingTable = TensorPtr(), TensorPtr tasks = TensorPtr(), TensorPtr vocabSize = TensorPtr())
            : embeddingTable{ std::move(embeddingTable) }
            , tasks{ std::move(tasks) }
        , vocabSize{ std::move(vocabSize) } {};

        TensorPtr embeddingTable;
        TensorPtr tasks;
        TensorPtr vocabSize;

        std::vector<bool> promptTuningEnabled;
    };

    /// <summary>
    /// Class for specific prompt tuning parameters.
    /// </summary>
    class PromptTuningParams : public GenericPromptTuningParams<ITensor::SharedPtr>
    {
    public:
        using TensorPtr = ITensor::SharedPtr;
        using SizeType = GenericPromptTuningParams::SizeType;

        /// <summary>
        /// Constructor for PromptTuningParams.
        /// </summary>
        /// <param name="embeddingTable">Pointer to the embedding table tensor.</param>
        /// <param name="tasks">Pointer to the tasks tensor.</param>
        /// <param name="vocabSize">Pointer to the vocabSize tensor.</param>
        explicit PromptTuningParams(
            TensorPtr embeddingTable = nullptr, TensorPtr tasks = nullptr, TensorPtr vocabSize = nullptr)
            : GenericPromptTuningParams(std::move(embeddingTable), std::move(tasks), std::move(vocabSize))
        {
        }

        /// <summary>
        /// Fill the tasks tensor with data.
        /// </summary>
        /// <param name="tasksHost">Pointer to the tasks host tensor.</param>
        /// <param name="batchSize">Size of the batch.</param>
        /// <param name="numContextRequests">Number of context requests.</param>
        /// <param name="reqBeamWidths">Vector of requested beam widths.</param>
        /// <param name="reqPromptLengths">Vector of requested prompt lengths.</param>
        /// <param name="manager">Buffer manager object.</param>
        /// <param name="packedInput">Boolean indicating packed input.</param>
        void fillTasksTensor(TensorPtr tasksHost, const SizeType batchSize, const SizeType numContextRequests,
            const std::vector<SizeType>& reqBeamWidths, const std::vector<SizeType>& reqPromptLengths,
            BufferManager const& manager, bool packedInput);
    };
}