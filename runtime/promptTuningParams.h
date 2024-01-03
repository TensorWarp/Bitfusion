
#pragma once

#include "../common/cudaUtils.h"
#include "bufferManager.h"
#include "common.h"
#include "iTensor.h"
#include "tllmBuffers.h"

#include <optional>
#include <utility>

namespace bitfusion::runtime
{

template <typename TTensor>
class GenericPromptTuningParams
{
public:
    using TensorPtr = TTensor;
    using SizeType = bitfusion::runtime::SizeType;

    explicit GenericPromptTuningParams(
        TensorPtr embeddingTable = TensorPtr(), TensorPtr tasks = TensorPtr(), TensorPtr vocabSize = TensorPtr())
        : embeddingTable{std::move(embeddingTable)}
        , tasks{std::move(tasks)}
        , vocabSize{std::move(vocabSize)} {};

    TensorPtr embeddingTable;
    TensorPtr tasks;
    TensorPtr vocabSize;

    std::vector<bool>
        promptTuningEnabled;
};

class PromptTuningParams : public GenericPromptTuningParams<ITensor::SharedPtr>
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using SizeType = GenericPromptTuningParams::SizeType;

    explicit PromptTuningParams(
        TensorPtr embeddingTable = nullptr, TensorPtr tasks = nullptr, TensorPtr vocabSize = nullptr)
        : GenericPromptTuningParams(std::move(embeddingTable), std::move(tasks), std::move(vocabSize))
    {
    }

    void fillTasksTensor(TensorPtr tasksHost, const SizeType batchSize, const SizeType numContextRequests,
        const std::vector<SizeType>& reqBeamWidths, const std::vector<SizeType>& reqPromptLengths,
        BufferManager const& manager, bool packedInput);
};

}
