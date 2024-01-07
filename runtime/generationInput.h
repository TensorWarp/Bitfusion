
#pragma once

#include "../common/cudaUtils.h"
#include "common.h"
#include "iTensor.h"
#include "promptTuningParams.h"

#include <optional>
#include <utility>

namespace bitfusion::runtime
{

template <typename TTensor, typename PromptTuningParams>
class GenericGenerationInput
{
public:
    using TensorPtr = TTensor;

    explicit GenericGenerationInput(
        SizeType const endId, SizeType const padId, TensorPtr ids, TensorPtr lengths, bool packed = false)
        : endId{endId}
        , padId{padId}
        , ids{std::move(ids)}
        , lengths{std::move(lengths)}
        , packed{packed}
        , maxNewTokens(std::nullopt)
    {
        CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
        CHECK_WITH_INFO(static_cast<bool>(this->lengths), "Invalid lengths tensor");
    }

    SizeType endId;
    SizeType padId;
    TensorPtr ids;
    TensorPtr lengths;
    bool packed;

    TensorPtr embeddingBias;
    TensorPtr badWordsList;
    TensorPtr stopWordsList;
    std::optional<SizeType> maxNewTokens;

    PromptTuningParams promptTuningParams;
};

class GenerationInput : public GenericGenerationInput<ITensor::SharedPtr, PromptTuningParams>
{
public:
    using Base = GenericGenerationInput<ITensor::SharedPtr, PromptTuningParams>;
    using TensorPtr = Base::TensorPtr;

    explicit GenerationInput(
        SizeType const endId, SizeType const padId, TensorPtr ids, TensorPtr lengths, bool packed = false)
        : GenericGenerationInput(endId, padId, std::move(ids), std::move(lengths), packed)
    {
    }
};

}
