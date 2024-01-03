
#pragma once

#include "../common/cudaUtils.h"
#include "common.h"
#include "iTensor.h"

#include <functional>
#include <utility>

namespace bitfusion::runtime
{

template <typename TTensor>
class GenericGenerationOutput
{
public:
    using TensorPtr = TTensor;
    using Callback = std::function<void(TensorPtr const& ids, SizeType step, bool finished)>;

    explicit GenericGenerationOutput(TensorPtr ids, TensorPtr lengths)
        : ids{std::move(ids)}
        , lengths{std::move(lengths)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->lengths), "Invalid lengths tensor");
    }

    TensorPtr ids;
    TensorPtr lengths;

    TensorPtr cumLogProbs;
    TensorPtr logProbs;
    TensorPtr contextLogits;
    TensorPtr generationLogits;

    Callback onTokenGenerated;
};

class GenerationOutput : public GenericGenerationOutput<ITensor::SharedPtr>
{
public:
    using Base = GenericGenerationOutput<ITensor::SharedPtr>;
    using TensorPtr = Base::TensorPtr;

    explicit GenerationOutput(TensorPtr ids, TensorPtr lengths)
        : GenericGenerationOutput(std::move(ids), std::move(lengths))
    {
    }
};

}
