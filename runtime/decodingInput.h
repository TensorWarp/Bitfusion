
#pragma once

#include "../common/cudaUtils.h"
#include "common.h"
#include "iTensor.h"

#include <memory>

namespace bitfusion::runtime
{
class DecodingInput
{
public:
    using TensorPtr = std::shared_ptr<ITensor const>;

    DecodingInput(
        SizeType maxLength, SizeType maxAttentionWindow, SizeType batchSize, TensorPtr logits, TensorPtr endIds)
        : step{maxLength}
        , maxLength{maxLength}
        , maxAttentionWindow{maxAttentionWindow}
        , batchSize{batchSize}
        , logits{std::move(logits)}
        , endIds{std::move(endIds)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->endIds), "Invalid endIds tensor");
    }

    SizeType step;
    SizeType maxLength;
    SizeType maxAttentionWindow;
    SizeType batchSize;
    TensorPtr logits;
    TensorPtr endIds;

    TensorPtr finished;
    TensorPtr sequenceLimitLength;
    TensorPtr embeddingBias;
    TensorPtr lengths;
    TensorPtr badWordsList;
    TensorPtr stopWordsList;
    TensorPtr noRepeatNgramSize;

    TensorPtr cacheIndirection;
};

}
