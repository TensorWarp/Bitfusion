
#pragma once

#include "../common/tensor.h"

#include <optional>
#include <vector>

namespace tc = bitfusion::common;

namespace bitfusion::layers
{

class DecodingSetupParams
{
public:
    std::optional<std::vector<float>> temperature;
    std::optional<std::vector<std::int32_t>> min_length;
    std::optional<std::vector<float>> repetition_penalty;
    std::optional<std::vector<float>> presence_penalty;
};

class DecodingParams
{
public:
    DecodingParams(int step, int ite, tc::Tensor logits, tc::Tensor end_ids)
        : step{step}
        , ite{ite}
        , logits{std::move(logits)}
        , end_ids{std::move(end_ids)}
    {
    }

    int step;
    int ite;
    tc::Tensor logits;
    tc::Tensor end_ids;
    std::optional<tc::Tensor> finished;
};

class DecodingOutputParams
{
public:
    explicit DecodingOutputParams(tc::Tensor outputIds)
        : output_ids{std::move(outputIds)}
    {
    }

    tc::Tensor output_ids;

    std::optional<tc::Tensor> finished;
    std::optional<tc::Tensor> sequence_length;
    std::optional<tc::Tensor> cum_log_probs;
    std::optional<tc::Tensor>
        output_log_probs;
    std::optional<tc::Tensor> parent_ids;

    tc::Tensor output_ids_ptr;
};

}
