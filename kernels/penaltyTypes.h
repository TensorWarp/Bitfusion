

#pragma once

#include <string>
#include <unordered_map>

namespace bitfusion
{
namespace kernels
{

enum class RepetitionPenaltyType
{
    Additive,       // the presence penalty
    Multiplicative, // the repetition penalty
    None            // No repetition penalty.
};

inline float getDefaultPenaltyValue(RepetitionPenaltyType penalty_type)
{
    switch (penalty_type)
    {
    case RepetitionPenaltyType::Additive: return 0.0f;
    case RepetitionPenaltyType::Multiplicative: return 1.0f;
    default: break;
    }
    return 0.0f;
}

} // namespace kernels
} // namespace bitfusion
