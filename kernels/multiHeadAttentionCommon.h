#pragma once

#include <limits.h>
#include <stdint.h>

namespace bitfusion
{
namespace kernels
{
enum Data_type
{
    DATA_TYPE_BOOL,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32,
    DATA_TYPE_BF16,
    DATA_TYPE_E4M3,
    DATA_TYPE_E5M2
};

constexpr int32_t kSM_70 = 70;
constexpr int32_t kSM_72 = 72;
constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_89 = 89;
constexpr int32_t kSM_90 = 90;

}
}
