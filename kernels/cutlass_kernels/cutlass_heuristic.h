

#pragma once

#include "../../cutlass_extensions/gemm_configs.h"
#include "../../common/cudaUtils.h"

namespace bitfusion
{
namespace kernels
{
namespace cutlass_kernels
{

std::vector<bitfusion::cutlass_extensions::CutlassGemmConfig> get_candidate_configs(int sm,
    const bool is_weight_only, const bool simt_configs_only, const bool int8_configs_only = false,
    const int max_split_k = 1);

bitfusion::cutlass_extensions::CutlassGemmConfig estimate_best_config_from_occupancies(
    const std::vector<bitfusion::cutlass_extensions::CutlassGemmConfig>& candidate_configs,
    const std::vector<int>& occupancies, const int64_t m, const int64_t n, const int64_t k, const int64_t num_experts,
    const int split_k_limit, const size_t workspace_bytes, const int multi_processor_count, const int is_weight_only);

} // namespace cutlass_kernels
} // namespace kernels
} // namespace bitfusion
