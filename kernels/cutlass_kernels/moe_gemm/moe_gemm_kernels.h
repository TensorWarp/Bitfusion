#pragma once
#include "../../cutlass_extensions/gemm_configs.h"
#include <cuda_runtime_api.h>
#include <optional>

namespace bitfusion
{

// Note update moe.py to match
enum class ActivationType
{
    Gelu = 0,
    Relu,
    Silu,
    Swiglu,
    Geglu,
    Identity,
    InvalidType
};

constexpr bool isGatedActivation(ActivationType activation_type)
{
    return activation_type == ActivationType::Swiglu || activation_type == ActivationType::Geglu;
}

template <typename T, /*The type used for activations/scales/compute*/
    typename WeightType /* The type for the MoE weights */>
class MoeGemmRunner
{
public:
    MoeGemmRunner();

    void setBestConfig(std::optional<cutlass_extensions::CutlassGemmConfig> best_config)
    {
        best_config_ = std::move(best_config);
    }

    void moeGemmBiasAct(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        ActivationType activation_type, cudaStream_t stream);

    void moeGemm(const T* A, const WeightType* B, const T* weight_scales, T* C, int64_t* total_rows_before_expert,
        int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, cudaStream_t stream);

    std::vector<cutlass_extensions::CutlassGemmConfig> getConfigs();

private:
    template <typename EpilogueTag>
    void dispatchToArch(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        cutlass_extensions::CutlassGemmConfig gemm_config, cudaStream_t stream, int* occupancy = nullptr);

    template <typename EpilogueTag>
    void runGemm(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        cudaStream_t stream);

private:
    int sm_;
    int multi_processor_count_;
    std::optional<cutlass_extensions::CutlassGemmConfig> best_config_{};
};

} // namespace bitfusion
