
#pragma once
#include "cutlass/gemm/gemm.h"
#include "../../common/assert.h"
#include "../../kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "../../plugins/common/gemmPluginProfiler.h"
#include <cuda_runtime_api.h>
#include <optional>

namespace bitfusion::kernels
{

static inline size_t pad_to_multiple_of_16(const size_t& input)
{
    static constexpr int ALIGNMENT = 16;
    return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

template <typename T>
void topk_gating_softmax_kernelLauncher(const T* input, const bool* finished, T* output, T* softmax_temp_out,
    int* indices, int* source_row, const int num_rows, const int num_experts, const int k, cudaStream_t stream);

class CubKeyValueSorter
{
public:
    CubKeyValueSorter();

    CubKeyValueSorter(const int num_experts);

    void updateNumExperts(const int num_experts);

    static size_t getWorkspaceSize(const size_t num_key_value_pairs, const int num_experts);

    void run(void* workspace, const size_t workspace_size, const int* keys_in, int* keys_out, const int* values_in,
        int* values_out, const size_t num_key_value_pairs, cudaStream_t stream);

private:
    int num_experts_;
    int num_bits_;
};

enum class MOEParallelismMode : int
{
    NONE = 0,
    EXPERT_PARALLELISM,
    TENSOR_PARALLELISM,
};

enum class MOEExpertScaleNormalizationMode : int
{
    NONE = 0,
    RENORMALIZE,
};

struct MOEParallelismConfig
{
    constexpr static MOEParallelismConfig TensorParallelism(int tp_size, int tp_rank)
    {
        return {tp_size, tp_rank, 1, 0};
    }

    constexpr static MOEParallelismConfig ExpertParallelism(int ep_size, int ep_rank)
    {
        return {1, 0, ep_size, ep_rank};
    }

    const int tp_size = 1;
    const int tp_rank = 0;
    const int ep_size = 1;
    const int ep_rank = 0;
};

class CutlassMoeFCRunnerInterface
{
public:
    virtual ~CutlassMoeFCRunnerInterface() = default;
    virtual size_t getWorkspaceSize(const int num_rows, const int hidden_size, const int fc1_output_size,
        const int num_experts, const int k, ActivationType activation_type,
        MOEParallelismConfig parallelism_config) const
        = 0;
    virtual void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm_config) = 0;
    virtual std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() = 0;

    virtual void runMoe(const void* input_activations, const float* gating_output, const void* fc1_expert_weights,
        const void* fc1_scales, const void* fc1_expert_biases, ActivationType fc1_activation_type,
        const void* fc2_expert_weights, const void* fc2_scales, const void* fc2_expert_biases, const int num_rows,
        const int hidden_size, const int inter_size, const int num_experts, const int k, char* workspace_ptr,
        void* final_output, void* fc2_result, const bool* finished, const int active_rows, void* expert_scales,
        int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row,
        MOEParallelismConfig parallelism_config, MOEExpertScaleNormalizationMode normalization_mode,
        cudaStream_t stream)
        = 0;
};

template <typename T,
    typename WeightType,
    typename Enable = void>
class CutlassMoeFCRunner : public CutlassMoeFCRunnerInterface
{
public:
    CutlassMoeFCRunner() = default;
    ~CutlassMoeFCRunner() override = default;

    size_t getWorkspaceSize(const int num_rows, const int hidden_size, const int fc1_output_size, const int num_experts,
        const int k, ActivationType activation_type, MOEParallelismConfig parallelism_config) const override;

    void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm_config) override
    {
        moe_gemm_runner_.setBestConfig(std::move(gemm_config));
    }

    std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() override
    {
        return moe_gemm_runner_.getConfigs();
    }

    void runMoe(const void* input_activations, const float* gating_output, const void* fc1_expert_weights,
        const void* fc1_scales, const void* fc1_expert_biases, ActivationType fc1_activation_type,
        const void* fc2_expert_weights, const void* fc2_scales, const void* fc2_expert_biases, const int num_rows,
        const int hidden_size, const int inter_size, const int num_experts, const int k, char* workspace_ptr,
        void* final_output, void* fc2_result, const bool* finished, const int active_rows, void* expert_scales,
        int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row,
        MOEParallelismConfig parallelism_config, MOEExpertScaleNormalizationMode normalization_mode,
        cudaStream_t stream) override;

private:
    void computeTotalRowsBeforeExpert(const int* sorted_indices, const int total_indices, const int num_experts,
        int64_t* total_rows_before_expert, cudaStream_t stream);
    std::vector<size_t> getWorkspaceBufferSizes(const int num_rows, const int hidden_size, const int inter_size,
        const int num_experts, const int num_experts_per_node, const int k, ActivationType activation_type) const;
    void configureWsPtrs(char* ws_ptr, const int num_rows, const int hidden_size, const int inter_size,
        const int num_experts, const int num_experts_per_node, const int k, ActivationType activation_type);

private:
    CubKeyValueSorter sorter_;
    MoeGemmRunner<T, WeightType> moe_gemm_runner_;

    int* source_rows_;
    int* permuted_rows_;
    int* permuted_experts_;
    char* sorter_ws_;
    T* permuted_data_;
    float* softmax_out_;

    int64_t* total_rows_before_expert_;

    T* fc1_result_;
    T* glu_inter_result_;
};

template <typename WeightType>
class CutlassMoeFCRunner<float, WeightType, typename std::enable_if_t<!std::is_same<float, WeightType>::value>>
    : public CutlassMoeFCRunnerInterface
{
public:
    CutlassMoeFCRunner() = default;

    size_t getWorkspaceSize(const int num_rows, const int hidden_size, const int fc1_output_size, const int num_experts,
        const int k, ActivationType activation_type, MOEParallelismConfig parallelism_config) const override
    {
        return 0;
    }

    void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm_config) override
    {
        return;
    }

    void runMoe(const void* input_activations, const float* gating_output, const void* fc1_expert_weights,
        const void* fc1_scales, const void* fc1_expert_biases, ActivationType fc1_activation_type,
        const void* fc2_expert_weights, const void* fc2_scales, const void* fc2_expert_biases, const int num_rows,
        const int hidden_size, const int inter_size, const int num_experts, const int k, char* workspace_ptr,
        void* final_output, void* fc2_result, const bool* finished, const int active_rows, void* expert_scales,
        int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row,
        MOEParallelismConfig parallelism_config, MOEExpertScaleNormalizationMode normalization_mode,
        cudaStream_t stream) override
    {
        TLLM_THROW("FP32 MoE with different precision weights is not supported.");
    }
};

void makeLoadBalancedRoutingConfiguration(
    void* data_void, int num_experts, int num_tokens, int k, nvinfer1::DataType type, cudaStream_t stream);

}
