

#pragma once

#include "../../../cutlass_extensions/gemm_configs.h"
#include "../../../common/quantization.h"
#include <cuda_runtime_api.h>

namespace tk = bitfusion::common;
namespace tkc = bitfusion::cutlass_extensions;

namespace bitfusion
{
namespace kernels
{
namespace cutlass_kernels
{

/*
  This runner supports:
  int8_t inputs (A and B)
  float alpha scalings (either per-col, or per-col x per-row)
  T output (D) where T = {float, half, __nv_bfloat16} // TODO

  Activations, biases, scales and outputs are all assumed to be row-major.
  Weights are assumed to be column-major.
*/

class CutlassInt8GemmRunnerInterface
{
public:
    CutlassInt8GemmRunnerInterface() {}

    virtual ~CutlassInt8GemmRunnerInterface() {}

    virtual void gemm(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, void* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream)
        = 0;

    // Returns desired workspace size in bytes.
    virtual size_t getWorkspaceSize(const int m, const int n, const int k) = 0;

    virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;

protected:
    static constexpr int SPLIT_K_LIMIT = 7;
    static constexpr int MIN_M_TILE = 32;
    static constexpr int MIN_N_TILE = 64;
};

template <typename T>
class CutlassInt8GemmRunner : public virtual CutlassInt8GemmRunnerInterface
{
public:
    CutlassInt8GemmRunner();
    ~CutlassInt8GemmRunner();

    void gemm(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol, const float* alphaRow,
        void* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream) override;

    // Returns desired workspace size in bytes.
    size_t getWorkspaceSize(const int m, const int n, const int k) override;

    std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

private:
    void dispatchToArch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, T* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

    int mSm;
    int mMultiProcessorCount;
};

} // namespace cutlass_kernels
} // namespace kernels
} // namespace bitfusion
