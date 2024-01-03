
#pragma once

#include "cutlass/gemm_coord.h"
#include <NvInferRuntime.h>

namespace bitfusion
{
namespace kernels
{

void gropuedGemm(std::vector<cutlass::gemm::GemmCoord> problem_sizes, std::vector<void*> ptrA, std::vector<void*> ptrB,
    std::vector<void*> ptrC, std::vector<void*> ptrD, void* workspace, int64_t workSpaceSize, void* cublasWorkSpace,
    int64_t cublasWorkspaceSize, bool isLoraIn, nvinfer1::DataType dataType, cudaStream_t stream);

} // namespace kernels

} // namespace bitfusion
