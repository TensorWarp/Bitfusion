

#include "../common/reduceKernelUtils.cuh"
#include "decodingCommon.h"
#include <stdio.h>

using namespace bitfusion::common;

namespace bitfusion
{
namespace kernels
{

__global__ void curandInitialize(curandState_t* state, const int size, const unsigned long long randomSeed)
{
    if (threadIdx.x + blockIdx.x * blockDim.x < size)
    {
        curand_init(randomSeed, 0, 0, &state[blockIdx.x * blockDim.x + threadIdx.x]);
    }
}

void invokeCurandInitialize(
    curandState_t* state, const size_t batchSize, const unsigned long long randomSeed, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((int) (ceil(batchSize * 1.0 / 256)));
    curandInitialize<<<grid, block, 0, stream>>>(state, batchSize, randomSeed);
}

__global__ void curandBatchInitialize(curandState_t* states, const int size, const unsigned long long* randomSeeds)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        curand_init(randomSeeds[idx], 0, 0, &states[idx]);
    }
}

void invokeCurandBatchInitialize(
    curandState_t* states, const size_t batchSize, const unsigned long long* randomSeeds, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((int) (ceil(batchSize * 1.0 / 256)));
    curandBatchInitialize<<<grid, block, 0, stream>>>(states, batchSize, randomSeeds);
}

template <typename T>
__global__ void addBiasSoftMax(T* logits, T* probs, const T* bias, const int* endIds, const FinishedState* finished,
    const int vocabSize, const int vocabSizePadded)
{
    int bid = blockIdx.x;

    if (finished != nullptr)
    {
        FinishedState finishState = finished[bid];

        if (finishState.isSkipDecoding())
        {
            // Skip decoding for this block
            return;
        }
    }

    int offset = bid * vocabSizePadded;

    float maxVal = -1 * FLT_MAX;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    __shared__ float sMaxVal;
    __shared__ float sSumVal;

    for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
    {
        if (tid < vocabSize)
        {
            if ((finished == nullptr || !finished[bid].isFinished()) || endIds == nullptr || tid != endIds[bid])
            {
                T bias_val = (bias != nullptr) ? bias[tid] : (T)0.0f;
                logits[offset + tid] += bias_val;
            }
            else
            {
                logits[offset + tid] = MAX_T_VAL;
            }
        }
        else
        {
            logits[offset + tid] = -MAX_T_VAL;
        }
        maxVal = max(maxVal, (float)logits[offset + tid]);
    }

    maxVal = blockReduceMax<float>((float)maxVal);
    if (threadIdx.x == 0)
    {
        sMaxVal = maxVal;
    }
    __syncthreads();

    float sumVal = 0.0f;
    for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
    {
        probs[offset + tid] = __expf((float)logits[offset + tid] - sMaxVal);
        sumVal += (float)probs[offset + tid];
    }

    sumVal = blockReduceSum<float>(sumVal);
    if (threadIdx.x == 0)
    {
        sSumVal = sumVal;
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
    {
        probs[offset + tid] = ((float)probs[offset + tid] / (sSumVal + 1e-6f));
    }
}

template <typename T>
void invokeAddBiasSoftMax(T* logits, T* probs, const T* bias, const int* endIds, const FinishedState* finished,
    const int batchSize, const int vocabSize, const int vocabSizePadded, cudaStream_t stream)
{
    dim3 grid(batchSize);
    dim3 block(min(vocabSize, 1024));
    // vocabSize, e.g., 30000, 7000.... vocabSize is usually very big.
    addBiasSoftMax<<<grid, block, 0, stream>>>(logits, probs, bias, endIds, finished, vocabSize, vocabSizePadded);
}

template void invokeAddBiasSoftMax(float* logits, float* probs, const float* bias, const int* endIds,
    const FinishedState* finished, const int m, const int nPadded, const int n, cudaStream_t stream);

template void invokeAddBiasSoftMax(half* logits, half* probs, const half* bias, const int* endIds,
    const FinishedState* finished, const int m, const int nPadded, const int n, cudaStream_t stream);

} // namespace kernels
} // namespace bitfusion
