#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "Types.h"

uint32_t CalculateBlocks(uint64_t size);
template<typename T> __device__ T sgn(T x) { return (x > 0) - (x < 0); }

void SetKernelsGpuData();
void GetKernelsGpuData();
void SetKLossGpuData();
void GetKLossGpuData();
void SetKActivationGpuData();
void GetKActivationGpuData();
void SetKDeltaGpuData();
void GetKDeltaGpuData();

void kScaleAndBias(float* pData, uint64_t size, float scale, float bias);
void kAddBias(float* pUnit, float* pBias, uint32_t stride, uint32_t batch);
void kAddDualBias(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch);
void kAddTripleBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch);
void kAddQuadBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch);
void kClearUnit(float* pUnit, float* pBias, uint32_t stride, uint32_t batch);
void kClearDualSourceUnit(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch);
void kClearTripleSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch);
void kClearQuadSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch);
void kUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias);
void invokeExamples(float* pOutputKey, float *pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k);
void invokeExamples(float* pOutputKey, float* pOutputValue, float *pKey, float* pValue, uint32_t batch, uint32_t width, uint32_t k);
void invokeExamples(float* pOutputKey, uint32_t* pOutputValue, float *pKey, uint32_t * pValue, uint32_t batch, uint32_t width, uint32_t k);
void invokeKSparse(float* pUnit, uint32_t batch, uint32_t stride, uint32_t kSparse);
void kAddScaleBuffers(float* pDest, float* pSrc, float scale, uint64_t size);
void kAddBuffers(float* pDest, float* pSrc, uint64_t size, cudaStream_t stream = 0);
void kAddBuffers2D(float* pDest, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream = 0);
void kCopy2D(float* pDest, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream = 0);
template<typename KeyType, typename ValueType> size_t kInitSort(uint32_t items, GpuBuffer<KeyType>* pbKey, GpuBuffer<ValueType>* pbValue);
template<typename KeyType, typename ValueType> bool kSort(uint32_t items, KeyType* pKey0, KeyType* pKey1, ValueType* pValue0, ValueType* pValue1, char* pTemp, size_t tempBytes);
template<typename T> void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData);
template<typename T> void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData);
void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);
void kLoadIndexedSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);
void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom);
void kLoadIndexedSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom);
template<typename T> void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);
template<typename T> void kLoadIndexedSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);
template<typename T> void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom);
template<typename T> void kLoadIndexedSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom);
void invokeSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta);
void invokeIndexedSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta);
template<typename T> void invokeSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta);
template<typename T> void invokeIndexedSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta);
void invokeSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta);
void invokeIndexedSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta);
template<typename T> void invokeSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta);
template<typename T> void invokeIndexedSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta);
void invokeSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);
void invokeIndexedSparseTransposedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);
void invokeSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);
void invokeIndexedSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);
void invokeSparseTransposedWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pDelta, float* pWeightGradient);
template<typename T> void invokeSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);
template<typename T> void invokeIndexedSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);
template<typename T> void invokeSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);
template<typename T> void invokeIndexedSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);
void invokeSparseTransposedAnalogWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData, float* pDelta, float* pWeightGradient);
template<typename T> float invokeL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float invokeIndexedL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float invokeL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float invokeIndexedL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float invokeL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float invokeIndexedL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float invokeCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float invokeIndexedCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float invokeScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float invokeIndexedScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float invokeMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float invokeIndexedMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float invokeMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float invokeIndexedMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float invokeHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float invokeIndexedHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
float invokeSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float invokeIndexedSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float invokeSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float invokeIndexedSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float invokeSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float invokeIndexedSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float invokeSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float invokeIndexedSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float invokeSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float invokeIndexedSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float invokeSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);
float invokeIndexedSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);
float invokeSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);
float invokeIndexedSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);
template<typename T> float invokeSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float invokeIndexedSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float invokeSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float invokeIndexedSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float invokeSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float invokeIndexedSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float invokeSparseAnalogCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float invokeIndexedSparseAnalogCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float invokeSparseAnalogScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float invokeIndexedSparseAnalogScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float invokeSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);
template<typename T> float invokeIndexedSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);
template<typename T> float invokeSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);
template<typename T> float invokeIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);
template<typename T> float invokeSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float invokeIndexedSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);
float invokeRegularizationError(float lambda, float lambda1, float* pWeight, uint64_t size);
void kNormalizeWeights(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight);
void invokeWeightMagnitudes(uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude);
void kNormalizeWeightMagnitudes(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude);
void kNormalizeDeltas(float norm, uint32_t batch, uint32_t stride, float* pDelta);
void invokeDeltaMagnitudes(uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude);
void kNormalizeDeltaMagnitudes(float norm, uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude);
void invokeScaledBiasedDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target, float a, float b);
void invokeDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target);
template<typename T> void invokeL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void invokeIndexedL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void invokeCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight);
template<typename T> void invokeIndexedCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> void invokeScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight);
template<typename T> void invokeIndexedScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> void invokeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void invokeIndexedOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void invokeL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void invokeIndexedL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void invokeHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight);
template<typename T> void invokeIndexedHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight);
void invokeSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
void invokeIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
void invokeSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
void invokeIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
void invokeSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
void invokeIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
void invokeSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
void invokeIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
void invokeSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
void invokeIndexedSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void invokeSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float scope, float alpha, float lambda);
template<typename T> void invokeIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float scope, float alpha, float lambda);
template<typename T> void invokeSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void invokeIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void invokeSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void invokeIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void invokeSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void invokeIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void invokeSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void invokeIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void invokeSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void invokeIndexedSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void invokeSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void invokeIndexedSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
void invokeSparsenessPenalty(uint32_t batch,  uint32_t stride, float* pUnit, float* pDelta, float p, float beta);
void invokeHadamardProduct(Activation activation, uint64_t size, float scale, float* pUnit, float* pDelta, float slope, float alpha, float lambda);
void invokeSigmoidActivation(float* pData, uint64_t size);
void invokeTanhActivation(float* pData, uint64_t size);
void invokeRELUActivation(float* pData, uint64_t size);
void invokeELUActivation(float* pData, uint64_t size, float alpha);
void invokeSELUActivation(float* pData, uint64_t size, float alpha, float lambda);
void invokeLRELUActivation(float* pData, uint64_t size, float slope);
void invokeSoftMaxActivation(float* pData, uint32_t batch, uint32_t stride);
void kSGDUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightGradient, float* pWeight);
void kSGDUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias);
void kMomentumUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);
void kMomentumUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);
void kAdaGradUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);
void kAdaGradUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);
void kNesterovShiftWeights(float mu, uint64_t size, float* pWeightVelocity, float* pWeight);
void kNesterovShiftBiases(float mu, uint32_t width, float* pBiasVelocity, float* pBias);
void kNesterovUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);
void kNesterovUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);
void kRMSPropUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);
void kRMSPropUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);
void kAdaDeltaUpdateWeights(float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight);
void kAdaDeltaUpdateBiases(float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias);
void kAdamUpdateWeights(float alpha, float lambda, float lambda1, float mu, float mu1, float t, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight);
void kAdamUpdateBiases(float alpha, float mu, float mu1, float t, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias);
void invokeMaxout(float* pSrc, size_t size, float* pDst);
void invokeCosine(float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDPOut, float* pAOut, float* pBOut, uint32_t outStride);                        
void invokeDotProduct(float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDPOut, uint32_t outStride);                        
void invokeMaxoutDelta(float* pSrc, float* pSrcDelta, size_t size, float beta, float* pDst, float* pDstDelta);
void invokeDotProductDelta(float* pDPDelta, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride);
void invokeCosineDelta(float* pDPDelta, float* pDP, float* pA, float* pB, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride);

template <typename T>
void invokeBanRepeatNgram(T* logits, const int** output_ids_buf, const bool* finished_buf, const int* parent_ids_buf,
    int batch_size, int local_batch_size, int beam_width, const int* no_repeat_ngram_size_buf, int id_offset,
    int vocab_size_padded, size_t step, cudaStream_t stream);

template <typename T, typename Idx>
void invokeLookUp(T* out, const Idx* input, const T* weight, const Idx batch_size, const Idx offset, const Idx size,
    const int n_embed, cudaStream_t stream = 0);

struct gatherTreeParam
{
    int* beams = nullptr;
    int* sequence_lengths = nullptr;
    int max_sequence_length_final_step = 0;
    const int* input_lengths = nullptr;
    int* response_input_lengths = nullptr;
    int max_seq_len = 0;
    int batch_size = 0;
    int beam_width = 0;
    const int* step_ids = nullptr;
    const int* parent_ids = nullptr;
    const int* end_tokens = nullptr;
    int* output_ids = nullptr;
    cudaStream_t stream;
    float* cum_log_probs = nullptr;
    float length_penalty = 1.0f;
};

void invokeGatherTree(gatherTreeParam param);

void invokeFinalize(int* output_ids, int* sequence_lengths, float* cum_log_probs, float* output_log_probs,
    const int* topk_output_ids, const int* topk_sequence_lengths, const float* scores, const float* topk_cum_log_probs,
    const float* topk_log_probs, const int* num_beams, const int* input_lengths, const int beam_width,
    const int max_seq_len, const int batch_size, cudaStream_t stream);

void invokeInitializeOutput(int* output_ids, const int* end_ids, int batch_beam, int max_seq_len, cudaStream_t stream);

void invokeCopyNextStepIds(int* next_step_ids, int** output_ids_ptr, const int* sequence_lengths, int batch_size,
    int beam_width, int max_seq_len, cudaStream_t stream);

template <typename T>
void invokeGeneralLayerNorm(T* out, const T* input, const T* gamma, const T* beta, const float eps, const int tokens,
    const int hidden_dim, cudaStream_t stream = 0, bool use_diff_of_squares = true, const float* scale = nullptr,
    float* dynamic_scale = nullptr, int8_t* out_quant = nullptr);

template <typename T>
void apply_per_channel_scale_kernel_launcher(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream = 0);

template <typename T>
void invokeQuantization(
    int8_t* dst, const T* src, const int64_t size, const float* scalePtr, cudaStream_t stream = 0, int maxGirdSize = 0);

template <typename T>
void invokePerTokenQuantization(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, cudaStream_t stream = 0);

template <typename T>
void invokeGeneralRmsNorm(T* out, const T* input, const T* gamma, const T* beta, const float eps, const int tokens,
    const int hidden_dim, cudaStream_t stream = 0, const float* scale = nullptr, float* dynamic_scale = nullptr,
    int8_t* out_quant = nullptr);

void invokeStopWordsCriterion(const int** output_ids, const int** parent_ids, const int* stop_words, bool* finished,
    const int* sequence_lengths, size_t id_offset, size_t stop_words_len, int batch_size, int beam_width,
    int max_seq_len, cudaStream_t stream);

void invokeLengthCriterion(bool* finished, int* finished_sum, const uint32_t* sequence_limit_length,
    const int* sequence_lengths, int batch_size, int beam_width, cudaStream_t stream);

template <typename T>
void invokeBanBadWords(T* logits, const int** output_ids_ptr, const int** parent_ids_ptr, int batch_size,
    int local_batch_size, int beam_width, const int* bad_words, bool share_words, size_t bad_words_len,
    int vocab_size_padded, const int* sequence_lengths, int max_seq_len, cudaStream_t stream);


__device__ inline uint64_t llitoulli(int64_t l)
{
    uint64_t u;
    asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
    return u;
}

__device__ inline int64_t ullitolli(uint64_t u)
{
    int64_t l;
    asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
    return l;
}

#if (CUDA_VERSION >= 9000)
#define SHFL(x, lane) __shfl_sync(0xffffffff, (x), (lane))
#define BALLOT(predicate) __ballot_sync(0xffffffff, (predicate))
#define ANY(predicate) __any_sync(0xffffffff, (predicate))
#else
#define SHFL(x, lane) __shfl((x), (lane))
#define BALLOT(predicate) __ballot(predicate)
#define ANY(predicate) __any(predicate)
#endif


#define REDUCEERROR(error) \
    if (ANY(error != (float)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        error                  += SHFL(error, tgx ^ 1); \
        error                  += SHFL(error, tgx ^ 2); \
        error                  += SHFL(error, tgx ^ 4); \
        error                  += SHFL(error, tgx ^ 8); \
        error                  += SHFL(error, tgx ^ 16); \
        if (tgx == 0) \
        { \
            atomicAdd(cData._pAccumulator, llitoulli(llrintf(ERRORSCALEF * error))); \
        } \
    } 


#define REDUCE(a) \
    if (ANY((a) != (float)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        a                      += SHFL((a), tgx ^ 1); \
        a                      += SHFL((a), tgx ^ 2); \
        a                      += SHFL((a), tgx ^ 4); \
        a                      += SHFL((a), tgx ^ 8); \
        a                      += SHFL((a), tgx ^ 16); \
    } 


#endif
