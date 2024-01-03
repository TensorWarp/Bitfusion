
#pragma once

#include "bufferManager.h"
#include "common.h"
#include "cudaStream.h"
#include "iTensor.h"

namespace bitfusion::runtime::kernels
{

template <typename T>
void invokeFill(IBuffer& buffer, T value, CudaStream const& stream);

template <typename T>
void invokeFillBatch(
    IBuffer& buffer, IBuffer const& indices, std::size_t stride, IBuffer const& values, CudaStream const& stream);

void invokeCopyBatch(IBuffer const& srcBuffer, IBuffer& dstBuffer, IBuffer const& srcOffsets, IBuffer const& dstOffsets,
    IBuffer const& sizes, std::size_t maxStride, CudaStream const& stream);

template <typename T>
void invokeAdd(IBuffer& buffer, T value, CudaStream const& stream);

void reduce(IBuffer& output, IBuffer const& input, CudaStream const& stream);

void invokeTranspose(ITensor& output, ITensor const& input, CudaStream const& stream);

void invokeTransposeWithOutputOffset(
    ITensor& output, ITensor const& input, SizeType outputOffset, CudaStream const& stream);

void invokeTransposeWithInputOffset(
    ITensor& output, ITensor const& input, SizeType inputOffset, CudaStream const& stream);

void invokeInclusiveSum(IBuffer& output, IBuffer const& input, BufferManager const& manager, CudaStream const& stream);

void invokeBuildTokenMask(
    ITensor& tokenMask, ITensor const& inputLengths, SizeType maxInputLength, CudaStream const& stream);

void invokeBuildAttentionMask(ITensor& attentionMask, SizeType padId, CudaStream const& stream);

void invokeExtendAttentionMask(ITensor& newMask, ITensor const& oldMask, CudaStream const& stream);

void invokeCopyInputToOutputTransposed(
    ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths, SizeType padId, CudaStream const& stream);

void invokeCopyPackedInputToOutputTransposed(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputOffsets,
    SizeType maxInputLength, SizeType padId, CudaStream const& stream);

void invokeCopyInputToOutput(
    ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths, SizeType padId, CudaStream const& stream);

void invokeCopyPackedInputToOutput(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputOffsets,
    SizeType maxInputLength, SizeType padId, CudaStream const& stream);

void initOutputIds(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths,
    ITensor const& inputOffsets, TokenIdType padId, TokenIdType endId, SizeType maxInputLength, bool inputPacked,
    CudaStream const& stream);

void scatterTensor(ITensor& output, ITensor const& input, SizeType beamWidth, CudaStream const& stream);

void tileTensor(ITensor& output, ITensor const& input, SizeType beamWidth, CudaStream const& stream);

void tileTensorInplace(ITensor& tensor, SizeType beamWidth, CudaStream const& stream);

void gatherLastTokenLogits(
    ITensor& output, ITensor const& input, ITensor const& lastTokenIds, CudaStream const& stream);

void copyLatestTokenLogitsInGeneration(ITensor& output, ITensor const& input, SizeType step, SizeType firstBatchSlotIdx,
    SizeType microBatchSize, SizeType beamWidth, CudaStream const& stream);

}
