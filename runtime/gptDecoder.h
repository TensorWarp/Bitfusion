
#pragma once

#include "../common/cudaAllocator.h"
#include "../runtime/bufferManager.h"
#include "decodingInput.h"
#include "decodingOutput.h"
#include "samplingConfig.h"
#include <curand_kernel.h>

#include <cstdint>
#include <memory>

#include <NvInferRuntime.h>

namespace bitfusion
{

namespace layers
{
template <typename T>
class DynamicDecodeLayer;
}

namespace runtime
{

class IGptDecoder
{
public:
    virtual ~IGptDecoder() = default;

    virtual void setup(SamplingConfig const& samplingConfig, size_t batchSize, SizeType maxSequenceLength) = 0;

    virtual bool forward(DecodingOutput& output, DecodingInput const& input) = 0;

    virtual void forwardAsync(DecodingOutput& output, DecodingInput const& input) = 0;

    virtual void gatherTree(ITensor& finalOutputIds, DecodingOutput const& decodingOutput,
        DecodingInput const& decodingInput, BufferManager const& manager)
        = 0;

    virtual const SamplingConfig& getSamplingConfig() = 0;

    static void acceptDraftTokensByIds(const ITensor& targetTokenIds, const ITensor& draftTokenIds,
        const ITensor& contextLengths, const ITensor& numDraftTokens, ITensor& sequenceLengths,
        const ITensor& finishedVec, ITensor& finishedFinal, ITensor& finishedSum,
        BufferManager::CudaStreamPtr const& stream);

    static void acceptDraftTokensByLogits(ITensor& draftLogits, const ITensor& targetLogits, ITensor& draftProbs,
        ITensor& targetProbs, const ITensor& numDraftTokens, ITensor& finished, SizeType vocabSize,
        SizeType vocabSizePadded, bool useRandomAcceptThreshold, float randomAcceptThreshold,
        curandState_t* curandState, BufferManager::CudaStreamPtr const& stream);

    static std::unique_ptr<IGptDecoder> create(
        nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded, BufferManager::CudaStreamPtr const& stream);
};

template <typename T>
class GptDecoder : public virtual IGptDecoder
{

public:
    using CudaStreamPtr = BufferManager::CudaStreamPtr;
    using TensorPtr = std::shared_ptr<ITensor>;

    GptDecoder(size_t vocabSize, size_t vocabSizePadded, CudaStreamPtr const& stream);

    void setup(SamplingConfig const& samplingConfig, size_t batchSize, SizeType maxSequenceLength) override;

    bool forward(DecodingOutput& output, DecodingInput const& input) override;

    void forwardAsync(DecodingOutput& output, DecodingInput const& input) override;

    void gatherTree(ITensor& finalOutputIds, DecodingOutput const& decodingOutput, DecodingInput const& decodingInput,
        BufferManager const& manager) override;

    const SamplingConfig& getSamplingConfig() override
    {
        return mSamplingConfig;
    }

private:
    BufferManager mManager;

    common::CudaAllocator mAllocator;
    std::shared_ptr<bitfusion::layers::DynamicDecodeLayer<T>> mDynamicDecodeLayer;

    TensorPtr mLogProbsTiled;
    SamplingConfig mSamplingConfig;
};

inline std::unique_ptr<IGptDecoder> IGptDecoder::create(
    nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded, BufferManager::CudaStreamPtr const& stream)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return std::make_unique<GptDecoder<float>>(vocabSize, vocabSizePadded, stream);
    case nvinfer1::DataType::kHALF: return std::make_unique<GptDecoder<half>>(vocabSize, vocabSizePadded, stream);
    default: return nullptr;
    }
}
}
}
