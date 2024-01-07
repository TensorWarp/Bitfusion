
#pragma once

#include "cudaStream.h"
#include "generationInput.h"
#include "generationOutput.h"
#include "iTensor.h"
#include "samplingConfig.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <NvInferRuntime.h>

namespace bitfusion::runtime
{

namespace decoder
{

class Input
{
public:
    using TensorPtr = std::shared_ptr<ITensor const>;

    explicit Input(TensorPtr logits)
        : logits{std::move(logits)}
    {
        CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor");
    }

    TensorPtr logits;

    TensorPtr cacheIndirection;
};

class Output
{
public:
    using TensorPtr = std::shared_ptr<ITensor>;

    Output() = default;

    TensorPtr cacheIndirection;
    TensorPtr sequenceLengths;
};
}

class IStatefulDecoder
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;

    virtual void setup(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxAttentionWindow,
        SizeType maxSequenceLength, SizeType maxTokensPerStep, nvinfer1::DataType dtype)
        = 0;

    virtual void newBatch(
        GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig)
        = 0;

    virtual void forwardAsync(decoder::Output& output, decoder::Input const& input) = 0;

    virtual void forwardSync() = 0;

    virtual void forward(decoder::Output& output, decoder::Input const& input)
    {
        forwardAsync(output, input);
        return forwardSync();
    }

    virtual void finalize() const = 0;

    virtual TensorPtr getOutputIds() const = 0;

    virtual TensorPtr getCumLogProbs() const = 0;

    virtual TensorPtr getLogProbs() const = 0;

    virtual TensorPtr getNewTokens(SizeType iter = 0) const = 0;

    virtual TensorPtr getAllNewTokens() const = 0;

    virtual TensorPtr getNbFinished() const = 0;

    virtual ~IStatefulDecoder() = default;

protected:
    IStatefulDecoder() = default;
};

}
