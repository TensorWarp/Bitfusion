
#pragma once

#include "bufferManager.h"
#include "cudaEvent.h"
#include "cudaStream.h"
#include "iStatefulGptDecoder.h"
#include "iTensor.h"
#include "utils/sessionUtils.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace bitfusion::runtime
{

namespace decoder_batch
{
class Request
{
public:
    using ConstTensorPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;
    using BufferPtr = IBuffer::SharedPtr;

    explicit Request(ConstTensorPtr ids, SizeType inputLen, std::optional<SizeType> maxNewTokens = std::nullopt,
        std::optional<SizeType> endId = std::nullopt)
        : ids{std::move(ids)}
        , inputLen(inputLen)
        , maxNewTokens{maxNewTokens}
        , endId{endId}
        , computeCumLogProbs(false)
        , computeLogProbs(false)
    {
    }

    SizeType generatedTokensPerStep() const
    {
        return draftTokens ? draftTokens->getSize() + 1 : 1;
    }

    ConstTensorPtr ids;
    SizeType inputLen;

    std::optional<SizeType> maxNewTokens;
    std::optional<SizeType> endId;
    BufferPtr draftTokens;
    std::optional<TensorPtr>
        draftLogits;
    TensorPtr embeddingBias;
    TensorPtr badWordsList;
    TensorPtr stopWordsList;

    bool computeCumLogProbs;
    bool computeLogProbs;
};

class Input
{
public:
    using TensorConstPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;

    explicit Input(std::vector<TensorConstPtr> const& logits, std::vector<bool> const& active)
        : logits{logits}
        , active{active}
    {
        TLLM_CHECK_WITH_INFO(
            this->active.size() == logits.size(), "'active' vector size does not match logits vector size");
    }

    explicit Input(std::vector<TensorConstPtr> const& logits)
        : Input{logits, std::vector<bool>(logits.size(), true)}
    {
    }

    explicit Input(std::vector<TensorPtr> const& logits, std::vector<bool> const& active)
        : Input{
            utils::transformVector(logits, [](auto& x) { return std::const_pointer_cast<ITensor const>(x); }), active}
    {
    }

    explicit Input(std::vector<TensorPtr> const& logits)
        : Input{logits, std::vector<bool>(logits.size(), true)}
    {
    }

    std::vector<TensorConstPtr>
        logits;

    std::vector<bool> active;

    TensorConstPtr cacheIndirection;
};

using Output = decoder::Output;

class Token
{
public:
    explicit Token(CudaEvent&& event, std::vector<bool> const& active)
        : event(std::move(event))
        , active(active)
    {
    }

    CudaEvent event;
    std::vector<bool> active;
};
}

class IGptDecoderBatch : public virtual IStatefulGptDecoder
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;
    using TokenPtr = std::unique_ptr<decoder_batch::Token const>;

    virtual void newRequest(
        SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
        = 0;

    virtual TokenPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) = 0;

    virtual void forwardSync(decoder_batch::Token const& token) = 0;

    virtual void forward(decoder_batch::Output& output, decoder_batch::Input const& input)
    {
        forwardSync(*forwardAsync(output, input));
    }

    virtual TensorPtr getOutputIds(SizeType batchIdx) const = 0;

    virtual CudaEvent finalize(SizeType batchIdx) const = 0;

    virtual std::vector<bool> getFinished() const = 0;

    virtual TensorPtr getCumLogProbs() const = 0;

    virtual TensorPtr getCumLogProbs(SizeType batchIdx) const = 0;

    virtual TensorPtr getLogProbs() const = 0;

    virtual TensorPtr getLogProbs(SizeType batchIdx) const = 0;

    virtual TensorPtr getParentIds() const = 0;

    virtual std::vector<SizeType> getNbSteps() const = 0;

protected:
    IGptDecoderBatch() = default;
};

}
