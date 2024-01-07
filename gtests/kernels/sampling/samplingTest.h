#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <random>

#include "../../../kernels/decodingCommon.h"
#include "../../../kernels/samplingPenaltyKernels.h"
#include "../../../kernels/samplingTopKKernels.h"
#include "../../../kernels/samplingTopPKernels.h"
#include "../../../runtime/bufferManager.h"
#include "../../../runtime/cudaStream.h"
#include "../../../runtime/runtimeKernels.h"
#include "../../../runtime/tllmLogger.h"

namespace bitfusion::tests::kernels::sampling
{

typedef testing::Types<float, half> FloatAndHalfTypes;

constexpr float EPSILON = 1e-20f;

inline bool almostEqual(float a, float b, float atol = 1e-5, float rtol = 1e-8)
{
    if (isnan(a) && isnan(b))
    {
        return true;
    }
    return fabs(a - b) <= (atol + rtol * fabs(b));
}

template <typename T>
void initRandom(T* ptr, size_t size, float minval, float maxval)
{
    for (size_t i = 0; i < size; ++i)
    {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        val *= (maxval - minval);
        ptr[i] = static_cast<T>(minval + val);
    }
}

inline void initRandomInt(int* ptr, size_t size, int minval, int maxval)
{
    assert(minval < maxval);
    int mod = maxval - minval;
    for (size_t i = 0; i < size; ++i)
    {
        ptr[i] = minval + rand() % mod;
    }
}

template <typename T>
bool checkResult(std::string name, T* out, T* ref, size_t size)
{
    bool isFp32 = sizeof(T) == 4;
    float atol = isFp32 ? 1e-4f : 1e-3f;
    float rtol = isFp32 ? 1e-2f : 1e-1f;

    size_t failures = 0;
    float relativeGap = 0.0f;
    ;

    for (size_t i = 0; i < size; ++i)
    {
        float a = (float) out[i];
        float b = (float) ref[i];

        bool ok = almostEqual(a, b, atol, rtol);
        if (!ok && failures < 4)
        {
            TLLM_LOG_ERROR(">> invalid result for i=%lu:", i);
            TLLM_LOG_ERROR(">>    found......: %10.6f", a);
            TLLM_LOG_ERROR(">>    expected...: %10.6f", b);
            TLLM_LOG_ERROR(">>    error......: %.6f", fabsf(a - b));
            TLLM_LOG_ERROR(">>    tol........: %.6f", atol + rtol * fabs(b));
        }
        failures += ok ? 0 : 1;
        relativeGap += fabsf(a - b) / (fabsf(b) + EPSILON);
    }

    relativeGap /= size;

    size_t tolFailures = (size_t) (0.0 * size);
    TLLM_LOG_DEBUG("check...%6s : %-50s (failures: %.2f%% atol: %.2e rtol: %.2e rel_gap: %.2e%%)",
        failures <= tolFailures ? "....OK" : "FAILED", name.c_str(), 100. * failures / size, atol, rtol,
        100. * relativeGap);
    return failures <= tolFailures;
}


template <typename T>
void computeProb(T* probs, const T* logits, int batchSize, int vocabSize)
{
    for (int bidx = 0; bidx < batchSize; ++bidx)
    {
        float maxval = -FLT_MAX;
        for (int i = 0; i < vocabSize; ++i)
        {
            float logit = static_cast<float>(logits[bidx * vocabSize + i]);
            if (logit > maxval)
            {
                maxval = logit;
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < vocabSize; ++i)
        {
            sum += expf(static_cast<float>(logits[bidx * vocabSize + i]) - maxval);
        }
        for (int i = 0; i < vocabSize; ++i)
        {
            int idx = bidx * vocabSize + i;
            float logit = static_cast<float>(logits[idx]) - maxval;
            probs[idx] = static_cast<T>(expf(logit) / (sum + EPSILON));
        }
    }
}

template <typename T>
void computeLogProb(T* logprobs, const T* logits, int batchSize, int vocabSize)
{
    for (int bidx = 0; bidx < batchSize; ++bidx)
    {
        float maxval = -FLT_MAX;
        for (int i = 0; i < vocabSize; ++i)
        {
            float logit = static_cast<float>(logits[bidx * vocabSize + i]);
            if (logit > maxval)
            {
                maxval = logit;
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < vocabSize; ++i)
        {
            sum += expf(static_cast<float>(logits[bidx * vocabSize + i]) - maxval);
        }
        for (int i = 0; i < vocabSize; ++i)
        {
            int idx = bidx * vocabSize + i;
            float logit = static_cast<float>(logits[idx]) - maxval;
            logprobs[idx] = static_cast<T>(logit - logf(sum + EPSILON));
        }
    }
}

struct SamplingKernelTestParam
{
    int32_t batchSize;
    int32_t vocabSize;
    uint32_t topK;
    float topP;
    int32_t outputLen;

    SamplingKernelTestParam& setBatchSize(int32_t bs)
    {
        batchSize = bs;
        return *this;
    }

    SamplingKernelTestParam& setVocabSize(int32_t vs)
    {
        vocabSize = vs;
        return *this;
    }

    SamplingKernelTestParam& setTopK(uint32_t tk)
    {
        topK = tk;
        return *this;
    }

    SamplingKernelTestParam& setTopP(float tp)
    {
        topP = tp;
        return *this;
    }

    SamplingKernelTestParam& setOutputLen(int32_t ol)
    {
        outputLen = ol;
        return *this;
    }

    std::string toString() const
    {
        return bitfusion::common::fmtstr("SamplingKernelTestParam[batch=%d, vocab=%d, k=%u, p=%3.1f, output_len=%d]",
            batchSize, vocabSize, topK, topP, outputLen);
    }
};

template <typename T>
class SamplingKernelTest : public testing::Test
{
public:
    using TensorPtr = bitfusion::runtime::ITensor::SharedPtr;

    void SetUp() override;
    void TearDown() override;

    void runTest(const SamplingKernelTestParam& param);

protected:
    virtual size_t getWorkspaceSize(const SamplingKernelTestParam& param)
    {
        throw std::logic_error("Not implemented");
    };

    virtual void callTestedFunction(
        const SamplingKernelTestParam& param, bool hasDiffRuntimeArgs, size_t workspaceSize, TensorPtr& workspaceDevice)
    {
        throw std::logic_error("Not implemented");
    }

    void allocateBuffers(int32_t batchSize, int32_t vocabSize, int32_t maxSeqLen, int32_t outputLen);

    void setupBuffers(int32_t batchSize, int32_t vocabSize, int32_t maxSeqLen, int32_t outputLen, int32_t topK,
        float topP, bool useSkipDecode, bool hasDiffRuntimeArgs, std::mt19937& gen,
        std::uniform_int_distribution<>& endIdsDistr);

    void verifyCurrentStep(int32_t batchSize, int32_t vocabSize, int32_t maxSeqLen, int32_t step, bool greedySearch,
        bool useSkipDecode, bool hasDiffRuntimeArgs, std::vector<bitfusion::kernels::FinishedState>& refFinished,
        std::vector<int32_t>& refSeqLength,
        const std::vector<bitfusion::kernels::FinishedState>& finishedCurrentStep);

private:
    void runTest(const SamplingKernelTestParam& param, bool hasDiffRuntimeArgs, bool useSkipDecode);

protected:
    std::shared_ptr<bitfusion::runtime::BufferManager> mBufferManager;
    std::shared_ptr<bitfusion::runtime::CudaStream> mStream;
    uint32_t mSeed = 0;

    struct cudaDeviceProp mDeviceProp;

    TensorPtr mSeqLengthsHost;
    TensorPtr mSeqLengthsDevice;

    TensorPtr mFinishedHost;
    TensorPtr mFinishedDevice;

    TensorPtr mOutputIdsHost;
    TensorPtr mOutputIdsDevice;

    TensorPtr mProbsHost;
    TensorPtr mProbsDevice;

    TensorPtr mCumLogProbsDevice;
    TensorPtr mOutputLogProbsDevice;
    TensorPtr mTopPIdValsDevice;
    TensorPtr mZeroParentIdsDevice;
    TensorPtr mBeginOffsetsDevice;
    TensorPtr mEndOffsetsDevice;

    TensorPtr mLogitsHost;
    TensorPtr mLogProbsHost;
    TensorPtr mIdsPtrHost;

    TensorPtr mEndIdsHost;
    TensorPtr mEndIdsDevice;

    TensorPtr mTopKsHost;
    TensorPtr mTopKsDevice;

    TensorPtr mTopPsHost;
    TensorPtr mTopPsDevice;

    TensorPtr mSkipDecodeHost;
    TensorPtr mSkipDecodeDevice;

    TensorPtr mExpectedCumLogProbsHost;

    int32_t mMaxTopK;
    float mMaxTopP;

    curandState_t* mCurandStatesDevice;
};

}
