#include "samplingTest.h"
#include <random>

namespace tc = bitfusion::common;
namespace tk = bitfusion::kernels;
namespace trk = bitfusion::runtime::kernels;

using namespace bitfusion::runtime;
using namespace bitfusion::tests::kernels::sampling;

namespace
{

template <typename T>
class TopKSamplingKernelTest : public SamplingKernelTest<T>
{

protected:
    const int32_t endId = 0;
    using SamplingKernelTest<T>::mSeed;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mBufferManager;

    size_t getWorkspaceSize(const SamplingKernelTestParam& params) override
    {
        size_t workspaceSize;
        tk::invokeTopKSampling<T>(nullptr, workspaceSize, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, this->mMaxTopK, 1.0f, params.vocabSize, nullptr, this->mStream->get(), params.batchSize, nullptr);
        return workspaceSize;
    }

    void callTestedFunction(const SamplingKernelTestParam& params, bool hasDiffRuntimeArgs, size_t workspaceSize,
        bitfusion::runtime::ITensor::SharedPtr& workspaceDevice) override
    {
        tk::invokeBatchTopKSampling(workspaceDevice->data(), workspaceSize,
            bufferCast<T>(*this->mProbsDevice), bufferCast<int*>(*this->mIdsPtrHost),
            bufferCast<int32_t>(*this->mSeqLengthsDevice),
            reinterpret_cast<bitfusion::kernels::FinishedState*>(
                bufferCast<bitfusion::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice)),
            reinterpret_cast<bitfusion::kernels::FinishedState*>(
                bufferCast<bitfusion::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice)),
            bufferCast<float>(*this->mCumLogProbsDevice), bufferCast<float>(*this->mOutputLogProbsDevice),
            this->mCurandStatesDevice, this->mMaxTopK,
            hasDiffRuntimeArgs ? bufferCast<int32_t>(*this->mTopKsDevice) : nullptr, params.topP,
            hasDiffRuntimeArgs ? bufferCast<float>(*this->mTopPsDevice) : nullptr, params.vocabSize,
            bufferCast<int32_t>(*this->mEndIdsDevice), this->mStream->get(), params.batchSize,
            bufferCast<bool>(*this->mSkipDecodeDevice));
    }
};

TYPED_TEST_SUITE(TopKSamplingKernelTest, FloatAndHalfTypes);

TYPED_TEST(TopKSamplingKernelTest, CorrectnessGreedy)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(1).setTopP(1.0f).setOutputLen(1));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessGreedyLarge)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(16).setVocabSize(51200).setTopK(1).setTopP(1.0f).setOutputLen(8));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessAncestral)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(4).setTopP(1.0f).setOutputLen(1));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessLargeK63)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(16).setVocabSize(51200).setTopK(63).setTopP(1.0f).setOutputLen(8));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessLargeK1024)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(16).setVocabSize(51200).setTopK(1024).setTopP(1.0f).setOutputLen(8));
};

TYPED_TEST(TopKSamplingKernelTest, CorrectnessTopKTopP)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(16).setVocabSize(4000).setTopK(63).setTopP(0.3f).setOutputLen(8));
};

TYPED_TEST(TopKSamplingKernelTest, NotSupportedLargerThanK1024)
{
    EXPECT_THROW(
        this->runTest(
            SamplingKernelTestParam().setBatchSize(16).setVocabSize(4000).setTopK(1025).setTopP(1.0f).setOutputLen(8)),
        std::domain_error);
};

}
