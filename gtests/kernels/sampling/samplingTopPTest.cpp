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
class TopPSamplingKernelTest : public SamplingKernelTest<T>
{

protected:
    const int32_t endId = 0;
    using SamplingKernelTest<T>::mSeed;
    using SamplingKernelTest<T>::mStream;
    using SamplingKernelTest<T>::mBufferManager;

private:
    size_t getWorkspaceSize(const SamplingKernelTestParam& params) override
    {
        size_t workspaceSize;
        size_t cubTempStorageSize;
        tk::invokeBatchTopPSampling<T>(nullptr,
            workspaceSize, cubTempStorageSize,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            bufferCast<int32_t>(*this->mTopPIdValsDevice), bufferCast<int32_t>(*this->mEndOffsetsDevice),
            bufferCast<int32_t>(*this->mBeginOffsetsDevice), this->mCurandStatesDevice, params.batchSize,
            params.vocabSize, nullptr, this->mMaxTopP, bufferCast<float>(*this->mTopPsDevice), this->mStream->get(),
            nullptr);
        return workspaceSize;
    }

    void callTestedFunction(const SamplingKernelTestParam& params, bool hasDiffRuntimeArgs, size_t workspaceSize,
        bitfusion::runtime::ITensor::SharedPtr& workspaceDevice) override
    {
        size_t cubTempStorageSize;
        tk::invokeBatchTopPSampling<T>(nullptr,
            workspaceSize, cubTempStorageSize,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            bufferCast<int32_t>(*this->mTopPIdValsDevice), bufferCast<int32_t>(*this->mEndOffsetsDevice),
            bufferCast<int32_t>(*this->mBeginOffsetsDevice), this->mCurandStatesDevice, params.batchSize,
            params.vocabSize, nullptr, this->mMaxTopP, bufferCast<float>(*this->mTopPsDevice), this->mStream->get(),
            nullptr);

        tk::invokeTopPInitialize(bufferCast<int32_t>(*this->mTopPIdValsDevice),
            bufferCast<int32_t>(*this->mEndOffsetsDevice), bufferCast<int32_t>(*this->mBeginOffsetsDevice),
            params.batchSize, params.vocabSize, this->mStream->get());

        tk::invokeBatchTopPSampling<T>(workspaceDevice->data(), workspaceSize, cubTempStorageSize,
            bufferCast<int*>(*this->mIdsPtrHost), bufferCast<int32_t>(*this->mSeqLengthsDevice),
            reinterpret_cast<bitfusion::kernels::FinishedState*>(
                bufferCast<bitfusion::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice)),
            reinterpret_cast<bitfusion::kernels::FinishedState*>(
                bufferCast<bitfusion::kernels::FinishedState::UnderlyingType>(*this->mFinishedDevice)),
            bufferCast<float>(*this->mCumLogProbsDevice), bufferCast<float>(*this->mOutputLogProbsDevice),
            bufferCast<T>(*this->mProbsDevice), bufferCast<int32_t>(*this->mTopPIdValsDevice),
            bufferCast<int32_t>(*this->mEndOffsetsDevice), bufferCast<int32_t>(*this->mBeginOffsetsDevice),
            this->mCurandStatesDevice, params.batchSize, params.vocabSize, bufferCast<int32_t>(*this->mEndIdsDevice),
            this->mMaxTopP, hasDiffRuntimeArgs ? bufferCast<float>(*this->mTopPsDevice) : nullptr, this->mStream->get(),
            bufferCast<bool>(*this->mSkipDecodeDevice));
    }
};

TYPED_TEST_SUITE(TopPSamplingKernelTest, FloatAndHalfTypes);

TYPED_TEST(TopPSamplingKernelTest, CorrectnessSmallP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(0.2f).setOutputLen(1));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeP)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(0.9f).setOutputLen(1));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessAncestral)
{
    this->runTest(SamplingKernelTestParam().setBatchSize(6).setVocabSize(4).setTopK(0).setTopP(1.0f).setOutputLen(1));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeVocabSmallP)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopK(0).setTopP(0.2f).setOutputLen(16));
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeVocabLargeP)
{
    this->runTest(
        SamplingKernelTestParam().setBatchSize(32).setVocabSize(51200).setTopK(0).setTopP(0.9f).setOutputLen(16));
};

class TopPSamplingKernelUtilsTest : public SamplingKernelTest<float>
{
};

TEST_F(TopPSamplingKernelUtilsTest, invokeTopPInitialize)
{
    const int32_t batchSize = 8;
    const int32_t vocabSize = 256;

    const auto topPIdValsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize, vocabSize}), nvinfer1::DataType::kINT32);
    const auto beginOffsetsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);
    const auto endOffsetsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);

    tk::invokeTopPInitialize(bufferCast<int32_t>(*topPIdValsDevice), bufferCast<int32_t>(*endOffsetsDevice),
        bufferCast<int32_t>(*beginOffsetsDevice), batchSize, vocabSize, this->mStream->get());

    const auto topPIdValsHost = this->mBufferManager->copyFrom(*topPIdValsDevice, MemoryType::kCPU);
    const auto endOffsetsHost = this->mBufferManager->copyFrom(*endOffsetsDevice, MemoryType::kCPU);
    const auto beginOffsetsHost = this->mBufferManager->copyFrom(*beginOffsetsDevice, MemoryType::kCPU);

    this->mStream->synchronize();

    const auto topPIdValsHostPtr = bufferCast<int32_t>(*topPIdValsHost);
    const auto endOffsetsHostPtr = bufferCast<int32_t>(*endOffsetsHost);
    const auto beginOffsetsHostPtr = bufferCast<int32_t>(*beginOffsetsHost);

    for (int32_t bi = 0; bi < batchSize + 1; ++bi)
    {
        EXPECT_EQ(endOffsetsHostPtr[bi], bi * vocabSize);
        EXPECT_EQ(beginOffsetsHostPtr[bi], bi * vocabSize);
    }
    for (int32_t bi = 0; bi < batchSize; ++bi)
    {
        for (int32_t vi = 0; vi < vocabSize; ++vi)
        {
            EXPECT_EQ(topPIdValsHostPtr[bi * vocabSize + vi], vi);
        }
    }
};

}
