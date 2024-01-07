#include <gtest/gtest.h>

#include "../../common/memoryUtils.h"
#include "../../kernels/decodingCommon.h"
#include "../../kernels/stopCriteriaKernels.h"
#include "../../runtime/bufferManager.h"
#include <algorithm>
#include <curand_kernel.h>
#include <random>

namespace tk = bitfusion::kernels;
namespace tc = bitfusion::common;

using namespace bitfusion::runtime;

namespace
{

class StopCriteriaKernelsTest : public testing::Test
{
public:
    using TensorPtr = bitfusion::runtime::ITensor::SharedPtr;

    void SetUp() override
    {
        mStream = std::make_shared<bitfusion::runtime::CudaStream>();
        mBufferManager = std::make_shared<bitfusion::runtime::BufferManager>(mStream);
    }

    void TearDown() override {}

    void initData(SizeType seed, const std::vector<std::vector<std::vector<SizeType>>>& stopWords,
        SizeType stopWordsLen, SizeType batchSize, SizeType beamWidth)
    {
        std::mt19937 generator(seed);
        std::uniform_int_distribution<int> seqLenDistr(0, mMaxSeqLen);

        mSequenceLengths
            = mBufferManager->pinned(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32);
        mSequenceLengthLimits = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
        mFinished = mBufferManager->pinned(
            ITensor::makeShape({batchSize, beamWidth}), TRTDataType<tk::FinishedState::UnderlyingType>::value);
        mFinishedSum = mBufferManager->pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

        mOutputIds = mBufferManager->pinned(
            ITensor::makeShape({batchSize, beamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);
        mOutputIdsPtr = mBufferManager->pinned(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT64);

        mParentIds = mBufferManager->pinned(
            ITensor::makeShape({batchSize, beamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);
        mParentIdsPtr = mBufferManager->pinned(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT64);

        mRefOutputIds = mBufferManager->pinned(
            ITensor::makeShape({batchSize, beamWidth, mMaxSeqLen}), nvinfer1::DataType::kINT32);

        mStopWords
            = mBufferManager->pinned(ITensor::makeShape({batchSize, 2, stopWordsLen}), nvinfer1::DataType::kINT32);

        auto sequenceLengthsPtr = bufferCast<SizeType>(*mSequenceLengths);
        auto sequenceLengthLimitsPtr = bufferCast<SizeType>(*mSequenceLengthLimits);
        auto finishedPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished));
        auto finishedSumPtr = bufferCast<SizeType>(*mFinishedSum);

        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            for (SizeType ri = 0; ri < beamWidth; ri++)
            {
                sequenceLengthsPtr[bi * beamWidth + ri]
                    = stopWordsLen == 0 ? seqLenDistr(generator) : mMaxSeqLen - (bi + ri) % mMaxSeqLen;
                finishedPtr[bi * beamWidth + ri] = tk::FinishedState::empty();
            }
        }
        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            sequenceLengthLimitsPtr[bi] = stopWordsLen == 0 ? seqLenDistr(generator) : mMaxSeqLen - bi % mMaxSeqLen;
        }
        finishedSumPtr[0] = 0;

        auto outputIdsPtrsData = reinterpret_cast<void**>(bufferCast<int64_t>(*mOutputIdsPtr));
        auto parentIdsPtrsData = reinterpret_cast<void**>(bufferCast<int64_t>(*mParentIdsPtr));
        auto outputIdsData = bufferCast<int32_t>(*mOutputIds);
        auto refOutputIdsData = bufferCast<int32_t>(*mRefOutputIds);
        auto parentIdsData = bufferCast<int32_t>(*mParentIds);

        for (SizeType bi = 0; bi < batchSize; bi++)
        {
            for (SizeType ri = 0; ri < beamWidth; ri++)
            {
                for (SizeType si = 0; si < mMaxSeqLen; si++)
                {
                    auto const idx = tc::flat_index3(bi, ri, si, beamWidth, mMaxSeqLen);
                    outputIdsData[idx] = ri * mMaxSeqLen + si;
                    parentIdsData[idx] = 0;
                }
            }
        }

        for (SizeType bi = 0; bi < batchSize; bi++)
        {
            outputIdsPtrsData[bi] = outputIdsData + bi * beamWidth * mMaxSeqLen;
            parentIdsPtrsData[bi] = parentIdsData + bi * beamWidth * mMaxSeqLen;
        }

        auto stopWordsData = bufferCast<int32_t>(*mStopWords);
        std::fill(stopWordsData, stopWordsData + batchSize * 2 * stopWordsLen, -1);
        for (SizeType bi = 0; bi < stopWords.size(); bi++)
        {
            SizeType totalLen = 0;
            for (SizeType wi = 0; wi < stopWords[bi].size(); ++wi)
            {
                for (SizeType si = 0; si < stopWords[bi][wi].size(); ++si)
                {
                    stopWordsData[bi * 2 * stopWordsLen + 0 * stopWordsLen + totalLen + si] = stopWords[bi][wi][si];
                }
                totalLen += stopWords[bi][wi].size();
                if (totalLen > 0)
                {
                    stopWordsData[bi * 2 * stopWordsLen + 1 * stopWordsLen + wi] = totalLen;
                }
            }
            if (stopWords[bi].size() == totalLen)
            {
                stopWordsData[bi * 2 * stopWordsLen + 1 * stopWordsLen + totalLen] = totalLen + 1;
            }
        }
    }

    void verifyMaxSeqLenStopCriteriaResults(SizeType seed, SizeType batchSize, SizeType beamWidth)
    {
        mStream->synchronize();

        auto sequenceLengthsPtr = bufferCast<SizeType>(*mSequenceLengths);
        auto sequenceLengthLimitsPtr = bufferCast<SizeType>(*mSequenceLengthLimits);
        auto finishedPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished));
        auto finishedSumPtr = bufferCast<SizeType>(*mFinishedSum);

        int32_t refSumFinished = 0;
        for (SizeType bi = 0; bi < batchSize * beamWidth; ++bi)
        {
            auto const batchIdx = bi / beamWidth;
            auto const beamIdx = bi % beamWidth;
            const auto limitExceeded = sequenceLengthsPtr[bi] >= sequenceLengthLimitsPtr[batchIdx];
            refSumFinished += limitExceeded;
            if (limitExceeded)
            {
                EXPECT_TRUE(finishedPtr[bi].isFinishedMaxLength())
                    << " batchIdx: " << batchIdx << " beamIdx: " << beamIdx << " seed: " << seed;
            }
        }
        EXPECT_EQ(refSumFinished, finishedSumPtr[0]);
    }

    bool isSubsequence(const SizeType* sequence, SizeType n, const std::vector<int>& subsequence)
    {
        auto it = std::search(sequence, sequence + n, subsequence.begin(), subsequence.end());
        return it != sequence + n;
    }

    void verifyStopWordsStopCriteriaResults(SizeType seed,
        const std::vector<std::vector<std::vector<SizeType>>>& stopWords, SizeType stopWordsLen, SizeType batchSize,
        SizeType beamWidth)
    {
        mStream->synchronize();

        auto outputIdsData = bufferCast<int32_t>(*mOutputIds);
        auto finishedPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished));
        auto sequenceLengthsPtr = bufferCast<SizeType>(*mSequenceLengths);

        for (SizeType bi = 0; bi < batchSize; bi++)
        {
            for (SizeType bwi = 0; bwi < beamWidth; bwi++)
            {
                auto outputIdsBatchBeam = outputIdsData + bi * beamWidth * mMaxSeqLen + bwi * mMaxSeqLen;
                bool found = false;
                for (SizeType wi = 0; wi < stopWords[bi].size(); ++wi)
                {
                    auto const wordLen = stopWords[bi][wi].size();
                    auto const seqLen = sequenceLengthsPtr[bi * beamWidth + bwi];
                    auto const offset = seqLen - wordLen;
                    found |= isSubsequence(outputIdsBatchBeam + offset, wordLen, stopWords[bi][wi]);
                    if (found)
                    {
                        EXPECT_TRUE(finishedPtr[bi * beamWidth + bwi].isFinishedStopWords());
                        break;
                    }
                }
                if (!found)
                {
                    EXPECT_FALSE(finishedPtr[bi * beamWidth + bwi].isFinished());
                }
            }
        }
    }

    void runStopWordsCriteriaTest(
        const std::vector<std::vector<std::vector<SizeType>>>& stopWords, SizeType batchSize, SizeType beamWidth)
    {
        SizeType maxStopWordsLen = 0;
        for (const auto& batchStopWords : stopWords)
        {
            SizeType stopWordsLen = 0;
            for (const auto& words : batchStopWords)
            {
                stopWordsLen += words.size();
            }
            if (stopWordsLen == batchStopWords.size())
            {
                stopWordsLen += 1;
            }
            maxStopWordsLen = std::max(maxStopWordsLen, stopWordsLen);
        }

        initData(0, stopWords, maxStopWordsLen, batchSize, beamWidth);

        tk::invokeStopWordsCriterion(reinterpret_cast<const int**>(bufferCast<int64_t>(*mOutputIdsPtr)),
            reinterpret_cast<const int**>(bufferCast<int64_t>(*mParentIdsPtr)), bufferCast<SizeType>(*mStopWords),
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished)),
            bufferCast<SizeType>(*mSequenceLengths), maxStopWordsLen, batchSize, beamWidth, mMaxSeqLen, mStream->get());

        verifyStopWordsStopCriteriaResults(0, stopWords, maxStopWordsLen, batchSize, beamWidth);
    }

    void runMaxLengthCriteriaTest(SizeType seed, SizeType batchSize, SizeType beamWidth)
    {
        initData(seed, {}, 0, batchSize, beamWidth);

        tk::invokeLengthCriterion(
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinished)),
            bufferCast<SizeType>(*mFinishedSum),
            reinterpret_cast<const uint32_t*>(bufferCast<SizeType>(*mSequenceLengthLimits)),
            bufferCast<SizeType>(*mSequenceLengths), batchSize, beamWidth, mStream->get());

        verifyMaxSeqLenStopCriteriaResults(seed, batchSize, beamWidth);
    }

protected:
    std::shared_ptr<bitfusion::runtime::BufferManager> mBufferManager;
    std::shared_ptr<bitfusion::runtime::CudaStream> mStream;

    TensorPtr mSequenceLengths;
    TensorPtr mSequenceLengthLimits;
    TensorPtr mFinished;
    TensorPtr mFinishedSum;

    TensorPtr mOutputIds;
    TensorPtr mRefOutputIds;
    TensorPtr mOutputIdsPtr;
    TensorPtr mParentIds;
    TensorPtr mParentIdsPtr;
    TensorPtr mStopWords;

    static constexpr SizeType mMaxSeqLen{16};
    static constexpr SizeType mVocabSize{32};
};

TEST_F(StopCriteriaKernelsTest, maxLengthCriteriaBS1BW1Test)
{
    constexpr SizeType seeds = 64;
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    for (SizeType seed = 0; seed < seeds; ++seed)
    {
        this->runMaxLengthCriteriaTest(seed, batchSize, beamWidth);
    }
}

TEST_F(StopCriteriaKernelsTest, maxLengthCriteriaBS1BW2Test)
{
    constexpr SizeType seeds = 64;
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 2;
    for (SizeType seed = 0; seed < seeds; ++seed)
    {
        this->runMaxLengthCriteriaTest(seed, batchSize, beamWidth);
    }
}

TEST_F(StopCriteriaKernelsTest, maxLengthCriteriaBS1024BW1Test)
{
    constexpr SizeType seeds = 64;
    constexpr SizeType batchSize = 1024;
    constexpr SizeType beamWidth = 1;
    for (SizeType seed = 0; seed < seeds; ++seed)
    {
        this->runMaxLengthCriteriaTest(seed, batchSize, beamWidth);
    }
}

TEST_F(StopCriteriaKernelsTest, maxLengthCriteriaBS1024BW2Test)
{
    constexpr SizeType seeds = 64;
    constexpr SizeType batchSize = 1024;
    constexpr SizeType beamWidth = 2;
    for (SizeType seed = 0; seed < seeds; ++seed)
    {
        this->runMaxLengthCriteriaTest(seed, batchSize, beamWidth);
    }
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS1SingleTokenSingleWordTest)
{
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    this->runStopWordsCriteriaTest({{{2}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS1SingleTokenMultipleWordsTest)
{
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    this->runStopWordsCriteriaTest({{{145}, {4}, {1}, {15}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS1MultipleTokensSingleWordTest)
{
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    this->runStopWordsCriteriaTest({{{2, 3}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS1MultipleTokensMultipleWordsMatchTest)
{
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    this->runStopWordsCriteriaTest({{{1, 4}, {2, 3}, {13, 14, 15}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS1MultipleTokensMultipleWordsNotMatchTest)
{
    constexpr SizeType batchSize = 1;
    constexpr SizeType beamWidth = 1;
    this->runStopWordsCriteriaTest({{{1, 4}, {2, 3}, {12, 14, 15}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS4MultipleTokensMultipleWordsTest)
{
    constexpr SizeType batchSize = 4;
    constexpr SizeType beamWidth = 1;
    this->runStopWordsCriteriaTest({{{2}}, {{}}, {{15}, {12, 13}}, {{1}, {8, 9}}}, batchSize, beamWidth);
}

TEST_F(StopCriteriaKernelsTest, stopWordsCriteriaBS4BW2MultipleTokensMultipleWordsTest)
{
    constexpr SizeType batchSize = 4;
    constexpr SizeType beamWidth = 2;
    this->runStopWordsCriteriaTest({{{2}}, {{}}, {{11}, {12, 13}}, {{27}, {11, 12}}}, batchSize, beamWidth);
}

}
