
#include <gtest/gtest.h>

#include "../../common/memoryUtils.h"
#include "../../runtime/bufferManager.h"
#include "../../runtime/gptDecoder.h"
#include "../../runtime/gptModelConfig.h"
#include "../../runtime/worldConfig.h"

using namespace bitfusion::runtime;

namespace tc = bitfusion::common;

namespace
{

void testDecoder(nvinfer1::DataType const dtype, SamplingConfig const& samplingConfig)
{
    SizeType constexpr tensorParallelism{1};
    SizeType constexpr pipelineParallelism{1};
    SizeType constexpr localRank{0};
    WorldConfig const worldConfig{tensorParallelism, pipelineParallelism, localRank};

    SizeType constexpr vocabSize{51200};
    SizeType constexpr nbLayers{2};
    SizeType constexpr nbHeads{16};
    SizeType constexpr hiddenSize{1024};
    GptModelConfig modelConfig{vocabSize, nbLayers, nbHeads, hiddenSize, dtype};
    modelConfig.useGptAttentionPlugin(false);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
    auto decoder = IGptDecoder::create(modelConfig.getDataType(), vocabSize, vocabSizePadded, streamPtr);
    ASSERT_TRUE(static_cast<bool>(decoder));

    auto const beamWidth = samplingConfig.beamWidth;
    SizeType constexpr batchSize{4};

    SizeType constexpr maxInputLength{8};
    SizeType constexpr maxNewTokens{2};
    auto constexpr maxSeqLength = maxInputLength + maxNewTokens;
    decoder->setup(samplingConfig, batchSize, maxSeqLength);

    auto logits = std::shared_ptr(
        manager.gpu(ITensor::makeShape({batchSize, beamWidth, vocabSizePadded}), modelConfig.getDataType()));
    manager.setZero(*logits);

    int constexpr endId{50257};
    std::vector<int> const endIdsVec(batchSize * beamWidth, endId);
    auto endIds
        = std::shared_ptr(manager.copyFrom(endIdsVec, ITensor::makeShape({batchSize, beamWidth}), MemoryType::kGPU));

    DecodingInput inputs{maxInputLength, maxSeqLength, batchSize, logits, endIds};
    std::vector<std::int32_t> sequenceLimitLengthsVec(batchSize, maxSeqLength);
    inputs.sequenceLimitLength
        = manager.copyFrom(sequenceLimitLengthsVec, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    if (beamWidth > 1)
    {
        auto srcCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32));
        manager.setZero(*srcCacheIndirection);
        inputs.cacheIndirection = srcCacheIndirection;
    }

    auto outputIds = std::shared_ptr(
        manager.gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32));
    manager.setZero(*outputIds);
    DecodingOutput outputs{outputIds};
    auto newTokens
        = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32));
    manager.setZero(*newTokens);
    outputs.newTokens = newTokens;

    std::vector<int> sequenceLengthsVec(batchSize * beamWidth, maxInputLength);
    outputs.lengths
        = manager.copyFrom(sequenceLengthsVec, ITensor::makeShape({batchSize, beamWidth}), MemoryType::kGPU);
    outputs.finished = manager.gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kBOOL);
    inputs.finished = ITensor::view(outputs.finished);
    manager.setZero(*outputs.finished);
    outputs.finishedSum = BufferManager::pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    auto* finishedSumHost = bufferCast<std::int32_t>(*outputs.finishedSum);
    *finishedSumHost = -1;

    if (beamWidth > 1)
    {
        auto tgtCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32));
        manager.setZero(*tgtCacheIndirection);
        outputs.cacheIndirection = tgtCacheIndirection;

        auto cumLogProbs
            = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kFLOAT));
        manager.setZero(*cumLogProbs);
        outputs.cumLogProbs = cumLogProbs;

        auto parentIds = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32));
        manager.setZero(*parentIds);
        outputs.parentIds = parentIds;
    }

    EXPECT_FALSE(decoder->forward(outputs, inputs));
    inputs.step += 1;
    EXPECT_EQ(*finishedSumHost, 0);

    auto outputsIdsHost = manager.copyFrom(*outputs.ids, MemoryType::kCPU);
    auto output = bufferCast<std::int32_t>(*outputsIdsHost);
    manager.getStream().synchronize();

    for (auto b = 0; b < batchSize; ++b)
    {
        for (auto bw = 0; bw < beamWidth; ++bw)
        {
            auto const result = (beamWidth == 1) ? 1023 : bw;

            bool anyMismatch = false;
            for (auto i = 0; i < maxInputLength; ++i)
            {
                auto const outputIndex = tc::flat_index3(b, bw, i, beamWidth, maxSeqLength);
                EXPECT_EQ(output[outputIndex], 0) << " b: " << b << " bw: " << bw << " i: " << i;
                anyMismatch |= (output[outputIndex] != 0);
            }
            for (auto i = 0; i < maxNewTokens - 1; ++i)
            {
                auto const index = tc::flat_index3(b, bw, maxInputLength + i, beamWidth, maxSeqLength);
                EXPECT_EQ(output[index], result) << " b: " << b << " bw: " << bw << " i: " << i;
                anyMismatch |= (output[index] != result);
            }
            ASSERT_FALSE(anyMismatch);
        }
    }

    EXPECT_TRUE(decoder->forward(outputs, inputs));
    EXPECT_EQ(*finishedSumHost, outputs.finished->getSize());
}

}

class ParamTest : public ::testing::TestWithParam<std::tuple<nvinfer1::DataType, SizeType>>
{
};

TEST_P(ParamTest, Test)
{
    nvinfer1::DataType const dtype{std::get<0>(GetParam())};
    SizeType const beamWidth{std::get<1>(GetParam())};
    SamplingConfig const samplingConfig{beamWidth};

    testDecoder(dtype, samplingConfig);
}

INSTANTIATE_TEST_SUITE_P(GptDecoderTest, ParamTest,
    testing::Combine(testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF), testing::Values(1, 3)),
    [](const testing::TestParamInfo<ParamTest::ParamType>& info)
    {
        std::string name{std::get<0>(info.param) == nvinfer1::DataType::kFLOAT ? "Float" : "Half"};
        auto const beamWidth = std::get<1>(info.param);
        name.append(beamWidth == 1 ? "Sampling" : "BeamWidth" + std::to_string(beamWidth));
        return name;
    });
