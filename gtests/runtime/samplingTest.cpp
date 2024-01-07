#include <gtest/gtest.h>

#include "../../common/cudaAllocator.h"
#include "../../common/memoryUtils.h"
#include "../../common/tensor.h"
#include "../../common/tensorConversion.h"
#include "../../kernels/decodingKernels.h"
#include "../../layers/dynamicDecodeLayer.h"
#include "../../runtime/bufferManager.h"
#include "../../runtime/cudaStream.h"
#include "../../runtime/tllmLogger.h"

using namespace bitfusion::runtime;

namespace tc = bitfusion::common;
namespace tl = bitfusion::layers;

class SamplingTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();

        if (mDeviceCount == 0)
            GTEST_SKIP() << "No GPUs found";

        mLogger = std::make_shared<TllmLogger>();
    }

    void TearDown() override {}

    int mDeviceCount;
    std::shared_ptr<nvinfer1::ILogger> mLogger{};
};

typename tl::DynamicDecodeLayer<float>::OutputParams dynamicDecodeTest(BufferManager& manager,
    tc::CudaAllocator& allocator, size_t vocabSize, size_t vocabSizePadded, size_t batchSize, size_t beamWidth,
    int step, int ite, int maxInputLength, size_t maxSeqLength, int localBatchSize, std::vector<int>& cpuOutputIds,
    std::vector<float> cpuLogits, int noRepeatNgramSizeValue = 0)
{
    constexpr int endId = 1;
    constexpr bool isFreeBufferAfterForward{false};
    cudaDeviceProp prop;
    tc::check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    std::vector<int> cpuEndIds(batchSize, endId);
    std::vector<int> cpuSequenceLengths(batchSize, maxInputLength);
    std::vector<int> cpuNoRepeatNgramSize(batchSize, noRepeatNgramSizeValue);

    float* gpuLogits = nullptr;
    int* gpuEndIds = nullptr;
    int* gpuOutputIds = nullptr;
    int* gpuSequenceLengths = nullptr;
    int* gpuNewTokens = nullptr;
    int* gpuNoRepeatNgramSize = nullptr;

    gpuLogits = allocator.reMalloc(gpuLogits, batchSize * beamWidth * vocabSizePadded * sizeof(float));
    gpuEndIds = allocator.reMalloc(gpuEndIds, batchSize * sizeof(int));
    gpuOutputIds = allocator.reMalloc(gpuOutputIds, batchSize * beamWidth * maxSeqLength * sizeof(int));
    gpuSequenceLengths = allocator.reMalloc(gpuSequenceLengths, batchSize * sizeof(int));
    gpuNewTokens = allocator.reMalloc(gpuNewTokens, batchSize * beamWidth * sizeof(int));
    gpuNoRepeatNgramSize = allocator.reMalloc(gpuNoRepeatNgramSize, batchSize * sizeof(int));

    cudaMemcpy(gpuLogits, cpuLogits.data(), cpuLogits.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuEndIds, cpuEndIds.data(), cpuEndIds.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(
        gpuSequenceLengths, cpuSequenceLengths.data(), cpuSequenceLengths.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuOutputIds, cpuOutputIds.data(), cpuOutputIds.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuNoRepeatNgramSize, cpuNoRepeatNgramSize.data(), cpuNoRepeatNgramSize.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    tc::Tensor logits{tc::MEMORY_GPU, tc::TYPE_FP32, {batchSize, beamWidth, vocabSizePadded}, gpuLogits};
    tc::Tensor endIds{tc::MEMORY_GPU, tc::TYPE_INT32, {batchSize}, gpuEndIds};
    tc::Tensor outputIds{tc::MEMORY_GPU, tc::TYPE_INT32, {batchSize, beamWidth, maxSeqLength}, gpuOutputIds};
    tc::Tensor sequenceLengths{tc::MEMORY_GPU, tc::TYPE_INT32, {batchSize}, gpuSequenceLengths};
    tc::Tensor newTokens{tc::MEMORY_GPU, tc::TYPE_INT32, {batchSize}, gpuNewTokens};
    tc::Tensor noRepeatNgramSize{tc::MEMORY_GPU, tc::TYPE_INT32, {batchSize}, gpuNoRepeatNgramSize};

    auto ddLayer = tl::DynamicDecodeLayer<float>(
        vocabSize, vocabSizePadded, manager.getStream().get(), &allocator, isFreeBufferAfterForward, &prop);

    typename tl::DynamicDecodeLayer<float>::SetupParams setupParams;

    ddLayer.setup(batchSize, beamWidth, setupParams);

    typename tl::DynamicDecodeLayer<float>::ForwardParams forwardParams(
        step, ite, maxInputLength, static_cast<int>(maxSeqLength), localBatchSize, logits, endIds);
    forwardParams.no_repeat_ngram_size = noRepeatNgramSize;

    typename tl::DynamicDecodeLayer<float>::OutputParams outputParams(outputIds);
    outputParams.sequence_length = sequenceLengths;
    outputParams.newTokens = newTokens;

    ddLayer.forward(outputParams, forwardParams);

    return outputParams;
}

TEST_F(SamplingTest, SamplingWithNoRepeatNGramSize)
{
    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);
    tc::CudaAllocator allocator(manager);

    constexpr size_t vocabSize{200};
    constexpr size_t vocabSizePadded{256};
    constexpr size_t batchSize{1};
    constexpr size_t beamWidth{1};
    constexpr int step{8};
    constexpr int ite{0};
    constexpr int maxInputLength{8};
    constexpr int maxSeqLength{9};
    constexpr int localBatchSize{batchSize};
    constexpr int noRepeatNgramSize{3};

    std::vector<int> cpuOutputIds(batchSize * beamWidth * maxSeqLength);
    int ids[maxInputLength] = {10, 11, 12, 40, 41, 42, 40, 41};
    for (int i = 0; i < maxInputLength; i++)
    {
        cpuOutputIds[i] = ids[i];
    }

    std::vector<float> cpuLogits(batchSize * beamWidth * vocabSizePadded, 1.0);
    cpuLogits[42] = 10.0;
    cpuLogits[43] = 5.0;

    auto outputParams = dynamicDecodeTest(manager, allocator, vocabSize, vocabSizePadded, batchSize, beamWidth, step,
        ite, maxInputLength, maxSeqLength, localBatchSize, cpuOutputIds, cpuLogits, noRepeatNgramSize);

    cudaMemcpy(cpuOutputIds.data(), outputParams.output_ids.getPtr<int>(), cpuOutputIds.size() * sizeof(int),
        cudaMemcpyDeviceToHost);

    EXPECT_EQ(cpuOutputIds[maxSeqLength - 1], 43);
}
