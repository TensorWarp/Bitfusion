
#include <cstdlib>
#include <gtest/gtest.h>

#include "../../common/memoryUtils.h"
#include "../../kernels/unfusedAttentionKernels.h"
#include "../../runtime/bufferManager.h"
#include <cuda_fp8.h>

using namespace bitfusion::runtime;
using namespace bitfusion::kernels;

namespace tc = bitfusion::common;

namespace
{

template <typename T>
void randomInitVector(std::vector<T>& vec, float range)
{
    for (auto& v : vec)
    {
        float r = range * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

        if (std::is_same_v<T, float>)
        {
            v = r;
        }
        else if (std::is_same_v<T, half>)
        {
            v = __float2half(r);
        }
    }
}

template void randomInitVector(std::vector<float>& vec, float scale);
template void randomInitVector(std::vector<half>& vec, float scale);

std::vector<void*> pointerArrayFromPageTable(const std::unordered_map<int, int>& pageTable, void* memoryPool,
    int32_t batchSize, int32_t blocksPerSeq, int32_t blockSizeInBytes, int32_t blocksPerPool)
{
    const auto pointerArrayElts = pageTable.size();
    std::vector<void*> pointers(2 * pointerArrayElts);
    for (int i = 0; i < pointerArrayElts; ++i)
    {
        const int pageIdx = pageTable.find(i)->second;
        auto kPtr = reinterpret_cast<void*>(reinterpret_cast<int8_t*>(memoryPool) + pageIdx * blockSizeInBytes);
        auto vPtr = reinterpret_cast<void*>(
            reinterpret_cast<int8_t*>(memoryPool) + pageIdx * blockSizeInBytes + blocksPerPool * blockSizeInBytes);
        const int batchIdx = i / batchSize;
        const int seqIdx = i % blocksPerSeq;
        pointers[batchIdx * blocksPerSeq * 2 + 0 * blocksPerSeq + seqIdx] = kPtr;
        pointers[batchIdx * blocksPerSeq * 2 + 1 * blocksPerSeq + seqIdx] = vPtr;
    }
    return pointers;
}

template <typename T, typename T_DST>
T_DST castTo(T value)
{
    return value;
}

template <>
int8_t castTo(float value)
{
    const auto clipped = std::min(127.f, std::max(value, -128.f));
    const auto rounded = std::round(clipped);
    return static_cast<int8_t>(rounded);
}

template <>
__nv_fp8_e4m3 castTo(float value)
{
    return __nv_fp8_e4m3(value);
}

template <>
float castTo(__nv_fp8_e4m3 value)
{
    return float(value);
}

template <typename T, typename T_DST, typename KVCacheBuffer>
void verifyKVTransposed(int batchSize, int headsNum, int dimsPerHead, int seqLen, int maxSeqLen, KVCacheBuffer& buffer,
    const std::vector<T>& refKCacheVec, const std::vector<T>& vTransposedCacheVec, bool b8bitKVCache,
    float kvScaleOrigQuant)
{
    for (int bi = 0; bi < batchSize; ++bi)
    {
        for (int hi = 0; hi < headsNum; ++hi)
        {
            constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;
            for (int di = 0; di < dimsPerHead / X_ELEMS; ++di)
            {
                for (int li = 0; li < seqLen; ++li)
                {
                    const T_DST* blockKPtr = reinterpret_cast<T_DST*>(buffer.getKBlockPtr(bi, li));
                    const T_DST* blockVPtr = reinterpret_cast<T_DST*>(buffer.getVBlockPtr(bi, li));

                    for (int xi = 0; xi < X_ELEMS; ++xi)
                    {
                        const int refKVIdx = bi * headsNum * seqLen * dimsPerHead + hi * seqLen * dimsPerHead
                            + li * dimsPerHead + di * X_ELEMS + xi;

                        const int kVIdx = buffer.getKVLocalIdx(li, hi, dimsPerHead, di * X_ELEMS + xi);

                        T refK = refKCacheVec[refKVIdx];
                        T refV = vTransposedCacheVec[refKVIdx];
                        if (b8bitKVCache)
                        {
                            refK = castTo<float, T>(castTo<T, float>(refK) * kvScaleOrigQuant);
                            refV = castTo<float, T>(castTo<T, float>(refV) * kvScaleOrigQuant);
                        }

                        const T_DST castedRefK = castTo<T, T_DST>(refK);
                        const T_DST castedRefV = castTo<T, T_DST>(refV);

                        const auto outK = blockKPtr[kVIdx];
                        const auto outV = blockVPtr[kVIdx];

                        const float outK_float = castTo<T_DST, float>(outK);
                        const float outV_float = castTo<T_DST, float>(outV);
                        const float castedRefK_float = castTo<T_DST, float>(castedRefK);
                        const float castedRefV_float = castTo<T_DST, float>(castedRefV);
                        EXPECT_EQ(outK_float, castedRefK_float);
                        EXPECT_EQ(outV_float, castedRefV_float);
                    }
                }
            }
        }
    }
}

template <typename T, typename T_DST>
void testTransposeBatch4dPaged(bool multiQueryMode, bool int8KVCache, bool fp8KVCache)
{
    srand(42);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    constexpr int32_t tokensPerBlock{8};
    constexpr int32_t maxBlocksPerSeq{64};
    constexpr int32_t maxSeq{64};
    constexpr int32_t batchSize{2};
    const int32_t headsNum = multiQueryMode ? 1 : 8;
    constexpr int32_t seqLen{16};
    constexpr int32_t maxSeqLen{2 * seqLen};
    constexpr int32_t dimsPerHead{256};
    constexpr int32_t blockSizeBytes = tokensPerBlock * dimsPerHead * sizeof(T_DST);

    TLLM_CHECK_WITH_INFO(batchSize <= maxSeq, "Batch size is larger than max number of allowed sequence");
    TLLM_CHECK_WITH_INFO(headsNum * seqLen <= maxBlocksPerSeq * tokensPerBlock,
        "Total amount of tokens is less than max amount of tokens is cache per sequence");

    KVBlockArray blockArray(maxSeq, maxBlocksPerSeq, tokensPerBlock, dimsPerHead * headsNum * sizeof(T_DST));

    const auto pointerArrayElts = maxSeq * maxBlocksPerSeq;
    const auto pointerArraySize = 2 * pointerArrayElts * sizeof(void*);
    cudaMalloc(&blockArray.data, pointerArraySize);
    cudaMemset(blockArray.data, 0, pointerArraySize);

    const auto blocksPerPool = maxBlocksPerSeq * maxSeq;
    const auto kvPoolSize = 2 * blockSizeBytes * blocksPerPool;
    void* kvMemoryPool;
    cudaMalloc(&kvMemoryPool, kvPoolSize);
    cudaMemset(kvMemoryPool, 0, kvPoolSize);

    std::unordered_map<int, int> mapIndicesTable;
    for (int i = 0; i < pointerArrayElts; ++i)
    {
        int value;
        int idx = i;
        if (idx % 2 == 0)
        {
            value = idx / 2;
        }
        else
        {
            value = pointerArrayElts / 2 + idx / 2;
        }

        mapIndicesTable[idx] = value;
    }

    const auto pointers = pointerArrayFromPageTable(
        mapIndicesTable, kvMemoryPool, maxSeq, maxBlocksPerSeq, blockSizeBytes, blocksPerPool);
    cudaMemcpy(blockArray.data, pointers.data(), pointerArraySize, cudaMemcpyHostToDevice);

    float kvScaleOrigQuant = 1.0f;
    float* kvScaleOrigQuantPtr = nullptr;
    if (int8KVCache || fp8KVCache)
    {
        kvScaleOrigQuant = 0.1f;
        cudaMalloc(&kvScaleOrigQuantPtr, sizeof(float));
        cudaMemcpy(kvScaleOrigQuantPtr, &kvScaleOrigQuant, sizeof(float), cudaMemcpyHostToDevice);
    }
    int* sequenceLengths = nullptr;
    cudaMalloc(&sequenceLengths, sizeof(int) * batchSize);
    bitfusion::common::deviceFill(sequenceLengths, batchSize, seqLen, streamPtr->get());

    std::vector<T> kTransposedCacheVec(batchSize * headsNum * seqLen * dimsPerHead);
    std::vector<T> vTransposedCacheVec(batchSize * headsNum * seqLen * dimsPerHead);
    randomInitVector(kTransposedCacheVec, 1.f / kvScaleOrigQuant);
    randomInitVector(vTransposedCacheVec, 1.f / kvScaleOrigQuant);

    auto kTransposedCache = std::shared_ptr(manager.copyFrom(
        kTransposedCacheVec, ITensor::makeShape({batchSize, headsNum, seqLen, dimsPerHead}), MemoryType::kGPU));
    auto vTransposedCache = std::shared_ptr(manager.copyFrom(
        vTransposedCacheVec, ITensor::makeShape({batchSize, headsNum, seqLen, dimsPerHead}), MemoryType::kGPU));

    const KvCacheDataType cache_type
        = int8KVCache ? KvCacheDataType::INT8 : (fp8KVCache ? KvCacheDataType::FP8 : KvCacheDataType::BASE);
    invokeTranspose4dBatchMajor(bufferCast<T>(*kTransposedCache), bufferCast<T>(*vTransposedCache), blockArray,
        batchSize, seqLen, maxSeqLen, dimsPerHead, headsNum, cache_type, kvScaleOrigQuantPtr, sequenceLengths,
        streamPtr->get());

    streamPtr->synchronize();

    std::vector<T_DST> kvMemoryPoolHost(kvPoolSize);
    cudaMemcpy(kvMemoryPoolHost.data(), kvMemoryPool, kvPoolSize, cudaMemcpyDeviceToHost);
    KVBlockArray blockArrayHost = blockArray;

    auto pointersHost = pointerArrayFromPageTable(mapIndicesTable, reinterpret_cast<void*>(kvMemoryPoolHost.data()),
        maxSeq, maxBlocksPerSeq, blockSizeBytes, blocksPerPool);
    blockArrayHost.data = reinterpret_cast<int64_t*>(pointersHost.data());

    verifyKVTransposed<T, T_DST>(batchSize, headsNum, dimsPerHead, seqLen, maxSeqLen, blockArrayHost,
        kTransposedCacheVec, vTransposedCacheVec, int8KVCache || fp8KVCache, kvScaleOrigQuant);

    cudaFree(sequenceLengths);
    if (int8KVCache || fp8KVCache)
    {
        cudaFree(kvScaleOrigQuantPtr);
    }
}

template <typename T, typename T_DST>
void testTransposeBatch4dContiguous(bool multiQueryMode, bool int8KVCache, bool fp8KVCache)
{
    srand(42);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    constexpr int32_t batchSize{2};
    const int32_t headsNum = multiQueryMode ? 1 : 8;
    constexpr int32_t seqLen{16};
    constexpr int32_t maxSeqLen{2 * seqLen};
    constexpr int32_t dimsPerHead{256};

    KVLinearBuffer kvLinearBuffer(batchSize, 1, maxSeqLen, dimsPerHead * headsNum * sizeof(T_DST));

    const auto kvPoolElts = 2 * batchSize * maxSeqLen * dimsPerHead * headsNum;
    const auto kvPoolSize = kvPoolElts * sizeof(T_DST);
    cudaMalloc(&kvLinearBuffer.data, kvPoolSize);
    cudaMemset(kvLinearBuffer.data, 0, kvPoolSize);

    float kvScaleOrigQuant = 1.0f;
    float* kvScaleOrigQuantPtr = nullptr;
    if (int8KVCache || fp8KVCache)
    {
        kvScaleOrigQuant = 0.1f;
        cudaMalloc(&kvScaleOrigQuantPtr, sizeof(float));
        cudaMemcpy(kvScaleOrigQuantPtr, &kvScaleOrigQuant, sizeof(float), cudaMemcpyHostToDevice);
    }
    int* sequenceLengths = nullptr;
    cudaMalloc(&sequenceLengths, sizeof(int) * batchSize);
    bitfusion::common::deviceFill(sequenceLengths, batchSize, seqLen, streamPtr->get());

    std::vector<T> kTransposedCacheVec(batchSize * headsNum * seqLen * dimsPerHead);
    std::vector<T> vTransposedCacheVec(batchSize * headsNum * seqLen * dimsPerHead);
    randomInitVector(kTransposedCacheVec, 1.f / kvScaleOrigQuant);
    randomInitVector(vTransposedCacheVec, 1.f / kvScaleOrigQuant);

    auto kTransposedCache = std::shared_ptr(manager.copyFrom(
        kTransposedCacheVec, ITensor::makeShape({batchSize, headsNum, seqLen, dimsPerHead}), MemoryType::kGPU));
    auto vTransposedCache = std::shared_ptr(manager.copyFrom(
        vTransposedCacheVec, ITensor::makeShape({batchSize, headsNum, seqLen, dimsPerHead}), MemoryType::kGPU));

    const KvCacheDataType cache_type
        = int8KVCache ? KvCacheDataType::INT8 : (fp8KVCache ? KvCacheDataType::FP8 : KvCacheDataType::BASE);
    invokeTranspose4dBatchMajor(bufferCast<T>(*kTransposedCache), bufferCast<T>(*vTransposedCache), kvLinearBuffer,
        batchSize, seqLen, maxSeqLen, dimsPerHead, headsNum, cache_type, kvScaleOrigQuantPtr, sequenceLengths,
        streamPtr->get());

    streamPtr->synchronize();

    std::vector<T_DST> kvMemoryPoolHost(kvPoolElts);
    cudaMemcpy(kvMemoryPoolHost.data(), kvLinearBuffer.data, kvPoolSize, cudaMemcpyDeviceToHost);
    KVLinearBuffer kvLinearBufferHost = kvLinearBuffer;

    kvLinearBufferHost.data = reinterpret_cast<int8_t*>(kvMemoryPoolHost.data());

    verifyKVTransposed<T, T_DST>(batchSize, headsNum, dimsPerHead, seqLen, maxSeqLen, kvLinearBufferHost,
        kTransposedCacheVec, vTransposedCacheVec, int8KVCache || fp8KVCache, kvScaleOrigQuant);

    cudaFree(sequenceLengths);
    if (int8KVCache || fp8KVCache)
    {
        cudaFree(kvScaleOrigQuantPtr);
    }
}

}

TEST(AttentionKernelTest, transposeBatch4dPagedFloat)
{
    testTransposeBatch4dPaged<float, float>(false, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dPagedHalf)
{
    testTransposeBatch4dPaged<half, half>(false, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dPagedMultiQuery)
{
    testTransposeBatch4dPaged<half, half>(true, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dPagedInt8)
{
    testTransposeBatch4dPaged<float, int8_t>(false, true, false);
}

#ifdef ENABLE_FP8
TEST(AttentionKernelTest, transposeBatch4dPagedFp8)
{
    testTransposeBatch4dPaged<float, __nv_fp8_e4m3>(false, false, true);
}
#endif

TEST(AttentionKernelTest, transposeBatch4dContiguousFloat)
{
    testTransposeBatch4dContiguous<float, float>(false, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dContiguousHalf)
{
    testTransposeBatch4dContiguous<half, half>(false, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dContiguousMultiQuery)
{
    testTransposeBatch4dContiguous<half, half>(true, false, false);
}

TEST(AttentionKernelTest, transposeBatch4dContiguousInt8)
{
    testTransposeBatch4dContiguous<float, int8_t>(false, true, false);
}

#ifdef ENABLE_FP8
TEST(AttentionKernelTest, transposeBatch4dContiguousFp8)
{
    testTransposeBatch4dContiguous<float, __nv_fp8_e4m3>(false, false, true);
}
#endif
