#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <future>
#include <stdexcept>
#include <queue>
#include <functional>
#include "../system/ThreadPool.h"

namespace astdl
{
    namespace cuda_util
    {
        constexpr int bytesInMb = 1024 * 1024;
        constexpr int numStreams = 4;
        constexpr int memoryPoolSize = 16; // Size of the GPU memory pool (in MB)

        struct GpuInfo
        {
            int device;
            size_t free;
            size_t total;
        };

        class GpuResourceManager
        {
        public:
            GpuResourceManager(int device) : device_(device)
            {
                cudaSetDevice(device_);
                streams_.resize(numStreams);

                for (int i = 0; i < numStreams; ++i)
                {
                    cudaStreamCreate(&streams_[i]);
                }

                initializeMemoryPool();
            }

            ~GpuResourceManager()
            {
                for (auto stream : streams_)
                {
                    cudaStreamDestroy(stream);
                }
            }

            void allocateMemoryAsync(void** ptr, size_t sizeInMb, int streamIndex)
            {
                if (sizeInMb > memoryPoolSize)
                {
                    throw std::runtime_error("Memory allocation size exceeds the memory pool size.");
                }

                std::unique_lock<std::mutex> lock(memoryMutex_);

                while (memoryPool_.empty())
                {
                    memoryPoolCV_.wait(lock);
                }

                *ptr = memoryPool_.front();
                memoryPool_.pop();
                lock.unlock();

                cudaMemsetAsync(*ptr, 0, sizeInMb * bytesInMb, streams_[streamIndex]);
            }

            void freeMemoryAsync(void* ptr)
            {
                std::lock_guard<std::mutex> lock(memoryMutex_);
                memoryPool_.push(ptr);
                memoryPoolCV_.notify_one();
            }

            void printMemInfo(const char* header)
            {
                GpuInfo gpuInfo;
                cudaMemGetInfo(&gpuInfo.free, &gpuInfo.total);

                long freeMb = gpuInfo.free / bytesInMb;
                long usedMb = (gpuInfo.total - gpuInfo.free) / bytesInMb;
                long totalMb = gpuInfo.total / bytesInMb;

                std::cout << "--" << std::setw(50) << header << " GPU [" << device_ << "] Mem Used: " << usedMb << " MB. Free: " << freeMb
                    << " MB. Total: " << totalMb << " MB\n";
            }

        private:
            int device_;
            std::vector<cudaStream_t> streams_;
            std::mutex memoryMutex_;
            std::queue<void*> memoryPool_;
            std::condition_variable memoryPoolCV_;

            void initializeMemoryPool()
            {
                for (int i = 0; i < memoryPoolSize; ++i)
                {
                    void* gpuMemory;
                    cudaMalloc(&gpuMemory, bytesInMb);
                    memoryPool_.push(gpuMemory);
                }
            }
        };

        std::vector<GpuInfo> getAllGpuInfo()
        {
            int deviceCount;
            cudaGetDeviceCount(&deviceCount);
            std::vector<GpuInfo> gpuInfoList(deviceCount);

            ThreadPool threadPool(deviceCount);

            for (int i = 0; i < deviceCount; ++i)
            {
                GpuInfo& gpuInfo = gpuInfoList[i];
                gpuInfo.device = i;

                threadPool.enqueue([&gpuInfo]() {
                    try
                    {
                        GpuResourceManager resourceManager(gpuInfo.device);
                        resourceManager.printMemInfo("Async Mem Info");

                        for (int streamIndex = 0; streamIndex < numStreams; ++streamIndex)
                        {
                            void* gpuMemory;
                            resourceManager.allocateMemoryAsync(&gpuMemory, 256, streamIndex);
                            resourceManager.printMemInfo("After Allocation");
                            resourceManager.freeMemoryAsync(gpuMemory);
                            resourceManager.printMemInfo("After Deallocation");
                        }
                    }
                    catch (const std::exception& ex)
                    {
                        std::cerr << "Exception caught: " << ex.what() << std::endl;
                    }
                    });
            }

            return gpuInfoList;
        }
    } // namespace cuda_util
} // namespace astdl
