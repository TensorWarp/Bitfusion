#ifdef __CUDACC__

#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "CudaUtil.h"

#include <stdexcept>

namespace astdl {
    namespace math {

        __global__ void kFloatToHalf_kernel(const float* src, size_t length, half* dst) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < length) {
                dst[idx] = __float2half(src[idx]);
            }
        }

        void kFloatToHalf(const float* hSource, size_t sourceSizeInBytes, half* dDest, float* dBuffer, size_t bufferSizeInBytes) {
            if (sourceSizeInBytes % sizeof(float) != 0) {
                throw std::invalid_argument("sourceSizeInBytes must be divisible by sizeof(float)");
            }

            if (bufferSizeInBytes % sizeof(float) != 0) {
                throw std::invalid_argument("bufferSizeInBytes must be divisible by sizeof(float)");
            }

            dim3 threads(128);
            size_t bufferLen = bufferSizeInBytes / sizeof(float);
            dim3 blocks(static_cast<unsigned int>((bufferLen + static_cast<unsigned int>(threads.x) - 1) / static_cast<unsigned int>(threads.x)));

            size_t srcLeftBytes = sourceSizeInBytes;
            size_t offset = 0;

            while (srcLeftBytes > 0) {
                size_t cpyBytes = srcLeftBytes < bufferSizeInBytes ? srcLeftBytes : bufferSizeInBytes;
                size_t cpyLength = cpyBytes / sizeof(float);

                cudaMemcpy(dBuffer, hSource + offset, cpyBytes, cudaMemcpyHostToDevice);

                offset += cpyLength;
                srcLeftBytes -= cpyBytes;
            }
        }

        void kFloatToHalf(const float* hSource, size_t sourceSizeInBytes, half* dDest, size_t bufferSizeInBytes) {
            float* dBuffer;
            cudaMalloc(&dBuffer, bufferSizeInBytes);
            kFloatToHalf(hSource, sourceSizeInBytes, dDest, dBuffer, bufferSizeInBytes);
            cudaFree(dBuffer);
        }

        __global__ void kHalfToFloat_kernel(const half* src, size_t length, float* dst) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < length) {
                dst[idx] = __half2float(src[idx]);
            }
        }

        void kHalfToFloat(const half* dSource, size_t sourceSizeInBytes, float* hDest, size_t bufferSizeInBytes) {
            if (sourceSizeInBytes % sizeof(half) != 0) {
                throw std::invalid_argument("sourceSizeInBytes must be divisible by sizeof(half)");
            }

            if (bufferSizeInBytes % sizeof(float) != 0) {
                throw std::invalid_argument("bufferSizeInBytes must be divisible by sizeof(float)");
            }

            dim3 threads(128);
            size_t bufferLen = bufferSizeInBytes / sizeof(float);
            dim3 blocks(static_cast<unsigned int>((bufferLen + static_cast<unsigned int>(threads.x) - 1) / static_cast<unsigned int>(threads.x)));

            float* dBuffer;
            cudaMalloc(&dBuffer, bufferLen * sizeof(float));

            size_t sourceLength = sourceSizeInBytes / sizeof(half);
            size_t srcLeftBytes = sourceLength * sizeof(float);
            size_t offset = 0;

            while (srcLeftBytes > 0) {
                size_t cpyBytes = srcLeftBytes < bufferSizeInBytes ? srcLeftBytes : bufferSizeInBytes;
                size_t cpyLength = cpyBytes / sizeof(float);

                cudaMemcpy(hDest + offset, dBuffer, cpyBytes, cudaMemcpyDeviceToHost);

                offset += cpyLength;
                srcLeftBytes -= cpyBytes;
            }

            cudaFree(dBuffer);
        }

    } // namespace math
} // namespace astdl


#endif
