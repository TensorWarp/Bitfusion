#pragma once

#include <cuda_fp16.h>

namespace astdl
{
    namespace math
    {
        /// <summary>
        /// Converts an array of single-precision floating-point numbers to half-precision floating-point numbers.
        /// </summary>
        /// <param name="hSource">Pointer to the source array of single-precision floating-point numbers.</param>
        /// <param name="sourceLength">The number of elements in the source array.</param>
        /// <param name="dDest">Pointer to the destination array of half-precision floating-point numbers.</param>
        /// <param name="bufferSizeInBytes">Optional buffer size in bytes; defaults to 4 * 1024 * 1024 bytes.</param>
        void kFloatToHalf(const float* hSource, size_t sourceLength, half* dDest, size_t bufferSizeInBytes = 4 * 1024 * 1024);

        /// <summary>
        /// Converts an array of single-precision floating-point numbers to half-precision floating-point numbers
        /// using an intermediate buffer.
        /// </summary>
        /// <param name="hSource">Pointer to the source array of single-precision floating-point numbers.</param>
        /// <param name="sourceSizeInBytes">The size of the source array in bytes.</param>
        /// <param name="dDest">Pointer to the destination array of half-precision floating-point numbers.</param>
        /// <param name="dBuffer">Pointer to the intermediate buffer for temporary storage.</param>
        /// <param name="bufferSizeInBytes">The size of the intermediate buffer in bytes.</param>
        void kFloatToHalf(const float* hSource, size_t sourceSizeInBytes, half* dDest, float* dBuffer, size_t bufferSizeInBytes);

        /// <summary>
        /// Converts an array of half-precision floating-point numbers to single-precision floating-point numbers.
        /// </summary>
        /// <param name="dSource">Pointer to the source array of half-precision floating-point numbers.</param>
        /// <param name="sourceLength">The number of elements in the source array.</param>
        /// <param name="hDest">Pointer to the destination array of single-precision floating-point numbers.</param>
        /// <param name="bufferSizeInBytes">Optional buffer size in bytes; defaults to 4 * 1024 * 1024 bytes.</param>
        void kHalfToFloat(const half* dSource, size_t sourceLength, float* hDest, size_t bufferSizeInBytes = 4 * 1024 * 1024);
    }
}

