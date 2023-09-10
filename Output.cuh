#pragma once

#include <cstdint>
#include <iostream>

/// <summary>
/// Number of threads per block for SM 3.x architecture.
/// </summary>
static constexpr int SM_3X_THREADS_PER_BLOCK = 128;

/// <summary>
/// Number of threads per block for SM 5.x architecture.
/// </summary>
static constexpr int SM_5X_THREADS_PER_BLOCK = 128;

/// <summary>
/// Number of threads per block for SM 6.x architecture.
/// </summary>
static constexpr int SM_6X_THREADS_PER_BLOCK = 128;

/// <summary>
/// Set launch bounds based on the CUDA architecture.
/// </summary>
#if (__CUDA_ARCH__ >= 600)
#define LAUNCH_BOUNDS() __launch_bounds__(SM_6X_THREADS_PER_BLOCK, 8)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#elif (__CUDA_ARCH__ >= 500)
#define LAUNCH_BOUNDS() __launch_bounds__(SM_5X_THREADS_PER_BLOCK, 8)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#else
#define LAUNCH_BOUNDS() __launch_bounds__(SM_3X_THREADS_PER_BLOCK, 10)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 4)
#endif

/// <summary>
/// Set launch bounds for 512 threads.
/// </summary>
#define LAUNCH_BOUNDS512() __launch_bounds__(512, 2)

/// <summary>
/// Set launch bounds for 1024 threads.
/// </summary>
#define LAUNCH_BOUNDS1024() __launch_bounds__(1024, 1)

/// <summary>
/// Macro for error handling during kernel launch.
///
/// This macro checks for CUDA errors and, if synchronous mode is enabled,
/// it also synchronizes the device. It throws a runtime error in case of
/// a CUDA error.
/// </summary>
/// <param name="s">The name of the kernel being launched.</param>
#ifdef SYNCHRONOUS
#define LAUNCHERROR(s)                             \
    {                                               \
        cudaError_t status = cudaGetLastError();    \
        if (status != cudaSuccess) {                \
            std::cerr << "Error: " << cudaGetErrorString(status) << " launching kernel " << s << std::endl; \
            throw std::runtime_error("CUDA kernel launch error"); \
        }                                           \
        cudaDeviceSynchronize();                   \
    }
#else
#define LAUNCHERROR(s)                             \
    {                                               \
        cudaError_t status = cudaGetLastError();    \
        if (status != cudaSuccess) {                \
            std::cerr << "Error: " << cudaGetErrorString(status) << " launching kernel " << s << std::endl; \
            throw std::runtime_error("CUDA kernel launch error"); \
        }                                           \
    }
#endif

/// <summary>
/// Calculate the output based on input data and parameters.
/// </summary>
/// <param name="pOutput">Pointer to the output data.</param>
/// <param name="pKey">Pointer to the key data.</param>
/// <param name="pValue">Pointer to the value data.</param>
/// <param name="batch">Batch size.</param>
/// <param name="width">Width of the data.</param>
/// <param name="widthPadding">Width padding.</param>
/// <param name="k">Parameter 'k'.</param>
void CalculateOutput(float* pOutput, float *pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t widthPadding, uint32_t k);
