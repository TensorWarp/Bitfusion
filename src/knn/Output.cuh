#pragma once

#include <cstdint>
#include <iostream>

#ifdef __HIP_PLATFORM_HCC__
// HIP-specific includes or definitions go here
#define HIP_LAUNCH_BOUNDS() __launch_bounds__(256, 5)
#else
// CUDA-specific includes or definitions go here
#define CUDA_LAUNCH_BOUNDS() __launch_bounds__(256, 5)
#endif

/// <summary>
/// Set launch bounds based on the CUDA architecture or HIP.
/// </summary>
#if defined(__HIP_PLATFORM_HCC__)
#define LAUNCH_BOUNDS() HIP_LAUNCH_BOUNDS()
#elif (__CUDA_ARCH__ >= 600)
#define LAUNCH_BOUNDS() CUDA_LAUNCH_BOUNDS()
#elif (__CUDA_ARCH__ >= 500)
#define LAUNCH_BOUNDS() CUDA_LAUNCH_BOUNDS()
#else
#define LAUNCH_BOUNDS() CUDA_LAUNCH_BOUNDS()
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
/// This macro checks for CUDA or HIP errors and, if synchronous mode is enabled,
/// it also synchronizes the device. It throws a runtime error in case of
/// a CUDA or HIP error.
/// </summary>
/// <param name="s">The name of the kernel being launched.</param>
#ifdef SYNCHRONOUS
#define LAUNCHERROR(s)                             \
    {                                               \
        auto status = __HIP_PLATFORM_HCC__ ? hipGetLastError() : cudaGetLastError(); \
        if (status != (__HIP_PLATFORM_HCC__ ? hipSuccess : cudaSuccess)) { \
            std::cerr << "Error: " << (__HIP_PLATFORM_HCC__ ? hipGetErrorString(status) : cudaGetErrorString(status)) << " launching kernel " << s << '\n'; \
            throw std::runtime_error(__HIP_PLATFORM_HCC__ ? "HIP kernel launch error" : "CUDA kernel launch error"); \
        }                                           \
        if (!__HIP_PLATFORM_HCC__) cudaDeviceSynchronize(); \
    }
#else
#define LAUNCHERROR(s)                             \
    {                                               \
        auto status = __HIP_PLATFORM_HCC__ ? hipGetLastError() : cudaGetLastError(); \
        if (status != (__HIP_PLATFORM_HCC__ ? hipSuccess : cudaSuccess)) { \
            std::cerr << "Error: " << (__HIP_PLATFORM_HCC__ ? hipGetErrorString(status) : cudaGetErrorString(status)) << " launching kernel " << s << '\n'; \
            throw std::runtime_error(__HIP_PLATFORM_HCC__ ? "HIP kernel launch error" : "CUDA kernel launch error"); \
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
void CalculateOutput(float* pOutput, float* pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t widthPadding, uint32_t k);

