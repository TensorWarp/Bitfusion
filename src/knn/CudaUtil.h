#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

/// <summary>
/// Checks for CUDA errors and exits the application if an error is encountered.
/// </summary>
static void CHECK_ERR2(cudaError_t e, const char* fname, int line)
{
    if (e != cudaSuccess)
    {
        std::cerr << "FATAL ERROR: CUDA failure (" << e << "): " << cudaGetErrorString(e)
            << " in " << fname << "#" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/// <summary>
/// Checks for cuBLAS status errors and exits the application if an error is encountered.
/// </summary>
static void STATUS_ERR2(cublasStatus_t e, const char* fname, int line)
{
    if (e != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "FATAL ERROR: cuBLAS failure " << e << " in " << fname << "#" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/// <summary>
/// Checks for CUDA launch errors and exits the application if an error is encountered.
/// </summary>
static void LAUNCH_ERR2(const char* kernelName, const char* fname, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        std::cerr << "FATAL ERROR: " << cudaGetErrorString(e) << " launching kernel: " << kernelName
            << " in " << fname << "#" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/// <summary>
/// Macro to check for CUDA errors and exit the application if an error is encountered.
/// </summary>
#define CHECK_ERR(e) {CHECK_ERR2(e, __FILE__, __LINE__);}

/// <summary>
/// Macro to check for cuBLAS status errors and exit the application if an error is encountered.
/// </summary>
#define STATUS_ERR(e) {STATUS_ERR2(e, __FILE__, __LINE__);}

/// <summary>
/// Macro to check for CUDA launch errors and exit the application if an error is encountered.
/// </summary>
#define LAUNCH_ERR(expression) { \
    expression; \
    LAUNCH_ERR2(#expression, __FILE__, __LINE__); \
  }

namespace astdl
{
    namespace cuda_util
    {
        /// <summary>
        /// Prints GPU memory information.
        /// </summary>
        void printMemInfo(const char* header = "");

        /// <summary>
        /// Retrieves GPU device memory information in megabytes.
        /// </summary>
        void getDeviceMemoryInfoInMb(int device, size_t* total, size_t* free);

        /// <summary>
        /// Retrieves the number of available GPU devices.
        /// </summary>
        int getDeviceCount();

        /// <summary>
        /// Checks if the system has one or more GPUs available.
        /// </summary>
        bool hasGpus();

    }
}

/// <summary>
/// Macro to require at least one GPU. Returns from the current function if no GPUs are available.
/// </summary>
#define REQUIRE_GPU if(!astdl::cuda_util::hasGpus()) return;

/// <summary>
/// Macro to require a minimum number of GPUs. Returns from the current function if the requirement is not met.
/// </summary>
#define REQUIRE_GPUS(numGpus) if(astdl::cuda_util::getDeviceCount() < numGpus) return;