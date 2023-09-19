#ifdef __CUDACC__

#include "GpuTypes.h"
#include "Types.h"
#include <limits>
#include <stdexcept>

static __constant__ GpuData cData;

/// <summary>
/// Minimum activation value.
/// </summary>
#define MIN_ACTIVATION 0.1f

/// <summary>
/// Maximum activation value.
/// </summary>
#define MAX_ACTIVATION 0.9f

/// <summary>
/// Utility function to check and handle CUDA errors.
/// </summary>
/// <param name="status">The CUDA status.</param>
/// <param name="errorMsg">The error message to display if there's an error.</param>
inline void checkCudaStatus(cudaError_t status, const std::string& errorMsg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(errorMsg + ": " + std::string(cudaGetErrorString(status)));
    }
}

/// <summary>
/// Generalized function to set GPU data.
/// </summary>
/// <param name="funcName">The name of the calling function for error handling.</param>
void SetGpuData(const std::string& funcName) {
    cudaError_t status = cudaMemcpyToSymbol(cData, &(getGpu()._data), sizeof(GpuData));
    checkCudaStatus(status, "cudaMemcpyToSymbol in " + funcName + " failed");
}

/// <summary>
/// Generalized function to get GPU data.
/// </summary>
/// <param name="funcName">The name of the calling function for error handling.</param>
void GetGpuData(const std::string& funcName) {
    cudaError_t status = cudaMemcpyFromSymbol(&(getGpu()._data), cData, sizeof(GpuData));
    checkCudaStatus(status, "cudaMemcpyFromSymbol in " + funcName + " failed");
}

/// <summary>
/// Calculates the number of blocks needed for the given size.
/// </summary>
/// <param name="size">The size of the data.</param>
/// <returns>The number of blocks needed.</returns>
uint32_t CalculateBlocks(uint64_t size) {
    return (size + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
}

/// <summary>
/// Atomically computes the maximum of the two values and stores the result in the specified address.
/// </summary>
/// <param name="address">The address to store the result in.</param>
/// <param name="val">The value to compare.</param>
/// <returns>The maximum value.</returns>
__device__ float atomicMax(float* address, float val) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old_as_int = __float_as_int(*address);
    int assumed_as_int = old_as_int;
    int max_as_int = __float_as_int(val);

    do {
        old_as_int = assumed_as_int;
        assumed_as_int = atomicCAS(address_as_int, old_as_int, max_as_int);
    } while (assumed_as_int != old_as_int);

    return __int_as_float(old_as_int);
}

/// <summary>
/// Sets GPU data related to KDelta.
/// </summary>
void SetKDeltaGpuData() { SetGpuData("SetKDeltaGpuData"); }

/// <summary>
/// Gets GPU data related to KDelta.
/// </summary>
void GetKDeltaGpuData() { GetGpuData("GetKDeltaGpuData"); }

/// <summary>
/// Sets GPU data related to KLoss.
/// </summary>
void SetKLossGpuData() { SetGpuData("SetKLossGpuData"); }

/// <summary>
/// Gets GPU data related to KLoss.
/// </summary>
void GetKLossGpuData() { GetGpuData("GetKLossGpuData"); }

/// <summary>
/// Sets GPU data related to Kernels.
/// </summary>
void SetKernelsGpuData() { SetGpuData("SetKernelsGpuData"); }

/// <summary>
/// Gets GPU data related to Kernels.
/// </summary>
void GetKernelsGpuData() { GetGpuData("GetKernelsGpuData"); }

/// <summary>
/// Sets GPU data related to KActivation.
/// </summary>
void SetKActivationGpuData() { SetGpuData("SetKActivationGpuData"); }

/// <summary>
/// Gets GPU data related to KActivation.
/// </summary>
void GetKActivationGpuData() { GetGpuData("GetKActivationGpuData"); }

//////////////////// CUDA Delta kernels ////////////////////

/// <summary>
/// Template kernel function for calculating the sigmoid output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    // Define constants for the kernel.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Define integer quantization parameters.
    const int INT_BITS = 4;
    const int FRAC_BITS = 4;
    const int SCALE = 1 << FRAC_BITS;

    // Declare a shared memory array to hold a tile of data.
    __shared__ int tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Determine a scaling factor based on the data type T.
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get the thread's x and y indices within the block.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute indices and offsets for vectorized processing.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Declare arrays for storing vectors of quantized weights, activations, and target values.
    int w_vec[VEC_SIZE];
    int a_vec[VEC_SIZE];
    int t_vec[VEC_SIZE];

    // Loop to quantize and load data into vectors for vectorized processing.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float scaled_weight = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) * SCALE : SCALE;
        float scaled_activation = pUnit[uOffset + row + i] * SCALE;
        float scaled_target = static_cast<float>(pData[dOffset + row + i]) * SCALE;

        w_vec[i] = static_cast<int>(scaled_weight);
        a_vec[i] = static_cast<int>(scaled_activation);
        t_vec[i] = static_cast<int>(scaled_target);
    }

    // Declare an array to store quantized delta values.
    int delta_vec[VEC_SIZE];

    // Loop to calculate quantized delta values for each element in the vectors.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = (w_vec[i] * a_vec[i] * (SCALE - a_vec[i]) * (a_vec[i] - t_vec[i])) / SCALE;
    }

    // Store quantized delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Reduce quantized delta values within each thread block using parallel reduction.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final quantized delta value in the global memory.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Calculate the column index for the quantized delta value.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the x-axis of the shared memory tile to get the final quantized delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final quantized delta value in the global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

// TODO: Implement quantization

/// <summary>
/// Template kernel function for calculating the hyperbolic tangent (tanh) output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const float* pUnit, float* pDelta, const T* pData, const float* pDataWeight)
{
    // Define constants for the kernel.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory array to hold a tile of data.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Determine a scaling factor based on the data type T.
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get the thread's x and y indices within the block.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute indices and offsets for vectorized processing.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Declare arrays for storing vectors of weights, activations, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load data into vectors for vectorized processing.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Declare an array to store delta values.
    float delta_vec[VEC_SIZE];

    // Loop to calculate delta values for each element in the vectors.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * (1.0f - a_vec[i] * a_vec[i]);
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Reduce delta values within each thread block using parallel reduction.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in the shared memory tile for row-wise storage.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Calculate the column index for the delta value.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the x-axis of the shared memory tile to get the final delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in the global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

/// <summary>
/// Template kernel function for calculating the linear output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, const float* pUnit, float* pDelta, const T* pData, const float* pDataWeight)
{
    // Define constants for the kernel.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory array to hold a tile of data.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices for data and weights.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Determine a scaling factor based on the data type T.
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get the thread's x and y indices within the block.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute indices and offsets for vectorized processing.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Declare arrays for storing vectors of weights, activations, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load data into vectors for vectorized processing.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Declare an array to store delta values.
    float delta_vec[VEC_SIZE];

    // Loop to calculate delta values for each element in the vectors.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * (1.0f - a_vec[i] * a_vec[i]);
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Reduce delta values within each thread block using parallel reduction.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in the shared memory tile for row-wise storage.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Calculate the column index for the delta value.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the x-axis of the shared memory tile to get the final delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in the global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

/// <summary>
/// Template kernel function for calculating the Rectified Linear Unit (ReLU) output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    // Define some constants for thread block configuration.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory array 'tile' for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate offsets based on the thread block and thread IDs.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Calculate a scaling factor based on the data type.
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Calculate indices and offsets for vectorized operations.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Declare arrays to store values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load and calculate values into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Declare an array to store delta values for vectorized operations.
    float delta_vec[VEC_SIZE];

    // Loop to calculate delta values for vectorized operations.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * (a_vec[i] > 0.0f);
    }

    // Copy delta values into the shared memory 'tile' array.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the thread block.
    __syncthreads();

    // Reduce delta values within the thread block using a parallel reduction.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in 'tile' to global memory for the current row.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the thread block again.
    __syncthreads();

    // Calculate the column index and reduce delta values within the thread block for the current column.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory for the current thread.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

/// <summary>
/// Template kernel function for calculating the Leaky Rectified Linear Unit (LReLU) output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
/// <param name="slope">The slope for negative activations.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope)
{
    // Define some constants for thread block configuration.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory array 'tile' for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate offsets based on the thread block and thread IDs.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Calculate a scaling factor based on the data type.
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Calculate indices and offsets for vectorized operations.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Declare arrays to store values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load and calculate values into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Declare an array to store delta values for vectorized operations.
    float delta_vec[VEC_SIZE];

    // Loop to calculate delta values using the LRELU formula.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * ((a_vec[i] > 0.0f) + (a_vec[i] <= 0.0f) * slope);
    }

    // Copy delta values into the shared memory 'tile' array.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the thread block.
    __syncthreads();

    // Reduce delta values within the thread block using a parallel reduction.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in 'tile' to global memory for the current row.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the thread block again.
    __syncthreads();

    // Calculate the column index and reduce delta values within the thread block for the current column.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory for the current thread.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

/// <summary>
/// Template kernel function for calculating the Exponential Linear Unit (ELU) output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
/// <param name="alpha">The alpha value for negative activations.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha)
{
    // Define some constants for thread block configuration.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory array 'tile' for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate offsets based on the thread block and thread IDs.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Calculate a scaling factor based on the data type.
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Calculate indices and offsets for vectorized operations.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Declare arrays to store values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load and calculate values into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Declare an array to store delta values for vectorized operations.
    float delta_vec[VEC_SIZE];

    // Loop to calculate delta values using the ELU formula.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * ((a_vec[i] >= 0.0f) + (a_vec[i] < 0.0f) * (a_vec[i] + alpha));
    }

    // Copy delta values into the shared memory 'tile' array.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the thread block.
    __syncthreads();

    // Reduce delta values within the thread block using a parallel reduction.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in 'tile' to global memory for the current row.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the thread block again.
    __syncthreads();

    // Calculate the column index and reduce delta values within the thread block for the current column.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory for the current thread.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

/// <summary>
/// Template kernel function for calculating the Scaled Exponential Linear Unit (SELU) output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
/// <param name="alpha">The alpha value for the SELU activation.</param>
/// <param name="lambda">The lambda value for the SELU activation.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha, float lambda)
{
    // Define some constants for thread block configuration.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory array 'tile' for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate offsets based on the thread block and thread IDs.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Calculate a scaling factor based on the data type.
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Calculate indices and offsets for vectorized operations.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Declare arrays to store values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load and calculate values into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Declare an array to store delta values for vectorized operations.
    float delta_vec[VEC_SIZE];

    // Loop to calculate delta values using the SELU formula.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * ((a_vec[i] >= 0.0f) * lambda + (a_vec[i] < 0.0f) * (lambda * alpha * expf(a_vec[i])));
    }

    // Copy delta values into the shared memory 'tile' array.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the thread block.
    __syncthreads();

    // Reduce delta values within the thread block using a parallel reduction.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in 'tile' to global memory for the current row.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the thread block again.
    __syncthreads();

    // Calculate the column index and reduce delta values within the thread block for the current column.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Loop to perform a parallel reduction along the X-axis.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory for the current thread.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

/// <summary>
/// Template kernel function for calculating the Softmax output delta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    // Define some constants for thread block configuration.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory array 'tile' for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate offsets based on the thread block and thread IDs.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Calculate a scaling factor based on the data type.
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Calculate indices and offsets for vectorized operations.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Declare arrays to store values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load and calculate values into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Declare an array to store delta values for vectorized operations.
    float delta_vec[VEC_SIZE];

    // Loop to calculate delta values based on Softmax.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]);
    }

    // Copy delta values into the shared memory 'tile' array.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the thread block.
    __syncthreads();

    // Reduce delta values within the thread block using a parallel reduction.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in 'tile' to global memory for the current row.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the thread block again.
    __syncthreads();

    // Calculate the column index and reduce delta values within the thread block for the current column.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Loop to perform a parallel reduction along the X-axis.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory for the current thread.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

/// <summary>
/// Calculates the output delta using the specified activation function and updates pDelta.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="activation">The activation function to be applied.</param>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data (output).</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
/// <param name="slope">The slope for Leaky Rectified Linear Unit (LReLU) activation.</param>
/// <param name="alpha">The alpha value for Exponential Linear Unit (ELU) activation.</param>
/// <param name="lambda">The lambda value for Scaled Exponential Linear Unit (SELU) activation.</param>
template<typename T> void kCalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);

    try {
        switch (activation)
        {
        case Sigmoid:
            kCalculateSigmoidOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateTanhOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Linear:
            kCalculateLinearOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateRELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateLRELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateSELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha, lambda);
            break;

        case SoftMax:
            kCalculateSoftMaxOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

/// <summary>
/// Template kernel function for calculating the Sigmoid output delta with indexed data.
/// </summary>
/// <typeparam name="T">The data type used for pData.</typeparam>
/// <param name="position">The position within the data.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">Pointer to the unit data.</param>
/// <param name="pDelta">Pointer to the delta data.</param>
/// <param name="pIndex">Pointer to the index data for data lookup.</param>
/// <param name="pData">Pointer to the data.</param>
/// <param name="pDataWeight">Pointer to the data weights.</param>
template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    // Define constants for thread block dimensions and vectorization.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory array for data tile storage.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate offsets based on block and thread indices.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the data type.
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Calculate vectorization-related indices.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Declare arrays for weight, activation, and target vectors.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Load data into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Calculate delta values for each element in the vector.
    float delta_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * a_vec[i] * (1.0f - a_vec[i]);
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Reduce delta values across threads in the block using parallel reduction.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the reduced delta values back into shared memory.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Calculate column index for storing the final delta value.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform another reduction across threads in the block.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    // Define constants for thread and block dimensions.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory tile for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate data offsets and scaling factor for the current thread.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute vectorized indices and data offsets.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Initialize arrays for weight, activation, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load data into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Initialize an array to store delta values.
    float delta_vec[VEC_SIZE];

    // Compute delta values for each element.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * (1.0f - a_vec[i] * a_vec[i]);
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Perform reduction to sum delta values within the block.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in the tile.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Compute the column index.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the columns to obtain the final delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    // Define constants for thread and block dimensions.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory tile for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate data offsets and scaling factor for the current thread.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute vectorized indices and data offsets.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Initialize arrays for weight, activation, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load data into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Initialize an array to store delta values.
    float delta_vec[VEC_SIZE];

    // Compute delta values for each element.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]);
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Perform reduction to sum delta values within the block.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in the tile.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Compute the column index.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the columns to obtain the final delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    // Define constants for thread and block dimensions.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory tile for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate data offsets and scaling factor for the current thread.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute vectorized indices and data offsets.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Initialize arrays for weight, activation, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load data into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Initialize an array to store delta values.
    float delta_vec[VEC_SIZE];

    // Compute delta values for each element using ReLU activation.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * (a_vec[i] > 0.0f);
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Perform reduction to sum delta values within the block.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in the tile.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Compute the column index.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the columns to obtain the final delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope)
{
    // Define constants for thread and block dimensions.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory tile for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate data offsets and scaling factor for the current thread.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute vectorized indices and data offsets.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Initialize arrays for weight, activation, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load data into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Initialize an array to store delta values.
    float delta_vec[VEC_SIZE];

    // Compute delta values for each element using Leaky ReLU activation with slope.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * ((a_vec[i] > 0.0f) + (a_vec[i] <= 0.0f) * slope);
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Perform reduction to sum delta values within the block.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in the tile.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Compute the column index.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the columns to obtain the final delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha)
{
    // Define constants for thread and block dimensions.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory tile for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate data offsets and scaling factor for the current thread.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute vectorized indices and data offsets.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Initialize arrays for weight, activation, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load data into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Initialize an array to store delta values.
    float delta_vec[VEC_SIZE];

    // Compute delta values for each element using ELU activation with slope 'alpha'.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * ((a_vec[i] >= 0.0f) + (a_vec[i] < 0.0f) * (a_vec[i] + alpha));
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Perform reduction to sum delta values within the block.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in the tile.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Compute the column index.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the columns to obtain the final delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha, float lambda)
{
    // Define constants for thread and block dimensions.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory tile for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate data offsets and scaling factor for the current thread.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute vectorized indices and data offsets.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Initialize arrays for weight, activation, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load data into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Initialize an array to store delta values.
    float delta_vec[VEC_SIZE];

    // Compute delta values for each element using SELU activation.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        // Calculate the SELU derivative for each element.
        float selu_derivative = (a_vec[i] >= 0.0f) ? lambda : (lambda * alpha * expf(a_vec[i]));
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * selu_derivative;
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Perform reduction to sum delta values within the block.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in the tile.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Compute the column index.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the columns to obtain the final delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    // Define constants for thread and block dimensions.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory tile for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate data offsets and scaling factor for the current thread.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute vectorized indices and data offsets.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Initialize arrays for weight, activation, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load data into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Initialize an array to store delta values.
    float delta_vec[VEC_SIZE];

    // Compute delta values for each element using Softmax activation.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]);
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Perform reduction to sum delta values within the block.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in the tile.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Compute the column index.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the columns to obtain the final delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template<typename T> void kCalculateIndexedOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);

    try {
        switch (activation)
        {
        case Sigmoid:
            kCalculateIndexedSigmoidOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateIndexedTanhOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Linear:
            kCalculateIndexedLinearOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateIndexedRELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateIndexedLRELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateIndexedELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateIndexedSELUOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha, lambda);
            break;

        case SoftMax:
            kCalculateIndexedSoftMaxOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    // Define constants for thread and block dimensions.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory tile for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate data offsets and determine if shuffling is required.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T).
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute vectorized indices and data offsets.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Initialize arrays for weight, activation, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load data into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Initialize an array to store delta values.
    float delta_vec[VEC_SIZE];

    // Loop to calculate delta values for each element using Sigmoid L2 Hinge activation.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        // Calculate the difference between activation and the absolute value of the target value.
        float diff = a_vec[i] - fabsf(t_vec[i]);

        // Modify the diff based on the sign of the target value (t).
        if (t_vec[i] > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Compute the delta value for this element.
        delta_vec[i] = w_vec[i] * diff * a_vec[i] * (1.0f - a_vec[i]);
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Perform reduction to sum delta values within the block.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in the tile.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Compute the column index.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the columns to obtain the final delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    // Define constants for thread and block dimensions.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Declare a shared memory tile for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate data offsets and determine if shuffling is required.
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T).
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Get thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Compute vectorized indices and data offsets.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Initialize arrays for weight, activation, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

    // Loop to load data into the arrays.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    // Initialize an array to store delta values.
    float delta_vec[VEC_SIZE];

    // Loop to calculate delta values for each element using Tanh L2 Hinge activation.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        // Calculate the difference between activation and the absolute value of the target value.
        float diff = a_vec[i] - fabsf(t_vec[i]);

        // Modify the diff based on the sign of the target value (t).
        if (t_vec[i] > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Compute the delta value for this element.
        delta_vec[i] = w_vec[i] * diff * (1.0f - a_vec[i] * a_vec[i]);
    }

    // Store delta values in the shared memory tile.
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Perform reduction to sum delta values within the block.
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    // Store the final delta value in the tile.
    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block again.
    __syncthreads();

    // Compute the column index.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

    // Perform a reduction along the columns to obtain the final delta value.
#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    // Store the final delta value in global memory.
    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T)
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]) * scalingFactor;
    }

    float delta_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float diff = a_vec[i] - fabsf(t_vec[i]);

        // Modify the diff based on the sign of t
        if (t_vec[i] > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        delta_vec[i] = w_vec[i] * diff;
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    __syncthreads();

#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    __syncthreads();

    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T)
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]);
    }

    float delta_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float diff = a_vec[i] - fabsf(t_vec[i] * scalingFactor);

        // Modify the diff based on the sign of t
        if (t_vec[i] > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        delta_vec[i] = w_vec[i] * diff * (a_vec[i] > 0.0f);
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    __syncthreads();

#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    __syncthreads();

    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T)
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]);
    }

    float delta_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float diff = a_vec[i] - fabsf(t_vec[i] * scalingFactor);

        // Modify the diff based on the sign of t
        if (t_vec[i] > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        delta_vec[i] = w_vec[i] * diff * ((a_vec[i] > 0.0f) + (a_vec[i] <= 0.0f) * slope);
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    __syncthreads();

#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    __syncthreads();

    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T)
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]);
    }

    float delta_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float diff = a_vec[i] - fabsf(t_vec[i] * scalingFactor);

        // Modify the diff based on the sign of t
        if (t_vec[i] > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        delta_vec[i] = w_vec[i] * diff * ((a_vec[i] >= 0.0f) + (a_vec[i] < 0.0f) * (a_vec[i] + alpha));
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    __syncthreads();

#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    __syncthreads();

    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha, float lambda)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T)
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]);
    }

    float delta_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float diff = a_vec[i] - fabsf(t_vec[i] * scalingFactor);

        // Modify the diff based on the sign of t
        if (t_vec[i] > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Compute the SELU-specific update
        delta_vec[i] = w_vec[i] * diff * ((a_vec[i] >= 0.0f) * lambda + (a_vec[i] < 0.0f) * (lambda * alpha * expf(a_vec[i])));
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    __syncthreads();

#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    __syncthreads();

    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T)
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]);
    }

    float delta_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float diff = a_vec[i] - fabsf(t_vec[i] * scalingFactor);

        // Modify the diff based on the sign of t
        if (t_vec[i] > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        delta_vec[i] = w_vec[i] * diff;
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    __syncthreads();

#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    __syncthreads();

    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template<typename T> void kCalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);

    try {
        switch (activation)
        {
        case Sigmoid:
            kCalculateSigmoidL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateTanhL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Linear:
            kCalculateLinearL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateRELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateLRELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateSELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha, lambda);
            break;

        case SoftMax:
            kCalculateSoftMaxL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T)
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]);
    }

    float delta_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float diff = a_vec[i] - fabsf(t_vec[i] * scalingFactor);

        // Modify the diff based on the sign of t
        if (t_vec[i] > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        delta_vec[i] = w_vec[i] * diff * a_vec[i] * (1.0f - a_vec[i]);
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    __syncthreads();

#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    __syncthreads();

    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T)
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]);
    }

    float delta_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float diff = a_vec[i] - fabsf(t_vec[i] * scalingFactor);

        // Modify the diff based on the sign of t
        if (t_vec[i] > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        delta_vec[i] = w_vec[i] * diff * (1.0f - a_vec[i] * a_vec[i]);
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    __syncthreads();

#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    __syncthreads();

    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    if (tx == 0) {
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    // Define constants for the tile size, block rows, and vector size.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory tile for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate the offset for the current block.
    uint64_t uOffset = blockIdx.x * stride;

    // Calculate the data position based on whether shuffling indices is enabled.
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];

    // Calculate the data offset based on data position.
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T).
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Calculate vectorized index and data offset for vectorized operations.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Arrays for weights, activations, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        // Load weights and perform a check if pDataWeight is not NULL.
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        // Load activations and target values.
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]);
    }

    // Array for storing delta values.
    float delta_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        // Calculate the difference between activation and scaled target value.
        float diff = a_vec[i] - fabsf(t_vec[i] * scalingFactor);

        // Modify the diff based on the sign of t_vec[i].
        if (t_vec[i] > 0.0f)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Calculate the delta using weights and modified diff.
        delta_vec[i] = w_vec[i] * diff;
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        // Store delta values in the shared memory tile.
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            // Perform a parallel reduction within the warp.
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            // Store the reduced delta values in shared memory.
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Calculate the column index.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            // Perform a parallel reduction within the warp for column values.
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    if (tx == 0) {
        // Store the final delta value in global memory.
        pDelta[uOffset + row] = delta_vec[0];
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS()
kCalculateIndexedRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    // Define constants for the tile size, block rows, and vector size.
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory tile for storing intermediate values.
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate the offset for the current block.
    uint64_t uOffset = blockIdx.x * stride;

    // Calculate the data position based on whether shuffling indices is enabled.
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];

    // Calculate the data offset based on data position.
    uint64_t dOffset = dpos * stride;

    // Determine the scaling factor based on the type of data (T).
    float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

    // Thread indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;

    // Calculate vectorized index and data offset for vectorized operations.
    int vectorized_idx = tx / VEC_SIZE;
    int data_offset = vectorized_idx * VEC_SIZE;

    // Arrays for weights, activations, and target values.
    float w_vec[VEC_SIZE];
    float a_vec[VEC_SIZE];
    float t_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        // Load weights and perform a check if pDataWeight is not NULL.
        w_vec[i] = (pDataWeight != NULL) ? __ldg(&pDataWeight[dpos + i]) : 1.0f;
        // Load activations and target values.
        a_vec[i] = pUnit[uOffset + row + i];
        t_vec[i] = static_cast<float>(pData[dOffset + row + i]);
    }

    // Array for storing delta values.
    float delta_vec[VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        // Calculate the difference between activation and scaled target value.
        float diff = a_vec[i] - fabsf(t_vec[i] * scalingFactor);

        // Apply ReLU activation and calculate the final delta.
        if (a_vec[i] > 0.0f)
            delta_vec[i] = w_vec[i] * fminf(0.0f, diff);
        else
            delta_vec[i] = 0.0f;
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        // Store delta values in the shared memory tile.
        tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
    }

    // Synchronize threads within the block.
    __syncthreads();

#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        int mask = (1 << i) - 1;
        for (int j = 0; j < VEC_SIZE; ++j) {
            // Perform a parallel reduction within the warp.
            delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
        }
    }

    if (ty == 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            // Store the reduced delta values in shared memory.
            tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
        }
    }

    // Synchronize threads within the block.
    __syncthreads();

    // Calculate the column index.
    int col = blockIdx.x * TILE_DIM + tx;
    delta_vec[0] = tile[ty][col];

#pragma unroll
    for (int i = 1; i < TILE_DIM; i *= 2) {
        int offset = 2 * i * tx;
        if (offset < TILE_DIM) {
            // Perform a parallel reduction within the warp for column values.
            delta_vec[0] += tile[ty][col + offset];
        }
    }

    if (tx == 0) {
        // Store the final delta value in global memory.
        pDelta[uOffset + row] = delta_vec[0];
    }
} // TODO: Optimize kernels

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(static_cast<float>(t) * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Apply LRELUL2 Hinge activation and calculate the final delta
        pDelta[uOffset + pos] = w * diff * ((a > 0.0f) + (a <= 0.0f) * slope);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(static_cast<float>(t) * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Apply Indexed ELUL2 Hinge activation and calculate the final delta
        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) + (a < 0.0f) * (a + alpha));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(static_cast<float>(t) * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Apply Indexed SELUL2 Hinge activation and calculate the final delta
        pDelta[uOffset + pos] = w * diff * ((a >= 0.0f) * lambda + (a < 0.0f) * (lambda * alpha * expf(a)));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];

        // Determine the scaling factor based on the type of data (T)
        float scalingFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        float diff = a - fabsf(static_cast<float>(t) * scalingFactor);

        // Modify the diff based on the sign of t
        if (t > 0)
            diff = fminf(0.0f, diff);
        else
            diff = fmaxf(0.0f, diff);

        // Calculate the final delta
        pDelta[uOffset + pos] = w * diff;
    }
}

template<typename T> void kCalculateIndexedL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);

    try {
        switch (activation)
        {
        case Sigmoid:
            kCalculateIndexedSigmoidL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateIndexedTanhL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Linear:
            kCalculateIndexedLinearL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateIndexedRELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateIndexedLRELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateIndexedELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateIndexedSELUL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha, lambda);
            break;

        case SoftMax:
            kCalculateIndexedSoftMaxL2HingeOutputDelta_kernel << <grid, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos = threadIdx.x;
    uint64_t uOffset = blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t dOffset = dpos * stride;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
    pUnit += uOffset;
    pDelta += uOffset;
    pData += dOffset;

    float scalingFactor = 1.0;
    if (std::is_same<T, unsigned char>::value)
    {
        scalingFactor = 1.0f / 256.0f;
    }
    else if (std::is_same<T, char>::value)
    {
        scalingFactor = 1.0f / 128.0f;
    }

    while (pos < stride)
    {
        float a = pUnit[pos];
        float t = static_cast<float>(pData[pos]) * scalingFactor;
        pDelta[pos] = w * ((a < 0.0f) ? -t : 0.0f);
        pos += blockDim.x;
    }
}


template<typename T> void kCalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    unsigned long threads = std::max(32UL, std::min(static_cast<unsigned long>(stride), static_cast<unsigned long>(getGpu()._threadsPerBlock)));

    try {
        kCalculateHingeOutputDelta_kernel << <batch, threads >> > (position, batch, stride, pUnit, pDelta, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dpos               = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset            = dpos * stride;
    float w                   = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;    
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        float a               = pUnit[pos];
        float t               = pData[pos];
        pDelta[pos]             = w * ((a < (float)0.0) ? -t : (float)0.0);
        pos                    += blockDim.x;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
    {
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dpos               = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset            = dpos * stride;
    float w                   = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0; 
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        float a               = pUnit[pos];
        float t               = (float)pData[pos] * (float)(1.0 / 256.0);
        pDelta[pos]             = w * ((a < (float)0.0) ? -t : (float)0.0);
        pos                    += blockDim.x;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedHingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = threadIdx.x;
    uint64_t uOffset            = blockIdx.x * stride;
    uint64_t dpos               = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t dOffset            = dpos * stride;
    float w                   = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0; 
    pUnit                      += uOffset;
    pDelta                     += uOffset;
    pData                      += dOffset;
    while (pos < stride)
    {
        float a               = pUnit[pos];
        float t               = (float)pData[pos] * (float)(1.0 / 128.0);
        pDelta[pos]             = w * ((a < (float)0.0) ? -t : (float)0.0);
        pos                    += blockDim.x;
    }
}


template<typename T> void kCalculateIndexedHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    unsigned long threads = std::max(32UL, std::min(static_cast<unsigned long>(stride), static_cast<unsigned long>(getGpu()._threadsPerBlock)));

    try {
        kCalculateIndexedHingeOutputDelta_kernel << <batch, threads >> > (position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * a * a * ((float)1.0 - a);      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawTanhOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];    
        pDelta[pos]             = w * a * ((float)1.0 - a * a);       
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLinearOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];    
        pDelta[pos]             = w * a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawRELUOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];         
        pDelta[pos]             = w * a * (a > (float)0.0);   
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLRELUOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float slope)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * a * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawELUOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float alpha)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * a * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSELUOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * a * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * (a - (float)1.0) * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSoftMaxOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];    
        pDelta[pos]             = w * a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = a - w;   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = {
            &position,
            &batch,
            &stride,
            &size,
            &pUnit,
            &pDelta,
            &pSparseStart,
            &pSparseEnd,
            &pSparseIndex,
            &pDataWeight,
            &slope,
            &alpha,
            &lambda
        };

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSigmoidOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroTanhOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroLinearOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroRELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroLRELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSoftMaxOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * (a - (float)1.0) * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = a - w;   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = {
            &position,
            &batch,
            &stride,
            &size,
            &pUnit,
            &pDelta,
            &pIndex,
            &pSparseStart,
            &pSparseEnd,
            &pSparseIndex,
            &pDataWeight,
            &slope,
            &alpha,
            &lambda
        };

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSigmoidOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroTanhOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroLinearOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroRELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroLRELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSELUOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSoftMaxOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) * scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * a_vec[i] * (1.0f - a_vec[i]) * (a_vec[i] - t_vec[i]);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) * scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * (1.0f - a_vec[i] * a_vec[i]);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) * scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) * scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * (a_vec[i] > 0.0f);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float slope)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) * scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * ((a_vec[i] > 0.0f) + (a_vec[i] <= 0.0f) * slope);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float alpha)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) * scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * ((a_vec[i] > 0.0f) + (a_vec[i] <= 0.0f) * (a_vec[i] + alpha));
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float alpha, float lambda)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) * scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * ((a_vec[i] > 0.0f) * lambda + (a_vec[i] <= 0.0f) * lambda * alpha * exp(a_vec[i]));
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) * scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template<typename T>
void kCalculateSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, slope);
            }
            kCalculateSparseAnalogNonZeroLRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, slope);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha);
            }
            kCalculateSparseAnalogNonZeroELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha, lambda);
            }
            kCalculateSparseAnalogNonZeroSELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha, lambda);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroSigmoidOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) * scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * a_vec[i] * (1.0f - a_vec[i]) * (a_vec[i] - t_vec[i]);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroTanhOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) / scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * (1.0f - a_vec[i] * a_vec[i]);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroLinearOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) / scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) / scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * (a_vec[i] > 0.0f);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroLRELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float slope)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) / scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * ((a_vec[i] > 0.0f) + (a_vec[i] <= 0.0f) * slope);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float alpha)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) / scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * ((a_vec[i] > 0.0f) + (a_vec[i] <= 0.0f) * (a_vec[i] + alpha));
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroSELUOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float alpha, float lambda)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) / scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]) * ((a_vec[i] > 0.0f) * lambda + (a_vec[i] <= 0.0f) * lambda * alpha * expf(a_vec[i]));
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroSoftMaxOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) / scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            delta_vec[i] = w_vec[i] * (a_vec[i] - t_vec[i]);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroSigmoidOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroTanhOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroLinearOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, slope);
            }
            kCalculateIndexedSparseAnalogNonZeroLRELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, slope);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha);
            }
            kCalculateIndexedSparseAnalogNonZeroELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha, lambda);
            }
            kCalculateIndexedSparseAnalogNonZeroSELUOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha, lambda);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroSoftMaxOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);
        pDelta[pos]             = w * a * a * ((float)1.0 - a);      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawTanhL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);    
        pDelta[pos]             = w * a * ((float)1.0 - a * a);       
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);                
            pDelta[pos2]        = w * diff * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLinearL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);    
        pDelta[pos]             = w * a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff;   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawRELUL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);         
        pDelta[pos]             = w * a * (a > (float)0.0);   
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLRELUL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float slope)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);
        pDelta[pos]             = w * a * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawELUL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float alpha)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        float diff            = min((float)0.0, a - (float)1.0);         
        pDelta[pos]             = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSELUL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);
        pDelta[pos]             = w * a * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));            
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSoftMaxL2HingeOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);    
        pDelta[pos]             = w * a;         
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - w);             
            pDelta[pos2]        = diff;   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = {
            &position,
            &batch,
            &stride,
            &size,
            &pUnit,
            &pDelta,
            &pSparseStart,
            &pSparseEnd,
            &pSparseIndex,
            &pDataWeight,
            &slope,
            &alpha,
            &lambda
        };

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSigmoidL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroTanhL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroLinearL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroRELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroLRELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSoftMaxL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * a * ((float)1.0 - a);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((float)1.0 - a * a);   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff;   
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * (a > (float)0.0); 
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);             
            pDelta[pos2]        = w * diff * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
            pos1               += cData._warpSize;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - w);             
            pDelta[pos2]        = diff;   
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);

    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, &slope, &alpha, &lambda };

        dim3 blockDim(getGpu()._threadsPerBlock);


        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSigmoidL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroTanhL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroLinearL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroRELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroLRELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSELUL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxL2HingeOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSoftMaxL2HingeOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            delta_vec[i] = w_vec[i] * diff * a * (1.0f - a);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            delta_vec[i] = w_vec[i] * diff * (1.0f - a * a);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            delta_vec[i] = w_vec[i] * diff;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            delta_vec[i] = w_vec[i] * diff * (a > 0.0f);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float slope)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            delta_vec[i] = w_vec[i] * diff * ((a > 0.0f) + (a <= 0.0f) * slope);
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float alpha)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            pDelta[pos2] = w * diff * ((a > 0.0f) + (a <= 0.0f) * (a + alpha));
            pos1 += cData._warpSize;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroSELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float alpha, float lambda)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            pDelta[pos2] = w * diff * ((a > 0.0f) * lambda + (a <= 0.0f) * lambda * alpha * expf(a));
            pos1 += cData._warpSize;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            pDelta[pos2] = w * diff * alpha + (w * diff * lambda) * (expf(a) / expf(alpha));
            pos1 += cData._warpSize;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template<typename T>
void kCalculateSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size = static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride);

    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, slope);
            }
            kCalculateSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, slope);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha);
            }
            kCalculateSparseAnalogNonZeroELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha, lambda);
            }
            kCalculateSparseAnalogNonZeroSELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha, lambda);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float alpha, float lambda)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            pDelta[pos2] = w * diff * a * (1.0f - a) * alpha + w * diff * lambda;
            pos1 += cData._warpSize;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float alpha, float lambda)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            pDelta[pos2] = w * diff * (1.0f - a * a) * alpha + w * diff * lambda;
            pos1 += cData._warpSize;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float alpha, float lambda)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            pDelta[pos2] = w * diff * alpha + w * diff * lambda;
            pos1 += cData._warpSize;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float alpha, float lambda)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            pDelta[pos2] = w * diff * alpha + w * diff * lambda;
            pos1 += cData._warpSize;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float slope)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            pDelta[pos2] = w * diff * (alpha * (a <= 0.0) + (a > 0.0) + slope * (a <= 0.0));
            pos1 += cData._warpSize;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < TILE_DIM; i *= 2) {
            int offset = 2 * i * tx;
            if (offset < TILE_DIM) {
                delta_vec[0] += tile[ty][col + offset];
            }
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float alpha)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            pDelta[pos2] = w * diff * ((a > 0.0) + (a <= 0.0) * (a + alpha));
            pos1 += cData._warpSize;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the shared memory tile to get the final delta value
        for (int i = 1; i < BLOCK_ROWS; i++) {
            delta_vec[0] += tile[i][col];
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroELUL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float alpha)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = fabsf(static_cast<float>(pSparseData[pos1 + i]) / scaleFactor);
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            pDelta[pos2] = w * diff * ((a > 0.0) * lambda + (a <= 0.0) * lambda * alpha * expf(a));
            pos1 += cData._warpSize;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the tile
        for (int i = 1; i < BLOCK_ROWS; i++) {
            delta_vec[0] += tile[i][col];
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}


template <typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    // Constants for the kernel
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int VEC_SIZE = 32;

    // Shared memory for a tile of data
    __shared__ float tile[TILE_DIM][TILE_DIM * VEC_SIZE];

    // Calculate memory offsets and indices
    uint64_t uOffset = (blockIdx.x * stride);
    uint64_t pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        // Determine the scale factor based on the type 'T'
        float scaleFactor = (sizeof(T) == sizeof(unsigned char)) ? 1.0f / 256.0f : 1.0f / 128.0f;

        // Thread indices
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_DIM + ty;

        // Compute indices and offsets for vectorized processing
        int vectorized_idx = tx / VEC_SIZE;
        int data_offset = vectorized_idx * VEC_SIZE;

        // Arrays for storing vectors of weights, activations, and target values
        float w_vec[VEC_SIZE];
        float a_vec[VEC_SIZE];
        float t_vec[VEC_SIZE];

        // Load data into vectors for vectorized processing
        for (int i = 0; i < VEC_SIZE; ++i) {
            w_vec[i] = w;
            a_vec[i] = pUnit[uOffset + row + i];
            t_vec[i] = static_cast<float>(pSparseData[pos1 + i]) / scaleFactor;
        }

        // Array to store delta values
        float delta_vec[VEC_SIZE];

        // Calculate delta values for each element in the vectors
        for (int i = 0; i < VEC_SIZE; ++i) {
            float a = a_vec[i];
            T t = pSparseData[pos1 + i];
            float diff = a - t_vec[i];
            diff = (t > static_cast<T>(0.0)) ? fminf(0.0f, diff) : fmaxf(0.0f, diff);
            pDelta[pos2] = w * diff;
            pos1 += cData._warpSize;
        }

        // Store delta values in the shared memory tile
        for (int i = 0; i < VEC_SIZE; ++i) {
            tile[ty][tx * VEC_SIZE + i] = delta_vec[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Reduce delta values within each thread block using parallel reduction
        for (int i = 16; i > 0; i /= 2) {
            int mask = (1 << i) - 1;
            for (int j = 0; j < VEC_SIZE; ++j) {
                delta_vec[j] += __shfl_down_sync(mask, delta_vec[j], i);
            }
        }

        // Store the final delta value in the global memory
        if (ty == 0) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                tile[tx][i * TILE_DIM + row % TILE_DIM] = delta_vec[i];
            }
        }

        // Synchronize threads within the block again
        __syncthreads();

        // Calculate the column index for the delta value
        int col = blockIdx.x * TILE_DIM + tx;
        delta_vec[0] = tile[ty][col];

        // Perform a reduction along the x-axis of the tile
        for (int i = 1; i < BLOCK_ROWS; i++) {
            delta_vec[0] += tile[i][col];
        }

        // Store the final delta value in the global memory
        if (tx == 0) {
            pDelta[uOffset + row] = delta_vec[0];
        }
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    uint64_t size               = (uint64_t)batch * (uint64_t)stride;
    dim3 grid1(CalculateBlocks(size));
    dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

    try {
        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroSigmoidL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawTanhL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroTanhL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLinearL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroLinearL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawRELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroRELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawLRELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, slope);
            }
            kCalculateIndexedSparseAnalogNonZeroLRELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, slope);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha);
            }
            kCalculateIndexedSparseAnalogNonZeroELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSELUL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta, alpha, lambda);
            }
            kCalculateIndexedSparseAnalogNonZeroSELUL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, alpha, lambda);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSoftMaxL2HingeOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroSoftMaxL2HingeOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * (a - t);      
    }
}
        
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * (a - t);      
    }
        }

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * (a - t);      
    }
}

template<typename T> void kCalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
        case SoftMax:
            kCalculateSigmoidCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * (a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * (a - t);      
    }
        }

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * (a - t);      
    }
}

template<typename T> 
void kCalculateIndexedCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
        case SoftMax:
            kCalculateIndexedSigmoidCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * a;       
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };

        dim3 blockDim(getGpu()._threadsPerBlock);

        switch (activation)
        {
        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
                break;
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSoftMaxOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
                break;
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * (a - (float)1.0);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };

        dim3 blockDim(getGpu()._threadsPerBlock);

        switch (activation)
        {
        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
                break;
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSoftMaxOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
                break;
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSigmoidCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
        }

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }
    }
}

template<typename T>
void kCalculateSparseAnalogCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case SoftMax:
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
        }

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint32_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            pDelta[pos2]        = w * (a - t);
            pos1               += cData._warpSize;
        }      
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        switch (activation)
        {
        case SoftMax:
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(position, pDataWeight, stride, size, pUnit, pDelta);
            }
            kCalculateIndexedSparseAnalogNonZeroSigmoidCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float output          = (float)0.0;
        if ((t == (float)1.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t); 
        pDelta[uOffset + pos]   = w * output;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float output          = (float)0.0;
        if ((t == (float)1.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t); 
        pDelta[uOffset + pos]   = w * output;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;
    }
}

template<typename T> void kCalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
            kCalculateSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case SoftMax:
            kCalculateSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float output          = (float)0.0;
        if ((t == (float)1.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t); 
        pDelta[uOffset + pos]   = w * output;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float output          = (float)0.0;
        if ((t == (float)1.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t); 
        pDelta[uOffset + pos]   = w * output;
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float output          = (float)0.0;
        if ((t > (float)0.0) && (a < cData._SMCE_oneTarget))
            output              = cData._SMCE_oneScale * (a - t);
        else if ((t == (float)0.0) && (a > cData._SMCE_zeroTarget))
            output              = cData._SMCE_zeroScale * (a - t);   
        pDelta[uOffset + pos]   = w * output;
    }
}

template<typename T> void kCalculateIndexedScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
            kCalculateIndexedSigmoidScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case SoftMax:
            kCalculateIndexedSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._SMCE_zeroScale;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        float output          = (float)0.0;
        if (a > cData._SMCE_zeroTarget)
            output              = w * a;
        pDelta[pos]             = output;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float output      = (float)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = w * (a - (float)1.0);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float a               = pUnit[pos];
        float output          = (float)0.0;
        if (a > cData._SMCE_zeroTarget)
            output              = cData._SMCE_zeroScale * a;
        pDelta[pos]             = output;   
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1));
        uint64_t offset         = pos * stride;
        pos1                   += threadIdx.x & cData._warpMask;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float output      = (float)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = (a - w);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };

        dim3 blockDim(getGpu()._threadsPerBlock);

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float output      = (float)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = w * (a - (float)1.0);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0 / (float)(end - pos1);
        uint64_t offset         = pos * stride;
        pos1                   += threadIdx.x & cData._warpMask;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float output      = (float)0.0;
            if (a < cData._SMCE_oneTarget)
                output          = cData._SMCE_oneScale * (a - w);
            pDelta[pos2]        = output;
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight };

        dim3 blockDim(getGpu()._threadsPerBlock);

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidScaledMarginalCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSigmoidScaledMarginalCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case SoftMax:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSoftMaxScaledMarginalCrossEntropyOutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel(uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
          float a               = pUnit[pos];
          float output          = (float)0.0;
          if (a > cData._SMCE_zeroTarget)
          {
              output              = cData._SMCE_zeroScale * a;
          }
          pDelta[pos]             = output;
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
              uint64_t pos2       = offset + pSparseIndex[pos1];
              float a           = pUnit[pos2];
              T t                 = pSparseData[pos1];
              float output      = (float)0.0;
              if (a < cData._SMCE_oneTarget)
              {
                  output          = cData._SMCE_oneScale * t * (a - (float)1.0);
              }
              pDelta[pos2]        = output;
              pos1               += cData._warpSize;
        }
    }
}

template<typename T>
void kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            }
            kCalculateSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            break;

        case SoftMax:
            std::cout << "unsupported activation for this cost function" << '\n';
            getGpu().Shutdown();
            exit(-1);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
              uint64_t pos2       = offset + pSparseIndex[pos1];
              float a           = pUnit[pos2];
              T t                 = pSparseData[pos1];
              float output      = (float)0.0;
              if (a < cData._SMCE_oneTarget)
              {
                  output          = cData._SMCE_oneScale * t * (a - (float)1.0);
              }
              pDelta[pos2]        = output;
              pos1               += cData._warpSize;
        }
    }
}

template<typename T>
void kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;
        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                kCalculateSparseRawSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel<<<grid1, getGpu()._threadsPerBlock>>>(size, pUnit, pDelta);
            }
            kCalculateIndexedSparseNonZeroSigmoidDataScaledMarginalCrossEntropyOutputDelta_kernel<<<grid2, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
            break;

        case SoftMax:
            std::cout << "unsupported activation for this cost function" << '\n';
            getGpu().Shutdown();
            exit(-1);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((float)1.0 - a * a);      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t);  
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a- t) * ((float)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t); 
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);   
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, unsigned char* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((float)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);  
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, char* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<typename T> void kCalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
            kCalculateSigmoidL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateTanhL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case Linear:
            kCalculateLinearL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateLRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateSELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pData, pDataWeight, alpha, lambda);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((float)1.0 - a * a);      
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t);  
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = pData[dOffset + pos];
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}


template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a- t) * ((float)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t); 
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);   
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, unsigned char* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 256.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * a * ((float)1.0 - a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((float)1.0 - a * a);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t);      
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * (a > (float)0.0);  
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight, float slope)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight, float alpha)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, char* pData, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * float(1.0 / 128.0);
        pDelta[uOffset + pos]   = w * sgn(a - t) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

template<typename T> void kCalculateIndexedL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda)
{
    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        switch (activation)
        {
        case Sigmoid:
            kCalculateIndexedSigmoidL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Tanh:
            kCalculateIndexedTanhL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case Linear:
            kCalculateIndexedLinearL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case RectifiedLinear:
            kCalculateIndexedRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight);
            break;

        case LeakyRectifiedLinear:
            kCalculateIndexedLRELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, slope);
            break;

        case ExponentialLinear:
            kCalculateIndexedELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha);
            break;

        case ScaledExponentialLinear:
            kCalculateIndexedSELUL1OutputDelta_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, batch, stride, pUnit, pDelta, pIndex, pData, pDataWeight, alpha, lambda);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSigmoidL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w *  sgn(a) * a * ((float)1.0 - a);      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0) * a * ((float)1.0 - a);      
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawTanhL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit,  float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];    
        pDelta[pos]             = w * sgn(a) * ((float)1.0 - a * a);          
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((float)1.0 - a * a);     
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLinearL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];  
        pDelta[pos]             = w * sgn(a);
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0);    
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawRELUL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * (a > (float)0.0);          
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (float)1.0) * (a > (float)0.0);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawELUL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta, float alpha)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * sgn(a) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));    
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a > (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawSELUL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta, float alpha, float lambda)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * sgn(a) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));  
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawLRELUL1OutputDelta_kernel(uint32_t position, float* pDataWeight, uint32_t stride, uint64_t size, float* pUnit, float* pDelta, float slope)
{
    uint64_t pos                = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < size)
    {
        float w               = cData._deltaBoost_zero;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        pDelta[pos]             = w * sgn(a) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroRawLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];  
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, &slope, &alpha, &lambda };

        dim3 blockDim(getGpu()._threadsPerBlock);

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSigmoidL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroTanhL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroLinearL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroRELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroRawLRELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateSparseNonZeroSELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSigmoidL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0) * a * ((float)1.0 - a);      
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroTanhL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((float)1.0 - a * a);     
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroLinearL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a - (float)1.0);    
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (float)1.0) * (a > (float)0.0);
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            pDelta[pos2]        = w * sgn(a > (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * (a + alpha));
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroSELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float alpha, float lambda)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((a > (float)0.0) * lambda + (a <= (float)0.0) * lambda * alpha * exp(a));  
            pos1               += cData._warpSize;
        }      
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroRawLRELUL1OutputDelta_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t *pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, float slope)
{
    uint64_t pos                = ((blockIdx.x * blockDim.x) + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._deltaBoost_one * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];  
            pDelta[pos2]        = w * sgn(a - (float)1.0) * ((a > (float)0.0) + (a <= (float)0.0) * slope);
            pos1               += cData._warpSize;
        }      
    }
}

void kCalculateIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda)
{
    try {
        uint64_t size = (uint64_t)batch * (uint64_t)stride;

        dim3 grid1(CalculateBlocks(size));
        dim3 grid2(CalculateBlocks(batch * getGpu()._data._warpSize));

        dim3 blockDim(getGpu()._threadsPerBlock);

        if (bSparseIgnoreZero)
        {
            cudaMemset(pDelta, 0, size * sizeof(float));
        }

        void* args[] = { &position, &batch, &stride, &size, pUnit, pDelta, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, &slope, &alpha, &lambda };

        switch (activation)
        {
        case Sigmoid:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSigmoidL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSigmoidL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawTanhL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroTanhL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case Linear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLinearL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroLinearL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case RectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawRELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroRELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawLRELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroRawLRELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            if (!bSparseIgnoreZero)
            {
                cudaLaunchKernel((void*)kCalculateSparseRawSELUL1OutputDelta_kernel, grid1, blockDim, args, 0, nullptr);
            }
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroSELUL1OutputDelta_kernel, grid2, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparsenessPenalty_kernel(uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float p, float beta)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos < stride)
    {
        float pi              = (float)0.0;
        for (int i = 0; i < batch; i++)
        {
            pi                 += pUnit[pos];
            pos                += stride;
        }

        pi                     /= (float)batch;
        pi                      = max(MIN_ACTIVATION, min(MAX_ACTIVATION, pi));
        float penalty         = beta * (-p / pi + ((float)1.0 - p) / ((float)1.0 - pi));
        
        pos                     = blockIdx.x * blockDim.x + threadIdx.x;
        for (int i = 0; i < batch; i++)
        {
            pDelta[pos]        += penalty;
            pos                += stride;
        }
    }
}


void kCalculateSparsenessPenalty(uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, float p, float beta)
{
    try {
        dim3 grid1(CalculateBlocks(stride));

        void* args[] = {
            &batch,
            &stride,
            &pUnit,
            &pDelta,
            &p,
            &beta
        };

        cudaLaunchKernel((void*)kCalculateSparsenessPenalty_kernel, grid1, blockDim, args, 0, nullptr);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSigmoidHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];
        float d               = pDelta[pos];
        pDelta[pos]             = x * ((float)1.0 - x) * d;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateTanhHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta, float scale, float oneOverScale)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];
        float d               = pDelta[pos];
        x                      *= oneOverScale;
        pDelta[pos]             = scale * ((float)1.0 - x * x) * d;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateRELUHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];
        if (x <= (float)0.0)
            pDelta[pos]         = (float)0.0;
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateLRELUHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta, float slope)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];
        if (x <= (float)0.0)
        {
            pDelta[pos]         *= slope;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateELUHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta, float alpha)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];
        if (x <= (float)0.0)
            pDelta[pos]        *= (x + alpha);            
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSELUHadamardProduct_kernel(uint64_t size, float* pUnit, float* pDelta, float alpha, float lambda)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x               = pUnit[pos];        
        float delta           = pDelta[pos];
        if (x > (float)0.0)
        {
            delta              *= lambda;
        }
        else
        {
            delta              *= (x + lambda * alpha);
        }
        pDelta[pos]             = delta;
    }
}

void kCalculateHadamardProduct(Activation activation, uint64_t size, float scale, float* pUnit, float* pDelta, float slope, float alpha, float lambda)
{
    try {
        uint32_t blocks = CalculateBlocks(size);
        float oneOverScale = 1.0f / scale;

        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &size, pUnit, pDelta, &oneOverScale, &slope, &alpha, &lambda };

        switch (activation)
        {
        case Sigmoid:
            cudaLaunchKernel((void*)kCalculateSigmoidHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;

        case Tanh:
            cudaLaunchKernel((void*)kCalculateTanhHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;

        case Linear:
            break;

        case RectifiedLinear:
            cudaLaunchKernel((void*)kCalculateRELUHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;

        case LeakyRectifiedLinear:
            cudaLaunchKernel((void*)kCalculateLRELUHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;

        case ExponentialLinear:
            cudaLaunchKernel((void*)kCalculateELUHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;

        case ScaledExponentialLinear:
            cudaLaunchKernel((void*)kCalculateSELUHadamardProduct_kernel, gridDim, blockDim, args, 0, nullptr);
            break;
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kNormalizeDeltas_kernel(float norm, uint32_t batch, uint32_t stride, float* pDelta)
{
    uint32_t dpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;
    pDelta += dpos * stride;

    if (dpos < batch)
    {
        uint32_t pos = tgx;
        float r2 = 0.0f;

        while (pos < stride)
        {
            float x = pDelta[pos];
            r2 += x * x;
            pos += cData._warpSize;
        }

        for (int offset = cData._warpSize / 2; offset > 0; offset /= 2)
        {
            r2 += __shfl_down_sync(0xffffffff, r2, offset);
        }

        if (tgx == 0)
        {
            r2 = sqrtf(r2);
            if (r2 > norm)
            {
                norm = 1.0f / r2;
                pos = tgx;

                while (pos < stride)
                {
                    pDelta[pos] *= norm;
                    pos += cData._warpSize;
                }
            }
        }
    }
}

void kNormalizeDeltas(float norm, uint32_t batch, uint32_t stride, float* pDelta)
{
    try {
        uint32_t blocks = (batch + 3) / 4;
        unsigned long threadsPerBlock = 128;
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &norm, &batch, &stride, &pDelta };
        cudaLaunchKernel((void*)kNormalizeDeltas_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateDeltaMagnitudes_kernel(uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude)
{
    uint32_t dpos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;
    pDelta += dpos * stride;

    if (dpos < batch)
    {
        uint32_t pos = tgx;
        float r2 = 0.0f;

        while (pos < stride)
        {
            float x = pDelta[pos];
            r2 += x * x;
            pos += cData._warpSize;
        }

        for (int offset = cData._warpSize / 2; offset > 0; offset /= 2)
        {
            r2 += __shfl_down_sync(0xffffffff, r2, offset);
        }

        if (tgx == 0)
            pMagnitude[dpos] = sqrtf(r2);
    }
}

void kCalculateDeltaMagnitudes(uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude)
{
    try {
        uint32_t blocks = (batch + 3) / 4;
        unsigned long threadsPerBlock = 128;
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &batch, &stride, &pDelta, &pMagnitude };
        cudaLaunchKernel((void*)kCalculateDeltaMagnitudes_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kNormalizeDeltaMagnitudes_kernel(float norm, uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude)
{
    uint32_t dpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    pDelta                                 += dpos * stride;
    if (dpos < batch)
    {    
        float r2                          = pMagnitude[dpos];
        if (r2 > norm * norm)
        {
            norm                           *= rsqrt(r2);
            uint32_t pos                    = tgx;
            while (pos < stride)
            {
                pDelta[pos]                *= norm;
                pos                        += cData._warpSize;
            }
        }    
        
   }  
}

void kNormalizeDeltaMagnitudes(float norm, uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude)
{
    try {
        uint32_t blocks = (batch + 3) / 4;
        unsigned long threadsPerBlock = 128;
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &norm, &batch, &stride, &pDelta, &pMagnitude };
        cudaLaunchKernel((void*)kNormalizeDeltaMagnitudes_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();  // Wait for the kernel to finish
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateMaxoutDelta_kernel(float* pSrc, float* pSrcDelta, size_t size, float beta, float* pDst, float* pDstDelta)
{
    uint64_t pos                        = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float s = pSrc[pos];
        float sdelta = pSrcDelta[pos];
        float d = pDst[pos];
        float delta                   = (s == d) ? sdelta : (float)0;
        
        if (beta == (float)0)
            pDstDelta[pos]              = delta;
        else if (delta != (float)0.0)
            pDstDelta[pos]              = beta * pDstDelta[pos] + delta;
    }
}


void kCalculateMaxoutDelta(float* pSrc, float* pSrcDelta, size_t size, float beta, float* pDst, float* pDstDelta)
{
    try {
        unsigned long blocks = CalculateBlocks(size);
        unsigned long threadsPerBlock = getGpu()._threadsPerBlock;
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &pSrc, &pSrcDelta, &size, &beta, &pDst, &pDstDelta };
        cudaLaunchKernel((void*)kCalculateMaxoutDelta_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateCosineDelta_kernel(float* pDPDelta, float* pDP, float* pA, float* pB, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride)
{
    p0Vector               += blockIdx.x * inputStride + threadIdx.x;
    pVector                += blockIdx.x * inputStride + threadIdx.x;
    pDPDelta               += blockIdx.x * stride;
    pDP                    += blockIdx.x * stride;
    pA                     += blockIdx.x * stride;
    pB                     += blockIdx.x * stride;    
    pDelta0                += blockIdx.x * inputStride + threadIdx.x;
    pDelta                 += blockIdx.x * inputStride + threadIdx.x;    
    uint32_t pos            = threadIdx.x;
    float dp              = *pDP;
    float dpDelta         = *pDPDelta;
    float a               = *pA;
    float b               = *pB;
    float ab              = a * b;
    float a2              = a * a;
    float b2              = b * b;
    
    while (pos < inputStride)
    {
        float ai          = *p0Vector;
        float bi          = *pVector;

        float delta0      = dpDelta * ((bi / ab) - (ai * dp / a2));
        float delta       = dpDelta * ((ai / ab) - (bi * dp / b2));
        if (beta0 == (float)0)
            *pDelta0        = delta0;
        else
            *pDelta0        = *pDelta0 + beta0 * delta0;
        if (beta == (float)0)
            *pDelta         = delta;
        else
            *pDelta         = *pDelta + beta * delta;        
    
        pDelta0            += blockDim.x;
        pDelta             += blockDim.x;     
        p0Vector           += blockDim.x;
        pVector            += blockDim.x;
        pos                += blockDim.x;
    }
}

void kCalculateCosineDelta(float* pDPDeltaIn, float* pDPIn, float* pA, float* pB, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride)
{
    try {
        unsigned long blocks = batch;
        unsigned long threadsPerBlock = std::min(stride, getGpu()._threadsPerBlock);
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &pDPDeltaIn, &pDPIn, &pA, &pB, &p0Vector, &pVector, &batch, &stride, &pDelta0, &beta0, &pDelta, &beta, &inputStride };
        cudaLaunchKernel((void*)kCalculateCosineDelta_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();  // Wait for the kernel to finish
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateDotProductDelta_kernel(float* pDPDelta, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride)
{
    p0Vector               += blockIdx.x * inputStride + threadIdx.x;
    pVector                += blockIdx.x * inputStride + threadIdx.x;
    pDPDelta               += blockIdx.x * stride; 
    pDelta0                += blockIdx.x * inputStride + threadIdx.x;
    pDelta                 += blockIdx.x * inputStride + threadIdx.x;    
    uint32_t pos            = threadIdx.x;
    float dpDelta         = *pDPDelta;
    
    while (pos < inputStride)
    {
        float ai          = *p0Vector;
        float bi          = *pVector;
        float delta0      = dpDelta * bi;
        float delta       = dpDelta * ai;
        if (beta0 == (float)0)
            *pDelta0        = delta0;
        else
            *pDelta0        = *pDelta0 + beta0 * delta0;
        if (beta == (float)0)
            *pDelta         = delta;
        else
            *pDelta         = *pDelta + beta * delta;        
    
        pDelta0            += blockDim.x;
        pDelta             += blockDim.x;     
        p0Vector           += blockDim.x;
        pVector            += blockDim.x;
        pos                += blockDim.x;
    }
}

void kCalculateDotProductDelta(float* pDPDelta, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride)
{
    try {
        unsigned long blocks = batch;
        unsigned long threadsPerBlock = std::min(stride, getGpu()._threadsPerBlock);
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &pDPDelta, &p0Vector, &pVector, &batch, &stride, &pDelta0, &beta0, &pDelta, &beta, &inputStride };
        cudaLaunchKernel((void*)kCalculateDotProductDelta_kernel, gridDim, blockDim, args, 0, nullptr);

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}


/////////////////////// Loss KERNELS ///////////////////////

/// <summary>
/// CUDA kernel to calculate sparse raw L1 error.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="size">The size parameter.</param>
__global__ void LAUNCH_BOUNDS() kCalculateSparseRawL1Error_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint64_t stride, uint64_t size)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos < size)
    {
        float w = 1.0f;

        if (pDataWeight != NULL)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            sharedWeight[threadIdx.x] = pDataWeight[dpos];
            __syncthreads();
            w *= sharedWeight[threadIdx.x];
        }

        float a = pUnit[pos];
        float error = 0.0f;

#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            error += w * fabsf(a);
        }
    }
}

/// <summary>
/// CUDA kernel to calculate sparse L1 error for non-zero elements.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pSparseStart">Pointer to the sparse data start indices.</param>
/// <param name="pSparseEnd">Pointer to the sparse data end indices.</param>
/// <param name="pSparseIndex">Pointer to the sparse data indices.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
__global__ void LAUNCH_BOUNDS() kCalculateSparseNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;

        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float diff = fabsf(a - 1.0f) - fabsf(a);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}

/// <summary>
/// CUDA kernel to calculate sparse L1 error for non-zero elements.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pSparseStart">Pointer to the sparse data start indices.</param>
/// <param name="pSparseEnd">Pointer to the sparse data end indices.</param>
/// <param name="pSparseIndex">Pointer to the sparse data indices.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
__global__ void LAUNCH_BOUNDS() kCalculateSparseOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        uint64_t offset = pos * stride;

        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float diff = fabsf(a - 1.0f);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}

/// <summary>
/// Function to calculate sparse L1 error with optional zero element exclusion.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pSparseStart">Pointer to the sparse data start indices.</param>
/// <param name="pSparseEnd">Pointer to the sparse data end indices.</param>
/// <param name="pSparseIndex">Pointer to the sparse data indices.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="bSparseIgnoreZero">Flag to ignore zero elements.</param>
/// <returns>The calculated sparse L1 error.</returns>
float kCalculateSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseOnlyNonZeroL1Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL1Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseNonZeroL1Error_kernel, gridDim, blockDim, args2, 0, nullptr);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float diff = fabsf(a - 1.0f) - fabsf(a);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}

__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            error += sharedWeight[threadIdx.x] * fabsf(a - 1.0f);
            pos1 += cData._warpSize;
        }
    }
}

float kCalculateIndexedSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseOnlyNonZeroL1Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL1Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroL1Error_kernel, gridDim, blockDim, args2, 0, nullptr);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            error += sharedWeight[threadIdx.x] * fabsf(a - t);
            pos1 += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            float diff = fabsf(a - t) - fabsf(a);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (1.0f / 256.0f);
            float diff = fabsf(a - t);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}


template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (1.0f / 256.0f);
            float diff = fabsf(a - t) - fabsf(a);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (1.0f / 128.0f);
            float diff = fabsf(a - t);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (1.0f / 128.0f);
            float diff = fabsf(a - t) - fabsf(a);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}

template<typename T>
float kCalculateSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogOnlyNonZeroL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogNonZeroL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseAnalogL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            float diff = fabsf(a - t);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            float diff = fabsf(a - t) - fabsf(a);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (1.0f / 256.0f);
            float diff = fabsf(a - t);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}


template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (1.0f / 256.0f);
            float diff = fabsf(a - t) - fabsf(a);
            error += sharedWeight[threadIdx.x] * diff;
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (1.0f / 128.0f);
            error += sharedWeight[threadIdx.x] * fabsf(a - t);
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL1Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        uint64_t offset = pos * stride;

        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f;
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (1.0f / 128.0f);
            error += sharedWeight[threadIdx.x] * (fabsf(a - t) - fabsf(a));
            pos1 += cData._warpSize;
        }
    }
}

template<typename T>
float kCalculateIndexedSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogOnlyNonZeroL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogNonZeroL1Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseAnalogL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void LAUNCH_BOUNDS() kCalculateSparseRawL2Error_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = blockDim.x * blockIdx.x + threadIdx.x;
    float error = 0.0f;

    if (pos < size)
    {
        float w = 0.5f;
        if (pDataWeight != NULL)
        {
            uint64_t dpos = (pos / stride) + position;
            dpos = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w *= pDataWeight[dpos];
        }
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < cData._warpSize; ++i)
        {
            float a = pUnit[pos + i];
            error += sharedWeight[i] * a * a;
        }
    }
}

__global__ void LAUNCH_BOUNDS() kCalculateSparseOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];

        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = pos * stride + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float diff = a - 1.0f;
            error += sharedWeight[threadIdx.x] * (diff * diff);
            pos1 += cData._warpSize;
        }
    }
}

__global__ void LAUNCH_BOUNDS() kCalculateSparseNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];

        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = pos * stride + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float diff = a - 1.0f;
            error += sharedWeight[threadIdx.x] * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }
}


float kCalculateSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseOnlyNonZeroL2Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL2Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseNonZeroL2Error_kernel, gridDim, blockDim, args2, 0, nullptr);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseL2Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];

        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = pos * stride + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            float diff = a - t;
            error += sharedWeight[threadIdx.x] * (diff * diff);
            pos1 += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];

        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = pos * stride + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            float diff = a - t;
            error += sharedWeight[threadIdx.x] * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];

        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = pos * stride + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = static_cast<float>(pSparseData[pos1]) * (1.0f / 256.0f);
            float diff = a - t;
            error += sharedWeight[threadIdx.x] * (diff * diff);
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];

        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = pos * stride + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = static_cast<float>(pSparseData[pos1]) * (1.0f / 256.0f);
            float diff = a - t;
            error += w * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    extern __shared__ float sharedWeight[];

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = 0.5f;

        if (pDataWeight != NULL)
        {
            sharedWeight[threadIdx.x] = pDataWeight[dpos];
            __syncthreads();
            w *= sharedWeight[threadIdx.x];
        }

        uint64_t offset = pos * stride;

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = static_cast<float>(pSparseData[pos1]) * (1.0f / 128.0f);
            float diff = a - t;
            error += w * (diff * diff);
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    extern __shared__ float sharedWeight[];

    if (pos < batch)
    {
        uint32_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = 0.5f;

        if (pDataWeight != NULL)
        {
            sharedWeight[threadIdx.x] = pDataWeight[dpos];
            __syncthreads();
            w *= sharedWeight[threadIdx.x];
        }

        uint64_t offset = pos * stride;

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = static_cast<float>(pSparseData[pos1]) * (1.0f / 128.0f);
            float diff = a - t;
            error += w * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }
}


template<typename T>
float kCalculateSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogOnlyNonZeroL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = batch * stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogNonZeroL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseAnalogL2Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float diff = a - 1.0f;
            error += sharedWeight[threadIdx.x] * (diff * diff);
            pos1 += cData._warpSize;
        }
    }
}

__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float diff = a - 1.0f;
            error += sharedWeight[threadIdx.x] * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }
}


#include <stdexcept>

float kCalculateIndexedSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseOnlyNonZeroL2Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL2Error_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroL2Error_kernel, gridDim, blockDim, args2, 0, nullptr);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseL2Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    extern __shared__ float sharedWeight[];

    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

        sharedWeight[threadIdx.x] = w;
        __syncthreads();

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            float diff = a - t;
            error += sharedWeight[threadIdx.x] * (diff * diff);
            pos1 += cData._warpSize;
        }
    }
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            T t = pSparseData[pos1];
            float diff = a - t;
            error += w * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff = a - t;
            error += w * (diff * diff);
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff = a - t;
            error += w * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogOnlyNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff = a - t;
            error += w * (diff * diff);
            pos1 += cData._warpSize;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedSparseAnalogNonZeroL2Error_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error = 0.0f;

    if (pos < batch)
    {
        uint32_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1 = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end = pSparseEnd[dpos];
        float w = 0.5f * ((pDataWeight != NULL) ? pDataWeight[dpos] : 1.0f);
        uint64_t offset = pos * stride;

#pragma unroll
        while (pos1 < end)
        {
            uint64_t pos2 = offset + pSparseIndex[pos1];
            float a = pUnit[pos2];
            float t = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff = a - t;
            error += w * (diff * diff - a * a);
            pos1 += cData._warpSize;
        }
    }
}

template<typename T>
float kCalculateIndexedSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogOnlyNonZeroL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = batch * stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogNonZeroL2Error_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseAnalogL2Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawL2HingeError_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    float error               = (float)0.0;
    if (pos < size)
    {
        float w               = (float)0.5;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = max((float)0.0, pUnit[pos]);
        error                   = w * a * a;
    }
    
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);
            a                   = max((float)0.0, a);
            error              += w * (diff * diff - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseOnlyNonZeroL2HingeError_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL2HingeError_kernel, gridDim, blockDim, args, 0, nullptr);

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseNonZeroL2HingeError_kernel, gridDim, blockDim, args2, 0, nullptr);

            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseL2HingeError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            a                   = max((float)0.0, a);
            diff                = (t > (T)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);          
            error              += w * (diff * diff - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - t;
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);             
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - t;
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);   
            a                   = max((float)0.0, a);  
            error              += w * (diff * diff - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf((float)t);
            diff                = (t > (float)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);           
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            a                   = max((float)0.0, a);
            diff                = (t > (float)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);       
            error              += w * (diff * diff - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}


#include <stdexcept>

template<typename T>
float kCalculateSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogOnlyNonZeroL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = batch * stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateSparseAnalogNonZeroL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseAnalogL2HingeError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float diff        = min((float)0.0, pUnit[pos2] - (float)1.0);
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float diff        = min((float)0.0, a - (float)1.0);
            a                   = max((float)0.0, a);
            error              += w * (diff * diff - a * a); 
            pos1               += cData._warpSize;
        }
    }  

}


float kCalculateIndexedSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseOnlyNonZeroL2HingeError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawL2HingeError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroL2HingeError_kernel, gridDim, blockDim, args2, 0, nullptr);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseL2HingeError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            diff                = (t > (T)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);         
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            float diff        = a - fabsf(t);
            a                   = max((float)0.0, a);
            diff                = (t > (T)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);          
            error              += w * (diff * diff - a * a);               
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - t;
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff); 
            error              += w * diff * diff;   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            float diff        = a - t;
            diff                = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);    
            a                   = max((float)0.0, a);  
            error              += w * (diff * diff - a * a);   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            diff                = (t > (float)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);      
            error              += w * diff * diff;  
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            float diff        = a - fabsf(t);
            a                   = max((float)0.0, a);
            diff                = (t > (float)0.0) ? min((float)0.0f , diff) : max((float)0.0, diff);          
            error              += w * (diff * diff - a * a);  
            pos1               += cData._warpSize;
        }
    }  

}


template<typename T>
float kCalculateIndexedSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogOnlyNonZeroL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = batch * stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, pDataWeight, pUnit, stride, size);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            kCalculateIndexedSparseAnalogNonZeroL2HingeError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseAnalogL2HingeError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawCrossEntropyError_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    float error               = (float)0.0;
    if (pos < size)
    {
        float w               = (float)1.0;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = (pos / stride) + position;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[dpos] : dpos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        error                   = -w * log(max(MIN_ERROR, (float)1.0 - a));     
    }

}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += -w * log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  


}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += w * (-log(max(MIN_ERROR, a)) + log(max(MIN_ERROR, (float)1.0 - a)));   
            pos1               += cData._warpSize;
        }
    }  

}

#include <stdexcept>

float kCalculateSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseOnlyNonZeroCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseNonZeroCrossEntropyError_kernel, gridDim, blockDim, args2, 0, nullptr);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseOnlyNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += -w * log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += w * (-log(max(MIN_ERROR, a)) + log(max(MIN_ERROR, (float)1.0 - a)));   
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateIndexedSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseOnlyNonZeroCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroCrossEntropyError_kernel, gridDim, blockDim, args2, 0, nullptr);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight == NULL) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos];
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += -w * log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
        cudaLaunchKernel((void*)kCalculateSparseMultinomialCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseMultinomialCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight == NULL) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos];
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            error              += -w * log(max(MIN_ERROR, a));   
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateIndexedSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
        cudaLaunchKernel((void*)kCalculateIndexedSparseMultinomialCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseMultinomialCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            error              += w * (-t * log(max(MIN_ERROR, a)));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * (-t * log(max(MIN_ERROR, a)));   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * (-t * log(max(MIN_ERROR, a)));   
            pos1               += cData._warpSize;
        }
    }  

}


template<typename T>
float kCalculateSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseAnalogMultinomialCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseAnalogMultinomialCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];
            error              += w * (-t * log(max(MIN_ERROR, a)));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 256.0);
            error              += w * (-t * log(max(MIN_ERROR, a)));   
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{

    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = (float)pSparseData[pos1] * (float)(1.0 / 128.0);
            error              += w * (-t * log(max(MIN_ERROR, a)));   
            pos1               += cData._warpSize;
        }
    }  

}


template<typename T>
float kCalculateIndexedSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseAnalogMultinomialCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseAnalogMultinomialCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawScaledMarginalCrossEntropyError_kernel(uint32_t position, float* pDataWeight, float* pUnit, uint32_t stride, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    float error               = (float)0.0;
    if (pos < size)
    {
        float w               = cData._SMCE_zeroScale;
        if (pDataWeight != NULL)
        {
            uint64_t dpos       = pos / stride;
            dpos                = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
            w                  *= pDataWeight[dpos];
        }
        float a               = pUnit[pos];
        if (a > cData._SMCE_zeroTarget)
            error               = -w * log(max(MIN_ERROR, (float)1.0 - a));     
    }
    
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            if (a < cData._SMCE_oneTarget)
                error          += -w * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            if (a > cData._SMCE_zeroTarget)
            {
                error          += w * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
            }
            if (a < cData._SMCE_oneTarget)
            {
                error          += -w * cData._SMCE_oneScale * log(max(MIN_ERROR, a));
            }
            pos1               += cData._warpSize;
        }


    }  

}

float kCalculateSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateSparseNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args2, 0, nullptr);
            cudaDeviceSynchronize();
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            if (a < cData._SMCE_oneTarget)
                error          += -w * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            if (a > cData._SMCE_zeroTarget)
            {
                error          += w * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
            }
            if (a < cData._SMCE_oneTarget)
            {
                error          += -w * cData._SMCE_oneScale * log(max(MIN_ERROR, a));
            }
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateIndexedSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (bSparseIgnoreZero)
        {
            uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseOnlyNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        else
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            dim3 blockDim(getGpu()._threadsPerBlock);
            dim3 gridDim(blocks);

            void* args[] = { &position, &pDataWeight, &pUnit, &stride, &size };
            cudaLaunchKernel((void*)kCalculateSparseRawScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }

            blocks = CalculateBlocks(batch * getGpu()._warpSize);
            gridDim = dim3(blocks);

            void* args2[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
            cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args2, 0, nullptr);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus2 = cudaGetLastError();
            if (cudaStatus2 != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus2)));
            }
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseRawDataScaledMarginalCrossEntropyError_kernel(float* pUnit, uint64_t size)
{
    uint64_t pos                = blockDim.x * blockIdx.x + threadIdx.x;
    float error               = (float)0.0;
    if (pos < size)
    {
          float a               = pUnit[pos];
          if (a > cData._SMCE_zeroTarget)
          {
              error               = -cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
          }
    }

}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseNonZeroDataScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
              uint64_t pos2       = offset + pSparseIndex[pos1];
              float a           = pUnit[pos2];
              T t                 = pSparseData[pos1];

              if (a > cData._SMCE_zeroTarget)
              {
                  error          += cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
              }

              if (a < cData._SMCE_oneTarget)
              {
                  error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
              }
              pos1               += cData._warpSize;
        }
    }

}

template<typename T>
float kCalculateSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (!bSparseIgnoreZero)
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawDataScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (pUnit, size);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseNonZeroDataScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus2 = cudaGetLastError();
        if (cudaStatus2 != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus2)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseDataScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}



template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseNonZeroDataScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
              uint64_t pos2       = offset + pSparseIndex[pos1];
              float a           = pUnit[pos2];
              T t                 = pSparseData[pos1];

              if (a > cData._SMCE_zeroTarget)
              {
                  error          += cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a));
              }

              if (a < cData._SMCE_oneTarget)
              {
                  error          += -cData._SMCE_oneScale * t * log(max(MIN_ERROR, a));
              }
              pos1               += cData._warpSize;
        }
    }

}

template<typename T>
float kCalculateIndexedSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        if (!bSparseIgnoreZero)
        {
            uint64_t size = (uint64_t)batch * (uint64_t)stride;
            uint32_t blocks = CalculateBlocks(size);
            kCalculateSparseRawDataScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (pUnit, size);
            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseNonZeroDataScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus2 = cudaGetLastError();
        if (cudaStatus2 != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus2)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseDataScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight == NULL) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos]);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            if (a < cData._SMCE_oneTarget)
                error          += -w * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
        cudaLaunchKernel((void*)kCalculateSparseNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseMultinomialScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos];
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight == NULL) ? (float)1.0 / (float)(end - pos1) : pDataWeight[dpos]);
        pos1                   += threadIdx.x & cData._warpMask;
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2]; 
            if (a < cData._SMCE_oneTarget)
                error          += -w * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

float kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 gridDim(blocks);

        void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };
        cudaLaunchKernel((void*)kCalculateIndexedSparseNonZeroScaledMarginalCrossEntropyError_kernel, gridDim, blockDim, args, 0, nullptr);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}




template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];  
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = pSparseData[pos1] * (float)(1.0 / 256.0);
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos;
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = pSparseData[pos1] * (float)(1.0 / 128.0);
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
float kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            T t                 = pSparseData[pos1];  
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = pSparseData[pos1] * (float)(1.0 / 256.0);
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float *pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData)
{
    uint64_t pos                = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    float error               = (float)0.0;
    if (pos < batch)
    {
        uint32_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + pos] : position + pos];
        uint64_t pos1           = pSparseStart[dpos] + (threadIdx.x & cData._warpMask);
        uint64_t end            = pSparseEnd[dpos];
        float w               = cData._SMCE_oneScale * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        uint64_t offset         = pos * stride;
        while (pos1 < end)
        {
            uint64_t pos2       = offset + pSparseIndex[pos1];
            float a           = pUnit[pos2];
            float t           = pSparseData[pos1] * (float)(1.0 / 128.0);
            if (a < cData._SMCE_oneTarget)
                error          += -w * t * log(max(MIN_ERROR, a));
            pos1               += cData._warpSize;
        }
    }  

}

template<typename T>
float kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);
        kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error               = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;    
        float a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        error                   = w * fabsf(a - t);        
    }

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error               = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;        
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error                   = w * fabsf(a - t);        
    }

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error               = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;            
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error                   = w * fabsf(a - t);        
    }

}

template<typename T>
float kCalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateL1Error_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error               = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        T t                     = pData[dOffset + pos];
        error                   = w * fabsf(a - t);        
    }

}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos                = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error               = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset        = blockIdx.x * stride;
        uint64_t dpos           = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset        = dpos * stride;
        float w               = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a               = pUnit[uOffset + pos];
        float t               = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error                   = w * fabsf(a - t);        
    }

}

/// <summary>
/// CUDA kernel to calculate the L1 error for indexed data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL1Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * fabsf(a - t);
    }
}

/// <summary>
/// Calculate the L1 error for indexed data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated L1 error.</returns>
template<typename T>
float kCalculateIndexedL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedL1Error_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedL1Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the L2 error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 error for data with unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array with unsigned char type.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 error for data with unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array with unsigned char type.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 error for data with unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array with unsigned char type.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2Error_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * (a - t) * (a - t);
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated indexed L2 error.</returns>
template<typename T>
float kCalculateIndexedL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedL2Error_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedL2Error: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the L2 hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        float diff = a - fabsf(t);
        diff = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error += w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 hinge error for data with an unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (unsigned char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float diff = a - t;
        diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error = w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 hinge error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float diff = a - fabsf(t);
        diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error += w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateL2HingeError_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateL2HingeError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateL2HingeError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        float diff = a - fabsf(t);
        diff = (t > (T)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error += w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 hinge error for data with an unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array (unsigned char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        float diff = a - t;
        diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error = w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the L2 hinge error for data with an unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array (unsigned char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedL2HingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (float)0.5 * ((pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0);
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        float diff = a - fabsf(t);
        diff = (t > (float)0.0) ? min((float)0.0f, diff) : max((float)0.0, diff);
        error += w * diff * diff;
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed L2 hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateIndexedL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedL2HingeError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedL2HingeError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = pData[pos];
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }
}

/// <summary>
/// CUDA kernel to calculate the hinge error for data with an unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (unsigned char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = pData[pos] * (float)(1.0 / 128.0);
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }
}

/// <summary>
/// CUDA kernel to calculate the hinge error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{

    pUnit += blockIdx.x * stride;
    uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = (float)pData[pos] * (float)(1.0 / 256.0);
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }

}

/// <summary>
/// Calculate hinge error using CUDA for a given batch of data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated hinge error.</returns>
template<typename T>
float kCalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        unsigned long threads = max(32, min(stride, 128));
        kCalculateHingeError_kernel << <batch, threads >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateHingeError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the indexed hinge error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = pData[pos];
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed hinge error for data with an unsigned char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data array.</param>
/// <param name="pData">Pointer to the data array (unsigned char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = pData[pos] * (float)(1.0 / 256.0);
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed hinge error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data array.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedHingeError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    pUnit += blockIdx.x * stride;
    uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    pData += dpos * stride;

    uint32_t pos = threadIdx.x;
    float loss = (float)0.0;
    float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;

    while (pos < stride)
    {
        float t = (float)pData[pos] * (float)(1.0 / 128.0);
        float y = pUnit[pos];
        loss += w * max((float)0.0, (float)1.0 - t * y);
        pos += blockDim.x;
    }
}

/// <summary>
/// CUDA function to calculate the indexed hinge error for a batch of data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated indexed hinge error.</returns>
template<typename T>
float kCalculateIndexedHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        unsigned long threads = max(32, min(stride, 128));
        kCalculateIndexedHingeError_kernel << <batch, threads >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedHingeError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the cross-entropy error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA function to calculate the cross-entropy error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated cross-entropy error.</returns>
template<typename T>
float kCalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the indexed cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed cross-entropy error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array (char data type).</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the indexed cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (-t * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// Calculate the indexed cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated indexed cross-entropy error.</returns>
template<typename T>
float kCalculateIndexedCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the multinomial cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// Calculate the multinomial cross-entropy error for data with a char data type.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array with char data type.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the multinomial cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// Calculate the multinomial cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error.</returns>
template<typename T>
float kCalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateMultinomialCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateMultinomialCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the multinomial cross-entropy error for indexed data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the multinomial cross-entropy error for indexed data with a specific data type (char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the multinomial cross-entropy error for indexed data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        error = w * (-t * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// Function to calculate the multinomial cross-entropy error for indexed data.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index array.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateIndexedMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedMultinomialCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedMultinomialCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        if (((t == (T)1.0) && (a < cData._SMCE_oneTarget)) ||
            ((t == (T)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for data with a specific data type (char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        if (((t == (float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (float)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for data with a specific data type (unsigned char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        if (((t == (float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (float)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// Function to calculate the scaled marginal cross-entropy error for data.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateScaledMarginalCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateScaledMarginalCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        if (((t == (T)1.0) && (a < cData._SMCE_oneTarget)) ||
            ((t == (T)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed data with a specific data type (char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        if (((t == (float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (float)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed data with a specific data type (unsigned char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        if (((t == (float)1.0) && (a < cData._SMCE_oneTarget)) || ((t == (float)0.0) && (a > cData._SMCE_zeroTarget)))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)) - ((float)1.0 - t) * cData._SMCE_zeroScale * log(max(MIN_ERROR, (float)1.0 - a)));
    }
}

/// <summary>
/// Function to calculate the scaled marginal cross-entropy error for indexed data.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateIndexedScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedScaledMarginalCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedScaledMarginalCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for multinomial data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        if ((t != (T)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for multinomial data with a specific data type (char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        if ((t != (float)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for multinomial data with a specific data type (unsigned char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        if ((t != (float)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// Function to calculate the scaled marginal cross-entropy error for multinomial data.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateMultinomialScaledMarginalCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateMultinomialScaledMarginalCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed multinomial data with a generic data type.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<typename T>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        T t = pData[dOffset + pos];
        if ((t != (T)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed multinomial data with a specific data type (char).
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 128.0);
        if ((t != (float)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// CUDA kernel to calculate the scaled marginal cross-entropy error for indexed multinomial data.
/// </summary>
/// <param name="position">The position parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
template<>
__global__ void LAUNCH_BOUNDS() kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData, float* pDataWeight)
{
    uint64_t pos = (blockIdx.y * blockDim.x) + threadIdx.x;
    float error = (float)0.0;
    if (pos < stride)
    {
        uint64_t uOffset = blockIdx.x * stride;
        uint64_t dpos = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
        uint64_t dOffset = dpos * stride;
        float w = (pDataWeight != NULL) ? pDataWeight[dpos] : (float)1.0;
        float a = pUnit[uOffset + pos];
        float t = (float)pData[dOffset + pos] * (float)(1.0 / 256.0);
        if ((t != (float)0.0) && (a < cData._SMCE_oneTarget))
            error = w * (-t * cData._SMCE_oneScale * log(max(MIN_ERROR, a)));
    }
}

/// <summary>
/// Function to calculate the scaled marginal cross-entropy error for indexed multinomial data.
/// </summary>
/// <typeparam name="T">The data type of pData.</typeparam>
/// <param name="position">The position parameter.</param>
/// <param name="batch">The batch size parameter.</param>
/// <param name="stride">The stride parameter.</param>
/// <param name="pUnit">Pointer to the input unit data.</param>
/// <param name="pIndex">Pointer to the index data.</param>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <returns>The calculated error value.</returns>
template<typename T>
float kCalculateIndexedMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight)
{
    cudaMemset(getGpu()._data._pAccumulator, 0, sizeof(uint64_t));

    try {
        dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
        kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel << <grid, getGpu()._threadsPerBlock >> > (position, stride, pUnit, pIndex, pData, pDataWeight);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error in kCalculateIndexedMultinomialScaledMarginalCrossEntropyError_kernel: " + std::string(e.what()));
    }

    getGpu()._pbAccumulator->Download();
    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

/////////////////////// KERNELS ///////////////////////

__global__ void
LAUNCH_BOUNDS()
kScaleAndBias_kernel(float* pData, uint64_t size, float scale, float bias)
{
    const uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < size)
    {
        float value = pData[offset];
        pData[offset] = scale * value - bias;
    }
}

void kScaleAndBias(float* pData, uint64_t size, float scale, float bias)
{
    const uint64_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &pData, &size, &scale, &bias };

    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kScaleAndBias_kernel,
        gridDim, blockDim,
        args,
        0,
        NULL
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}


__global__ void LAUNCH_BOUNDS() kClearUnit_kernel(float* pUnit, float* pBias, uint32_t stride, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        uint32_t bpos = pos % stride;
        pUnit[pos] = pBias[bpos];
    }
}

void kClearUnit(float* pUnit, float* pBias, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &pUnit, &pBias, &stride, &size };

    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kClearUnit_kernel,
        gridDim, blockDim,
        args,
        0,
        NULL
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

__global__ void LAUNCH_BOUNDS() kClearDualSourceUnit_kernel(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t size)
{
    const uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        const uint32_t bpos = pos % stride;
        pUnit[pos] = pBias1[bpos] + pBias2[bpos];
    }
}

void kClearDualSourceUnit(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    // Define custom launch configuration
    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &pUnit, &pBias1, &pBias2, &stride, &size };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kClearDualSourceUnit_kernel,
        gridDim, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

__global__ void LAUNCH_BOUNDS() kClearTripleSourceUnit_kernel(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    if (pos < size)
    {
        pUnit[pos] = pBias1[bpos] + pBias2[bpos] + pBias3[pos];
    }
}

void kClearTripleSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    // Define custom launch configuration
    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &pUnit, &pBias1, &pBias2, &pBias3, &stride, &size };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kClearTripleSourceUnit_kernel,
        gridDim, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

__global__ void LAUNCH_BOUNDS() kClearQuadSourceUnit_kernel(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos = pos % stride;

    if (pos < size)
    {
        pUnit[pos] = pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

void kClearQuadSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    // Define custom launch configuration
    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &pUnit, &pBias1, &pBias2, &pBias3, &pBias4, &stride, &size };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kClearQuadSourceUnit_kernel,
        gridDim, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

__global__ void LAUNCH_BOUNDS() kLoadSparseInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    const uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1                   = pos + position;                            
        pos1                            = cData._bShuffleIndices ?  cData._pShuffleIndex[pos1] : pos1;
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = (pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0;
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            const uint64_t pos2 = offset + pSparseIndex[start];
            pUnit[pos2] = w;
            start += cData._warpSize;
        }
    }
}

void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    const uint32_t last = position + batch;
    const uint32_t count = last - position;
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    // Define custom launch configuration
    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kLoadSparseInputUnit_kernel,
        gridDim, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

__global__ void LAUNCH_BOUNDS() kLoadIndexedSparseInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    const uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1                   = pos + position;                            
        pos1                            = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[pos1] : pos1];
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = (pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0;
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            const uint64_t pos2 = offset + pSparseIndex[start];
            pUnit[pos2] = w;
            start += cData._warpSize;
        }
    }
}

void kLoadIndexedSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight)
{
    const uint32_t last = position + batch;
    const uint32_t count = last - position;
    const uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    const uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    // Define custom launch configuration
    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kLoadIndexedSparseInputUnit_kernel,
        gridDim, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

template<typename T>
__global__ void LAUNCH_BOUNDS() kLoadSparseAnalogInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    const uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;

    if (pos < batch)
    {
        uint32_t pos1 = pos + position;
        pos1 = cData._bShuffleIndices ? cData._pShuffleIndex[pos1] : pos1;
        const uint64_t start = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        const uint64_t end = pSparseEnd[pos1];
        const float w = (pDataWeight != nullptr) ? pDataWeight[pos1] : 1.0f;
        const uint64_t offset = pos * stride;

        for (uint64_t i = start; i < end; i += cData._warpSize)
        {
            const uint64_t pos2 = offset + pSparseIndex[i];
            T data = pSparseData[i];
            pUnit[pos2] = w * data;
        }
    }
}

template<typename T>
void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    try {
        uint32_t last = position + batch;
        uint32_t count = last - position;
        uint32_t blocks = (count * getGpu()._warpSize + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock;
        cudaError_t status = cudaMemset(pUnit, 0, (uint64_t)batch * (uint64_t)stride * sizeof(float));
        kLoadSparseAnalogInputUnit_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kLoadSparseAnalogInputUnit: " + std::string(e.what()));
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadIndexedSparseAnalogInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {
        uint32_t pos1                   = pos + position;                            
        pos1                            = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[pos1] : pos1];
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = (pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0;
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            uint64_t pos2               = offset + pSparseIndex[start];
            T data                      = pSparseData[start];
            pUnit[pos2]                 = w * data;
            start                      += cData._warpSize;
        }
    }
}

template<typename T>
void kLoadIndexedSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    kLoadIndexedSparseAnalogInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

__global__ void
LAUNCH_BOUNDS()
kLoadSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0);
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            float value               = pRandom[start];
            uint64_t pos2               = offset + pSparseIndex[start];
            if (value >= cData._denoising_p)
                pUnit[pos2]             = w;
            start                      += cData._warpSize;
        }
    }
}


void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    // Define custom launch configuration
    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &position, &batch, &stride, &pUnit, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight, &pRandom };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kLoadSparseDenoisedInputUnit_kernel,
        gridDim, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

__global__ void
LAUNCH_BOUNDS()
kLoadIndexedSparseDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1                   = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0);
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            float value               = pRandom[start];
            uint64_t pos2               = offset + pSparseIndex[start];
            if (value >= cData._denoising_p)
                pUnit[pos2]             = w;
            start                      += cData._warpSize;
        }
    }
}

void kLoadIndexedSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    // Define custom launch configuration
    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &position, &batch, &stride, &pUnit, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight, &pRandom };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kLoadIndexedSparseDenoisedInputUnit_kernel,
        gridDim, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[pos + position] : pos + position;
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0);
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            float value               = pRandom[start];
            uint64_t pos2               = offset + pSparseIndex[start];
            T data                      = pSparseData[start];
            if (value >= cData._denoising_p)
                pUnit[pos2]             = w * data;
            start                      += cData._warpSize;
        }
    }
}

template<typename T>
void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    kLoadSparseAnalogDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadIndexedSparseAnalogDenoisedInputUnit_kernel(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom)
{
    uint32_t pos                        = (blockIdx.x * blockDim.x + threadIdx.x) / cData._warpSize;
    if (pos < batch)
    {                           
        uint32_t pos1                   = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[pos + position] : pos + position];
        uint64_t start                  = pSparseStart[pos1] + (threadIdx.x & cData._warpMask);
        uint64_t end                    = pSparseEnd[pos1];
        float w                       = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[pos1] : (float)1.0);
        uint64_t offset                 = pos * stride;
        while (start < end)
        {
            float value               = pRandom[start];
            uint64_t pos2               = offset + pSparseIndex[start];
            T data                      = pSparseData[start];
            if (value >= cData._denoising_p)
                pUnit[pos2]             = w * data;
            start                      += cData._warpSize;
        }
    }
}

template<typename T>
void kLoadIndexedSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom)
{
    uint32_t last = position + batch;
    uint32_t count = last - position;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (count * getGpu()._warpSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t status = cudaMemset(pUnit, 0, static_cast<uint64_t>(batch) * static_cast<uint64_t>(stride) * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemset failed");
    }

    kLoadIndexedSparseAnalogDenoisedInputUnit_kernel<<<blocks, threadsPerBlock>>>(position, batch, stride, pUnit, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, T* pData)
{
    uint64_t pos                        = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = pData[soffset];
    }
}

__global__ void
kLoadNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, unsigned char* pData)
{
    uint64_t pos                        = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = (float)pData[soffset] * (float)(1.0 / 256.0) - (float)0.5;
    }
}

__global__ void
kLoadNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, char* pData)
{
    uint64_t pos          = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position;
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = (float)pData[soffset] * (float)(1.0 / 128.0);
    }
}

template<typename T>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData)
{
    dim3 grid(batch, (stride + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);
    kLoadInputUnit_kernel<<<grid, getGpu()._threadsPerBlock>>>(position, stride, pUnit, pData);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

template <typename T>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t gridX = batch;
    uint32_t gridY = (stride + threadsPerBlock - 1) / threadsPerBlock;

    // Define custom launch configuration
    dim3 grid(gridX, gridY, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &position, &stride, &pUnit, &pData };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kLoadNormalizedInputUnit_kernel,
        grid, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS()
kLoadIndexedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData)
{
    uint64_t pos                        = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position];
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = pData[soffset];
    }
}

__global__ void
kLoadIndexedNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, unsigned char* pData)
{
    uint64_t pos                        = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position];
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = (float)pData[soffset] * (float)(1.0 / 256.0) - (float)0.5;
    }
}

__global__ void
kLoadIndexedNormalizedInputUnit_kernel(uint32_t position, uint32_t stride, float* pUnit, uint32_t* pIndex, char* pData)
{
    uint64_t pos          = (blockIdx.y * blockDim.x) + threadIdx.x;
    if (pos < stride)
    {
        uint32_t pos1                   = pIndex[cData._bShuffleIndices ?  cData._pShuffleIndex[blockIdx.x + position] : blockIdx.x + position];
        uint64_t soffset                = pos1 * stride + pos;
        uint64_t doffset                = blockIdx.x * stride + pos;
        pUnit[doffset]                  = (float)pData[soffset] * (float)(1.0 / 128.0);
    }
}

template<typename T>
void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t gridX = batch;
    uint32_t gridY = (stride + threadsPerBlock - 1) / threadsPerBlock;

    // Define custom launch configuration
    dim3 grid(gridX, gridY, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &position, &stride, &pUnit, &pIndex, &pData };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kLoadIndexedNormalizedInputUnit_kernel,
        grid, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

__global__ void
LAUNCH_BOUNDS()
kAddBias_kernel(float* pUnit, float* pBias, uint32_t stride, uint32_t size)
{
    uint32_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias[bpos];
    }
}


void kAddBias(float* pUnit, float* pBias, uint32_t stride, uint32_t batch)
{
    uint32_t size = stride * batch;
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &pUnit, &pBias, &stride, &size };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kAddBias_kernel,
        grid, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}


__global__ void
LAUNCH_BOUNDS()
kAddDualBias_kernel(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias1[bpos] + pBias2[bpos];
    }
}

void kAddDualBias(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &pUnit, &pBias1, &pBias2, &stride, &size };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kAddDualBias_kernel,
        grid, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

__global__ void
LAUNCH_BOUNDS()
kAddTripleBias_kernel(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias1[bpos] + pBias2[bpos] + pBias3[pos];
    }
}

void kAddTripleBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &pUnit, &pBias1, &pBias2, &pBias3, &stride, &size };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kAddTripleBias_kernel,
        grid, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

__global__ void
LAUNCH_BOUNDS()
kAddQuadBias_kernel(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t size)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bpos               = pos % stride;
    if (pos < size)
    {
        pUnit[pos]             += pBias1[bpos] + pBias2[bpos] + pBias3[pos] + pBias4[pos];
    }
}

void kAddQuadBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch)
{
    uint64_t size = static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch);
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = static_cast<uint32_t>((size + threadsPerBlock - 1) / threadsPerBlock);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &pUnit, &pBias1, &pBias2, &pBias3, &pBias4, &stride, &size };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kAddQuadBias_kernel,
        grid, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

#if (__CUDA_ARCH__ >= 600)
static const uint32_t MAXSPARSE = SM_6X_MAXSPARSE;
static const uint32_t MAXSPARSEANALOG = SM_6X_MAXSPARSEANALOG;
#elif (__CUDA_ARCH__ >= 500)
static const uint32_t MAXSPARSE = SM_5X_MAXSPARSE;
static const uint32_t MAXSPARSEANALOG = SM_5X_MAXSPARSEANALOG;
#else
static const uint32_t MAXSPARSE = SM_3X_MAXSPARSE;
static const uint32_t MAXSPARSEANALOG = SM_3X_MAXSPARSEANALOG;
#endif

__global__ void LAUNCH_BOUNDS256() kCalculateIndexedSparseZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSE];

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSE);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += w * pWeight[offset + opos];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (float)1.0;
    }
}


void kCalculateIndexedSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    uint32_t blocks = batch;

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threads, 1, 1);

    void* args[] = { &position, &stride, &pWeight, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight, &pUnit, &beta };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kCalculateIndexedSparseZ_kernel,
        grid, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ T sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit += blockIdx.x * stride;

    // New complexity: Add a loop to process multiple data blocks
    for (int blockIndex = 0; blockIndex < 2; ++blockIndex)
    {
        while (start < end)
        {
            sOpos = blockDim.x;
            uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
            uint64_t tend = start + inputs;
            uint64_t tstart = start + threadIdx.x;
            uint32_t pos = threadIdx.x;

            while (tstart < tend)
            {
                sOffset[pos] = pSparseIndex[tstart] * stride;
                sValue[pos] = w * pSparseData[tstart];
                pos += blockDim.x;
                tstart += blockDim.x;
            }

            __threadfence();
            __syncthreads();

            uint32_t tgx = threadIdx.x & cData._warpMask;
            uint32_t opos = threadIdx.x - tgx;

            while (opos < stride)
            {
                opos += tgx;
                if (opos < stride)
                {
                    float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);

                    for (uint32_t i = 0; i < inputs; i++)
                    {
                        uint32_t offset = sOffset[i];
                        unit += pWeight[offset + opos] * sValue[i];
                    }

                    unit = tanh(unit);

                    pUnit[opos] = unit;
                }
                opos -= tgx;

                if (tgx == 0)
                {
                    opos = atomicAdd(&sOpos, cData._warpSize);
                }
                opos = SHFL(opos, 0);
            }

            start = tend;
            if (start < end)
            {
                __threadfence();
                __syncthreads();
            }
            beta = (float)1.0;
        }
    }
}

template<>
__global__ void LAUNCH_BOUNDS256() kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * ((float)pSparseData[tstart] * (float)(1.0 / 256.0));
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (float)1.0;
    }
}

template<>
__global__ void LAUNCH_BOUNDS256() kCalculateSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * ((float)pSparseData[tstart] * (float)(1.0 / 256.0));
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (float)1.0;
    }
}

template<typename T>
void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogZ_kernel << <batch, threads >> > (position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS256()
kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ T sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        sOpos                       = blockDim.x;
        uint32_t inputs             = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend               = start + inputs;
        uint64_t tstart             = start + threadIdx.x;
        uint32_t pos                = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos]        = pSparseIndex[tstart] * stride;
            sValue[pos]         = w * pSparseData[tstart];
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit           += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<>
__global__ void LAUNCH_BOUNDS256() kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * ((float)pSparseData[tstart] * (float)(1.0 / 256.0));
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (float)1.0;
    }
}

template<>
__global__ void LAUNCH_BOUNDS256() kCalculateIndexedSparseAnalogZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseIndex[tstart] * stride;
            sValue[pos] = w * ((float)pSparseData[tstart] * (float)(1.0 / 128.0));
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (float)1.0;
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateIndexedSparseAnalogZ_kernel << <batch, threads >> > (position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}


__global__ void LAUNCH_BOUNDS256() kCalculateSparseDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSE];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos];
                }

                pUnit[opos] = w * unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (float)1.0;
    }
}

void kCalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    uint32_t blocks = batch;

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threads, 1, 1);

    void* args[] = { &position, &stride, &pWeight, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight, &pRandom, &pUnit, &beta };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kCalculateSparseDenoisedZ_kernel,
        grid, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

__global__ void LAUNCH_BOUNDS256() kCalculateIndexedSparseDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSE];

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;

    while (start < end)
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos];
                }

                pUnit[opos] = w * unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (float)1.0;
    }
}

void kCalculateIndexedSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    uint32_t blocks = batch;

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threads, 1, 1);

    void* args[] = { &position, &stride, &pWeight, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight, &pRandom, &pUnit, &beta };

    // Launch the kernel with custom launch parameters using cudaLaunchKernel
    cudaError_t launchStatus = cudaLaunchKernel(
        (const void*)kCalculateIndexedSparseDenoisedZ_kernel,
        grid, blockDim,
        args,  // Pass kernel arguments as an array of pointers
        0,     // Shared memory size (optional)
        NULL   // Stream identifier (optional, NULL for default stream)
    );

    cudaDeviceSynchronize();

    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS256()
kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ T sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs         = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend           = start + inputs;
        uint64_t tstart         = start + threadIdx.x;
        uint32_t pos            = threadIdx.x;

        while (tstart < tend)
        {
            float value       = pRandom[tstart];
            sOffset[pos]        = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos]         = pSparseData[tstart] * w;
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : pUnit[opos];
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit       += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<>
__global__ void LAUNCH_BOUNDS256() kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ int32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;
        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = (float)pSparseData[tstart] * (float)(1.0 / 256.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (float)1.0;
    }
}

template<>
__global__ void LAUNCH_BOUNDS256() kCalculateSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x;
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;
        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = (float)pSparseData[tstart] * (float)(1.0 / 128.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (float)1.0;
    }
}

template<typename T>
void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateSparseAnalogDenoisedZ_kernel << <batch, threads >> > (position, stride, pWeight, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}


template<typename T>
__global__ void
LAUNCH_BOUNDS256()
kCalculateIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
__shared__ uint32_t sOpos;
__shared__ uint32_t sOffset[MAXSPARSEANALOG];
__shared__ T sValue[MAXSPARSEANALOG];

    sOpos                       = blockDim.x;
    position                    = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];    
    uint64_t start              = pSparseStart[position];
    uint64_t end                = pSparseEnd[position];
    float w                   = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit                      += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs         = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend           = start + inputs;
        uint64_t tstart         = start + threadIdx.x;
        uint32_t pos            = threadIdx.x;

        while (tstart < tend)
        {
            float value       = pRandom[tstart];
            sOffset[pos]        = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos]         = pSparseData[tstart] * w;
            pos                += blockDim.x;
            tstart             += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx                = threadIdx.x & cData._warpMask;    
        uint32_t opos               = threadIdx.x - tgx;
        while (opos < stride)
        {        
            opos                   += tgx;
            if (opos < stride)
            {
                float unit        = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit       += pWeight[offset + opos] * sValue[i];  
                }

                pUnit[opos]         = unit;
            }
            opos                   -= tgx;

            if (tgx == 0)
            {
                opos                = atomicAdd(&sOpos, cData._warpSize);
            }
            opos                    = SHFL(opos, 0);
        }

        start                   = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta                    = (float)1.0;
    }
}

template<>
__global__ void LAUNCH_BOUNDS256() kCalculateIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ int32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = (float)pSparseData[tstart] * (float)(1.0 / 256.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl_sync(0xFFFFFFFF, opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (float)1.0;
    }
}

template<>
__global__ void LAUNCH_BOUNDS256() kCalculateIndexedSparseAnalogDenoisedZ_kernel(uint32_t position, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float* pRandom, float* pUnit, float beta)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    sOpos = blockDim.x;
    position = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + blockIdx.x] : position + blockIdx.x];
    uint64_t start = pSparseStart[position];
    uint64_t end = pSparseEnd[position];
    float w = cData._denoising_q * ((pDataWeight != NULL) ? pDataWeight[position] : (float)1.0);
    pUnit += blockIdx.x * stride;
    while (start < end)
    {
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            float value = pRandom[tstart];
            sOffset[pos] = (value < cData._denoising_p) ? cData._maxUint32_t : (int32_t)pSparseIndex[tstart] * stride;
            sValue[pos] = (float)pSparseData[tstart] * (float)(1.0 / 128.0) * w;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t tgx = threadIdx.x & cData._warpMask;
        uint32_t opos = threadIdx.x - tgx;
        while (opos < stride)
        {
            opos += tgx;
            if (opos < stride)
            {
                float unit = (beta == (float)0.0) ? (float)0.0 : (beta * pUnit[opos]);
                for (uint32_t i = 0; i < inputs; i++)
                {
                    uint32_t offset = sOffset[i];
                    if (offset != cData._maxUint32_t)
                        unit += pWeight[offset + opos] * sValue[i];
                }

                pUnit[opos] = unit;
            }
            opos -= tgx;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl(opos, 0);
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
        }
        beta = (float)1.0;
    }
}

template<typename T>
void kCalculateIndexedSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta)
{
    uint32_t threads = std::min(256u, ((stride + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    kCalculateIndexedSparseAnalogDenoisedZ_kernel << <batch, threads >> > (position, stride, pWeight, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pUnit, beta);

    cudaDeviceSynchronize();

    cudaError_t syncStatus = cudaGetLastError();
    if (syncStatus != cudaSuccess) {
        throw std::runtime_error("Device synchronization failed");
    }

    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            start                          += cData._warpSize;                   
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateWeightedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = pDataWeight[position];
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            pSparseTransposedData[opos]     = w;
            start                          += cData._warpSize;                   
        }
    }
}

void kCalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t threads = std::min(256u, ((batch * getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threads, 1, 1);

    void* args[] = { &position, &batch, &pSparseStart, &pSparseEnd, &pSparseIndex, &pSparseTransposedEnd, &pSparseTransposedIndex, &pDataWeight, &pSparseTransposedData };

    try {
        if (pDataWeight == NULL) {
            // Launch the appropriate kernel based on pDataWeight
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)kCalculateSparseTransposedMatrix_kernel,
                grid, blockDim,
                args,  // Pass kernel arguments as an array of pointers
                0,     // Shared memory size (optional)
                NULL   // Stream identifier (optional, NULL for default stream)
            );
        }
        else {
            // Launch the appropriate kernel based on pDataWeight
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)kCalculateWeightedSparseTransposedMatrix_kernel,
                grid, blockDim,
                args,  // Pass kernel arguments as an array of pointers
                0,     // Shared memory size (optional)
                NULL   // Stream identifier (optional, NULL for default stream)
            );
        }

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedMatrix: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            start                          += cData._warpSize;                   
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedWeightedSparseTransposedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = pDataWeight[position];
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            pSparseTransposedData[opos]     = w;
            start                          += cData._warpSize;                   
        }
    }
}

void kCalculateIndexedSparseTransposedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t threads = std::min(256u, ((batch * getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    uint32_t blocks = CalculateBlocks(batch);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threads, 1, 1);

    void* args[] = { &position, &batch, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pSparseTransposedEnd, &pSparseTransposedIndex, &pDataWeight, &pSparseTransposedData };

    try {
        if (pDataWeight == NULL) {
            // Launch the appropriate kernel based on pDataWeight
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)kCalculateIndexedSparseTransposedMatrix_kernel,
                grid, blockDim,
                args,  // Pass kernel arguments as an array of pointers
                0,     // Shared memory size (optional)
                NULL   // Stream identifier (optional, NULL for default stream)
            );
        }
        else {
            // Launch the appropriate kernel based on pDataWeight
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)kCalculateIndexedWeightedSparseTransposedMatrix_kernel,
                grid, blockDim,
                args,  // Pass kernel arguments as an array of pointers
                0,     // Shared memory size (optional)
                NULL   // Stream identifier (optional, NULL for default stream)
            );
        }

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateIndexedSparseTransposedMatrix: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
            }
            start                          += cData._warpSize;                   
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateWeightedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = cData._denoising_q * pDataWeight[position];
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w;
            }
            start                          += cData._warpSize;                   
        }
    }
}

void kCalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t threads = std::min(256u, ((batch * getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    uint32_t blocks = CalculateBlocks(batch);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threads, 1, 1);

    void* args[] = { &position, &batch, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight, &pRandom, &pSparseTransposedEnd, &pSparseTransposedIndex, &pSparseTransposedData };

    try {
        if (pDataWeight == NULL) {
            // Launch the appropriate kernel based on pDataWeight
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)kCalculateSparseTransposedDenoisedMatrix_kernel,
                grid, blockDim,
                args,  // Pass kernel arguments as an array of pointers
                0,     // Shared memory size (optional)
                NULL   // Stream identifier (optional, NULL for default stream)
            );
        }
        else {
            // Launch the appropriate kernel based on pDataWeight
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)kCalculateWeightedSparseTransposedDenoisedMatrix_kernel,
                grid, blockDim,
                args,  // Pass kernel arguments as an array of pointers
                0,     // Shared memory size (optional)
                NULL   // Stream identifier (optional, NULL for default stream)
            );
        }

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedDenoisedMatrix: " + std::string(e.what()));
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
            }
            start                          += cData._warpSize;                   
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedWeightedSparseTransposedDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = cData._denoising_q * pDataWeight[position];
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w;
            }
            start                          += cData._warpSize;                   
        }
    }
}

void kCalculateIndexedSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t threads = std::min(256u, ((batch * getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);
    uint32_t blocks = CalculateBlocks(batch);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threads, 1, 1);

    void* args[] = { &position, &batch, &pIndex, &pSparseStart, &pSparseEnd, &pSparseIndex, &pDataWeight, &pRandom, &pSparseTransposedEnd, &pSparseTransposedIndex, &pSparseTransposedData };

    try {
        if (pDataWeight == NULL) {
            // Launch the appropriate kernel based on pDataWeight
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)kCalculateIndexedSparseTransposedDenoisedMatrix_kernel,
                grid, blockDim,
                args,  // Pass kernel arguments as an array of pointers
                0,     // Shared memory size (optional)
                NULL   // Stream identifier (optional, NULL for default stream)
            );
        }
        else {
            // Launch the appropriate kernel based on pDataWeight
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)kCalculateIndexedWeightedSparseTransposedDenoisedMatrix_kernel,
                grid, blockDim,
                args,  // Pass kernel arguments as an array of pointers
                0,     // Shared memory size (optional)
                NULL   // Stream identifier (optional, NULL for default stream)
            );
        }

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateIndexedSparseTransposedDenoisedMatrix: " + std::string(e.what()));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            T value                         = pSparseData[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            pSparseTransposedData[opos]     = w * value;
            start                          += cData._warpSize;                   
        }
    }
}

template<typename T>
void kCalculateSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    try {
        kCalculateSparseTransposedAnalogMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedAnalogMatrix: " + std::string(e.what()));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];    
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            uint32_t index                  = pSparseIndex[start];
            T value                         = pSparseData[start];
            uint32_t opos                   = atomicAdd(&pSparseTransposedEnd[index], 1);
            pSparseTransposedIndex[opos]    = bpos;
            pSparseTransposedData[opos]     = w * value;
            start                          += cData._warpSize;                   
        }
    }
}

template<typename T>
void kCalculateIndexedSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    try {
        kCalculateIndexedSparseTransposedAnalogMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateIndexedSparseTransposedAnalogMatrix: " + std::string(e.what()));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                T value                     = pSparseData[start];
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value               = (float)pSparseData[start] * (float)(1.0 / 256.0);
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos;
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value               = (float)pSparseData[start] * (float)(1.0 / 128.0);
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}

template<typename T>
void kCalculateSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    try {
        kCalculateSparseTransposedAnalogDenoisedMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedAnalogDenoisedMatrix: " + std::string(e.what()));
    }
}

template<typename T>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                T value                     = pSparseData[start];
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}
template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, unsigned char* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value               = (float)pSparseData[start] * (float)(1.0 / 256.0);
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}

template<>
__global__ void
LAUNCH_BOUNDS()
kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, char* pSparseData, float *pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t bpos                           = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx                            = threadIdx.x & cData._warpMask;
    
    if (bpos < batch)
    {
        position                            = pIndex[cData._bShuffleIndices ? cData._pShuffleIndex[position + bpos] : position + bpos];
        uint64_t start                      = pSparseStart[position] + tgx;
        uint64_t end                        = pSparseEnd[position];
        float w                           = (pDataWeight != NULL) ? pDataWeight[position] : (float)1.0;
        while (start < end)
        {
            float rnd                     = pRandom[start];
            uint32_t index                  = pSparseIndex[start];
            if (rnd >= cData._denoising_p)
            {
                float value               = (float)pSparseData[start] * (float)(1.0 / 128.0);
                uint32_t opos               = atomicAdd(&pSparseTransposedEnd[index], 1);
                pSparseTransposedIndex[opos]= bpos;
                pSparseTransposedData[opos] = w * value;
            }
            start                          += cData._warpSize;                   
        }
    }
}

template<typename T>
void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData)
{
    uint32_t blocks = CalculateBlocks(batch * getGpu()._warpSize);

    try {
        kCalculateIndexedSparseTransposedAnalogDenoisedMatrix_kernel << <blocks, getGpu()._threadsPerBlock >> > (position, batch, pIndex, pSparseStart, pSparseEnd, pSparseIndex, pDataWeight, pSparseData, pRandom, pSparseTransposedEnd, pSparseTransposedIndex, pSparseTransposedData);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateIndexedSparseTransposedAnalogDenoisedMatrix: " + std::string(e.what()));
    }
}


__global__ void LAUNCH_BOUNDS256() kCalculateSparseTransposedWeightGradient_kernel(float alpha, float beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pDelta, float* pWeightGradient)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSE];

    uint64_t start = pSparseTransposedStart[blockIdx.x];
    uint64_t end = pSparseTransposedEnd[blockIdx.x];
    alpha *= cData._denoising_q;
    pWeightGradient += blockIdx.x * n;
    do
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSE);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseTransposedIndex[tstart] * n;
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t opos = threadIdx.x;
        uint32_t tgx = threadIdx.x & cData._warpMask;
        while (opos < n)
        {
            float oldgradient = (beta == (float)0.0) ? (float)0.0 : beta * pWeightGradient[opos];
            int64_t sum = 0;
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                sum += llrintf(ERRORSCALEF * pDelta[offset + opos]);
            }

            float fsum = alpha * (float)((double)sum * ONEOVERERRORSCALE);
            pWeightGradient[opos] = oldgradient + fsum;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl(opos, 0);
            opos += tgx;
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
            beta = (float)1.0;
        }
    } while (start < end);
}


void kCalculateSparseTransposedWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pDelta, float* pWeightGradient)
{
    uint32_t threads = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    // Define custom launch configuration
    dim3 grid(m, 1, 1);
    dim3 blockDim(threads, 1, 1);

    void* args[] = { &alpha, &beta, &n, &pSparseTransposedStart, &pSparseTransposedEnd, &pSparseTransposedIndex, &pDelta, &pWeightGradient };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kCalculateSparseTransposedWeightGradient_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedWeightGradient: " + std::string(e.what()));
    }
}

__global__ void LAUNCH_BOUNDS256() kCalculateSparseTransposedAnalogWeightGradient_kernel(float alpha, float beta, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData, float* pDelta, float* pWeightGradient)
{
    __shared__ uint32_t sOpos;
    __shared__ uint32_t sOffset[MAXSPARSEANALOG];
    __shared__ float sValue[MAXSPARSEANALOG];

    uint64_t start = pSparseTransposedStart[blockIdx.x];
    uint64_t end = pSparseTransposedEnd[blockIdx.x];
    alpha *= cData._denoising_q;
    pWeightGradient += blockIdx.x * n;
    do
    {
        sOpos = blockDim.x;
        uint32_t inputs = ullmin(end - start, (uint64_t)MAXSPARSEANALOG);
        uint64_t tend = start + inputs;
        uint64_t tstart = start + threadIdx.x;
        uint32_t pos = threadIdx.x;

        while (tstart < tend)
        {
            sOffset[pos] = pSparseTransposedIndex[tstart] * n;
            sValue[pos] = pSparseTransposedData[start];
            pos += blockDim.x;
            tstart += blockDim.x;
        }

        __threadfence();
        __syncthreads();

        uint32_t opos = threadIdx.x;
        uint32_t tgx = threadIdx.x & cData._warpMask;
        while (opos < n)
        {
            float oldgradient = (beta == (float)0.0) ? (float)0.0 : beta * pWeightGradient[opos];
            int64_t sum = 0;
            for (uint32_t i = 0; i < inputs; i++)
            {
                uint32_t offset = sOffset[i];
                float value = sValue[i];
                sum += llrintf(ERRORSCALEF * value * pDelta[offset + opos]);
            }

            float fsum = alpha * (float)((double)sum * ONEOVERERRORSCALE);
            pWeightGradient[opos] = oldgradient + fsum;

            if (tgx == 0)
            {
                opos = atomicAdd(&sOpos, cData._warpSize);
            }
            opos = __shfl(opos, 0);
            opos += tgx;
        }

        start = tend;
        if (start < end)
        {
            __threadfence();
            __syncthreads();
            beta = (float)1.0;
        }
    } while (start < end);
}

void kCalculateSparseTransposedAnalogWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData, float* pDelta, float* pWeightGradient)
{
    uint32_t threads = min(256, ((m + getGpu()._warpSize - 1) >> getGpu()._warpBits) << getGpu()._warpBits);

    // Define custom launch configuration
    dim3 grid(m, 1, 1);
    dim3 blockDim(threads, 1, 1);

    void* args[] = { &alpha, &beta, &n, &pSparseTransposedStart, &pSparseTransposedEnd, &pSparseTransposedIndex, &pSparseTransposedData, &pDelta, &pWeightGradient };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kCalculateSparseTransposedAnalogWeightGradient_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateSparseTransposedAnalogWeightGradient: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kUpdateBiases_kernel(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum             = (float)0.0;
        pDelta                 += pos;
        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }
        pBias[pos]             -= alpha * sum;
    }
}

void kUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(width);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &alpha, &batch, &width, &pDelta, &pBias };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kUpdateBiases_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kUpdateBiases: " + std::string(e.what()));
    }
}

__global__ void LAUNCH_BOUNDS() kCalculateRegularizationError_kernel(float* pWeight, uint64_t size, float lambda, float lambda1)
{
    extern __shared__ float sdata[];
    uint64_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    float error = (float)0.0;

    if (pos < size)
    {
        float w = pWeight[pos];
        error = lambda * w * w + lambda1 * abs(w);
    }

    sdata[threadIdx.x] = error;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        pWeight[blockIdx.x] = sdata[0];
    }
}

float kCalculateRegularizationError(float lambda, float lambda1, float* pWeight, uint64_t size)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &lambda, &lambda1, &pWeight, &size, &getGpu()._data._pAccumulator };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kCalculateRegularizationError_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }

        getGpu()._pbAccumulator->Download();
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateRegularizationError: " + std::string(e.what()));
    }

    return (float)((double)(getGpu()._pbAccumulator->_pSysData[0]) * ONEOVERERRORSCALE);
}

__global__ void
LAUNCH_BOUNDS()
kSGDUpdateWeights_kernel(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightGradient, float* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g               = pWeightGradient[pos];
        float w               = pWeight[pos];
        pWeight[pos]            = w + alpha * (g - lambda * w - lambda1 * sgn(w));
    }
}

void kSGDUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightGradient, float* pWeight)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &alpha, &lambda, &lambda1, &size, &pWeightGradient, &pWeight };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kSGDUpdateWeights_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kSGDUpdateWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kSGDUpdateBiases_kernel(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint32_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum             = 0.0f;
        pDelta                 += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }
        sum                    /= (float)batch;

        float bias            = pBias[pos];
        pBias[pos]              = bias - alpha * sum;
    }
}

void kSGDUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(width);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &alpha, &batch, &width, &pDelta, &pBias };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kSGDUpdateBiases_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kSGDUpdateBiases: " + std::string(e.what()));
    }
}


__global__ void
LAUNCH_BOUNDS()
kMomentumUpdateWeights_kernel(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g               = pWeightGradient[pos];
        float w               = pWeight[pos];
        float v               = pWeightVelocity[pos];
        v                       = mu * v + alpha * (g - lambda * w - lambda1 * sgn(w));
        pWeightVelocity[pos]    = v;
        pWeight[pos]            = w + v;
    }
}

void kMomentumUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(size);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &alpha, &lambda, &lambda1, &mu, &size, &pWeightVelocity, &pWeightGradient, &pWeight };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kMomentumUpdateWeights_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kMomentumUpdateWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kMomentumUpdateBiases_kernel(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint32_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum             = 0.0f;
        pDelta                 += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }
        sum                    /= (float)batch;

        float v               = pBiasVelocity[pos];
        v                       = mu * v - alpha * sum;
        pBiasVelocity[pos]      = v;
        pBias[pos]             += v;
    }
}

void kMomentumUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(width);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &alpha, &mu, &batch, &width, &pDelta, &pBiasVelocity, &pBias };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kMomentumUpdateBiases_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kMomentumUpdateBiases: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kAdaGradUpdateWeights_kernel(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g               = pWeightGradient[pos];
        float w               = pWeight[pos];
        float v               = pWeightVelocity[pos];
        g                      -= lambda * w + lambda1 * sgn(w);
        v                      += g * g;
        pWeightVelocity[pos]    = v;
        pWeight[pos]            = w + alpha * g * rsqrt(max(0.000000001f, v));
    }
}

void kAdaGradUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    unsigned long threadsPerBlock = getGpu()._threadsPerBlock;
    unsigned long blocks = CalculateBlocks(size);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &alpha, &lambda, &lambda1, &size, &pWeightVelocity, &pWeightGradient, &pWeight };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kAdaGradUpdateWeights_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAdaGradUpdateWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kAdaGradUpdateBiases_kernel(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum             = 0.0f;
        pDelta                 += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum                += *pDelta;
            pDelta             += width;
        }
        sum                    /= (float)batch;

        float v               = pBiasVelocity[pos];
        v                      += sum * sum;
        pBiasVelocity[pos]      = v;
        pBias[pos]             -= alpha * sum * rsqrt(max(0.000000001f, v));
    }
}

void kAdaGradUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    unsigned long threadsPerBlock = getGpu()._threadsPerBlock;
    unsigned long blocks = CalculateBlocks(width);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &alpha, &batch, &width, &pDelta, &pBiasVelocity, &pBias };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kAdaGradUpdateBiases_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAdaGradUpdateBiases: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kAdaDeltaUpdateWeights_kernel(float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight)
{
    uint64_t pos                = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float g                       = pWeightGradient[pos];
        float w                       = pWeight[pos];
        float v                       = pWeightVelocity[pos];
        float vg                      = pWeightGradientVelocity[pos];
        g                              -= lambda * w + lambda1 * sgn(w);
        vg                              = mu * vg + ((float)1.0 - mu) * g * g;
        float dw                      = sqrt(max((float)0.000000001, v) / max((float)0.000000001, vg)) * g;
        v                               = mu * v + ((float)1.0 - mu) * dw * dw;
        pWeightVelocity[pos]            = v;
        pWeightGradientVelocity[pos]    = vg;
        pWeight[pos]                    = w + dw;
    }
}

void kAdaDeltaUpdateWeights(float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight)
{
    unsigned long threadsPerBlock = getGpu()._threadsPerBlock;
    unsigned long blocks = CalculateBlocks(size);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &lambda, &lambda1, &mu, &size, &pWeightVelocity, &pWeightGradient, &pWeightGradientVelocity, &pWeight };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kAdaDeltaUpdateWeights_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAdaDeltaUpdateWeights: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kAdaDeltaUpdateBiases_kernel(float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias)
{
    uint64_t pos                    = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width)
    {
        float sum                 = (float)0.0;
        pDelta                     += pos;

        for (uint32_t i = 0; i < batch; i++)
        {
            sum                    += *pDelta;
            pDelta                 += width;
        }
        sum                        /= (float)batch;

        float v                   = pBiasVelocity[pos];
        float vg                  = pBiasGradientVelocity[pos];        
        vg                          = mu * vg + ((float)1.0 - mu) * sum * sum;
        float dw                  = sqrt(max((float)0.000000001, v) / max((float)0.000000001, vg)) * sum;        
        v                           = mu * v + ((float)1.0 - mu) * dw * dw;
        pBiasVelocity[pos]          = v;
        pBiasGradientVelocity[pos]  = vg;        
        pBias[pos]                 -= dw;
    }
}

void kAdaDeltaUpdateBiases(float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias)
{
    unsigned long threadsPerBlock = getGpu()._threadsPerBlock;
    uint32_t blocks = CalculateBlocks(width);

    // Define custom launch configuration
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    void* args[] = { &mu, &batch, &width, &pDelta, &pBiasVelocity, &pBiasGradientVelocity, &pBias };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kAdaDeltaUpdateBiases_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAdaDeltaUpdateBiases: " + std::string(e.what()));
    }
}

/// <summary>
/// CUDA kernel for updating weights using the Adam optimizer.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization term.</param>
/// <param name="lambda1">The L1 regularization term.</param>
/// <param name="beta1">The first momentum term.</param>
/// <param name="beta2">The second momentum term.</param>
/// <param name="t">The current time step.</param>
/// <param name="size">The size of the weight arrays.</param>
/// <param name="pWeightVelocity">Pointer to the weight velocity array.</param>
/// <param name="pWeightGradient">Pointer to the weight gradient array.</param>
/// <param name="pWeightGradientVelocity">Pointer to the weight gradient velocity array.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
__global__ void LAUNCH_BOUNDS() kAdamUpdateWeights_kernel(float alpha, float lambda, float lambda1, float beta1, float beta2, float t, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight)
{
    // Calculate the position in the weight arrays for this thread.
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current position is within the valid range.
    if (pos < size)
    {
        // Load the weight gradient and weight values for this position.
        float dw = pWeightGradient[pos];
        float w = pWeight[pos];
        float vdw = pWeightVelocity[pos];
        float sdw = pWeightGradientVelocity[pos];

        // Update the weight gradient with L2 and L1 regularization terms.
        dw -= lambda * w + lambda1 * sgn(w);

        // Update the first moment estimate (momentum) using beta1.
        vdw = beta1 * vdw + ((float)1.0 - beta1) * dw;

        // Update the second moment estimate using beta2.
        sdw = beta2 * sdw + ((float)1.0 - beta2) * dw * dw;

        // Increment the current time step.
        t += (float)1.0;

        // Update the weight velocity and gradient velocity arrays.
        pWeightVelocity[pos] = vdw;
        pWeightGradientVelocity[pos] = sdw;

        // Bias-corrected first and second moment estimates.
        vdw /= (float)1.0 - pow(beta1, t);
        sdw /= (float)1.0 - pow(beta2, t);

        // Compute the weight update using the Adam update rule.
        dw = alpha * vdw / (sqrt(sdw) + (float)1.0e-8);

        // Update the weight.
        pWeight[pos] = w + dw;
    }
}

/// <summary>
/// Update weights using the Adam optimizer on the GPU.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization term.</param>
/// <param name="lambda1">The L1 regularization term.</param>
/// <param name="beta1">The first momentum term.</param>
/// <param name="beta2">The second momentum term.</param>
/// <param name="t">The current time step.</param>
/// <param name="size">The size of the weight arrays.</param>
/// <param name="pWeightVelocity">Pointer to the weight velocity array.</param>
/// <param name="pWeightGradient">Pointer to the weight gradient array.</param>
/// <param name="pWeightGradientVelocity">Pointer to the weight gradient velocity array.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
void kAdamUpdateWeights(float alpha, float lambda, float lambda1, float beta1, float beta2, float t, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight)
{
    // Get the number of threads per block from the GPU configuration.
    unsigned long threadsPerBlock = getGpu()._threadsPerBlock;

    // Calculate the number of blocks required based on the size of the weight arrays.
    uint32_t blocks = CalculateBlocks(size);

    // Define custom launch configuration with a grid and block dimension.
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Create an array of pointers to pass kernel arguments.
    void* args[] = { &alpha, &lambda, &lambda1, &beta1, &beta2, &t, &size, &pWeightVelocity, &pWeightGradient, &pWeightGradientVelocity, &pWeight };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel.
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kAdamUpdateWeights_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        // Synchronize the device to ensure all operations are completed.
        cudaDeviceSynchronize();

        // Check for synchronization errors.
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        // Check for any kernel launch errors.
        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        // Catch and rethrow any exceptions with a descriptive error message.
        throw std::runtime_error("Error in kAdamUpdateWeights: " + std::string(e.what()));
    }
}

/// <summary>
/// CUDA kernel for updating biases using the Adam optimizer.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="beta1">The first momentum term.</param>
/// <param name="beta2">The second momentum term.</param>
/// <param name="t">The current time step.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the biases.</param>
/// <param name="pDelta">Pointer to the delta array.</param>
/// <param name="pBiasVelocity">Pointer to the bias velocity array.</param>
/// <param name="pBiasGradientVelocity">Pointer to the bias gradient velocity array.</param>
/// <param name="pBias">Pointer to the bias array.</param>
__global__ void LAUNCH_BOUNDS() kAdamUpdateBiases_kernel(float alpha, float beta1, float beta2, float t, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias)
{
    // Calculate the position in the bias arrays for this thread.
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current position is within the valid range.
    if (pos < width)
    {
        // Initialize a sum variable to accumulate delta values.
        float sum = (float)0.0;

        // Move the pDelta pointer to the current position.
        pDelta += pos;

        // Iterate through the batch and accumulate delta values.
        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }

        // Calculate the mean of delta values for the batch.
        sum /= (float)batch;

        // Load bias velocity and bias gradient velocity values.
        float vdw = pBiasVelocity[pos];
        float sdw = pBiasGradientVelocity[pos];

        // Update the first moment estimate (momentum) using beta1.
        vdw = beta1 * vdw + ((float)1.0 - beta1) * sum;

        // Update the second moment estimate using beta2.
        sdw = beta2 * sdw + ((float)1.0 - beta2) * sum * sum;

        // Increment the current time step.
        t += (float)1.0;

        // Update the bias velocity and bias gradient velocity arrays.
        pBiasVelocity[pos] = vdw;
        pBiasGradientVelocity[pos] = sdw;

        // Bias-corrected first and second moment estimates.
        vdw /= (float)1.0 - pow(beta1, t);
        sdw /= (float)1.0 - pow(beta2, t);

        // Compute the bias update using the Adam update rule.
        float dw = alpha * vdw / (sqrt(sdw) + (float)1.0e-8);

        // Update the bias.
        pBias[pos] -= dw;
    }
}

/// <summary>
/// Update biases using the Adam optimizer on the GPU.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="mu">The L2 regularization term.</param>
/// <param name="mu1">The L1 regularization term.</param>
/// <param name="t">The current time step.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the biases.</param>
/// <param name="pDelta">Pointer to the delta array.</param>
/// <param name="pBiasVelocity">Pointer to the bias velocity array.</param>
/// <param name="pBiasGradientVelocity">Pointer to the bias gradient velocity array.</param>
/// <param name="pBias">Pointer to the bias array.</param>
void kAdamUpdateBiases(float alpha, float mu, float mu1, float t, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias)
{
    // Get the number of threads per block from the GPU configuration.
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;

    // Calculate the number of blocks required based on the width of the biases.
    uint32_t blocks = CalculateBlocks(width);

    // Define custom launch configuration with a grid and block dimension.
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Create an array of pointers to pass kernel arguments.
    void* args[] = { &alpha, &mu, &mu1, &t, &batch, &width, &pDelta, &pBiasVelocity, &pBiasGradientVelocity, &pBias };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel.
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kAdamUpdateBiases_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        // Synchronize the device to ensure all operations are completed.
        cudaDeviceSynchronize();

        // Check for synchronization errors.
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        // Check for any kernel launch errors.
        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        // Catch and rethrow any exceptions with a descriptive error message.
        throw std::runtime_error("Error in kAdamUpdateBiases: " + std::string(e.what()));
    }
}

/// <summary>
/// CUDA kernel for updating weights using the Nesterov accelerated gradient descent.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization term.</param>
/// <param name="lambda1">The L1 regularization term.</param>
/// <param name="mu">The momentum term.</param>
/// <param name="size">The size of the weight arrays.</param>
/// <param name="pWeightVelocity">Pointer to the weight velocity array.</param>
/// <param name="pWeightGradient">Pointer to the weight gradient array.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
__global__ void LAUNCH_BOUNDS() kNesterovUpdateWeights_kernel(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    // Calculate the position in the weight arrays for this thread.
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current position is within the valid range.
    if (pos < size)
    {
        // Load the weight gradient and weight values for this position.
        float g = pWeightGradient[pos];
        float w = pWeight[pos];

        // Load the old velocity for this position.
        float vOld = pWeightVelocity[pos];

        // Compute the new velocity using Nesterov accelerated gradient descent.
        float vNew = mu * vOld + alpha * (g - lambda * w - lambda1 * sgn(w));

        // Update the weight velocity for this position.
        pWeightVelocity[pos] = vNew;

        // Update the weight using Nesterov accelerated gradient descent update rule.
        w = w + vNew + mu * (vNew - vOld);

        // Update the weight array with the new weight value.
        pWeight[pos] = w;
    }
}

/// <summary>
/// Update weights using the Nesterov accelerated gradient descent on the GPU.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization term.</param>
/// <param name="lambda1">The L1 regularization term.</param>
/// <param name="mu">The momentum term.</param>
/// <param name="size">The size of the weight arrays.</param>
/// <param name="pWeightVelocity">Pointer to the weight velocity array.</param>
/// <param name="pWeightGradient">Pointer to the weight gradient array.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
void kNesterovUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    // Get the number of threads per block from the GPU configuration.
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;

    // Calculate the number of blocks required based on the size of the weight arrays.
    uint32_t blocks = CalculateBlocks(size);

    // Define custom launch configuration with a grid and block dimension.
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Create an array of pointers to pass kernel arguments.
    void* args[] = { &alpha, &lambda, &lambda1, &mu, &size, &pWeightVelocity, &pWeightGradient, &pWeight };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel.
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kNesterovUpdateWeights_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        // Synchronize the device to ensure all operations are completed.
        cudaDeviceSynchronize();

        // Check for synchronization errors.
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        // Check for any kernel launch errors.
        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        // Catch and rethrow any exceptions with a descriptive error message.
        throw std::runtime_error("Error in kNesterovUpdateWeights: " + std::string(e.what()));
    }
}

/// <summary>
/// CUDA kernel for updating biases using the Nesterov accelerated gradient descent.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="mu">The momentum term.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the biases.</param>
/// <param name="pDelta">Pointer to the delta array.</param>
/// <param name="pBiasVelocity">Pointer to the bias velocity array.</param>
/// <param name="pBias">Pointer to the bias array.</param>
__global__ void LAUNCH_BOUNDS() kNesterovUpdateBiases_kernel(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    // Calculate the position in the bias arrays for this thread.
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current position is within the valid range.
    if (pos < width)
    {
        // Initialize a sum variable to accumulate delta values.
        float sum = 0.0f;

        // Move the pDelta pointer to the current position.
        pDelta += pos;

        // Iterate through the batch and accumulate delta values.
        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }

        // Calculate the mean of delta values for the batch.
        sum /= (float)batch;

        // Load the old velocity for this position.
        float vOld = pBiasVelocity[pos];

        // Compute the new velocity using Nesterov accelerated gradient descent.
        float vNew = mu * vOld - alpha * sum;

        // Update the bias velocity for this position.
        pBiasVelocity[pos] = vNew;

        // Update the bias using Nesterov accelerated gradient descent update rule.
        pBias[pos] += vNew + mu * (vNew - vOld);
    }
}

/// <summary>
/// Update biases using the Nesterov accelerated gradient descent on the GPU.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="mu">The momentum term.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the biases.</param>
/// <param name="pDelta">Pointer to the delta array.</param>
/// <param name="pBiasVelocity">Pointer to the bias velocity array.</param>
/// <param name="pBias">Pointer to the bias array.</param>
void kNesterovUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    // Get the number of threads per block from the GPU configuration.
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;

    // Calculate the number of blocks required based on the width of the biases.
    uint32_t blocks = CalculateBlocks(width);

    // Define custom launch configuration with a grid and block dimension.
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Create an array of pointers to pass kernel arguments.
    void* args[] = { &alpha, &mu, &batch, &width, &pDelta, &pBiasVelocity, &pBias };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel.
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kNesterovUpdateBiases_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        // Synchronize the device to ensure all operations are completed.
        cudaDeviceSynchronize();

        // Check for synchronization errors.
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        // Check for any kernel launch errors.
        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        // Catch and rethrow any exceptions with a descriptive error message.
        throw std::runtime_error("Error in kNesterovUpdateBiases: " + std::string(e.what()));
    }
}

/// <summary>
/// CUDA kernel for shifting weights using Nesterov momentum.
/// </summary>
/// <param name="mu">The momentum term.</param>
/// <param name="size">The size of the weight arrays.</param>
/// <param name="pWeightVelocity">Pointer to the weight velocity array.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
__global__ void LAUNCH_BOUNDS() kNesterovShiftWeights_kernel(float mu, uint64_t size, float* pWeightVelocity, float* pWeight)
{
    // Calculate the position in the weight arrays for this thread.
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current position is within the valid range.
    if (pos < size)
    {
        // Load the weight and velocity values for this position.
        float w = pWeight[pos];
        float v = pWeightVelocity[pos];

        // Update the weight using Nesterov momentum update rule.
        pWeight[pos] = w + mu * v;
    }
}

/// <summary>
/// Shift weights using Nesterov momentum on the GPU.
/// </summary>
/// <param name="mu">The momentum term.</param>
/// <param name="size">The size of the weight arrays.</param>
/// <param name="pWeightVelocity">Pointer to the weight velocity array.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
void kNesterovShiftWeights(float mu, uint64_t size, float* pWeightVelocity, float* pWeight)
{
    // Get the number of threads per block from the GPU configuration.
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;

    // Calculate the number of blocks required based on the size of the weight arrays.
    uint32_t blocks = CalculateBlocks(size);

    // Define custom launch configuration with a grid and block dimension.
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Create an array of pointers to pass kernel arguments.
    void* args[] = { &mu, &size, &pWeightVelocity, &pWeight };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel.
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kNesterovShiftWeights_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        // Synchronize the device to ensure all operations are completed.
        cudaDeviceSynchronize();

        // Check for synchronization errors.
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        // Check for any kernel launch errors.
        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        // Catch and rethrow any exceptions with a descriptive error message.
        throw std::runtime_error("Error in kNesterovShiftWeights: " + std::string(e.what()));
    }
}

/// <summary>
/// CUDA kernel for shifting biases using Nesterov momentum.
/// </summary>
/// <param name="mu">The momentum term.</param>
/// <param name="width">The width of the biases.</param>
/// <param name="pBiasVelocity">Pointer to the bias velocity array.</param>
/// <param name="pBias">Pointer to the bias array.</param>
__global__ void LAUNCH_BOUNDS() kNesterovShiftBiases_kernel(float mu, uint32_t width, float* pBiasVelocity, float* pBias)
{
    // Calculate the position in the bias arrays for this thread.
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current position is within the valid range.
    if (pos < width)
    {
        // Load the bias and velocity values for this position.
        float b = pBias[pos];
        float v = pBiasVelocity[pos];

        // Update the bias using Nesterov momentum update rule.
        pBias[pos] = b + mu * v;
    }
}

/// <summary>
/// Shift biases using Nesterov momentum on the GPU.
/// </summary>
/// <param name="mu">The momentum term.</param>
/// <param name="width">The width of the biases.</param>
/// <param name="pBiasVelocity">Pointer to the bias velocity array.</param>
/// <param name="pBias">Pointer to the bias array.</param>
void kNesterovShiftBiases(float mu, uint32_t width, float* pBiasVelocity, float* pBias)
{
    // Get the number of threads per block from the GPU configuration.
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;

    // Calculate the number of blocks required based on the width of the biases.
    uint32_t blocks = CalculateBlocks(width);

    // Define custom launch configuration with a grid and block dimension.
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Create an array of pointers to pass kernel arguments.
    void* args[] = { &mu, &width, &pBiasVelocity, &pBias };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel.
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kNesterovShiftBiases_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        // Synchronize the device to ensure all operations are completed.
        cudaDeviceSynchronize();

        // Check for synchronization errors.
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        // Check for any kernel launch errors.
        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        // Catch and rethrow any exceptions with a descriptive error message.
        throw std::runtime_error("Error in kNesterovShiftBiases: " + std::string(e.what()));
    }
}

/// <summary>
/// CUDA kernel for updating weights using the RMSProp optimization algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization term.</param>
/// <param name="lambda1">The L1 regularization term.</param>
/// <param name="mu">The decay factor for the moving average of squared gradients.</param>
/// <param name="size">The size of the weight arrays.</param>
/// <param name="pWeightVelocity">Pointer to the weight velocity array.</param>
/// <param name="pWeightGradient">Pointer to the weight gradient array.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
__global__ void LAUNCH_BOUNDS() kRMSPropUpdateWeights_kernel(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    // Calculate the position in the weight arrays for this thread.
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current position is within the valid range.
    if (pos < size)
    {
        // Load the gradient, weight, and velocity values for this position.
        float g = pWeightGradient[pos];
        float w = pWeight[pos];
        float v = pWeightVelocity[pos];

        // Apply L2 and L1 regularization to the gradient.
        g -= lambda * w + lambda1 * sgn(w);

        // Compute the updated velocity using RMSProp.
        v = mu * v + (1.0f - mu) * g * g;

        // Update the weight velocity for this position.
        pWeightVelocity[pos] = v;

        // Update the weight using the RMSProp update rule.
        pWeight[pos] = w + alpha * g * rsqrt(max(0.000000001f, v));
    }
}

/// <summary>
/// Update weights using the RMSProp optimization algorithm on the GPU.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization term.</param>
/// <param name="lambda1">The L1 regularization term.</param>
/// <param name="mu">The decay factor for the moving average of squared gradients.</param>
/// <param name="size">The size of the weight arrays.</param>
/// <param name="pWeightVelocity">Pointer to the weight velocity array.</param>
/// <param name="pWeightGradient">Pointer to the weight gradient array.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
void kRMSPropUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight)
{
    // Get the number of threads per block from the GPU configuration.
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;

    // Calculate the number of blocks required based on the size of the weight arrays.
    uint32_t blocks = CalculateBlocks(size);

    // Define custom launch configuration with a grid and block dimension.
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Create an array of pointers to pass kernel arguments.
    void* args[] = { &alpha, &lambda, &lambda1, &mu, &size, &pWeightVelocity, &pWeightGradient, &pWeight };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel.
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kRMSPropUpdateWeights_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        // Synchronize the device to ensure all operations are completed.
        cudaDeviceSynchronize();

        // Check for synchronization errors.
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        // Check for any kernel launch errors.
        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        // Catch and rethrow any exceptions with a descriptive error message.
        throw std::runtime_error("Error in kRMSPropUpdateWeights: " + std::string(e.what()));
    }
}

/// <summary>
/// CUDA kernel for updating biases using the RMSProp optimization algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="mu">The decay factor for the moving average of squared gradients.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the biases.</param>
/// <param name="pDelta">Pointer to the delta array.</param>
/// <param name="pBiasVelocity">Pointer to the bias velocity array.</param>
/// <param name="pBias">Pointer to the bias array.</param>
__global__ void LAUNCH_BOUNDS() kRMSPropUpdateBiases_kernel(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    // Calculate the position in the bias arrays for this thread.
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current position is within the valid range.
    if (pos < width)
    {
        // Initialize the sum for this position.
        float sum = 0.0f;

        // Update the delta pointer to the current position.
        pDelta += pos;

        // Calculate the sum of deltas over the batch.
        for (uint32_t i = 0; i < batch; i++)
        {
            sum += *pDelta;
            pDelta += width;
        }

        // Calculate the average of the sum over the batch.
        sum /= (float)batch;

        // Load the bias velocity value for this position.
        float v = pBiasVelocity[pos];

        // Update the bias velocity using RMSProp.
        v = mu * v + (1.0f - mu) * sum * sum;

        // Update the bias using the RMSProp update rule.
        pBiasVelocity[pos] = v;
        pBias[pos] -= alpha * sum * rsqrt(max(0.000000001f, v));
    }
}

/// <summary>
/// Update biases using the RMSProp optimization algorithm on the GPU.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="mu">The decay factor for the moving average of squared gradients.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the biases.</param>
/// <param name="pDelta">Pointer to the delta array.</param>
/// <param name="pBiasVelocity">Pointer to the bias velocity array.</param>
/// <param name="pBias">Pointer to the bias array.</param>
void kRMSPropUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias)
{
    // Get the number of threads per block from the GPU configuration.
    uint32_t threadsPerBlock = getGpu()._threadsPerBlock;

    // Calculate the number of blocks required based on the width of the biases.
    uint32_t blocks = CalculateBlocks(width);

    // Define custom launch configuration with a grid and block dimension.
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Create an array of pointers to pass kernel arguments.
    void* args[] = { &alpha, &mu, &batch, &width, &pDelta, &pBiasVelocity, &pBias };

    try {
        // Launch the kernel with custom launch parameters using cudaLaunchKernel.
        cudaError_t launchStatus = cudaLaunchKernel(
            (const void*)kRMSPropUpdateBiases_kernel,
            grid, blockDim,
            args,  // Pass kernel arguments as an array of pointers
            0,     // Shared memory size (optional)
            NULL   // Stream identifier (optional, NULL for default stream)
        );

        // Synchronize the device to ensure all operations are completed.
        cudaDeviceSynchronize();

        // Check for synchronization errors.
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        // Check for any kernel launch errors.
        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        // Catch and rethrow any exceptions with a descriptive error message.
        throw std::runtime_error("Error in kRMSPropUpdateBiases: " + std::string(e.what()));
    }
}

#include "Bitonic.cuh"

/// <summary>
/// CUDA kernel for calculating output with 32 threads per warp.
/// </summary>
/// <param name="pOutputBuffer">Pointer to the output buffer.</param>
/// <param name="pKeyBuffer">Pointer to the key buffer.</param>
/// <param name="pValueBuffer">Pointer to the value buffer.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the output.</param>
/// <param name="k">The number of top-K values to compute.</param>
__global__ void LAUNCH_BOUNDS() CalculateOutput_32_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    // Declare a shared memory array for keys and values.
    __shared__ volatile float sKey[64 * 4];
    __shared__ volatile uint32_t sValue[64 * 4];
    __shared__ uint32_t sharedCount;  // Added sharedCount

    // Calculate the position in the batch and the thread ID within a warp.
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        // Calculate pointers and initial values.
        float* pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[64 * offset];
        volatile uint32_t* psValue = &sValue[64 * offset];

        float k0 = -FLT_MAX;
        float k1 = -FLT_MAX;
        uint32_t v0 = 0;
        uint32_t v1 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += cData._warpSize;

        float minValue = -FLT_MAX;
        uint32_t rpos = 32;
        uint32_t bufferSize = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;

        if (tgx == 0)
            sharedCount = 0;
        __syncthreads();

        // Iterate over the width of the output.
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -FLT_MAX;
            uint32_t value = wpos;
            if (wpos < width)
            {
                key = pOutput[wpos];
            }

            uint32_t count = 0;

            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                atomicAdd(&count, 1);
                count += sharedCount;
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }

            atomicAdd(&sharedCount, __popc(count));
            __syncthreads();

            if (bufferSize >= 32)
            {
                k1 = psKey[tgx];
                v1 = psValue[tgx];
                bool flag;
                BITONICSORT64_64();

                minValue = __shfl(k0, cData._warpSize - 1);

                bufferSize -= 32;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 32];
                    psValue[tgx] = psValue[tgx + 32];
                }
            }

            rpos += cData._warpSize;
        }

        // Handle remaining elements in the buffer.
        if ((bufferSize > 0) || (width <= 32))
        {
            k1 = -FLT_MAX;
            v1 = 0;

            if (tgx < bufferSize)
            {
                k1 = psKey[tgx];
                v1 = psValue[tgx];
            }
            BITONICSORT64_64();
        }

        // Store the top-k values in the output buffers.
        float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
    }
}

/// <summary>
/// CUDA kernel for calculating output with 64 threads per warp.
/// </summary>
/// <param name="pOutputBuffer">Pointer to the output buffer.</param>
/// <param name="pKeyBuffer">Pointer to the key buffer.</param>
/// <param name="pValueBuffer">Pointer to the value buffer.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the output.</param>
/// <param name="k">The number of top-K values to compute.</param>
__global__ void LAUNCH_BOUNDS() CalculateOutput_64_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    // Declare a shared memory array for keys and values.
    __shared__ volatile float sKey[96 * 4];
    __shared__ volatile uint32_t sValue[96 * 4];
    __shared__ uint32_t sharedCount;

    // Calculate the position in the batch and the thread ID within a warp.
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        // Calculate pointers and initial values.
        float* pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[96 * offset];
        volatile uint32_t* psValue = &sValue[96 * offset];

        float k0 = -FLT_MAX;
        float k1 = -FLT_MAX;
        float k2 = -FLT_MAX;
        float k3 = -FLT_MAX;
        uint32_t v0 = 0;
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        uint32_t v3 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutput[wpos];
            v1 = wpos;
        }
        wpos += cData._warpSize;

        float minValue = -FLT_MAX;
        uint32_t rpos = 64;
        uint32_t bufferSize = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;

        if (tgx == 0)
            sharedCount = 0;
        __syncthreads();

        // Iterate over the width of the output.
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -FLT_MAX;
            uint32_t value = wpos;
            if (wpos < width)
            {
                key = pOutput[wpos];
            }

            uint32_t count = 0;

            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                atomicAdd(&count, 1);
                count += sharedCount;
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }

            atomicAdd(&sharedCount, __popc(count));
            __syncthreads();

            if (bufferSize >= 64)
            {
                k2 = psKey[tgx];
                v2 = psValue[tgx];
                k3 = psKey[tgx + cData._warpSize];
                v3 = psValue[tgx + cData._warpSize];
                bool flag;
                BITONICSORT128_128();

                minValue = __shfl(k1, cData._warpSize - 1);

                bufferSize -= 64;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 64];
                    psValue[tgx] = psValue[tgx + 64];
                }
            }

            rpos += cData._warpSize;
        }

        // Handle remaining elements in the buffer.
        if ((bufferSize > 0) || (width <= 64))
        {
            k2 = -FLT_MAX;
            k3 = -FLT_MAX;
            v2 = 0;
            v3 = 0;

            if (tgx < bufferSize)
            {
                k2 = psKey[tgx];
                v2 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k3 = psKey[tgx + cData._warpSize];
                v3 = psValue[tgx + cData._warpSize];
            }

            BITONICSORT128_128();
        }

        // Store the top-k values in the output buffers.
        float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += cData._warpSize;
    }
}

/// <summary>
/// CUDA kernel for calculating output with 128 threads per warp.
/// </summary>
/// <param name="pOutputBuffer">Pointer to the output buffer.</param>
/// <param name="pKeyBuffer">Pointer to the key buffer.</param>
/// <param name="pValueBuffer">Pointer to the value buffer.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the output.</param>
/// <param name="k">The number of top-K values to compute.</param>
__global__ void LAUNCH_BOUNDS() CalculateOutput_128_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    // Declare a shared memory array for keys and values.
    __shared__ volatile float sKey[160 * 4];
    __shared__ volatile uint32_t sValue[160 * 4];
    __shared__ uint32_t sharedCount;

    // Calculate the position in the batch and the thread ID within a warp.
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        // Calculate pointers and initial values.
        float* pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[160 * offset];
        volatile uint32_t* psValue = &sValue[160 * offset];

        float k0 = -FLT_MAX;
        float k1 = -FLT_MAX;
        float k2 = -FLT_MAX;
        float k3 = -FLT_MAX;
        float k4 = -FLT_MAX;
        float k5 = -FLT_MAX;
        float k6 = -FLT_MAX;
        float k7 = -FLT_MAX;
        uint32_t v0 = 0;
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        uint32_t v3 = 0;
        uint32_t v4 = 0;
        uint32_t v5 = 0;
        uint32_t v6 = 0;
        uint32_t v7 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutput[wpos];
            v1 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k2 = pOutput[wpos];
            v2 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k3 = pOutput[wpos];
            v3 = wpos;
        }

        float minValue = -FLT_MAX;
        uint32_t rpos = 128;
        uint32_t bufferSize = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;

        if (tgx == 0)
            sharedCount = 0;
        __syncthreads();

        // Iterate over the width of the output.
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -FLT_MAX;
            uint32_t value = wpos;
            if (wpos < width)
            {
                key = pOutput[wpos];
            }

            uint32_t count = 0;

            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                atomicAdd(&count, 1);
                count += sharedCount;
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }

            atomicAdd(&sharedCount, __popc(count));
            __syncthreads();

            if (bufferSize >= 128)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                minValue = __shfl(k3, cData._warpSize - 1);

                bufferSize -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 128];
                    psValue[tgx] = psValue[tgx + 128];
                }
            }

            rpos += cData._warpSize;
        }

        // Handle remaining elements in the buffer.
        if ((bufferSize > 0) || (width <= 128))
        {
            k4 = -FLT_MAX;
            k5 = -FLT_MAX;
            k6 = -FLT_MAX;
            k7 = -FLT_MAX;
            v4 = 0;
            v5 = 0;
            v6 = 0;
            v7 = 0;

            if (tgx < bufferSize)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
            }

            BITONICSORT256_256();
        }

        // Store the top-k values in the output buffers.
        float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k2;
            pValue[wpos] = v2;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k3;
            pValue[wpos] = v3;
        }
    }
}

/// <summary>
/// CUDA kernel for calculating output with 256 threads per warp.
/// </summary>
/// <param name="pOutputBuffer">Pointer to the output buffer.</param>
/// <param name="pKeyBuffer">Pointer to the key buffer.</param>
/// <param name="pValueBuffer">Pointer to the value buffer.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the output.</param>
/// <param name="k">The number of top-K values to compute.</param>
__global__ void LAUNCH_BOUNDS() CalculateOutput_256_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    // Declare a shared memory array for keys and values.
    __shared__ volatile float sKey[288 * 4];
    __shared__ volatile uint32_t sValue[288 * 4];
    __shared__ uint32_t sharedCount;

    // Calculate the position in the batch and the thread ID within a warp.
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        // Calculate pointers and initial values.
        float* pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[288 * offset];
        volatile uint32_t* psValue = &sValue[288 * offset];

        float k0 = -FLT_MAX;
        float k1 = -FLT_MAX;
        float k2 = -FLT_MAX;
        float k3 = -FLT_MAX;
        float k4 = -FLT_MAX;
        float k5 = -FLT_MAX;
        float k6 = -FLT_MAX;
        float k7 = -FLT_MAX;
        float k8 = -FLT_MAX;
        float k9 = -FLT_MAX;
        float k10 = -FLT_MAX;
        float k11 = -FLT_MAX;
        float k12 = -FLT_MAX;
        float k13 = -FLT_MAX;
        float k14 = -FLT_MAX;
        float k15 = -FLT_MAX;
        uint32_t v0 = 0;
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        uint32_t v3 = 0;
        uint32_t v4 = 0;
        uint32_t v5 = 0;
        uint32_t v6 = 0;
        uint32_t v7 = 0;
        uint32_t v8 = 0;
        uint32_t v9 = 0;
        uint32_t v10 = 0;
        uint32_t v11 = 0;
        uint32_t v12 = 0;
        uint32_t v13 = 0;
        uint32_t v14 = 0;
        uint32_t v15 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutput[wpos];
            v1 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k2 = pOutput[wpos];
            v2 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k3 = pOutput[wpos];
            v3 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k4 = pOutput[wpos];
            v4 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k5 = pOutput[wpos];
            v5 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k6 = pOutput[wpos];
            v6 = wpos;
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k7 = pOutput[wpos];
            v7 = wpos;
        }

        float minValue = -FLT_MAX;
        uint32_t rpos = 256;
        uint32_t bufferSize = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;

        if (tgx == 0)
            sharedCount = 0;
        __syncthreads();

        // Iterate over the width of the output.
        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -FLT_MAX;
            uint32_t value = wpos;
            if (wpos < width)
            {
                key = pOutput[wpos];
            }

            uint32_t count = 0;

            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                atomicAdd(&count, 1);
                count += sharedCount;
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }

            atomicAdd(&sharedCount, __popc(count));
            __syncthreads();

            if (bufferSize >= 256)
            {
                k8 = psKey[tgx];
                v8 = psValue[tgx];
                k9 = psKey[tgx + cData._warpSize];
                v9 = psValue[tgx + cData._warpSize];
                k10 = psKey[tgx + 2 * cData._warpSize];
                v10 = psValue[tgx + 2 * cData._warpSize];
                k11 = psKey[tgx + 3 * cData._warpSize];
                v11 = psValue[tgx + 3 * cData._warpSize];
                k12 = psKey[tgx + 4 * cData._warpSize];
                v12 = psValue[tgx + 4 * cData._warpSize];
                k13 = psKey[tgx + 5 * cData._warpSize];
                v13 = psValue[tgx + 5 * cData._warpSize];
                k14 = psKey[tgx + 6 * cData._warpSize];
                v14 = psValue[tgx + 6 * cData._warpSize];
                k15 = psKey[tgx + 7 * cData._warpSize];
                v15 = psValue[tgx + 7 * cData._warpSize];
                bool flag;
                BITONICSORT512_512();

                minValue = __shfl(k7, cData._warpSize - 1);

                bufferSize -= 256;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 256];
                    psValue[tgx] = psValue[tgx + 256];
                }
            }

            rpos += cData._warpSize;
            __syncthreads();
        }

        // Handle remaining elements in the buffer.
        if ((bufferSize > 0) || (width <= 256))
        {
            k8 = -FLT_MAX;
            k9 = -FLT_MAX;
            k10 = -FLT_MAX;
            k11 = -FLT_MAX;
            k12 = -FLT_MAX;
            k13 = -FLT_MAX;
            k14 = -FLT_MAX;
            k15 = -FLT_MAX;
            v8 = 0;
            v9 = 0;
            v10 = 0;
            v11 = 0;
            v12 = 0;
            v13 = 0;
            v14 = 0;
            v15 = 0;

            if (tgx < bufferSize)
            {
                k8 = psKey[tgx];
                v8 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k9 = psKey[tgx + cData._warpSize];
                v9 = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k10 = psKey[tgx + 2 * cData._warpSize];
                v10 = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k11 = psKey[tgx + 3 * cData._warpSize];
                v11 = psValue[tgx + 3 * cData._warpSize];
            }
            if (tgx + 4 * cData._warpSize < bufferSize)
            {
                k12 = psKey[tgx + 4 * cData._warpSize];
                v12 = psValue[tgx + 4 * cData._warpSize];
            }
            if (tgx + 5 * cData._warpSize < bufferSize)
            {
                k13 = psKey[tgx + 5 * cData._warpSize];
                v13 = psValue[tgx + 5 * cData._warpSize];
            }
            if (tgx + 6 * cData._warpSize < bufferSize)
            {
                k14 = psKey[tgx + 6 * cData._warpSize];
                v14 = psValue[tgx + 6 * cData._warpSize];
            }
            if (tgx + 7 * cData._warpSize < bufferSize)
            {
                k15 = psKey[tgx + 7 * cData._warpSize];
                v15 = psValue[tgx + 7 * cData._warpSize];
            }

            BITONICSORT512_512();
        }

        // Store the top-k values in the output buffers.
        float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k8;
            pValue[wpos] = v8;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k9;
            pValue[wpos] = v9;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k10;
            pValue[wpos] = v10;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k11;
            pValue[wpos] = v11;
        }
        if (wpos < k)
        {
            pKey[wpos] = k12;
            pValue[wpos] = v12;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k13;
            pValue[wpos] = v13;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k14;
            pValue[wpos] = v14;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k15;
            pValue[wpos] = v15;
        }
    }
}

/// <summary>
/// Calculate the output values for a given batch.
/// </summary>
/// <param name="pOutput">Pointer to the output buffer.</param>
/// <param name="pKey">Pointer to the key buffer.</param>
/// <param name="pValue">Pointer to the value buffer.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the output.</param>
/// <param name="k">The number of top-K values to compute.</param>
void CalculateOutput(float* pOutput, float* pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k)
{
    // Define the number of threads per block.
    uint32_t threadsPerBlock = 128;
    // Calculate the number of blocks.
    uint32_t blocks = (batch + 3) / 4;

    // Define custom launch configuration.
    dim3 grid(blocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Create an array of kernel arguments.
    void* args[] = { &pOutput, &pKey, &pValue, &batch, &width, &k };

    try {
        if (k <= 32) {
            // Launch the kernel with custom launch parameters using cudaLaunchKernel.
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)CalculateOutput_32_kernel,
                grid, blockDim,
                args,  // Pass kernel arguments as an array of pointers.
                0,     // Shared memory size (optional).
                NULL   // Stream identifier (optional, NULL for default stream).
            );
        }
        else if (k <= 64) {
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)CalculateOutput_64_kernel,
                grid, blockDim,
                args,
                0,
                NULL
            );
        }
        else if (k <= 128) {
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)CalculateOutput_128_kernel,
                grid, blockDim,
                args,
                0,
                NULL
            );
        }
        else {
            cudaError_t launchStatus = cudaLaunchKernel(
                (const void*)CalculateOutput_256_kernel,
                grid, blockDim,
                args,
                0,
                NULL
            );
        }

        // Synchronize the device to ensure kernel execution is completed.
        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in CalculateOutput: " + std::string(e.what()));
    }
}


__global__ void LAUNCH_BOUNDS() CalculateOutput_kernel(float* pOutputKey, float* pOutputValue, float* pKeyBuffer, float* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    __shared__ volatile float sKey[160 * 4];
    __shared__ volatile float sValue[160 * 4];
    __shared__ uint32_t sharedCount;

    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        pOutputKey += pos * width;
        pOutputValue += pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[160 * offset];
        volatile float* psValue = &sValue[160 * offset];

        float k0 = -FLT_MAX;
        float k1 = -FLT_MAX;
        float k2 = -FLT_MAX;
        float k3 = -FLT_MAX;
        float k4 = -FLT_MAX;
        float k5 = -FLT_MAX;
        float k6 = -FLT_MAX;
        float k7 = -FLT_MAX;
        float v0 = 0.0f;
        float v1 = 0.0f;
        float v2 = 0.0f;
        float v3 = 0.0f;
        float v4 = 0.0f;
        float v5 = 0.0f;
        float v6 = 0.0f;
        float v7 = 0.0f;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutputKey[wpos];
            v0 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutputKey[wpos];
            v1 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k2 = pOutputKey[wpos];
            v2 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k3 = pOutputKey[wpos];
            v3 = pOutputValue[wpos];
        }

        float minValue = -FLT_MAX;
        uint32_t rpos = 128;
        uint32_t bufferSize = 0;
        float key1, key2;
        float value1, value2;
        uint32_t otgx;
        bool flag;

        if (tgx == 0)
            sharedCount = 0;
        __syncthreads();

        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -FLT_MAX;
            float value = 0.0f;
            if (wpos < width)
            {
                key = pOutputKey[wpos];
                value = pOutputValue[wpos];
            }

            uint32_t count = 0;

            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                atomicAdd(&count, 1);
                count += sharedCount;
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }

            atomicAdd(&sharedCount, __popc(count));
            __syncthreads();

            if (bufferSize >= 128)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                bufferSize -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 128];
                    psValue[tgx] = psValue[tgx + 128];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 128))
        {
            k4 = -FLT_MAX;
            k5 = -FLT_MAX;
            k6 = -FLT_MAX;
            k7 = -FLT_MAX;
            v4 = 0.0f;
            v5 = 0.0f;
            v6 = 0.0f;
            v7 = 0.0f;

            if (tgx < bufferSize)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
            }
            BITONICSORT256_256();
        }

        float* pKey = pKeyBuffer + pos * k;
        float* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k2;
            pValue[wpos] = v2;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k3;
            pValue[wpos] = v3;
        }
    }
}

__global__ void LAUNCH_BOUNDS() CalculateOutput_kernel(float* pOutputKey, uint32_t* pOutputValue, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch, uint32_t width, uint32_t k)
{
    __shared__ volatile float sKey[160 * 4];
    __shared__ volatile uint32_t sValue[160 * 4];
    __shared__ uint32_t sharedCount;

    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> cData._warpBits;
    uint32_t tgx = threadIdx.x & cData._warpMask;

    if (pos < batch)
    {
        pOutputKey += pos * width;
        pOutputValue += pos * width;
        uint32_t offset = threadIdx.x >> cData._warpBits;
        volatile float* psKey = &sKey[160 * offset];
        volatile uint32_t* psValue = &sValue[160 * offset];

        float k0 = -FLT_MAX;
        float k1 = -FLT_MAX;
        float k2 = -FLT_MAX;
        float k3 = -FLT_MAX;
        float k4 = -FLT_MAX;
        float k5 = -FLT_MAX;
        float k6 = -FLT_MAX;
        float k7 = -FLT_MAX;
        uint32_t v0 = 0;
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        uint32_t v3 = 0;
        uint32_t v4 = 0;
        uint32_t v5 = 0;
        uint32_t v6 = 0;
        uint32_t v7 = 0;

        uint32_t wpos = tgx;
        if (wpos < width)
        {
            k0 = pOutputKey[wpos];
            v0 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k1 = pOutputKey[wpos];
            v1 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k2 = pOutputKey[wpos];
            v2 = pOutputValue[wpos];
        }
        wpos += cData._warpSize;
        if (wpos < width)
        {
            k3 = pOutputKey[wpos];
            v3 = pOutputValue[wpos];
        }

        float minValue = -FLT_MAX;
        uint32_t rpos = 128;
        uint32_t bufferSize = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;

        if (tgx == 0)
            sharedCount = 0;
        __syncthreads();

        while (rpos < width)
        {
            unsigned wpos = rpos + tgx;
            float key = -FLT_MAX;
            float value = 0.0f;
            if (wpos < width)
            {
                key = pOutputKey[wpos];
                value = pOutputValue[wpos];
            }

            uint32_t count = 0;

            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                atomicAdd(&count, 1);
                count += sharedCount;
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }

            atomicAdd(&sharedCount, __popc(count));
            __syncthreads();

            if (bufferSize >= 128)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
                bool flag;
                BITONICSORT256_256();

                bufferSize -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 128];
                    psValue[tgx] = psValue[tgx + 128];
                }
            }

            rpos += cData._warpSize;
        }

        if ((bufferSize > 0) || (width <= 128))
        {
            k4 = -FLT_MAX;
            k5 = -FLT_MAX;
            k6 = -FLT_MAX;
            k7 = -FLT_MAX;
            v4 = 0;
            v5 = 0;
            v6 = 0;
            v7 = 0;

            if (tgx < bufferSize)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
            }
            if (tgx + cData._warpSize < bufferSize)
            {
                k5 = psKey[tgx + cData._warpSize];
                v5 = psValue[tgx + cData._warpSize];
            }
            if (tgx + 2 * cData._warpSize < bufferSize)
            {
                k6 = psKey[tgx + 2 * cData._warpSize];
                v6 = psValue[tgx + 2 * cData._warpSize];
            }
            if (tgx + 3 * cData._warpSize < bufferSize)
            {
                k7 = psKey[tgx + 3 * cData._warpSize];
                v7 = psValue[tgx + 3 * cData._warpSize];
            }

            BITONICSORT256_256();
        }

        float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k2;
            pValue[wpos] = v2;
        }
        wpos += cData._warpSize;
        if (wpos < k)
        {
            pKey[wpos] = k3;
            pValue[wpos] = v3;
        }
    }
}

__global__ void
LAUNCH_BOUNDS()
kNormalizeWeights_kernel(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight)
{
    uint32_t pos                            = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        float r2                          = 0.0f;
        float* pEnd                       = pWeight + outputStride * inputStride; 
        pWeight                            += pos;
        float* p                          = pWeight;
        
        while (p < pEnd)
        {
            float x                       = *p;
            r2                             += x * x;
            p                              += outputStride;
        } 
        
        if (r2 > norm * norm)
        {
            norm                           *= rsqrt(r2);
            p                               = pWeight;
            while (p < pEnd)
            {
                *p                         *= norm;
                p                          += outputStride;
            }             
        }
    }

}

void kNormalizeWeights(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight)
{
    try {
        uint32_t blocks = (outputStride + 127) / 128;
        dim3 grid(blocks, 1, 1);
        dim3 block(128, 1, 1);

        void* args[] = { &norm, &outputStride, &inputStride, &pWeight };

        cudaLaunchKernel(
            (void*)kNormalizeWeights_kernel, // kernel function pointer
            grid, block,                     // grid and block dimensions
            args,                            // arguments to the kernel
            0, nullptr                       // shared memory size and stream
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kNormalizeWeights: " + std::string(e.what()));
    }
}


__global__ void
LAUNCH_BOUNDS()
kCalculateWeightMagnitudes_kernel(uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude)
{
    uint32_t pos                            = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        float r2                          = 0.0f;
        float* pEnd                       = pWeight + outputStride * inputStride; 
        pWeight                            += pos;
        float* p                          = pWeight;
        
        while (p < pEnd)
        {
            float x                       = *p;
            r2                             += x * x;
            p                              += outputStride;
        } 
        
        pMagnitude[pos]                     = r2;
    }

}

void kCalculateWeightMagnitudes(uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude)
{
    try {
        uint32_t blocks = (outputStride + 127) / 128;
        dim3 grid(blocks, 1, 1);
        dim3 block(128, 1, 1);

        void* args[] = { &outputStride, &inputStride, &pWeight, &pMagnitude };

        cudaLaunchKernel(
            (void*)kCalculateWeightMagnitudes_kernel, // kernel function pointer
            grid, block,                              // grid and block dimensions
            args,                                     // arguments to the kernel
            0, nullptr                                // shared memory size and stream
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateWeightMagnitudes: " + std::string(e.what()));
    }
}

__global__ void LAUNCH_BOUNDS() kNormalizeWeightMagnitudes_kernel(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude)
{
    uint32_t pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < outputStride)
    {
        float r2 = pMagnitude[pos];
        float* pEnd = pWeight + outputStride * inputStride;
        pWeight += pos;
        float* p = pWeight;

        if (r2 > norm * norm)
        {
            norm *= rsqrt(r2);
            p = pWeight;
            while (p < pEnd)
            {
                *p *= norm;
                p += outputStride;
            }
        }
    }
}

void kNormalizeWeightMagnitudes(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude)
{
    try {
        uint32_t blocks = (outputStride + 127) / 128;
        dim3 grid(blocks, 1, 1);
        dim3 block(128, 1, 1);

        void* args[] = { &norm, &outputStride, &inputStride, &pWeight, &pMagnitude };

        cudaLaunchKernel(
            (void*)kNormalizeWeightMagnitudes_kernel, // kernel function pointer
            grid, block,                              // grid and block dimensions
            args,                                     // arguments to the kernel
            0, nullptr                                // shared memory size and stream
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kNormalizeWeightMagnitudes: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateScaledBiasedDropout_kernel(float* pUnit, float* pRandom, float p, float target, float a, float b, size_t size)
{
    uint64_t pos                            = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float r                           = pRandom[pos];
        pUnit[pos]                          = (r < p) ? target : a * pUnit[pos] + b;
    }
}

void kCalculateScaledBiasedDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target, float a, float b)
{
    try {
        curandGenerateUniform(getGpu()._RNG, pRandom, batch * stride);
        unsigned long blocks = CalculateBlocks(batch * stride);
        dim3 grid(blocks, 1, 1);
        dim3 block(getGpu()._threadsPerBlock, 1, 1);

        float bias = a * target + b;
        uint32_t batch_stride = batch * stride;
        float p_copy = p;
        float a_copy = a;
        float b_copy = b;

        void* args[] = { &pUnit, &pRandom, &p_copy, &bias, &a_copy, &b_copy, &batch_stride };

        cudaLaunchKernel(
            (void*)kCalculateScaledBiasedDropout_kernel, // kernel function pointer
            grid, block,                                // grid and block dimensions
            args,                                       // arguments to the kernel
            0, nullptr                                  // shared memory size and stream
        );

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncStatus)));
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateScaledBiasedDropout: " + std::string(e.what()));
    }
}

__global__ void
LAUNCH_BOUNDS()
kCalculateDropout_kernel(float* pUnit, float* pRandom, float p, float scale, float target, size_t size)
{
    uint64_t pos                            = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float r                           = pRandom[pos];
        pUnit[pos]                          = (r < p) ? target : scale * pUnit[pos];
    }
}

void kCalculateDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target)
{
    try {
        if (batch == 0 || stride == 0) {
            return;
        }

        curandGenerateUniform(getGpu()._RNG, pRandom, batch * stride);
        unsigned long blocks = CalculateBlocks(batch * stride);
        float scale = (target == 0.0f) ? 1.0f / (1.0f - p) : 1.0f;

        // Define launch parameters
        dim3 grid(blocks);
        dim3 block(getGpu()._threadsPerBlock);
        size_t sharedMemBytes = 0;  // Assuming no dynamic shared memory for now

        uint32_t totalStride = batch * stride;
        void* kernelArgs[] = {
            &pUnit,
            &pRandom,
            &p,
            &scale,
            &target,
            &totalStride
        };

        // Using cudaLaunchKernel to launch the kernel
        cudaError_t launchStatus = cudaLaunchKernel(
            (void*)kCalculateDropout_kernel,
            grid,
            block,
            kernelArgs,
            sharedMemBytes,
            nullptr  // Default stream
        );

        if (launchStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateDropout: " + std::string(e.what()));
    }
}

__global__ void 
LAUNCH_BOUNDS()
kCalculateMaxout_kernel(float* pSrc, size_t size, float* pDst)
{
    uint64_t pos                        = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float s = pSrc[pos];
        float d = pDst[pos];
        if (s > d)
            pDst[pos]                   = s;
    }
}

void kCalculateMaxout(float* pSrc, size_t size, float* pDst)
{
    try {
        if (size == 0)
            return;

        unsigned long blocks = CalculateBlocks(size);
        dim3 gridDim(blocks);
        dim3 blockDim(getGpu()._threadsPerBlock);

        void* args[] = { (void*)&pSrc, (void*)&size, (void*)&pDst };

        cudaLaunchKernel((void*)kCalculateMaxout_kernel, gridDim, blockDim, args, 0, 0);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateMaxout: " + std::string(e.what()));
    }
}

__global__ void LAUNCH_BOUNDS() kCalculateCosine_kernel(float* pVector1, float* pVector2, uint32_t stride, float* pDPOut, float* pAOut, float* pBOut, uint32_t outStride)
{
    __shared__ float sDP[64];
    __shared__ float sA[64];
    __shared__ float sB[64];

    // Tiling parameters
    const int tileSize = 64;
    const int numTiles = stride / tileSize;

    // Calculate the starting positions for this thread block
    uint32_t blockOffset = blockIdx.x * stride;
    pVector1 += blockOffset + threadIdx.x;
    pVector2 += blockOffset + threadIdx.x;
    pDPOut += blockIdx.x * outStride;
    pAOut += blockIdx.x * outStride;
    pBOut += blockIdx.x * outStride;

    uint32_t pos = threadIdx.x;
    float dp = 0.0f;
    float al = 0.0f;
    float bl = 0.0f;

    // Loop unrolling for vectorized memory access
#pragma unroll
    for (int tile = 0; tile < numTiles; ++tile)
    {
        float4 a_vec = *((float4*)pVector1);
        float4 b_vec = *((float4*)pVector2);
        dp += a_vec.x * b_vec.x + a_vec.y * b_vec.y + a_vec.z * b_vec.z + a_vec.w * b_vec.w;
        al += a_vec.x * a_vec.x + a_vec.y * a_vec.y + a_vec.z * a_vec.z + a_vec.w * a_vec.w;
        bl += b_vec.x * b_vec.x + b_vec.y * b_vec.y + b_vec.z * b_vec.z + b_vec.w * b_vec.w;

        pVector1 += tileSize;
        pVector2 += tileSize;
        pos += tileSize;
    }

    // Warp-level reduction for dp, al, and bl
    uint32_t tgx = threadIdx.x & 31;
    dp += __shfl_xor_sync(0xFFFFFFFF, dp, tgx);
    al += __shfl_xor_sync(0xFFFFFFFF, al, tgx);
    bl += __shfl_xor_sync(0xFFFFFFFF, bl, tgx);

    // Store partial results in shared memory
    if (tgx == 0)
    {
        uint32_t index = threadIdx.x / 32;
        sDP[index] = dp;
        sA[index] = al;
        sB[index] = bl;
    }
    __syncthreads();

    // Thread 0 in each warp performs final reduction and normalization
    if (threadIdx.x < 32)
    {
        dp = sDP[threadIdx.x];
        al = sA[threadIdx.x];
        bl = sB[threadIdx.x];

        // Final warp-level reduction
        dp += __shfl_xor_sync(0xFFFFFFFF, dp, tgx);
        al += __shfl_xor_sync(0xFFFFFFFF, al, tgx);
        bl += __shfl_xor_sync(0xFFFFFFFF, bl, tgx);

        if (tgx == 0)
        {
            // Add small epsilon to avoid division by zero
            al = sqrtf(al + 1.0e-08f);
            bl = sqrtf(bl + 1.0e-08f);
            dp /= al * bl;

            // Write results to output
            *pAOut = al;
            *pBOut = bl;
            *pDPOut = dp;
        }
    }
}

void kCalculateCosine(float* pVector1In, float* pVector2In, uint32_t batch, uint32_t stride,
    float* pDPOut, float* pAOut, float* pBOut, uint32_t outStride)
{
    try {
        if (batch == 0 || stride == 0)
            return;

        unsigned long threads = max(32, min(stride, getGpu()._threadsPerBlock));
        dim3 gridDim(batch);
        dim3 blockDim(threads);

        void* args[] = { (void*)&pVector1In, (void*)&pVector2In, (void*)&stride,
                        (void*)&pDPOut, (void*)&pAOut, (void*)&pBOut, (void*)&outStride };

        cudaLaunchKernel((void*)kCalculateCosine_kernel, gridDim, blockDim, args, 0, 0);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateCosine: " + std::string(e.what()));
    }
}

__global__ void LAUNCH_BOUNDS() kCalculateDotProduct_kernel(float* pVector1In, float* pVector2In, uint32_t strideIn, float* pDPOut, uint32_t strideOut)
{
    __shared__ float sDP[32];

    pVector1In += blockIdx.x * strideIn + threadIdx.x;
    pVector2In += blockIdx.x * strideIn + threadIdx.x;
    pDPOut += blockIdx.x * strideOut;
    uint32_t pos = threadIdx.x;
    float dp = (float)0;

    while (pos < strideIn)
    {
        float a = *pVector1In;
        float b = *pVector2In;
        dp += a * b;
        pVector1In += blockDim.x;
        pVector2In += blockDim.x;
        pos += blockDim.x;
    }

    // Perform a parallel reduction within a warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        dp += __shfl_down_sync(0xFFFFFFFF, dp, offset);
    }

    // Store the reduced value in shared memory
    int warpIdx = threadIdx.x / warpSize;
    if (threadIdx.x % warpSize == 0)
    {
        sDP[warpIdx] = dp;
    }

    __syncthreads();

    // Perform a parallel reduction for all warps within a block
    if (threadIdx.x < warpSize)
    {
        dp = (threadIdx.x < blockDim.x / warpSize) ? sDP[threadIdx.x] : (float)0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            dp += __shfl_down_sync(0xFFFFFFFF, dp, offset);
        }

        // Store the final result in global memory
        if (threadIdx.x == 0)
        {
            *pDPOut = dp;
        }
    }
}

void kCalculateDotProduct(float* pVector1In, float* pVector2In, uint32_t batch, uint32_t strideIn,
    float* pDPOut, uint32_t strideOut)
{
    try {
        if (batch == 0 || strideIn == 0)
            return;

        unsigned long threads = max(32, min(strideIn, getGpu()._threadsPerBlock));
        dim3 gridDim(batch);
        dim3 blockDim(threads);

        void* args[] = { (void*)&pVector1In, (void*)&pVector2In, (void*)&strideIn,
                        (void*)&pDPOut, (void*)&strideOut };

        cudaLaunchKernel((void*)kCalculateDotProduct_kernel, gridDim, blockDim, args, 0, 0);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCalculateDotProduct: " + std::string(e.what()));
    }
}

#include "cub/util_allocator.cuh"
#include "cub/device/device_radix_sort.cuh"

template<typename KeyType, typename ValueType> size_t kInitSort(uint32_t items, GpuBuffer<KeyType>* pbKey, GpuBuffer<ValueType>* pbValue)
{
    uint32_t itemStride                     = ((items + 511) >> 9) << 9;
    size_t tempBytes;
    cub::DoubleBuffer<KeyType> d_keys(pbKey->_pDevData, pbKey->_pDevData + itemStride);
    cub::DoubleBuffer<ValueType> d_values(pbValue->_pDevData, pbValue->_pDevData + itemStride);
    cub::DeviceRadixSort::SortPairs(NULL, tempBytes, d_keys, d_values, items);
    return tempBytes;
}

template<typename KeyType, typename ValueType> bool kSort(uint32_t items, KeyType* pKey0, KeyType* pKey1, ValueType* pValue0, ValueType* pValue1, char* pTemp, size_t tempBytes)
{
    cub::DoubleBuffer<KeyType>  d_keys(pKey0, pKey1);
    cub::DoubleBuffer<ValueType> d_values(pValue0, pValue1);
    cub::DeviceRadixSort::SortPairs(pTemp, tempBytes, d_keys, d_values, items);
    return true;   
}

__global__ void
LAUNCH_BOUNDS()
kAddScaleBuffers_kernel(float* pDst, float* pSrc, float scale, uint64_t size)
{
    uint64_t pos                            = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        *(pDst + pos)                      += *(pSrc + pos) * scale;
}

void kAddScaleBuffers(float* pDst, float* pSrc, float scale, uint64_t size)
{
    try {
        if (size == 0)
            return;

        uint32_t blocks = CalculateBlocks(size);
        dim3 gridDim(blocks);
        dim3 blockDim(getGpu()._threadsPerBlock);

        void* args[] = { (void*)&pDst, (void*)&pSrc, (void*)&scale, (void*)&size };

        cudaLaunchKernel((void*)kAddScaleBuffers_kernel, gridDim, blockDim, args, 0, 0);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAddScaleBuffers: " + std::string(e.what()));
    }
}

__global__ void LAUNCH_BOUNDS() kAddBuffers_kernel(float* pDst, float* pSrc, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        *(pDst + pos) += *(pSrc + pos);
}

void kAddBuffers(float* pDst, float* pSrc, uint64_t size, cudaStream_t stream)
{
    if (size == 0)
        return;

    try {
        uint32_t blocks = CalculateBlocks(size);
        dim3 gridDim(blocks);
        dim3 blockDim(getGpu()._threadsPerBlock);

        void* args[] = { (void*)&pDst, (void*)&pSrc, (void*)&size };

        cudaLaunchKernel((void*)kAddBuffers_kernel, gridDim, blockDim, args, 0, stream);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAddBuffers: " + std::string(e.what()));
    }
}

__global__ void LAUNCH_BOUNDS() kAddBuffers2D_kernel(float* pDst, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width)
{
    uint64_t yOffset = blockIdx.y * blockDim.x + threadIdx.x;
    if (yOffset < width)
    {
        uint64_t dpos = blockIdx.x * dpitch + yOffset;
        uint64_t spos = blockIdx.x * spitch + yOffset;
        pDst[dpos] += pSrc[spos];
    }
}

void kAddBuffers2D(float* pDst, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream)
{
    if ((height == 0) || (width == 0))
        return;

    try {
        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 grid(height, (width + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);

        void* args[] = { (void*)&pDst, (void*)&dpitch, (void*)&pSrc, (void*)&spitch, (void*)&width };

        cudaLaunchKernel((void*)kAddBuffers2D_kernel, grid, blockDim, args, 0, stream);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kAddBuffers2D: " + std::string(e.what()));
    }
}

#include <cuda_fp16.h>

#define TILE_SIZE_X 32
#define TILE_SIZE_Y 16

__global__ void kCopy2D_kernel(float* pDst, uint32_t dpitch, const float* pSrc, uint32_t spitch, uint32_t width)
{
    // Calculate the global thread ID in both dimensions.
    uint64_t globalRow = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t globalCol = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory for tiling and cooperative loading.
    __shared__ float tile[TILE_SIZE_X][TILE_SIZE_Y];

    // Check if the thread is within the bounds of the destination matrix.
    if (globalRow < width)
    {
        // Calculate the starting positions in both source and destination matrices for this thread.
        uint64_t dpos = dpitch * globalRow;
        uint64_t spos = spitch * globalRow;

        // Complex data type for storing data with additional metadata.
        struct ComplexData
        {
            float value;
            int index;
        };

        ComplexData data;

        // Loop over tiles of the matrix with warp-level cooperation.
        for (uint64_t col = 0; col < width; col += TILE_SIZE_Y)
        {
            // Calculate the current column index.
            uint64_t currentCol = globalCol + col;

            // Cooperative loading into shared memory with advanced index calculations.
            if (currentCol < width)
            {
                // Calculate indices for advanced data loading.
                uint64_t srcIndex = spos + currentCol;
                float4* srcPtr = (float4*)(&pSrc[srcIndex]);

                // Load data into shared memory with a non-contiguous pattern.
                for (int i = 0; i < 4; ++i)
                {
                    // Create a ComplexData element with additional metadata.
                    data.value = srcPtr[i];
                    data.index = srcIndex + i;
                    // Store the ComplexData element into shared memory.
                    tile[threadIdx.x][threadIdx.y + i] = data;
                }
            }

            // Ensure all threads have loaded data into shared memory before proceeding.
            __syncthreads();

            // Copy data from shared memory to destination efficiently using vectorized memory access.
            if (currentCol < width)
            {
                // Calculate indices for advanced data storing.
                uint64_t dstIndex = dpos + currentCol;
                float4* dstPtr = (float4*)(&pDst[dstIndex]);

                // Perform the memory copy operation with advanced data manipulation.
                for (int i = 0; i < 4; ++i)
                {
                    // Extract the ComplexData element from shared memory.
                    data = tile[threadIdx.x][threadIdx.y + i];
                    // Store the value portion into the destination matrix.
                    dstPtr[i] = data.value;
                }
            }

            // Ensure all threads have finished copying data before proceeding.
            __syncthreads();
        }
    }
}

void kCopy2D(float* pDst, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream)
{
    if ((height == 0) || (width == 0))
        return;

    try {
        dim3 blockDim(getGpu()._threadsPerBlock);
        dim3 grid(height, (width + getGpu()._threadsPerBlock - 1) / getGpu()._threadsPerBlock);

        void* args[] = { (void*)&pDst, (void*)&dpitch, (void*)&pSrc, (void*)&spitch, (void*)&width };

        cudaLaunchKernel((void*)kCopy2D_kernel, grid, blockDim, args, 0, stream);

        cudaDeviceSynchronize();
        cudaError_t syncStatus = cudaGetLastError();
        if (syncStatus != cudaSuccess) {
            throw std::runtime_error("Device synchronization failed");
        }

        cudaError_t kernelStatus = cudaGetLastError();
        if (kernelStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error in kCopy2D: " + std::string(e.what()));
    }
}

//////////////////// CUDA kernels for calculating the activation functions ////////////////////

/// <summary>
/// CUDA kernel function for calculating the Sigmoid activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
__global__ void LAUNCH_BOUNDS() kCalculateSigmoidActivation_kernel(float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float a = 1.0f / (1.0f + exp(-pData[pos]));
        pData[pos] = a;
    }
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the Sigmoid activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
void kCalculateSigmoidActivation(float* pData, uint64_t size)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size
    };

    cudaLaunchKernel((void*)kCalculateSigmoidActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("kCalculateSigmoidActivation_kernel launch error: " + std::string(cudaGetErrorString(launchStatus)));
    }
}

/// <summary>
/// CUDA kernel function for calculating the Hyperbolic Tangent (tanh) activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
__global__ void LAUNCH_BOUNDS() kCalculateTanhActivation_kernel(float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        pData[pos] = tanh(pData[pos]);
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the tanh activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
void kCalculateTanhActivation(float* pData, uint64_t size)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size
    };

    cudaLaunchKernel((void*)kCalculateTanhActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("kCalculateTanhActivation_kernel launch error: " + std::string(cudaGetErrorString(launchStatus)));
    }
}

/// <summary>
/// CUDA kernel function for calculating the Rectified Linear Unit (ReLU) activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
__global__ void LAUNCH_BOUNDS() kCalculateRELUActivation_kernel(float* pData, uint64_t size)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
        pData[pos] = max(0.0f, pData[pos]);
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the ReLU activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
void kCalculateRELUActivation(float* pData, uint64_t size)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size
    };

    cudaLaunchKernel((void*)kCalculateRELUActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("kCalculateRELUActivation_kernel launch error: " + std::string(cudaGetErrorString(launchStatus)));
    }
}

/// <summary>
/// CUDA kernel function for calculating the Leaky Rectified Linear Unit (LReLU) activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="slope">Slope of the activation for negative inputs.</param>
__global__ void LAUNCH_BOUNDS() kCalculateLRELUActivation_kernel(float* pData, uint64_t size, float slope)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float val = pData[pos];
        pData[pos] = max(val, val * slope);
    }
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the LReLU activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="slope">Slope of the activation for negative inputs.</param>
void kCalculateLRELUActivation(float* pData, uint64_t size, float slope)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size,
        &slope
    };

    cudaLaunchKernel((void*)kCalculateLRELUActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("kCalculateLRELUActivation_kernel launch error: " + std::string(cudaGetErrorString(launchStatus)));
    }
}

/// <summary>
/// CUDA kernel function for calculating the Exponential Linear Unit (ELU) activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="alpha">Scaling factor for negative inputs.</param>
__global__ void LAUNCH_BOUNDS() kCalculateELUActivation_kernel(float* pData, uint64_t size, float alpha)
{
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size)
    {
        float x = pData[pos];
        pData[pos] = (x > (float)0.0) ? x : alpha * (exp(x) - (float)1.0);
    }
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the ELU activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="alpha">Scaling factor for negative inputs.</param>
void kCalculateELUActivation(float* pData, uint64_t size, float alpha)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size,
        &alpha
    };

    cudaLaunchKernel((void*)kCalculateELUActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("kCalculateELUActivation_kernel launch error: " + std::string(cudaGetErrorString(launchStatus)));
    }
}

// Define a constant for alpha to reduce redundancy
__device__ const float SELU_ALPHA = 1.67326f;
// Define a constant for lambda to reduce redundancy
__device__ const float SELU_LAMBDA = 1.0507f;

/// <summary>
/// Compute the SELU activation function for a given input.
/// </summary>
/// <param name="x">The input value.</param>
/// <returns>The result of the SELU activation function.</returns>
__device__ float computeSELUActivation(float x) {
    if (x > 0.0f) {
        return SELU_LAMBDA * x;
    }
    else {
        return SELU_LAMBDA * SELU_ALPHA * (expf(x) - 1.0f);
    }
}

/// <summary>
/// CUDA kernel function for calculating the Scaled Exponential Linear Unit (SELU) activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="alpha">Scaling factor for negative inputs.</param>
/// <param name="lambda">Scaling factor for positive inputs.</param>
__global__ void LAUNCH_BOUNDS() kCalculateSELUActivation_kernel(float* pData, uint64_t size, float alpha, float lambda)
{
    // Calculate the global position of the current thread
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Create an offset for strided access to data
    uint64_t stride = gridDim.x * blockDim.x;

    // Use a loop for parallel processing with strided access
    for (uint64_t pos = tid; pos < size; pos += stride)
    {
        float x = pData[pos];

        // Apply SELU activation function using a function call
        pData[pos] = computeSELUActivation(x);
    }
}

/// <summary>
/// Function for launching the CUDA kernel to calculate the SELU activation.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="size">Number of elements in the input data.</param>
/// <param name="alpha">Scaling factor for negative inputs.</param>
/// <param name="lambda">Scaling factor for positive inputs.</param>
void kCalculateSELUActivation(float* pData, uint64_t size, float alpha, float lambda)
{
    // Calculate the number of blocks needed based on the input size
    uint32_t blocks = CalculateBlocks(size);

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &size,
        &alpha,
        &lambda
    };

    cudaLaunchKernel((void*)kCalculateSELUActivation_kernel, blocks, blockDim, args, 0, nullptr);

    // Check for launch errors
    cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("kCalculateSELUActivation_kernel launch error: " + std::string(cudaGetErrorString(launchStatus)));
    }
}

/// <summary>
/// This CUDA kernel function calculates the SoftMax activation for a given array.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="stride">Stride of the input data.</param>
__global__ void kCalculateSoftMaxActivation_kernel(float* pData, uint32_t stride)
{
    // Shared memory for maximum value and sum of exponentials
    __shared__ float sMaxValue;
    __shared__ float sSumExp;

    // Initialize shared memory variables for the first thread in the block
    if (threadIdx.x == 0)
    {
        sMaxValue = -99999999.0f;
        sSumExp = 0.0f;
    }
    __syncthreads();

    // Calculate the starting position for this block
    pData += blockIdx.x * stride;
    uint32_t pos = threadIdx.x;
    float maxValue = -9999999999.0;

    // Compute the maximum value within the block
    while (pos < stride)
    {
        float z = pData[pos];
        maxValue = fmaxf(z, maxValue);
        pos += blockDim.x;
    }

    // Perform warp-level parallel reduction to find the block maximum
    uint32_t tgx = threadIdx.x & cData._warpMask;
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 1));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 2));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 4));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 8));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 16));

    // Store the block maximum in shared memory
    if (tgx == 0)
        sMaxValue = maxValue;
    __syncthreads();

    // Compute the sum of exponentials for the SoftMax function
    pos = threadIdx.x;
    float sum = 0.0f;
    while (pos < stride)
    {
        float z = pData[pos];
        sum += expf(z - sMaxValue);
        pos += blockDim.x;
    }

    // Perform warp-level parallel reduction to find the block sum
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 1);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 2);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 4);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 8);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 16);

    // Accumulate the sum of exponentials atomically
    atomicAdd(&sSumExp, sum);
    __syncthreads();

    // Normalize and apply SoftMax to the input data
    pos = threadIdx.x;
    while (pos < stride)
    {
        float z = pData[pos];
        float a = expf(z - sMaxValue);
        pData[pos] = fminf(1.0f, a / sSumExp);
        pos += blockDim.x;
    }
}

/// <summary>
/// This function calculates the SoftMax activation for a given data batch.
/// </summary>
/// <param name="pData">Pointer to the input data array.</param>
/// <param name="batch">Number of data points in the batch.</param>
/// <param name="stride">Stride of the input data.</param>
void kCalculateSoftMaxActivation(float* pData, uint32_t batch, uint32_t stride)
{
    // Calculate the number of warps per block
    uint32_t warps = getGpu()._threadsPerBlock / getGpu()._warpSize;

    // Define the kernel arguments
    void* args[] = {
        &pData,
        &batch,
        &stride
    };

    cudaLaunchKernel((void*)kCalculateSoftMaxActivation_kernel, batch, blockDim, args, 0, nullptr);

    // Check for launch errors
    cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
        throw std::runtime_error("kCalculateSoftMaxActivation_kernel launch error: " + std::string(cudaGetErrorString(launchStatus)));
    }
}

template size_t kInitSort<float, float>(uint32_t, GpuBuffer<float>*, GpuBuffer<float>*);
template size_t kInitSort<uint32_t, float>(uint32_t, GpuBuffer<uint32_t>*, GpuBuffer<float>*);
template size_t kInitSort<float, uint32_t>(uint32_t, GpuBuffer<float>*, GpuBuffer<uint32_t>*);
template size_t kInitSort<uint32_t, uint32_t>(uint32_t, GpuBuffer<uint32_t>*, GpuBuffer<uint32_t>*);

template bool kSort<float, float>(uint32_t, float*, float*, float*, float*, char*, size_t);
template bool kSort<float, uint32_t>(uint32_t, float*, float*, uint32_t*, uint32_t*, char*, size_t);
template bool kSort<uint32_t, float>(uint32_t, uint32_t*, uint32_t*, float*, float*, char*, size_t);
template bool kSort<uint32_t, uint32_t>(uint32_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, char*, size_t);

#define EXPLICITLY_INSTANTIATE_KERNELS(T) \
template void kCalculateL1OutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*, float, float, float);\
template void kCalculateIndexedL1OutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*, float, float, float);\
template void kCalculateL2HingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*, float, float, float);\
template void kCalculateIndexedL2HingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*, float, float, float);\
template void kCalculateCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*);\
template void kCalculateIndexedCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*);\
template void kCalculateScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*);\
template void kCalculateIndexedScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*);\
template void kCalculateOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*, float, float, float);\
template void kCalculateIndexedOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*, float, float, float);\
template void kCalculateHingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, T*, float*);\
template void kCalculateIndexedHingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, T*, float*);\
template void kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint64_t*, uint64_t*, uint32_t*, T*, bool);\
template void kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, T*, bool);\
template void kCalculateSparseAnalogOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*,  float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, bool, float, float, float);\
template void kCalculateIndexedSparseAnalogOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*,  float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, bool, float, float, float);\
template void kCalculateSparseAnalogL2HingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*,  float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, bool, float, float, float);\
template void kCalculateIndexedSparseAnalogL2HingeOutputDelta<T>(Activation, uint32_t, uint32_t, uint32_t, float*,  float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, bool, float, float, float);\
template float kCalculateL1Error<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedL1Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateL2Error<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedL2Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateHingeError<T>(uint32_t, uint32_t, uint32_t, float*, T*, float*);\
template float kCalculateIndexedHingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*, float*);\
template float kCalculateSparseAnalogL1Error<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateIndexedSparseAnalogL1Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateSparseAnalogL2Error<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateIndexedSparseAnalogL2Error<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateSparseAnalogL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateIndexedSparseAnalogL2HingeError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*, bool);\
template float kCalculateSparseAnalogMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*);\
template float kCalculateIndexedSparseAnalogMultinomialCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*);\
template float kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*);\
template float kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float* pDataWeight, T*);\
template float kCalculateSparseDataScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, T*, bool);\
template float kCalculateIndexedSparseDataScaledMarginalCrossEntropyError<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, T*, bool);\
template void kLoadSparseAnalogDenoisedInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*); \
template void kLoadIndexedSparseAnalogDenoisedInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*); \
template void kLoadSparseAnalogInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*); \
template void kLoadIndexedSparseAnalogInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*); \
template void kCalculateSparseAnalogZ<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float); \
template void kCalculateIndexedSparseAnalogZ<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float); \
template void kCalculateSparseAnalogDenoisedZ<T>(uint32_t, uint32_t, uint32_t, float*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float*, float); \
template void kCalculateIndexedSparseAnalogDenoisedZ<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, float*, float); \
template void kCalculateSparseTransposedAnalogMatrix<T>(uint32_t, uint32_t, uint64_t*, uint64_t*, uint32_t*, float*, T*, uint32_t*, uint32_t*, float*); \
template void kCalculateIndexedSparseTransposedAnalogMatrix<T>(uint32_t, uint32_t, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, uint32_t*, uint32_t*, float*); \
template void kCalculateSparseTransposedAnalogDenoisedMatrix<T>(uint32_t, uint32_t, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, uint32_t*, uint32_t*, float*); \
template void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix<T>(uint32_t, uint32_t, uint32_t*, uint64_t*, uint64_t*, uint32_t*, float*, T*, float*, uint32_t*, uint32_t*, float*); \
template void kLoadInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, T*); \
template void kLoadIndexedInputUnit<T>(uint32_t, uint32_t, uint32_t, float*, uint32_t*, T*);

EXPLICITLY_INSTANTIATE_KERNELS(float)
EXPLICITLY_INSTANTIATE_KERNELS(double)
EXPLICITLY_INSTANTIATE_KERNELS(unsigned char)
EXPLICITLY_INSTANTIATE_KERNELS(char)
EXPLICITLY_INSTANTIATE_KERNELS(uint32_t)
EXPLICITLY_INSTANTIATE_KERNELS(uint64_t)
EXPLICITLY_INSTANTIATE_KERNELS(int32_t)
EXPLICITLY_INSTANTIATE_KERNELS(int64_t)

#endif