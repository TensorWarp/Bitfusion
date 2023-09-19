#pragma once

/// <summary>
/// Calculates the number of blocks needed for a given size.
/// </summary>
/// <param name="size">The size for which to calculate the number of blocks.</param>
/// <returns>The number of blocks.</returns>
uint32_t CalculateBlocks(uint64_t size);

/// <summary>
/// Returns the sign of the input value.
/// </summary>
/// <typeparam name="T">The type of the input value.</typeparam>
/// <param name="x">The input value.</param>
/// <returns>1 if x is positive, -1 if x is negative, and 0 otherwise.</returns>
template<typename T> __device__ T sgn(T x);

/// <summary>
/// Sets the GPU data for kernels.
/// </summary>
void SetKernelsGpuData();

/// <summary>
/// Gets the GPU data for kernels.
/// </summary>
void GetKernelsGpuData();

/// <summary>
/// Sets the GPU data for KLoss.
/// </summary>
void SetKLossGpuData();

/// <summary>
/// Gets the GPU data for KLoss.
/// </summary>
void GetKLossGpuData();

/// <summary>
/// Sets the GPU data for KActivation.
/// </summary>
void SetKActivationGpuData();

/// <summary>
/// Gets the GPU data for KActivation.
/// </summary>
void GetKActivationGpuData();

/// <summary>
/// Sets the GPU data for KDelta.
/// </summary>
void SetKDeltaGpuData();

/// <summary>
/// Gets the GPU data for KDelta.
/// </summary>
void GetKDeltaGpuData();

/// <summary>
/// Scales and biases an array of floats.
/// </summary>
/// <param name="pData">Pointer to the data array.</param>
/// <param name="size">Size of the data array.</param>
/// <param name="scale">Scaling factor.</param>
/// <param name="bias">Bias value.</param>
void kScaleAndBias(float* pData, uint64_t size, float scale, float bias);

/// <summary>
/// Adds a bias to each unit in an array.
/// </summary>
/// <param name="pUnit">Pointer to the unit array.</param>
/// <param name="pBias">Pointer to the bias array.</param>
/// <param name="stride">Stride between units.</param>
/// <param name="batch">Number of batches.</param>
void kAddBias(float* pUnit, float* pBias, uint32_t stride, uint32_t batch);

/// <summary>
/// Adds two biases to each unit in an array.
/// </summary>
/// <param name="pUnit">Pointer to the unit array.</param>
/// <param name="pBias1">Pointer to the first bias array.</param>
/// <param name="pBias2">Pointer to the second bias array.</param>
/// <param name="stride">Stride between units.</param>
/// <param name="batch">Number of batches.</param>
void kAddDualBias(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch);

/// <summary>
/// Adds three biases to each unit in an array.
/// </summary>
/// <param name="pUnit">Pointer to the unit array.</param>
/// <param name="pBias1">Pointer to the first bias array.</param>
/// <param name="pBias2">Pointer to the second bias array.</param>
/// <param name="pBias3">Pointer to the third bias array.</param>
/// <param name="stride">Stride between units.</param>
/// <param name="batch">Number of batches.</param>
void kAddTripleBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch);

/// <summary>
/// Adds four biases to each unit in an array.
/// </summary>
/// <param name="pUnit">Pointer to the unit array.</param>
/// <param name="pBias1">Pointer to the first bias array.</param>
/// <param name="pBias2">Pointer to the second bias array.</param>
/// <param name="pBias3">Pointer to the third bias array.</param>
/// <param name="pBias4">Pointer to the fourth bias array.</param>
/// <param name="stride">Stride between units.</param>
/// <param name="batch">Number of batches.</param>
void kAddQuadBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch);

/// <summary>
/// Clears a unit by subtracting the bias.
/// </summary>
/// <param name="pUnit">Pointer to the unit array.</param>
/// <param name="pBias">Pointer to the bias array.</param>
/// <param name="stride">Stride between units.</param>
/// <param name="batch">Number of batches.</param>
void kClearUnit(float* pUnit, float* pBias, uint32_t stride, uint32_t batch);

/// <summary>
/// Clears a unit by subtracting two biases.
/// </summary>
/// <param name="pUnit">Pointer to the unit array.</param>
/// <param name="pBias1">Pointer to the first bias array.</param>
/// <param name="pBias2">Pointer to the second bias array.</param>
/// <param name="stride">Stride between units.</param>
/// <param name="batch">Number of batches.</param>
void kClearDualSourceUnit(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch);

/// <summary>
/// Clears a unit by subtracting three biases.
/// </summary>
/// <param name="pUnit">Pointer to the unit array.</param>
/// <param name="pBias1">Pointer to the first bias array.</param>
/// <param name="pBias2">Pointer to the second bias array.</param>
/// <param name="pBias3">Pointer to the third bias array.</param>
/// <param name="stride">Stride between units.</param>
/// <param name="batch">Number of batches.</param>
void kClearTripleSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch);

/// <summary>
/// Clears a unit by subtracting four biases.
/// </summary>
/// <param name="pUnit">Pointer to the unit array.</param>
/// <param name="pBias1">Pointer to the first bias array.</param>
/// <param name="pBias2">Pointer to the second bias array.</param>
/// <param name="pBias3">Pointer to the third bias array.</param>
/// <param name="pBias4">Pointer to the fourth bias array.</param>
/// <param name="stride">Stride between units.</param>
/// <param name="batch">Number of batches.</param>
void kClearQuadSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch);

/// <summary>
/// Updates biases using a delta and learning rate.
/// </summary>
/// <param name="alpha">Learning rate.</param>
/// <param name="batch">Number of batches.</param>
/// <param name="width">Width of the data.</param>
/// <param name="pDelta">Pointer to the delta array.</param>
/// <param name="pBias">Pointer to the bias array.</param>
void kUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias);

/// <summary>
/// Calculates the top-k elements from key-value pairs.
/// </summary>
/// <param name="pOutputKey">Pointer to the output key array.</param>
/// <param name="pKey">Pointer to the input key array.</param>
/// <param name="pValue">Pointer to the output value array.</param>
/// <param name="batch">Number of batches.</param>
/// <param name="width">Width of the data.</param>
/// <param name="k">Number of top elements to calculate.</param>
void CalculateOutput(float* pOutputKey, float* pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k);

/// <summary>
/// Calculates the top-k elements from key-value pairs.
/// </summary>
/// <param name="pOutputKey">Pointer to the output key array.</param>
/// <param name="pOutputValue">Pointer to the output value array.</param>
/// <param name="pKey">Pointer to the input key array.</param>
/// <param name="pValue">Pointer to the input value array.</param>
/// <param name="batch">Number of batches.</param>
/// <param name="width">Width of the data.</param>
/// <param name="k">Number of top elements to calculate.</param>
void CalculateOutput(float* pOutputKey, float* pOutputValue, float* pKey, float* pValue, uint32_t batch, uint32_t width, uint32_t k);

/// <summary>
/// Calculates the top-k elements from key-value pairs.
/// </summary>
/// <param name="pOutputKey">Pointer to the output key array.</param>
/// <param name="pOutputValue">Pointer to the output value array.</param>
/// <param name="pKey">Pointer to the input key array.</param>
/// <param name="pValue">Pointer to the input value array.</param>
/// <param name="batch">Number of batches.</param>
/// <param name="width">Width of the data.</param>
/// <param name="k">Number of top elements to calculate.</param>
void CalculateOutput(float* pOutputKey, uint32_t* pOutputValue, float* pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k);

/// <summary>
/// Calculates the k-sparse representation of a unit.
/// </summary>
/// <param name="pUnit">Pointer to the unit array.</param>
/// <param name="batch">Number of batches.</param>
/// <param name="stride">Stride between units.</param>
/// <param name="kSparse">Number of non-zero elements to keep.</param>
void kCalculateKSparse(float* pUnit, uint32_t batch, uint32_t stride, uint32_t kSparse);

/// <summary>
/// Adds scaled values from one buffer to another.
/// </summary>
/// <param name="pDest">Pointer to the destination buffer.</param>
/// <param name="pSrc">Pointer to the source buffer.</param>
/// <param name="scale">Scaling factor.</param>
/// <param name="size">Size of the buffers.</param>
void kAddScaleBuffers(float* pDest, float* pSrc, float scale, uint64_t size);

/// <summary>
/// Adds values from one buffer to another.
/// </summary>
/// <param name="pDest">Pointer to the destination buffer.</param>
/// <param name="pSrc">Pointer to the source buffer.</param>
/// <param name="size">Size of the buffers.</param>
/// <param name="stream">Optional CUDA stream for asynchronous operation (default is 0).</param>
void kAddBuffers(float* pDest, float* pSrc, uint64_t size, cudaStream_t stream = 0);

/// <summary>
/// Adds 2D values from one buffer to another.
/// </summary>
/// <param name="pDest">Pointer to the destination buffer.</param>
/// <param name="dpitch">Pitch of the destination buffer.</param>
/// <param name="pSrc">Pointer to the source buffer.</param>
/// <param name="spitch">Pitch of the source buffer.</param>
/// <param name="width">Width of the data to copy.</param>
/// <param name="height">Height of the data to copy.</param>
/// <param name="stream">Optional CUDA stream for asynchronous operation (default is 0).</param>
void kAddBuffers2D(float* pDest, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream = 0);

/// <summary>
/// Copies 2D values from one buffer to another.
/// </summary>
/// <param name="pDest">Pointer to the destination buffer.</param>
/// <param name="dpitch">Pitch of the destination buffer.</param>
/// <param name="pSrc">Pointer to the source buffer.</param>
/// <param name="spitch">Pitch of the source buffer.</param>
/// <param name="width">Width of the data to copy.</param>
/// <param name="height">Height of the data to copy.</param>
/// <param name="stream">Optional CUDA stream for asynchronous operation (default is 0).</param>
void kCopy2D(float* pDest, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream = 0);

/// <summary>
/// Initializes a sorting operation for an array of key-value pairs on the GPU.
/// </summary>
/// <typeparam name="KeyType">The type of keys in the key-value pairs.</typeparam>
/// <typeparam name="ValueType">The type of values in the key-value pairs.</typeparam>
/// <param name="items">The number of key-value pairs to sort.</param>
/// <param name="pbKey">Pointer to a GPU buffer containing the keys.</param>
/// <param name="pbValue">Pointer to a GPU buffer containing the values.</param>
/// <returns>The size in bytes of temporary storage required for sorting.</returns>
template<typename KeyType, typename ValueType>
size_t kInitSort(uint32_t items, GpuBuffer<KeyType>* pbKey, GpuBuffer<ValueType>* pbValue);

/// <summary>
/// Sorts an array of key-value pairs using parallel sorting on the GPU.
/// </summary>
/// <typeparam name="KeyType">The type of keys in the key-value pairs.</typeparam>
/// <typeparam name="ValueType">The type of values in the key-value pairs.</typeparam>
/// <param name="items">The number of key-value pairs to sort.</param>
/// <param name="pKey0">Pointer to the first key buffer.</param>
/// <param name="pKey1">Pointer to the second key buffer used for double buffering.</param>
/// <param name="pValue0">Pointer to the first value buffer.</param>
/// <param name="pValue1">Pointer to the second value buffer used for double buffering.</param>
/// <param name="pTemp">Pointer to temporary storage allocated for sorting.</param>
/// <param name="tempBytes">The size in bytes of the temporary storage.</param>
/// <returns>True if the sorting operation is successful, false otherwise.</returns>
template<typename KeyType, typename ValueType>
bool kSort(uint32_t items, KeyType* pKey0, KeyType* pKey1, ValueType* pValue0, ValueType* pValue1, char* pTemp, size_t tempBytes);

/// <summary>
/// Loads an input unit from generic data.
/// </summary>
/// <typeparam name="T">The type of the input data.</typeparam>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pUnit">Pointer to the unit array to load data into.</param>
/// <param name="pData">Pointer to the input data array.</param>
template<typename T>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData);

/// <summary>
/// Loads an input unit from indexed generic data.
/// </summary>
/// <typeparam name="T">The type of the input data.</typeparam>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pUnit">Pointer to the unit array to load data into.</param>
/// <param name="pIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pData">Pointer to the input data array.</param>
template<typename T>
void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData);

/// <summary>
/// Loads an input unit from sparse data with data weights.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pUnit">Pointer to the unit array to load data into.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);

/// <summary>
/// Loads an input unit from indexed sparse data with data weights.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pUnit">Pointer to the unit array to load data into.</param>
/// <param name="pIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
void kLoadIndexedSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);

/// <summary>
/// Loads an input unit from sparse denoised data with data weights and random values.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pUnit">Pointer to the unit array to load data into.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pRandom">Pointer to the random value array.</param>
void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom);

/// <summary>
/// Loads an input unit from indexed sparse denoised data with data weights and random values.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pUnit">Pointer to the unit array to load data into.</param>
/// <param name="pIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pRandom">Pointer to the random value array.</param>
void kLoadIndexedSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom);

/// <summary>
/// Loads an input unit from sparse analog data with data weights and analog data values.
/// </summary>
/// <typeparam name="T">The type of the sparse analog data.</typeparam>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pUnit">Pointer to the unit array to load data into.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pSparseData">Pointer to the sparse analog data array.</param>
template<typename T>
void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);

/// <summary>
/// Loads an input unit from indexed sparse analog data with data weights and analog data values.
/// </summary>
/// <typeparam name="T">The type of the sparse analog data.</typeparam>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pUnit">Pointer to the unit array to load data into.</param>
/// <param name="pIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pSparseData">Pointer to the sparse analog data array.</param>
template<typename T>
void kLoadIndexedSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);

/// <summary>
/// Loads an input unit from sparse analog denoised data with data weights, analog data values, and random values.
/// </summary>
/// <typeparam name="T">The type of the sparse analog data.</typeparam>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pUnit">Pointer to the unit array to load data into.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pSparseData">Pointer to the sparse analog data array.</param>
/// <param name="pRandom">Pointer to the random value array.</param>
template<typename T>
void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom);

/// <summary>
/// Loads an input unit from indexed sparse analog denoised data with data weights, analog data values, and random values.
/// </summary>
/// <typeparam name="T">The type of the sparse analog data.</typeparam>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pUnit">Pointer to the unit array to load data into.</param>
/// <param name="pIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pSparseData">Pointer to the sparse analog data array.</param>
/// <param name="pRandom">Pointer to the random value array.</param>
template<typename T>
void kLoadIndexedSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom);

/// <summary>
/// Calculates sparse Z values using weight, sparse data, and data weights.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pUnit">Pointer to the unit array to store the calculated values.</param>
/// <param name="beta">A constant factor for the calculation.</param>
void kCalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta);

/// <summary>
/// Calculates indexed sparse Z values using weight, index data, sparse data, and data weights.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
/// <param name="pIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pUnit">Pointer to the unit array to store the calculated values.</param>
/// <param name="beta">A constant factor for the calculation.</param>
void kCalculateIndexedSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta);

/// <summary>
/// Calculates sparse analog Z values using weight, sparse data, data weights, and analog data.
/// </summary>
/// <typeparam name="T">The type of the sparse analog data.</typeparam>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pSparseData">Pointer to the sparse analog data array.</param>
/// <param name="pUnit">Pointer to the unit array to store the calculated values.</param>
/// <param name="beta">A constant factor for the calculation.</param>
template<typename T>
void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta);

/// <summary>
/// Calculates indexed sparse analog Z values using weight, index data, sparse data, data weights, and analog data.
/// </summary>
/// <typeparam name="T">The type of the sparse analog data.</typeparam>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
/// <param name="pIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pSparseData">Pointer to the sparse analog data array.</param>
/// <param name="pUnit">Pointer to the unit array to store the calculated values.</param>
/// <param name="beta">A constant factor for the calculation.</param>
template<typename T>
void kCalculateIndexedSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta);

/// <summary>
/// Calculates denoised sparse Z values using weight, sparse data, data weights, and random values.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pRandom">Pointer to the random value array.</param>
/// <param name="pUnit">Pointer to the unit array to store the calculated values.</param>
/// <param name="beta">A constant factor for the calculation.</param>
void kCalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta);

/// <summary>
/// Calculates indexed denoised sparse Z values using weight, index data, sparse data, data weights, and random values.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
/// <param name="pIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pRandom">Pointer to the random value array.</param>
/// <param name="pUnit">Pointer to the unit array to store the calculated values.</param>
/// <param name="beta">A constant factor for the calculation.</param>
void kCalculateIndexedSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta);

/// <summary>
/// Calculates denoised sparse analog Z values using weight, sparse data, data weights, analog data, and random values.
/// </summary>
/// <typeparam name="T">The type of the sparse analog data.</typeparam>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pSparseData">Pointer to the sparse analog data array.</param>
/// <param name="pRandom">Pointer to the random value array.</param>
/// <param name="pUnit">Pointer to the unit array to store the calculated values.</param>
/// <param name="beta">A constant factor for the calculation.</param>
template<typename T>
void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta);

/// <summary>
/// Calculates indexed denoised sparse analog Z values using weight, index data, sparse data, data weights, analog data, and random values.
/// </summary>
/// <typeparam name="T">The type of the sparse analog data.</typeparam>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="stride">The stride between units.</param>
/// <param name="pWeight">Pointer to the weight array.</param>
/// <param name="pIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pSparseData">Pointer to the sparse analog data array.</param>
/// <param name="pRandom">Pointer to the random value array.</param>
/// <param name="pUnit">Pointer to the unit array to store the calculated values.</param>
/// <param name="beta">A constant factor for the calculation.</param>
template<typename T>
void kCalculateIndexedSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta);

/// <summary>
/// Calculates the transposed matrix from sparse data with data weights.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pSparseTransposedEnd">Pointer to the end indices of the transposed sparse data.</param>
/// <param name="pSparseTransposedIndex">Pointer to the index array specifying the data location for the transposed data.</param>
/// <param name="pSparseTransposedData">Pointer to the transposed sparse data array.</param>
void kCalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Calculates the transposed matrix from indexed sparse data with data weights.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="pIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pSparseTransposedEnd">Pointer to the end indices of the transposed sparse data.</param>
/// <param name="pSparseTransposedIndex">Pointer to the index array specifying the data location for the transposed data.</param>
/// <param name="pSparseTransposedData">Pointer to the transposed sparse data array.</param>
void kCalculateIndexedSparseTransposedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Calculates the transposed matrix from sparse denoised data with data weights and random values.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pRandom">Pointer to the random value array.</param>
/// <param name="pSparseTransposedEnd">Pointer to the end indices of the transposed sparse data.</param>
/// <param name="pSparseTransposedIndex">Pointer to the index array specifying the data location for the transposed data.</param>
/// <param name="pSparseTransposedData">Pointer to the transposed sparse data array.</param>
void kCalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Calculates the transposed matrix from indexed sparse denoised data with data weights and random values.
/// </summary>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="pIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pRandom">Pointer to the random value array.</param>
/// <param name="pSparseTransposedEnd">Pointer to the end indices of the transposed sparse data.</param>
/// <param name="pSparseTransposedIndex">Pointer to the index array specifying the data location for the transposed data.</param>
/// <param name="pSparseTransposedData">Pointer to the transposed sparse data array.</param>
void kCalculateIndexedSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Calculates the transposed weight gradient matrix from sparse transposed data and delta values.
/// </summary>
/// <param name="alpha">A scaling factor for the delta values.</param>
/// <param name="beta">A scaling factor for the transposed weight gradient.</param>
/// <param name="m">The number of rows in the original weight matrix.</param>
/// <param name="n">The number of columns in the original weight matrix.</param>
/// <param name="pSparseTransposedStart">Pointer to the start indices of the transposed sparse data.</param>
/// <param name="pSparseTransposedEnd">Pointer to the end indices of the transposed sparse data.</param>
/// <param name="pSparseTransposedIndex">Pointer to the index array specifying the data location for the transposed sparse data.</param>
/// <param name="pDelta">Pointer to the delta values.</param>
/// <param name="pWeightGradient">Pointer to the transposed weight gradient matrix.</param>
void kCalculateSparseTransposedWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pDelta, float* pWeightGradient);

/// <summary>
/// Calculates the transposed matrix from sparse analog data with data weights.
/// </summary>
/// <typeparam name="T">The type of the sparse analog data.</typeparam>
/// <param name="position">The position of the unit.</param>
/// <param name="batch">The batch number.</param>
/// <param name="pSparseStart">Pointer to the start indices of sparse data.</param>
/// <param name="pSparseEnd">Pointer to the end indices of sparse data.</param>
/// <param name="pSparseIndex">Pointer to the index array specifying the data location.</param>
/// <param name="pDataWeight">Pointer to the data weight array.</param>
/// <param name="pSparseData">Pointer to the sparse analog data array.</param>
/// <param name="pSparseTransposedEnd">Pointer to the end indices of the transposed sparse data.</param>
/// <param name="pSparseTransposedIndex">Pointer to the index array specifying the data location for the transposed data.</param>
/// <param name="pSparseTransposedData">Pointer to the transposed sparse data array.</param>
template<typename T>
void kCalculateSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

template<typename T> void kCalculateIndexedSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);
template<typename T> void kCalculateSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);
template<typename T> void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);
void kCalculateSparseTransposedAnalogWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData, float* pDelta, float* pWeightGradient);

template<typename T> float kCalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float kCalculateIndexedL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float kCalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float kCalculateIndexedL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float kCalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float kCalculateIndexedL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float kCalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float kCalculateIndexedCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float kCalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float kCalculateIndexedScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float kCalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float kCalculateIndexedMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float kCalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float kCalculateIndexedMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> float kCalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);
template<typename T> float kCalculateIndexedHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);

float kCalculateSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float kCalculateIndexedSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float kCalculateSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float kCalculateIndexedSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float kCalculateSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float kCalculateIndexedSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float kCalculateSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float kCalculateIndexedSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float kCalculateSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float kCalculateIndexedSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
float kCalculateSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);
float kCalculateIndexedSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);
float kCalculateSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);
float kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);
template<typename T> float kCalculateSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float kCalculateIndexedSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float kCalculateSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float kCalculateIndexedSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float kCalculateSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float kCalculateIndexedSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float kCalculateSparseAnalogCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float kCalculateIndexedSparseAnalogCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float kCalculateSparseAnalogScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float kCalculateIndexedSparseAnalogScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float kCalculateSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);
template<typename T> float kCalculateIndexedSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);
template<typename T> float kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);
template<typename T> float kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);
template<typename T> float kCalculateSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> float kCalculateIndexedSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);

float kCalculateRegularizationError(float lambda, float lambda1, float* pWeight, uint64_t size);

void kNormalizeWeights(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight);
void kCalculateWeightMagnitudes(uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude);
void kNormalizeWeightMagnitudes(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude);
void kNormalizeDeltas(float norm, uint32_t batch, uint32_t stride, float* pDelta);
void kCalculateDeltaMagnitudes(uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude);
void kNormalizeDeltaMagnitudes(float norm, uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude);

void kCalculateScaledBiasedDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target, float a, float b);
void kCalculateDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target);

template<typename T> void kCalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void kCalculateIndexedL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void kCalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight);
template<typename T> void kCalculateIndexedCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> void kCalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight);
template<typename T> void kCalculateIndexedScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight);
template<typename T> void kCalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void kCalculateIndexedOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void kCalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void kCalculateIndexedL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda);
template<typename T> void kCalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight);
template<typename T> void kCalculateIndexedHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight);

void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
void kCalculateIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
void kCalculateIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
void kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);
void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
void kCalculateIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
void kCalculateSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
void kCalculateIndexedSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float scope, float alpha, float lambda);
template<typename T> void kCalculateIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float scope, float alpha, float lambda);
template<typename T> void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void kCalculateIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void kCalculateIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);
template<typename T> void kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);

template<typename T> void kCalculateSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void kCalculateIndexedSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void kCalculateSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
template<typename T> void kCalculateIndexedSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);



void kCalculateSparsenessPenalty(uint32_t batch,  uint32_t stride, float* pUnit, float* pDelta, float p, float beta);

void kCalculateHadamardProduct(Activation activation, uint64_t size, float scale, float* pUnit, float* pDelta, float slope, float alpha, float lambda);

void kCalculateSigmoidActivation(float* pData, uint64_t size);
void kCalculateTanhActivation(float* pData, uint64_t size);
void kCalculateRELUActivation(float* pData, uint64_t size);
void kCalculateELUActivation(float* pData, uint64_t size, float alpha);
void kCalculateSELUActivation(float* pData, uint64_t size, float alpha, float lambda);
void kCalculateLRELUActivation(float* pData, uint64_t size, float slope);
void kCalculateSoftMaxActivation(float* pData, uint32_t batch, uint32_t stride);

void kSGDUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightGradient, float* pWeight);
void kSGDUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias);
void kMomentumUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);
void kMomentumUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);
void kAdaGradUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);
void kAdaGradUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);
void kNesterovShiftWeights(float mu, uint64_t size, float* pWeightVelocity, float* pWeight);
void kNesterovShiftBiases(float mu, uint32_t width, float* pBiasVelocity, float* pBias);
void kNesterovUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);
void kNesterovUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);
void kRMSPropUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);
void kRMSPropUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);
void kAdaDeltaUpdateWeights(float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight);
void kAdaDeltaUpdateBiases(float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias);
void kAdamUpdateWeights(float alpha, float lambda, float lambda1, float mu, float mu1, float t, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight);
void kAdamUpdateBiases(float alpha, float mu, float mu1, float t, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias);

void kCalculateMaxout(float* pSrc, size_t size, float* pDst);
void kCalculateCosine(float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDPOut, float* pAOut, float* pBOut, uint32_t outStride);                        
void kCalculateDotProduct(float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDPOut, uint32_t outStride);                        


void kCalculateMaxoutDelta(float* pSrc, float* pSrcDelta, size_t size, float beta, float* pDst, float* pDstDelta);
void kCalculateDotProductDelta(float* pDPDelta, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride);
void kCalculateCosineDelta(float* pDPDelta, float* pDP, float* pA, float* pB, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride);



#ifdef __NVCC__
__device__ inline uint64_t llitoulli(int64_t l)
{
    uint64_t u;
    asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
    return u;
}

__device__ inline int64_t ullitolli(uint64_t u)
{
    int64_t l;
    asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
    return l;
}

#if (CUDA_VERSION >= 9000)
#define SHFL(x, lane) __shfl_sync(0xffffffff, (x), (lane))
#define BALLOT(predicate) __ballot_sync(0xffffffff, (predicate))
#define ANY(predicate) __any_sync(0xffffffff, (predicate))
#else
#define SHFL(x, lane) __shfl((x), (lane))
#define BALLOT(predicate) __ballot(predicate)
#define ANY(predicate) __any(predicate)
#endif


#define REDUCEERROR(error) \
    if (ANY(error != (float)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        error                  += SHFL(error, tgx ^ 1); \
        error                  += SHFL(error, tgx ^ 2); \
        error                  += SHFL(error, tgx ^ 4); \
        error                  += SHFL(error, tgx ^ 8); \
        error                  += SHFL(error, tgx ^ 16); \
        if (tgx == 0) \
        { \
            atomicAdd(cData._pAccumulator, llitoulli(llrintf(ERRORSCALEF * error))); \
        } \
    } 


#define REDUCE(a) \
    if (ANY((a) != (float)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        a                      += SHFL((a), tgx ^ 1); \
        a                      += SHFL((a), tgx ^ 2); \
        a                      += SHFL((a), tgx ^ 4); \
        a                      += SHFL((a), tgx ^ 8); \
        a                      += SHFL((a), tgx ^ 16); \
    } 


#endif