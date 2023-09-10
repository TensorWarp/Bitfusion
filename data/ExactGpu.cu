#ifdef __CUDACC__

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <sstream>

#include "cudautil.h"
#include "ExactGpu.cuh"
#include "MathUtil.cuh"
#include "Output.cuh"

#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS cudaSuccess
#endif

namespace astdl {
    namespace knn {

        Knn::Knn(KnnData* data) :
            data(data) {
        }

        ExactGpu::ExactGpu(KnnData* data) :
            Knn(data) {
        }

        void ExactGpu::search(int k, const float* inputs, int size, std::string* keys, float* scores) {
            int maxK = data->maxK;
            int batchSize = data->batchSize;
            int numGpus = data->numGpus;

            if (k > maxK) {
                std::stringstream msg;
                msg << "k = " << k << " is > maxK = " << maxK;
                throw std::runtime_error(msg.str());
            }

            if (size > batchSize) {
                std::stringstream msg;
                msg << "size = " << size << " is > batchSize = " << batchSize;
                throw std::runtime_error(msg.str());
            }

            batchSize = size;

            std::vector<float*> allScores(numGpus);
            std::vector<uint32_t*> allIndexes(numGpus);

            omp_set_num_threads(numGpus);
#pragma omp parallel
            {
                int device = omp_get_thread_num();
                if (cudaSetDevice(device) != cudaSuccess) {
                    throw std::runtime_error("cudaSetDevice failed");
                }

                cublasHandle_t handle = data->cublasHandles[device];
                Matrix dCollectionPartition = data->dCollectionPartitions[device];
                Matrix dInputBatch = data->dInputBatches[device];
                Matrix dProducts = data->dProducts[device];
                Matrix dResultScores = data->dResultScores[device];
                Matrix dResultIndexes = data->dResultIndexes[device];
                Matrix hResultScores = data->hResultScores[device];
                Matrix hResultIndexes = data->hResultIndexes[device];
                uint32_t paddedRows = data->collectionRowsPadded[device];

                void* dA = dCollectionPartition.data;
                void* dB = dInputBatch.data;
                void* dC = dProducts.data;

                float* dScores = (float*)dResultScores.data;
                uint32_t* dIndexes = (uint32_t*)dResultIndexes.data;

                float* hScores = (float*)hResultScores.data;
                uint32_t* hIndexes = (uint32_t*)hResultIndexes.data;

                uint32_t aRows = dCollectionPartition.numRows;
                uint32_t bRows = batchSize;
                uint32_t cRows = batchSize;
                int aColumns = dCollectionPartition.numColumns;
                int bColumns = dInputBatch.numColumns;
                int cColumns = dProducts.numColumns;

                cudaDataType aType;
                cudaDataType bType;
                cudaDataType cType = CUDA_R_32F;

                if (data->dataType == astdl::knn::DataType::FP16) {
                    aType = CUDA_R_16F;
                    bType = CUDA_R_16F;

                    Matrix tmpBuffer = data->dInputBatchTmpBuffers[device];
                    astdl::math::kFloatToHalf(inputs, dInputBatch.getLength() * sizeof(float), (half*)dB,
                        (float*)tmpBuffer.data, tmpBuffer.getSizeInBytes());
                }
                else if (data->dataType == astdl::knn::DataType::FP32) {
                    aType = CUDA_R_32F;
                    bType = CUDA_R_32F;
                }
                else {
                    throw std::runtime_error("Unknown data type");
                }

                static const cublasOperation_t transa = CUBLAS_OP_N;
                static const cublasOperation_t transb = CUBLAS_OP_N;
                static const float alpha = 1.0f;
                static const float beta = 0.0f;

                cudaEvent_t start, stop;
                float elapsed;
                if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
                    throw std::runtime_error("cudaEventCreate failed");
                }
                if (cudaEventRecord(start, 0) != cudaSuccess) {
                    throw std::runtime_error("cudaEventRecord failed");
                }
                if (cublasSgemmEx(handle, transa, transb, aRows, bRows, aColumns, &alpha, dA, aType, aRows, dB, bType,
                    bColumns, &beta, dC, cType, cColumns) != CUDA_SUCCESS) {
                    throw std::runtime_error("cublasSgemmEx failed");
                }
                if (cudaEventRecord(stop, 0) != cudaSuccess) {
                    throw std::runtime_error("cudaEventRecord failed");
                }
                if (cudaEventSynchronize(stop) != cudaSuccess) {
                    throw std::runtime_error("cudaEventSynchronize failed");
                }
                if (cudaEventElapsedTime(&elapsed, start, stop) != cudaSuccess) {
                    throw std::runtime_error("cudaEventElapsedTime failed");
                }
                data->elapsedSgemm[device] = elapsed;

                if (cudaEventRecord(start, 0) != cudaSuccess) {
                    throw std::runtime_error("cudaEventRecord failed");
                }
                CalculateOutput((float*)dC, dScores, dIndexes, cRows, cColumns, paddedRows, maxK);
                if (cudaEventRecord(stop, 0) != cudaSuccess) {
                    throw std::runtime_error("cudaEventRecord failed");
                }
                if (cudaEventSynchronize(stop) != cudaSuccess) {
                    throw std::runtime_error("cudaEventSynchronize failed");
                }
                if (cudaEventElapsedTime(&elapsed, start, stop) != cudaSuccess) {
                    throw std::runtime_error("cudaEventElapsedTime failed");
                }
                data->elapsedTopK[device] = elapsed;

                if (cudaMemcpy(hScores, dScores, hResultScores.getSizeInBytes(), cudaMemcpyDeviceToHost) != cudaSuccess) {
                    throw std::runtime_error("cudaMemcpy failed");
                }
                if (cudaMemcpy(hIndexes, dIndexes, hResultIndexes.getSizeInBytes(), cudaMemcpyDeviceToHost) != cudaSuccess) {
                    throw std::runtime_error("cudaMemcpy failed");
                }

                allScores[device] = hScores;
                allIndexes[device] = hIndexes;
            }

            mergeKnn(k, batchSize, maxK, numGpus, allScores, allIndexes, data->hKeys, scores, keys);
        }

        void mergeKnn(int k, int batchSize, int width, int numGpus, const std::vector<float*>& allScores,
            const std::vector<uint32_t*>& allIndexes, const std::vector<std::vector<std::string>>& allKeys, float* scores,
            std::string* keys) {

            for (int i = 0; i < batchSize; ++i) {
                int* posIdxs = new int[numGpus];
                for (int n = 0; n < numGpus; n++) {
                    posIdxs[n] = i * width;
                }
                delete[] posIdxs;
                for (int col = 0; col < k; col++) {
                    int deviceId_0 = 0;
                    int posIdx_0 = posIdxs[deviceId_0];
                    float maxVal = allScores[deviceId_0][posIdx_0];
                    uint32_t maxIdx = allIndexes[deviceId_0][posIdx_0];
                    int maxDeviceId = deviceId_0;

                    for (int deviceId = 0; deviceId < numGpus; deviceId++) {
                        int posIdx = posIdxs[deviceId];
                        if (maxVal < allScores[deviceId][posIdx]) {
                            maxVal = allScores[deviceId][posIdx];
                            maxIdx = allIndexes[deviceId][posIdx];
                            maxDeviceId = deviceId;
                        }
                    }
                    ++posIdxs[maxDeviceId];
                    scores[i * k + col] = maxVal;
                    keys[i * k + col] = allKeys[maxDeviceId][maxIdx];
                }
            }
        }

    }  // namespace knn
}  // namespace astdl

#endif