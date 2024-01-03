#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <fstream>
#include <algorithm>

#include "recommendations.h"
#include "../../src/GpuTypes.h"
#include "utils.h"
#include "filters.h"
#include <driver_types.h>
#include "../../src/layer.h"

const string recommendations::DEFAULT_LAYER_RECS_GEN_LABEL = "Output";
const string recommendations::DEFAULT_SCORE_PRECISION = "4.3f";

const unsigned int recommendations::TOPK_SCALAR = 5;

recommendations::recommendations(unsigned int xBatchSize,
    unsigned int xK,
    unsigned int xOutputBufferSize,
    const string& layer,
    const string& precision)
    : pbKey(std::make_unique<GpuBuffer<float>>(xBatchSize* xK* TOPK_SCALAR, true)),
    pbUIValue(std::make_unique<GpuBuffer<unsigned int>>(xBatchSize* xK* TOPK_SCALAR, true)),
    pFilteredOutput(std::make_unique<GpuBuffer<float>>(xOutputBufferSize, true)),
    recsGenLayerLabel(layer),
    scorePrecision(precision) {
}

void recommendations::generateRecs(Network* xNetwork,
    unsigned int xK,
    const FilterConfig* xFilterSet,
    const vector<string>& xCustomerIndex,
    const vector<string>& xFeatureIndex) {
    int lBatch = xNetwork->GetBatch();
    int lExamples = xNetwork->GetExamples();
    int lPosition = xNetwork->GetPosition();
    if (lPosition + lBatch > lExamples) {
        lBatch = lExamples - lPosition;
    }

    bool bMultiGPU = (getGpu()._numprocs > 1);
    std::unique_ptr<GpuBuffer<float>> pbMultiKey;
    std::unique_ptr<GpuBuffer<unsigned int>> pbMultiUIValue;
    std::unique_ptr<GpuBuffer<unsigned int>> pbUIValueCache;
    float* pMultiKey = nullptr;
    unsigned int* pMultiUIValue = nullptr;
    unsigned int* pUIValueCache = nullptr;

    cudaIpcMemHandle_t keyMemHandle;
    cudaIpcMemHandle_t valMemHandle;
    const float* dOutput = xNetwork->GetUnitBuffer(recsGenLayerLabel);
    const Layer* pLayer = xNetwork->GetLayer(recsGenLayerLabel);
    auto [lx, ly, lz, lw] = pLayer->GetDimensions();
    int lOutputStride = lx * ly * lz * lw;
    auto [llx, lly, llz, llw] = pLayer->GetLocalDimensions();

    int lLocalOutputStride = llx * lly * llz * llw;
    unsigned int outputBufferSize = lLocalOutputStride * lBatch;
    if (!bMultiGPU) {
        outputBufferSize = xNetwork->GetBufferSize(recsGenLayerLabel);
    }

    vector<float> hOutputBuffer(outputBufferSize);

    if (bMultiGPU) {
        if (getGpu()._id == 0) {
            cudaError_t status;
            const size_t bufferSize = getGpu()._numprocs * lBatch * xK * TOPK_SCALAR;

            pbMultiKey = std::make_unique<GpuBuffer<float>>(bufferSize, true);
            pbMultiUIValue = std::make_unique<GpuBuffer<unsigned int>>(bufferSize, true);

            pMultiKey = pbMultiKey->_pDevData;
            pMultiUIValue = pbMultiUIValue->_pDevData;

            status = cudaIpcGetMemHandle(&keyMemHandle, pMultiKey);
            if (status != cudaSuccess) {
                fprintf(stderr, "cudaIpcGetMemHandle for keyMemHandle failed: %s\n", cudaGetErrorString(status));
            }

            status = cudaIpcGetMemHandle(&valMemHandle, pMultiUIValue);
            if (status != cudaSuccess) {
                fprintf(stderr, "cudaIpcGetMemHandle for valMemHandle failed: %s\n", cudaGetErrorString(status));
            }
        }

        MPI_Bcast(&keyMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&valMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        cudaError_t status;

        if (getGpu()._id != 0) {
            status = cudaIpcOpenMemHandle((void**)&pMultiKey, keyMemHandle, cudaIpcMemLazyEnablePeerAccess);
            if (status != cudaSuccess) {
                fprintf(stderr, "cudaIpcOpenMemHandle for pMultiKey failed: %s\n", cudaGetErrorString(status));
            }

            status = cudaIpcOpenMemHandle((void**)&pMultiUIValue, valMemHandle, cudaIpcMemLazyEnablePeerAccess);
            if (status != cudaSuccess) {
                fprintf(stderr, "cudaIpcOpenMemHandle for pMultiUIValue failed: %s\n", cudaGetErrorString(status));
            }
        }
    }
    cudaMemcpy(hOutputBuffer.data(), dOutput, outputBufferSize * sizeof(float), cudaMemcpyDeviceToHost);

    auto const start = std::chrono::steady_clock::now();
    for (int j = 0; j < lBatch; j++) {
        int sampleIndex = lPosition + j;

        int offset = getGpu()._id * lLocalOutputStride;
        xFilterSet->applySamplesFilter(hOutputBuffer.data() + j * lLocalOutputStride, sampleIndex, offset, lLocalOutputStride);
    }

    pFilteredOutput->Upload(hOutputBuffer.data());
    invokeExamples(pFilteredOutput->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, lBatch, lLocalOutputStride, xK * TOPK_SCALAR);

    if (bMultiGPU) {
        uint32_t offset = xK * TOPK_SCALAR * getGpu()._id;
        uint32_t kstride = xK * TOPK_SCALAR * getGpu()._numprocs;
        cudaMemcpy2D(pMultiKey + offset, kstride * sizeof(float), pbKey->_pDevData, xK * TOPK_SCALAR * sizeof(float), xK * TOPK_SCALAR * sizeof(float), lBatch, cudaMemcpyDefault);
        cudaMemcpy2D(pMultiUIValue + offset, kstride * sizeof(unsigned int), pbUIValue->_pDevData, xK * TOPK_SCALAR * sizeof(unsigned int), xK * TOPK_SCALAR * sizeof(unsigned int), lBatch, cudaMemcpyDefault);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        if (getGpu()._id == 0) {
            invokeExamples(pbMultiKey->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, lBatch, kstride, xK * TOPK_SCALAR);

            pbUIValueCache = std::make_unique<GpuBuffer<unsigned int>>(getGpu()._numprocs * lBatch * xK * TOPK_SCALAR, true);

            invokeExamples(pbMultiKey->_pDevData, pbMultiUIValue->_pDevData, pbKey->_pDevData, pbUIValueCache->_pDevData, lBatch, kstride, xK * TOPK_SCALAR);
        }
    }

    if (getGpu()._id == 0) {
        const char* fileName = xFilterSet->getOutputFileName().c_str();
        auto const now = std::chrono::steady_clock::now();
        std::cout << "Time Elapsed for Filtering and selecting Top " << xK << " recs: " << elapsed_seconds(start, now) << std::endl;
        std::cout << "Writing to " << fileName << std::endl;
        std::ofstream fp(fileName, std::ios::app);
        pbKey->Download();
        pbUIValue->Download();
        float* pKey = pbKey->_pSysData;
        unsigned int* pIndex = pbUIValue->_pSysData;

        if (bMultiGPU) {
            pbUIValueCache->Download();
            pUIValueCache = pbUIValueCache->_pSysData;
        }

        string strFormat = "%s,%" + scorePrecision + ":";
        for (int j = 0; j < lBatch; j++) {
            fp << xCustomerIndex[lPosition + j] << '\t';
            for (int x = 0; x < xK; ++x) {
                const size_t bufferPos = j * xK * TOPK_SCALAR + x;

                int finalIndex = pIndex[bufferPos];
                float value = pKey[bufferPos];
                if (bMultiGPU) {
                    int gpuId = finalIndex / (xK * TOPK_SCALAR);
                    int localIndex = pUIValueCache[bufferPos];
                    int globalIndex = gpuId * lLocalOutputStride + localIndex;
                    if (globalIndex < xFeatureIndex.size()) {
                        fp << xFeatureIndex[globalIndex] << ',' << value << ':';
                    }
                }
                else if (finalIndex < xFeatureIndex.size()) {
                    fp << xFeatureIndex[finalIndex] << ',' << value << ':';
                }
            }
            fp << '\n';
        }
        auto const end = std::chrono::steady_clock::now();
        std::cout << "Time Elapsed for Writing to file: " << elapsed_seconds(start, end) << std::endl;
    }

    if (bMultiGPU) {
        cudaError_t status;

        if (getGpu()._id != 0) {
            status = cudaIpcCloseMemHandle(pMultiKey);
            if (status != cudaSuccess) {
                fprintf(stderr, "cudaIpcCloseMemHandle for pMultiKey failed: %s\n", cudaGetErrorString(status));
            }

            status = cudaIpcCloseMemHandle(pMultiUIValue);
            if (status != cudaSuccess) {
                fprintf(stderr, "cudaIpcCloseMemHandle for pMultiUIValue failed: %s\n", cudaGetErrorString(status));
            }
        }
    }
}
