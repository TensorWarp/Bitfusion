#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>

#include "RecsGenerator.h"
#include "GpuTypes.h"
#include "Utils.h"
#include "Filters.h"

const std::string RecsGenerator::DEFAULT_LAYER_RECS_GEN_LABEL = "Output";
const std::string RecsGenerator::DEFAULT_SCORE_PRECISION = "4.3f";

const unsigned int RecsGenerator::TOPK_SCALAR = 5;

RecsGenerator::RecsGenerator(unsigned int xBatchSize,
    unsigned int xK,
    unsigned int xOutputBufferSize,
    const std::string& layer,
    const std::string& precision)
    : pbKey(std::make_unique<GpuBuffer<float>>(xBatchSize* xK* TOPK_SCALAR, true)),
    pbUIValue(std::make_unique<GpuBuffer<unsigned int>>(xBatchSize* xK* TOPK_SCALAR, true)),
    pFilteredOutput(std::make_unique<GpuBuffer<float>>(xOutputBufferSize, true)),
    recsGenLayerLabel(layer),
    scorePrecision(precision) {}

void RecsGenerator::generateRecs(Network* xNetwork,
    unsigned int xK,
    const FilterConfig* xFilterSet,
    const std::vector<std::string>& xCustomerIndex,
    const std::vector<std::string>& xFeatureIndex) {
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
    unsigned int lx, ly, lz, lw;
    std::tie(lx, ly, lz, lw) = pLayer->GetDimensions();
    int lOutputStride = lx * ly * lz * lw;
    unsigned int llx, lly, llz, llw;
    std::tie(llx, lly, llz, llw) = pLayer->GetLocalDimensions();

    int lLocalOutputStride = llx * lly * llz * llw;
    unsigned int outputBufferSize = lLocalOutputStride * lBatch;
    if (!bMultiGPU) {
        outputBufferSize = xNetwork->GetBufferSize(recsGenLayerLabel);
    }

    std::vector<float> hOutputBuffer(outputBufferSize);

    if (bMultiGPU) {
        if (getGpu()._id == 0) {
            const size_t bufferSize = getGpu()._numprocs * lBatch * xK * TOPK_SCALAR;

            pbMultiKey = std::make_unique<GpuBuffer<float>>(bufferSize, true);
            pbMultiUIValue = std::make_unique<GpuBuffer<unsigned int>>(bufferSize, true);

            pMultiKey = pbMultiKey->_pDevData;
            pMultiUIValue = pbMultiUIValue->_pDevData;
            cudaError_t status = cudaIpcGetMemHandle(&keyMemHandle, pMultiKey);
            RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiKey");
            status = cudaIpcGetMemHandle(&valMemHandle, pMultiUIValue);
            RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiUIValue");
        }

        MPI_Bcast(&keyMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&valMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (getGpu()._id != 0) {
            cudaError_t status = cudaIpcOpenMemHandle((void**)&pMultiKey, keyMemHandle, cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "cudaIpcOpenMemHandle: Unable to open key IPCMemHandle");
            status = cudaIpcOpenMemHandle((void**)&pMultiUIValue, valMemHandle, cudaIpcMemLazyEnablePeerAccess);
            RTERROR(status, "cudaIpcOpenMemHandle: Unable to open value IPCMemHandle");
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
    CalculateOutput(pFilteredOutput->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, lBatch, lLocalOutputStride, xK * TOPK_SCALAR);

    if (bMultiGPU) {
        uint32_t offset = xK * TOPK_SCALAR * getGpu()._id;
        uint32_t kstride = xK * TOPK_SCALAR * getGpu()._numprocs;
        cudaMemcpy2D(pMultiKey + offset, kstride * sizeof(float), pbKey->_pDevData, xK * TOPK_SCALAR * sizeof(float), xK * TOPK_SCALAR * sizeof(float), lBatch, cudaMemcpyDefault);
        cudaMemcpy2D(pMultiUIValue + offset, kstride * sizeof(unsigned int), pbUIValue->_pDevData, xK * TOPK_SCALAR * sizeof(unsigned int), xK * TOPK_SCALAR * sizeof(unsigned int), lBatch, cudaMemcpyDefault);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        if (getGpu()._id == 0) {
            CalculateOutput(pbMultiKey->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, lBatch, kstride, xK * TOPK_SCALAR);

            pbUIValueCache = std::make_unique<GpuBuffer<unsigned int>>(getGpu()._numprocs * lBatch * xK * TOPK_SCALAR, true);

            CalculateOutput(pbMultiKey->_pDevData, pbMultiUIValue->_pDevData, pbKey->_pDevData, pbUIValueCache->_pDevData, lBatch, kstride, xK * TOPK_SCALAR);
        }
    }

    if (getGpu()._id == 0) {
        const char* fileName = xFilterSet->getOutputFileName().c_str();
        auto const now = std::chrono::steady_clock::now();
        std::cout << "Time Elapsed for Filtering and selecting Top " << xK << " recs: " << elapsed_seconds(start, now) << std::endl;
        std::cout << "Writing to " << fileName << std::endl;
        std::ofstream outputFile(fileName, std::ios::app);
        pbKey->Download();
        pbUIValue->Download();
        float* pKey = pbKey->_pSysData;
        unsigned int* pIndex = pbUIValue->_pSysData;

        if (bMultiGPU) {
            pbUIValueCache->Download();
            pUIValueCache = pbUIValueCache->_pSysData;
        }

        std::string strFormat = "%s,%" + scorePrecision + ":";
        for (int j = 0; j < lBatch; j++) {
            outputFile << xCustomerIndex[lPosition + j] << '\t';
            for (int x = 0; x < xK; ++x) {
                const size_t bufferPos = j * xK * TOPK_SCALAR + x;

                int finalIndex = pIndex[bufferPos];
                float value = pKey[bufferPos];
                if (bMultiGPU) {
                    int gpuId = finalIndex / (xK * TOPK_SCALAR);
                    int localIndex = pUIValueCache[bufferPos];
                    int globalIndex = gpuId * lLocalOutputStride + localIndex;
                    if (globalIndex < xFeatureIndex.size()) {
                        outputFile << xFeatureIndex[globalIndex] << value;
                    }
                }
                else if (finalIndex < xFeatureIndex.size()) {
                    outputFile << xFeatureIndex[finalIndex] << value;
                }
            }
            outputFile << '\n';
        }
        outputFile.close();
        auto const end = std::chrono::steady_clock::now();
        std::cout << "Time Elapsed for Writing to file: " << elapsed_seconds(start, end) << std::endl;
    }

    if (bMultiGPU) {
        if (getGpu()._id != 0) {
            cudaError_t status = cudaIpcCloseMemHandle(pMultiKey);
            RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiKey IpcMemHandle");
            status = cudaIpcCloseMemHandle(pMultiUIValue);
            RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiFValue IpcMemHandle");
        }
    }
}