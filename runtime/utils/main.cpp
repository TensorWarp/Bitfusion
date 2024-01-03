#include "../../src/gpuTypes.h"
#include "../../src/types.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <cmath>
#include <mpi.h>
#include "cdl.h"

int main(int argc, char** argv) {
    getGpu().Startup(argc, argv);

    CDL cdl;

    getGpu().SetRandomSeed(cdl._randomSeed);

    float lambda1 = 0.0f;
    float mu1 = 0.0f;
    Network* pNetwork = nullptr;

    std::vector<DataSetBase*> vDataSet;
    vDataSet = LoadNetCDF(cdl._dataFileName);

    if (static_cast<Mode>(cdl._mode) == Mode::Validation)
        pNetwork = LoadNeuralNetworkNetCDF(cdl._networkFileName, cdl._batch);
    else
        pNetwork = LoadNeuralNetworkJSON(cdl._networkFileName, cdl._batch, vDataSet);

    pNetwork->LoadDataSets(vDataSet);
    pNetwork->SetCheckpoint(cdl._checkpointFileName, cdl._checkpointInterval);

    if (static_cast<Mode>(cdl._mode) == Mode::Validation) {
        pNetwork->SetTrainingMode(Nesterov);
        pNetwork->Validate();
    }
    else if (static_cast<Mode>(cdl._mode) == Mode::Training) {
        pNetwork->SetTrainingMode(cdl._optimizer);
        float alpha = cdl._alpha;
        int epochs = 0;
        while (epochs < cdl._epochs) {
            pNetwork->Train(cdl._alphaInterval, alpha, cdl._lambda, lambda1, cdl._mu, mu1);
            alpha *= cdl._alphaMultiplier;
            epochs += cdl._alphaInterval;
        }

        pNetwork->SaveNetCDF(cdl._resultsFileName);
    }
    else {
        bool bFilterPast = false;
        const Layer* pLayer = pNetwork->GetLayer("Output");
        uint32_t Nx, Ny, Nz, Nw;
        std::tie(Nx, Ny, Nz, Nw) = pLayer->GetLocalDimensions();
        const uint32_t STRIDE = Nx * Ny * Nz * Nw;

        unsigned int K = 10;

        size_t inputIndex = 0;
        while ((inputIndex < vDataSet.size()) && (vDataSet[inputIndex]->_name != "input"))
            inputIndex++;
        if (inputIndex == vDataSet.size()) {
            std::cerr << "Unable to find input dataset, exiting." << std::endl;
            exit(-1);
        }
        size_t outputIndex = 0;
        while ((outputIndex < vDataSet.size()) && (vDataSet[outputIndex]->_name != "output"))
            outputIndex++;
        if (outputIndex == vDataSet.size()) {
            std::cerr << "Unable to find output dataset, exiting." << std::endl;
            exit(-1);
        }

        int batch = cdl._batch;

        std::vector<float> vPrecision(K);
        std::vector<float> vRecall(K);
        std::vector<float> vNDCG(K);
        std::vector<uint32_t> vDataPoints(batch);
        std::unique_ptr<GpuBuffer<float>> pbTarget(new GpuBuffer<float>(batch * STRIDE, true));
        std::unique_ptr<GpuBuffer<float>> pbOutput(new GpuBuffer<float>(batch * STRIDE, true));
        DataSet<float>* pInputDataSet = dynamic_cast<DataSet<float>*>(vDataSet[inputIndex]);
        DataSet<float>* pOutputDataSet = dynamic_cast<DataSet<float>*>(vDataSet[outputIndex]);
        std::unique_ptr<GpuBuffer<float>> pbKey(new GpuBuffer<float>(batch * K, true));
        std::unique_ptr<GpuBuffer<unsigned int>> pbUIValue(new GpuBuffer<unsigned int>(batch * K, true));
        std::unique_ptr<GpuBuffer<float>> pbFValue(new GpuBuffer<float>(batch * K, true));
        float* pOutputValue = pbOutput->_pSysData;
        bool bMultiGPU = (getGpu()._numprocs > 1);
        std::unique_ptr<GpuBuffer<float>> pbMultiKey(nullptr);
        std::unique_ptr<GpuBuffer<float>> pbMultiFValue(nullptr);
        float* pMultiKey = nullptr;
        float* pMultiFValue = nullptr;
        cudaIpcMemHandle_t keyMemHandle;
        cudaIpcMemHandle_t valMemHandle;

        if (bMultiGPU) {
            if (getGpu()._id == 0) {
                pbMultiKey.reset(new GpuBuffer<float>(getGpu()._numprocs * batch * K, true));
                pbMultiFValue.reset(new GpuBuffer<float>(getGpu()._numprocs * batch * K, true));
                pMultiKey = pbMultiKey->_pDevData;
                pMultiFValue = pbMultiFValue->_pDevData;

                cudaError_t status = cudaIpcGetMemHandle(&keyMemHandle, pMultiKey);
                if (status != cudaSuccess) {
                    fprintf(stderr, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiKey: %s\n", cudaGetErrorString(status));
                }

                status = cudaIpcGetMemHandle(&valMemHandle, pMultiFValue);
                if (status != cudaSuccess) {
                    fprintf(stderr, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiFValue: %s\n", cudaGetErrorString(status));
                }
            }

            MPI_Bcast(&keyMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&valMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

            if (getGpu()._id != 0) {
                cudaError_t status = cudaIpcOpenMemHandle((void**)&pMultiKey, keyMemHandle, cudaIpcMemLazyEnablePeerAccess);
                if (status != cudaSuccess) {
                    fprintf(stderr, "cudaIpcOpenMemHandle: Unable to open key IPCMemHandle: %s\n", cudaGetErrorString(status));
                }
                status = cudaIpcOpenMemHandle((void**)&pMultiFValue, valMemHandle, cudaIpcMemLazyEnablePeerAccess);
                if (status != cudaSuccess) {
                    fprintf(stderr, "cudaIpcOpenMemHandle: Unable to open value IPCMemHandle: %s\n", cudaGetErrorString(status));
                }
            }
        }

        for (unsigned long long int pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch()) {
            pNetwork->SetPosition(pos);
            pNetwork->PredictBatch();
            unsigned int batch = pNetwork->GetBatch();
            if (pos + batch > pNetwork->GetExamples())
                batch = pNetwork->GetExamples() - pos;
            float* pTarget = pbTarget->_pSysData;
            std::memset(pTarget, 0, STRIDE * batch * sizeof(float));
            const float* pOutputKey = pNetwork->GetUnitBuffer("Output");
            float* pOut = pOutputValue;
            cudaError_t status = cudaMemcpy(pOut, pOutputKey, batch * STRIDE * sizeof(float), cudaMemcpyDeviceToHost);

            if (status != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy GpuBuffer::Download failed: %s\n", cudaGetErrorString(status));
            }

            for (int i = 0; i < batch; i++) {
                int j = pos + i;
                vDataPoints[i] = pOutputDataSet->_vSparseEnd[j] - pOutputDataSet->_vSparseStart[j];

                for (size_t k = pOutputDataSet->_vSparseStart[j]; k < pOutputDataSet->_vSparseEnd[j]; k++) {
                    pTarget[pOutputDataSet->_vSparseIndex[k]] = 1.0f;
                }

                if (bFilterPast) {
                    for (size_t k = pInputDataSet->_vSparseStart[j]; k < pInputDataSet->_vSparseEnd[j]; k++) {
                        pOut[pInputDataSet->_vSparseIndex[k]] = 0.0f;
                    }
                }
                pTarget += STRIDE;
                pOut += STRIDE;
            }
            pbTarget->Upload();
            pbOutput->Upload();
            invokeExamples(pbOutput->_pDevData, pbTarget->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, STRIDE, K);
            pbKey->Download();
            pbFValue->Download();

            if (bMultiGPU) {
                MPI_Reduce((getGpu()._id == 0) ? MPI_IN_PLACE : vDataPoints.data(), vDataPoints.data(), batch, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

                uint32_t offset = K * getGpu()._id;
                uint32_t kstride = K * getGpu()._numprocs;
                cudaMemcpy2D(pMultiKey + offset, kstride * sizeof(float), pbKey->_pDevData, K * sizeof(float), K * sizeof(float), batch, cudaMemcpyDefault);

                cudaMemcpy2D(pMultiFValue + offset, kstride * sizeof(float), pbFValue->_pDevData, K * sizeof(float), K * sizeof(float), batch, cudaMemcpyDefault);
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                if (getGpu()._id == 0) {
                    invokeExamples(pbMultiKey->_pDevData, pbMultiFValue->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, getGpu()._numprocs * K, K);
                }
            }

            if (getGpu()._id == 0) {
                pbKey->Download();
                pbFValue->Download();
                float* pKey = pbKey->_pSysData;
                float* pValue = pbFValue->_pSysData;
                for (int i = 0; i < batch; i++) {
                    float p = vDataPoints[i];
                    float tp = 0.0f;
                    float fp = 0.0f;
                    float idcg = 0.0f;
                    for (float pp = 0.0f; pp < p; pp++) {
                        idcg += 1.0f / std::log2(pp + 2.0f);
                    }
                    float dcg = 0.0f;
                    for (int j = 0; j < K; j++) {
                        if (pValue[j] == 1.0f) {
                            tp++;
                            dcg += 1.0f / std::log2(static_cast<float>(j + 2));
                        }
                        else
                            fp++;
                        vPrecision[j] += tp / (tp + fp);
                        vRecall[j] += tp / p;
                        vNDCG[j] += dcg / idcg;
                    }
                    pKey += K;
                    pValue += K;
                }
            }
        }

        if (bMultiGPU) {
            if (getGpu()._id != 0) {
                cudaError_t status = cudaIpcCloseMemHandle(pMultiKey);
                if (status != cudaSuccess) {
                    fprintf(stderr, "cudaIpcCloseMemHandle: Error closing MultiKey IpcMemHandle: %s\n", cudaGetErrorString(status));
                }
                status = cudaIpcCloseMemHandle(pMultiFValue);
                if (status != cudaSuccess) {
                    fprintf(stderr, "cudaIpcCloseMemHandle: Error closing MultiFValue IpcMemHandle: %s\n", cudaGetErrorString(status));
                }
            }
        }

        if (getGpu()._id == 0) {
            for (int i = 0; i < K; i++) {
                std::cout << i + 1 << "," << vPrecision[i] / pNetwork->GetExamples() << "," << vRecall[i] / pNetwork->GetExamples() << "," << vNDCG[i] / pNetwork->GetExamples() << std::endl;
            }
        }
    }

    delete pNetwork;

    for (auto p : vDataSet)
        delete p;

    getGpu().Shutdown();
    return 0;
}
