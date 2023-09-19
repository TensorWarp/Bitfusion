#include "../GpuTypes.h"
#include "../Types.h"
#include <time.h>
#include "CDL.h"
#include <format>

int main(int argc, char** argv) {
    getGpu().Startup(argc, argv);

    CDL cdl;

    if (argc == 2) {
        int err = cdl.Load_JSON(argv[1]);
        if (err != 0) {
            std::cout << std::format("*** Error, %s could parse CDC file %s\n", argv[0], argv[1]);
            return -1;
        }
    }
    else {
        cdl._mode = Prediction;
        cdl._optimizer = TrainingMode::Nesterov;
        cdl._networkFileName = "network.nc";
        cdl._alphaInterval = 20;
        cdl._alphaMultiplier = 0.8f;
        cdl._alpha = 0.025f;
        cdl._lambda = 0.0001f;
        cdl._mu = 0.5f;
        cdl._randomSeed = 12345;
        cdl._epochs = 60;
        cdl._dataFileName = "../../data/data_test.nc";
    }

    getGpu().SetRandomSeed(cdl._randomSeed);

    float lambda1 = 0.0f;
    float mu1 = 0.0f;
    Network* pNetwork;

    std::vector<DataSetBase*> vDataSet;
    vDataSet = LoadNetCDF(cdl._dataFileName);

#if 0        
    vector<tuple<uint64_t, uint64_t> > vMemory = vDataSet[0]->getMemoryUsage();
    uint64_t cpuMemory, gpuMemory;
    tie(cpuMemory, gpuMemory) = vMemory[0];
    cout << "CPUMem: " << cpuMemory << " GPUMem: " << gpuMemory << endl;
    exit(-1);
#endif    

    if (cdl._mode == Prediction)
        pNetwork = LoadNeuralNetworkNetCDF(cdl._networkFileName, cdl._batch);
    else
        pNetwork = LoadNeuralNetworkJSON(cdl._networkFileName, cdl._batch, vDataSet);

    int totalGPUMemory;
    int totalCPUMemory;
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << '\n';
    std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << '\n';
    pNetwork->LoadDataSets(vDataSet);
    pNetwork->SetCheckpoint(cdl._checkpointFileName, cdl._checkpointInterval);

    if (cdl._mode == Mode::Validation) {
        pNetwork->SetTrainingMode(Nesterov);
        pNetwork->Validate();
    }
    else if (cdl._mode == Training) {
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
            std::cout << std::format("Unable to find input dataset, exiting.\n");
            std::exit(-1);
        }
        size_t outputIndex = 0;
        while ((outputIndex < vDataSet.size()) && (vDataSet[outputIndex]->_name != "output"))
            outputIndex++;
        if (outputIndex == vDataSet.size()) {
            std::format("Unable to find output dataset, exiting.\n");
            std::exit(-1);
        }

        int batch = cdl._batch;

        std::vector<float> vPrecision(K);
        std::vector<float> vRecall(K);
        std::vector<float> vNDCG(K);
        std::vector<uint32_t> vDataPoints(batch);
        GpuBuffer<float>* pbTarget = new GpuBuffer<float>(batch * STRIDE, true);
        GpuBuffer<float>* pbOutput = new GpuBuffer<float>(batch * STRIDE, true);
        DataSet<float>* pInputDataSet = (DataSet<float>*)vDataSet[inputIndex];
        DataSet<float>* pOutputDataSet = (DataSet<float>*)vDataSet[outputIndex];
        GpuBuffer<float>* pbKey = new GpuBuffer<float>(batch * K, true);
        GpuBuffer<unsigned int>* pbUIValue = new GpuBuffer<unsigned int>(batch * K, true);
        GpuBuffer<float>* pbFValue = new GpuBuffer<float>(batch * K, true);
        float* pOutputValue = pbOutput->_pSysData;
        bool bMultiGPU = (getGpu()._numprocs > 1);
        GpuBuffer<float>* pbMultiKey = NULL;
        GpuBuffer<float>* pbMultiFValue = NULL;
        float* pMultiKey = NULL;
        float* pMultiFValue = NULL;
        cudaIpcMemHandle_t keyMemHandle;
        cudaIpcMemHandle_t valMemHandle;

        if (bMultiGPU) {
            if (getGpu()._id == 0) {
                pbMultiKey = new GpuBuffer<float>(getGpu()._numprocs * batch * K, true);
                pbMultiFValue = new GpuBuffer<float>(getGpu()._numprocs * batch * K, true);
                pMultiKey = pbMultiKey->_pDevData;
                pMultiFValue = pbMultiFValue->_pDevData;
                cudaError_t status = cudaIpcGetMemHandle(&keyMemHandle, pMultiKey);
                RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiKey");
                status = cudaIpcGetMemHandle(&valMemHandle, pMultiFValue);
                RTERROR(status, "cudaIpcGetMemHandle: Failed to get IPC mem handle on pMultiFValue");
            }
            MPI_Bcast(&keyMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&valMemHandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

            if (getGpu()._id != 0) {
                cudaError_t status = cudaIpcOpenMemHandle((void**)&pMultiKey, keyMemHandle, cudaIpcMemLazyEnablePeerAccess);
                RTERROR(status, "cudaIpcOpenMemHandle: Unable to open key IPCMemHandle");
                status = cudaIpcOpenMemHandle((void**)&pMultiFValue, valMemHandle, cudaIpcMemLazyEnablePeerAccess);
                RTERROR(status, "cudaIpcOpenMemHandle: Unable to open value IPCMemHandle");
            }
        }

        for (unsigned long long int pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch()) {
            pNetwork->SetPosition(pos);
            pNetwork->PredictBatch();
            unsigned int batch = pNetwork->GetBatch();
            if (pos + batch > pNetwork->GetExamples())
                batch = pNetwork->GetExamples() - pos;
            float* pTarget = pbTarget->_pSysData;
            memset(pTarget, 0, STRIDE * batch * sizeof(float));
            const float* pOutputKey = pNetwork->GetUnitBuffer("Output");
            float* pOut = pOutputValue;
            cudaError_t status = cudaMemcpy(pOut, pOutputKey, batch * STRIDE * sizeof(float), cudaMemcpyDeviceToHost);
            RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");

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
            CalculateOutput(pbOutput->_pDevData, pbTarget->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, STRIDE, K);
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
                    CalculateOutput(pbMultiKey->_pDevData, pbMultiFValue->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, getGpu()._numprocs * K, K);
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
                        idcg += 1.0f / log2(pp + 2.0f);
                    }
                    float dcg = 0.0f;
                    for (int j = 0; j < K; j++) {
                        if (pValue[j] == 1.0f) {
                            tp++;
                            dcg += 1.0f / log2((float)(j + 2));
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

        delete pbKey;
        delete pbFValue;
        delete pbUIValue;
        delete pbTarget;
        delete pbOutput;

        if (bMultiGPU) {
            if (getGpu()._id != 0) {
                cudaError_t status = cudaIpcCloseMemHandle(pMultiKey);
                RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiKey IpcMemHandle");
                status = cudaIpcCloseMemHandle(pMultiFValue);
                RTERROR(status, "cudaIpcCloseMemHandle: Error closing MultiFValue IpcMemHandle");
            }
            delete pbMultiKey;
            delete pbMultiFValue;
        }

        if (getGpu()._id == 0) {
            for (int i = 0; i < K; i++)
                std::cout << std::format("%d,%6.4f,%6.4f,%6.4f\n", i + 1, vPrecision[i] / pNetwork->GetExamples(), vRecall[i] / pNetwork->GetExamples(), vNDCG[i] / pNetwork->GetExamples());
        }
    }

    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    if (getGpu()._id == 0) {
        std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << '\n';
        std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << '\n';
    }

    delete pNetwork;

    for (auto p : vDataSet)
        delete p;

    getGpu().Shutdown();
    return 0;
}