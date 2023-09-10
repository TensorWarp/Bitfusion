#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <netcdf>
#include <stdexcept>
#include <unordered_map>
#include <iomanip>
#include "GpuTypes.h"
#include "NetCDFhelper.h"
#include "Types.h"
#include "Utils.h"

using namespace netCDF;
using namespace netCDF::exceptions;

void printUsageTrain() {
    std::cout << "Training: Initiates the training process for a neural network using specified configuration and dataset." << std::endl;
    std::cout << "Usage: train -d <dataset_name> -c <config_file> -n <network_file> -i <input_dataset> -o <output_dataset> [-b <batch_size>] [-e <epochs>]" << std::endl;
    std::cout << "    -c config_file: (required) JSON configuration file containing network training parameters." << std::endl;
    std::cout << "    -i input_dataset: (required) Path to the dataset in NetCDF format used as input for the network." << std::endl;
    std::cout << "    -o output_dataset: (required) Path to the dataset in NetCDF format representing the expected network output." << std::endl;
    std::cout << "    -n network_file: (required) Output file storing the trained neural network in NetCDF format." << std::endl;
    std::cout << "    -b batch_size: (default = 1024) Number of records or input rows to process in each batch." << std::endl;
    std::cout << "    -e epochs: (default = 40) Number of complete passes through the entire dataset during training." << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    float alpha = stof(getOptionalArgValue(argc, argv, "-alpha", "0.025f"));
    float lambda = stof(getOptionalArgValue(argc, argv, "-lambda", "0.0001f"));
    float lambda1 = stof(getOptionalArgValue(argc, argv, "-lambda1", "0.0f"));
    float mu = stof(getOptionalArgValue(argc, argv, "-mu", "0.5f"));
    float mu1 = stof(getOptionalArgValue(argc, argv, "-mu1", "0.0f"));

    if (isArgSet(argc, argv, "-h")) {
        printUsageTrain();
        exit(1);
    }

    string configFileName = getRequiredArgValue(argc, argv, "-c", "config file was not specified.", &printUsageTrain);
    if (!fileExists(configFileName)) {
        std::cout << "Error: Cannot read config file: " << configFileName << std::endl;
        return 1;
    }
    else {
        std::cout << "Train will use configuration file: " << configFileName << std::endl;
    }

    string inputDataFile = getRequiredArgValue(argc, argv, "-i", "input data file is not specified.", &printUsageTrain);
    if (!fileExists(inputDataFile)) {
        std::cout << "Error: Cannot read input feature index file: " << inputDataFile << std::endl;
        return 1;
    }
    else {
        std::cout << "Train will use input data file: " << inputDataFile << std::endl;
    }

    string outputDataFile = getRequiredArgValue(argc, argv, "-o", "output data  file is not specified.", &printUsageTrain);
    if (!fileExists(outputDataFile)) {
        std::cout << "Error: Cannot read output feature index file: " << outputDataFile << std::endl;
        return 1;
    }
    else {
        std::cout << "Train will use output data file: " << outputDataFile << std::endl;
    }

    string networkFileName = getRequiredArgValue(argc, argv, "-n", "the output network file path is not specified.", &printUsageTrain);
    if (fileExists(networkFileName)) {
        std::cout << "Error: Network file already exists: " << networkFileName << std::endl;
        return 1;
    }
    else {
        std::cout << "Train will produce networkFileName: " << networkFileName << std::endl;
    }

    unsigned int batchSize = stoi(getOptionalArgValue(argc, argv, "-b", "1024"));
    std::cout << "Train will use batchSize: " << batchSize << std::endl;

    unsigned int epoch = stoi(getOptionalArgValue(argc, argv, "-e", "40"));
    std::cout << "Train will use number of epochs: " << epoch << endl;
    std::cout << "Train alpha " << alpha << ", lambda " << lambda << ", mu " << mu << ".Please check CDL.txt for meanings" << std::endl;
    std::cout << "Train alpha " << alpha << ", lambda " << lambda << ", lambda1 " << lambda1 << ", mu " << mu << ", mu1 " << mu1 << ".Please check CDL.txt for meanings" << std::endl;

    getGpu().Startup(argc, argv);
    getGpu().SetRandomSeed(FIXED_SEED);

    std::vector<DataSetBase*> vDataSetInput = LoadNetCDF(inputDataFile);
    std::vector<DataSetBase*> vDataSetOutput = LoadNetCDF(outputDataFile);

    vDataSetInput.insert(vDataSetInput.end(), vDataSetOutput.begin(), vDataSetOutput.end());

    Network* pNetwork = LoadNeuralNetworkJSON(configFileName, batchSize, vDataSetInput);

    pNetwork->LoadDataSets(vDataSetInput);
    pNetwork->LoadDataSets(vDataSetOutput);
    pNetwork->SetCheckpoint(networkFileName, 10);

    pNetwork->SetPosition(0);
    pNetwork->PredictBatch();
    pNetwork->SaveNetCDF("initial_network.nc");

    TrainingMode mode = SGD;
    pNetwork->SetTrainingMode(mode);

    auto const start = std::chrono::steady_clock::now();

    for (unsigned int x = 0; x < epoch; ++x) {
        float error = pNetwork->Train(1, alpha, lambda, lambda1, mu, mu1);
    }

    auto const end = std::chrono::steady_clock::now();

    auto elapsed_duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << "Total Training Time: " << elapsed_duration.count() << " seconds" << std::endl;

    int totalGPUMemory;
    int totalCPUMemory;
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << std::endl;
    std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << std::endl;
    pNetwork->SaveNetCDF(networkFileName);
    delete pNetwork;
    getGpu().Shutdown();
    return 0;
}
