#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <memory>
#include "Train.h"

#include "../../src/gpuTypes.h"
#include "netCDFhelper.h"
#include "../../src/types.h"
#include "Utils.h"

namespace fs = std::filesystem;

void printUsageTrain() {
    std::cout << std::format("Training Program: Train a neural network with a configuration and dataset.\n"
        "Usage: train -d <dataset_name> -c <config_file> -n <network_file> -i <input_netcdf> -o <output_netcdf> [-b <batch_size>] [-e <num_epochs>]\n"
        "    -c config_file: (required) JSON configuration file with network training parameters.\n"
        "    -i input_netcdf: (required) path to the NetCDF file containing the input dataset for the network.\n"
        "    -o output_netcdf: (required) path to the NetCDF file where the expected output dataset will be saved.\n"
        "    -n network_file: (required) the output trained neural network stored in a NetCDF file.\n"
        "    -b batch_size: (default = 1024) the number of records or input rows to process in each batch.\n"
        "    -e num_epochs: (default = 40) the number of complete passes through the entire dataset.\n\n");
}

bool parseCommandLineArgs(int argc, char** argv, TrainingConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-alpha") {
            config.alpha = std::stof(argv[++i]);
        }
        else if (arg == "-lambda") {
            config.lambda = std::stof(argv[++i]);
        }
        else if (arg == "-lambda1") {
            config.lambda1 = std::stof(argv[++i]);
        }
        else if (arg == "-mu") {
            config.mu = std::stof(argv[++i]);
        }
        else if (arg == "-mu1") {
            config.mu1 = std::stof(argv[++i]);
        }
        else if (arg == "-c") {
            config.configFileName = argv[++i];
        }
        else if (arg == "-i") {
            config.inputDataFile = argv[++i];
        }
        else if (arg == "-o") {
            config.outputDataFile = argv[++i];
        }
        else if (arg == "-n") {
            config.networkFileName = argv[++i];
        }
        else if (arg == "-b") {
            config.batchSize = std::stoi(argv[++i]);
        }
        else if (arg == "-e") {
            config.epoch = std::stoi(argv[++i]);
        }
        else if (arg == "-h") {
            printUsageTrain();
            return false;
        }
    }
    return true;
}

bool fileExists(const std::string& filePath) {
    return fs::exists(filePath);
}

int main(int argc, char** argv) {
    TrainingConfig config;

    const float alpha = std::stof(getOptionalArgValue(argc, argv, "-alpha", "0.025f"));
    const float lambda = std::stof(getOptionalArgValue(argc, argv, "-lambda", "0.0001f"));
    const float lambda1 = std::stof(getOptionalArgValue(argc, argv, "-lambda1", "0.0f"));
    const float mu = std::stof(getOptionalArgValue(argc, argv, "-mu", "0.5f"));
    const float mu1 = std::stof(getOptionalArgValue(argc, argv, "-mu1", "0.0f"));

    if (isArgSet(argc, argv, "-h")) {
        printUsageTrain();
        return 1;
    }

    if (!parseCommandLineArgs(argc, argv, config)) {
        return 1;
    }

    std::cout << ("Starting training...");

    if (!fileExists(config.configFileName)) {
        std::cerr << ("Error: Cannot read config file: {}", config.configFileName);
        return 1;
    }

    if (!fileExists(config.inputDataFile)) {
        std::cerr << ("Error: Cannot read input feature index file: {}", config.inputDataFile);
        return 1;
    }

    if (!fileExists(config.outputDataFile)) {
        std::cerr << ("Error: Cannot read output feature index file: {}", config.outputDataFile);
        return 1;
    }

    if (fs::exists(config.networkFileName)) {
        std::cerr << ("Error: Network file already exists: {}", config.networkFileName);
        return 1;
    }

    getGpu().Startup(argc, argv);
    getGpu().SetRandomSeed(FIXED_SEED);

    std::vector<DataSetBase*> vDataSetInput = LoadNetCDF(config.inputDataFile);
    std::vector<DataSetBase*> vDataSetOutput = LoadNetCDF(config.outputDataFile);

    vDataSetInput.insert(vDataSetInput.end(), vDataSetOutput.begin(), vDataSetOutput.end());

    Network* pNetwork = LoadNeuralNetworkJSON(config.configFileName, config.batchSize, vDataSetInput);

    pNetwork->LoadDataSets(vDataSetInput);
    pNetwork->LoadDataSets(vDataSetOutput);
    pNetwork->SetCheckpoint(config.networkFileName, 10);

    pNetwork->SetPosition(0);
    pNetwork->PredictBatch();
    pNetwork->SaveNetCDF("initial_network.nc");

    TrainingMode mode = SGD;
    pNetwork->SetTrainingMode(mode);

    const auto start = std::chrono::steady_clock::now();
    for (unsigned int x = 0; x < config.epoch; ++x) {
        const float error = pNetwork->Train(1, config.alpha, config.lambda, config.lambda1, config.mu, config.mu1);
    }
    const auto end = std::chrono::steady_clock::now();
    std::cout << ("Total Training Time {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    pNetwork->SaveNetCDF(config.networkFileName);
    delete pNetwork;
    getGpu().Shutdown();

    return 0;
}
