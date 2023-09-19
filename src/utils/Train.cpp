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
#include <omp.h>
#include <mpi.h>
#include <format>
#include <cuda.h>

#include "Utils.h"
#include "NetCDFhelper.h"

#include "../GpuTypes.h"
#include "../Types.h"
#include "../ThreadPool.h"

/// <summary>
/// Prints the usage information for the train command.
/// </summary>
void printUsageTrain() {
    std::cout << std::format(
        "Training: Initiates the training process for a neural network using specified configuration and dataset.\n"
        "Usage: train -d <dataset_name> -c <config_file> -n <network_file> -i <input_dataset> -o <output_dataset> [-b <batch_size>] [-e <epochs>]\n"
        "    -c config_file: (required) JSON configuration file containing network training parameters.\n"
        "    -i input_dataset: (required) Path to the dataset in NetCDF format used as input for the network.\n"
        "    -o output_dataset: (required) Path to the dataset in NetCDF format representing the expected network output.\n"
        "    -n network_file: (required) Output file storing the trained neural network in NetCDF format.\n"
        "    -b batch_size: (default = 1024) Number of records or input rows to process in each batch.\n"
        "    -e epochs: (default = 40) Number of complete passes through the entire dataset during training.\n\n"
    );
}

/// <summary>
/// Represents a configuration for the training process.
/// </summary>
struct Configuration {
    float alpha;
    float lambda;
    float lambda1;
    float mu;
    float mu1;
    std::string configFileName;
    std::string inputDataFile;
    std::string outputDataFile;
    std::string networkFileName;
    unsigned int batchSize;
    unsigned int epoch;
};

/// <summary>
/// Parses command line arguments and returns a Configuration object.
/// </summary>
/// <param name="argc">The number of command line arguments.</param>
/// <param name="argv">An array of command line argument strings.</param>
/// <returns>A Configuration object with parsed values.</returns>
Configuration parseArgs(int argc, char** argv) {
    Configuration config;
    config.alpha = stof(getOptionalArgValue(argc, argv, "-alpha", "0.025f"));
    config.lambda = stof(getOptionalArgValue(argc, argv, "-lambda", "0.0001f"));
    config.lambda1 = stof(getOptionalArgValue(argc, argv, "-lambda1", "0.0f"));
    config.mu = stof(getOptionalArgValue(argc, argv, "-mu", "0.5f"));
    config.mu1 = stof(getOptionalArgValue(argc, argv, "-mu1", "0.0f"));
    config.configFileName = getRequiredArgValue(argc, argv, "-c", "config file was not specified.", &printUsageTrain);
    config.inputDataFile = getRequiredArgValue(argc, argv, "-i", "input data file is not specified.", &printUsageTrain);
    config.outputDataFile = getRequiredArgValue(argc, argv, "-o", "output data  file is not specified.", &printUsageTrain);
    config.networkFileName = getRequiredArgValue(argc, argv, "-n", "the output network file path is not specified.", &printUsageTrain);
    config.batchSize = stoi(getOptionalArgValue(argc, argv, "-b", "1024"));
    config.epoch = stoi(getOptionalArgValue(argc, argv, "-e", "40"));
    return config;
}

/// <summary>
/// Validates the configuration values.
/// </summary>
/// <param name="config">The Configuration object to validate.</param>
/// <returns>True if the configuration is valid, false otherwise.</returns>
bool validateConfig(const Configuration& config) {
    if (!fileExists(config.configFileName)) {
        std::cout << "Error: Cannot read config file: " << config.configFileName << '\n';
        return false;
    }
    if (!fileExists(config.inputDataFile)) {
        std::cout << "Error: Cannot read input feature index file: " << config.inputDataFile << '\n';
        return false;
    }
    if (!fileExists(config.outputDataFile)) {
        std::cout << "Error: Cannot read output feature index file: " << config.outputDataFile << '\n';
        return false;
    }
    if (fileExists(config.networkFileName)) {
        std::cout << "Error: Network file already exists: " << config.networkFileName << '\n';
        return false;
    }
    return true;
}

/// <summary>
/// Prints the details of the Configuration object.
/// </summary>
/// <param name="config">The Configuration object to print.</param>
void printConfigDetails(const Configuration& config) {
    std::cout << std::format("Train will use configuration file: {}\n", config.configFileName);
    std::cout << std::format("Train will use input data file: {}\n", config.inputDataFile);
    std::cout << std::format("Train will use output data file: {}\n", config.outputDataFile);
    std::cout << std::format("Train will produce networkFileName: {}\n", config.networkFileName);
    std::cout << std::format("Train will use batchSize: {}\n", config.batchSize);
    std::cout << std::format("Train will use number of epochs: {}\n", config.epoch);
    std::cout << std::format("Train alpha {}, lambda {}, mu {}.\n",
        config.alpha, config.lambda, config.mu);
    std::cout << std::format("Train alpha {}, lambda {}, lambda1 {}, mu {}, mu1 {}.\n",
        config.alpha, config.lambda, config.lambda1, config.mu, config.mu1);
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    // Get the rank and size of the MPI communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize CUDA and get the number of GPUs
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    // Check if the "-h" argument is set, and display usage if it is
    if (isArgSet(argc, argv, "-h")) {
        if (rank == 0) {
            printUsageTrain();
        }
        // Finalize MPI, CUDA, and exit the program with an error code
        MPI_Finalize();
        cudaDeviceReset();
        std::exit(1);
    }

    // Parse command-line arguments into a Configuration object
    Configuration config = parseArgs(argc, argv);

    // Validate the parsed configuration
    if (!validateConfig(config)) {
        // Finalize MPI, CUDA, and return with an error code
        MPI_Finalize();
        cudaDeviceReset();
        return 1;
    }

    // Print configuration details (only by rank 0)
    if (rank == 0) {
        printConfigDetails(config);
    }

    // Initialize CUDA on each GPU
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        getGpu().Startup(argc, argv);
        getGpu().SetRandomSeed(FIXED_SEED);
    }

    // Load input and output datasets from NetCDF files (assuming each GPU loads its own data)
    std::vector<DataSetBase*> vDataSetInput = LoadNetCDF(config.inputDataFile);
    std::vector<DataSetBase*> vDataSetOutput = LoadNetCDF(config.outputDataFile);

    // Combine input and output datasets (only by rank 0)
    if (rank == 0) {
        vDataSetInput.insert(vDataSetInput.end(), vDataSetOutput.begin(), vDataSetOutput.end());
    }

    // Load a neural network from a JSON configuration file (only by rank 0)
    Network* pNetwork = nullptr;
    if (rank == 0) {
        pNetwork = LoadNeuralNetworkJSON(config.configFileName, config.batchSize, vDataSetInput);
        pNetwork->LoadDataSets(vDataSetInput);
        pNetwork->LoadDataSets(vDataSetOutput);
        pNetwork->SetCheckpoint(config.networkFileName, 10);
    }

    // Set network position, predict, and save initial network state (only by rank 0)
    if (rank == 0) {
        pNetwork->SetPosition(0);
        pNetwork->PredictBatch();
        pNetwork->SaveNetCDF("initial_network.nc");
    }

    // Set the training mode to Stochastic Gradient Descent (SGD) (only by rank 0)
    const TrainingMode mode = SGD;
    if (rank == 0) {
        pNetwork->SetTrainingMode(mode);
    }

    // Start measuring time
    const auto start = std::chrono::steady_clock::now();

    // Calculate local epochs for this process
    int localEpochs = config.epoch / size;
    if (rank < config.epoch % size) {
        localEpochs++;
    }

    float totalError = 0.0f;

    // Parallelize training loop using OpenMP
#pragma omp parallel for reduction(+:totalError) num_threads(config.batchSize)
    for (int x = 0; x < localEpochs; ++x) {
        ThreadPool threadPool(std::thread::hardware_concurrency());
        std::vector<std::future<float>> futures;
        for (int i = 0; i < config.batchSize; ++i) {
            futures.push_back(threadPool.enqueue([&pNetwork, &config]() {
                return pNetwork->Train(1, config.alpha, config.lambda, config.lambda1, config.mu, config.mu1);
                }));
        }

        for (auto& future : futures) {
            totalError += future.get();
        }
    }

    float globalTotalError = 0.0f;
    // Reduce totalError across all MPI processes (only by rank 0)
    MPI_Allreduce(&totalError, &globalTotalError, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Stop measuring time
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    int totalGPUMemory;
    int totalCPUMemory;
    // Get GPU and CPU memory usage (only by rank 0)
    if (rank == 0) {
        getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    }

    // Display results (only by rank 0)
    if (rank == 0) {
        std::cout << std::format("Total Training Time: {} seconds\n", elapsed_duration.count());
        std::cout << std::format("Average Error: {}\n", globalTotalError / (config.epoch * size * config.batchSize * 1.0f));
        std::cout << std::format("GPU Memory Usage: {} KB\n", totalGPUMemory);
        std::cout << std::format("CPU Memory Usage: {} KB\n", totalCPUMemory);
        // Save the final network state and clean up
        pNetwork->SaveNetCDF(config.networkFileName);
        delete pNetwork;
    }

    // Finalize MPI and return success code
    MPI_Finalize();
    cudaDeviceReset();
    return 0;
}
