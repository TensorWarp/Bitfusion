#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <filesystem>
#include <format>
#include <mpi.h>

#include "NetCDFhelper.h"
#include "Utils.h"
#include "../ThreadPool.h"
#include "NetCDFGenerator.h"

namespace fs = std::filesystem;

const std::string DATASET_TYPE_INDICATOR = "indicator";

void printUsageNetCDFGenerator() {
    std::cout << R"(NetCDFGenerator: Converts a text dataset file into a more compressed NetCDF file.
Usage: generateNetCDF -d <dataset_name> -i <input_text_file> -o <output_netcdf_file> -f <features_index> -s <samples_index> [-c] [-m]
    -d dataset_name: (required) name for the dataset within the NetCDF file.
    -i input_text_file: (required) path to the input text file with records in data format.
    -o output_netcdf_file: (required) path to the output NetCDF file that we generate.
    -f features_index: (required) path to the features index file to read from/write to.
    -s samples_index: (required) path to the samples index file to read from/write to.
    -m : if set, we'll merge the feature index with new features found in the input_text_file. (Cannot be used with -c).
    -c : if set, we'll create a new feature index from scratch. (Cannot be used with -m).
    -t type: (default = 'indicator') the type of dataset to generate. Valid values are: ['indicator', 'analog'].
)" << '\n';
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (isArgSet(argc, argv, "-h")) {
        if (rank == 0) {
            printUsageNetCDFGenerator();
        }
        MPI_Finalize();
        exit(1);
    }

    std::string inputFile;
    std::string outputFile;
    std::string datasetName;
    std::string featureIndexFile;
    std::string sampleIndexFile;
    bool createFeatureIndex = false;
    bool mergeFeatureIndex = false;
    std::string dataType = DATASET_TYPE_INDICATOR;

    if (rank == 0) {
        inputFile = getRequiredArgValue(argc, argv, "-i", "input text file to convert.", &printUsageNetCDFGenerator);
        outputFile = getRequiredArgValue(argc, argv, "-o", "output NetCDF file to generate.", &printUsageNetCDFGenerator);
        datasetName = getRequiredArgValue(argc, argv, "-d", "dataset name for the NetCDF metadata.", &printUsageNetCDFGenerator);
        featureIndexFile = getOptionalArgValue(argc, argv, "-f", "");
        sampleIndexFile = getOptionalArgValue(argc, argv, "-s", "");
        createFeatureIndex = isArgSet(argc, argv, "-c");
        mergeFeatureIndex = isArgSet(argc, argv, "-m");
        dataType = getOptionalArgValue(argc, argv, "-t", DATASET_TYPE_INDICATOR);
    }

    MPI_Bcast(inputFile.data(), static_cast<int>(inputFile.size()) + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(outputFile.data(), static_cast<int>(outputFile.size()) + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(datasetName.data(), static_cast<int>(datasetName.size()) + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(featureIndexFile.data(), static_cast<int>(featureIndexFile.size()) + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(sampleIndexFile.data(), static_cast<int>(sampleIndexFile.size()) + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&createFeatureIndex, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mergeFeatureIndex, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(dataType.data(), static_cast<int>(dataType.size()) + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (createFeatureIndex && mergeFeatureIndex) {
        if (rank == 0) {
            std::cout << "Error: Cannot create (-c) and update existing (-u) feature index. Please select only one.";
            printUsageNetCDFGenerator();
        }
        MPI_Finalize();
        exit(1);
    }

    if (rank == 0) {
        std::cout << "Generating dataset of type: " << dataType << '\n';
    }

    auto const start = std::chrono::steady_clock::now();

    NetCDFGenerator netCDFGen(inputFile, featureIndexFile, sampleIndexFile, outputFile, datasetName, dataType);

    netCDFGen.initializeDataSources();

    // Implement an interactive menu
    bool continueProcessing = true;
    while (continueProcessing) {
        std::cout << "Choose an option:" << '\n';
        std::cout << "1. Start NetCDF generation" << '\n';
        std::cout << "2. Customize settings" << '\n';
        std::cout << "3. Exit" << '\n';

        int choice;
        std::cin >> choice;

        switch (choice) {
        case 1:
            netCDFGen.generateNetCDFDataset();
            break;
        case 2:
            std::cout << "Enter new values for settings:" << '\n';
            break;
        case 3:
            continueProcessing = false;
            break;
        default:
            std::cout << "Invalid choice. Please select a valid option." << '\n';
            break;
        }
    }

    auto const end = std::chrono::steady_clock::now();

    if (rank == 0) {
        std::cout << "Total time for generating NetCDF: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " secs. " << '\n';
    }

    MPI_Finalize();

    return 0;
}
