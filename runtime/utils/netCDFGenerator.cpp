#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <filesystem>

#include "netCDFhelper.h"
#include "Utils.h"

const std::string DATASET_TYPE_INDICATOR = "indicator";
const std::string DATASET_TYPE_ANALOG = "analog";

void printUsageNetCDFGenerator() {
    std::cout << std::format("CustomNetCDFGenerator: Transforms a text dataset file into a more compressed NetCDF file.\n");
    std::cout << std::format("Usage: generateCustomNetCDF -d <dataset_name> -i <input_text_file> -o <output_netcdf_file> -f <features_index> -s <samples_index> [-c] [-m]\n");
    std::cout << std::format("    -d dataset_name: (required) specify the dataset name within the NetCDF file.\n");
    std::cout << std::format("    -i input_text_file: (required) provide the path to the input text file containing records in data format.\n");
    std::cout << std::format("    -o output_netcdf_file: (required) specify the path to the output NetCDF file to be generated.\n");
    std::cout << std::format("    -f features_index: (required) provide the path to the features index file for reading/writing.\n");
    std::cout << std::format("    -s samples_index: (required) provide the path to the samples index file for reading/writing.\n");
    std::cout << std::format("    -m : if set, merge the feature index with new features found in the input_text_file. (Cannot be used with -c).\n");
    std::cout << std::format("    -c : if set, create a new feature index from scratch. (Cannot be used with -m).\n");
    std::cout << std::format("    -t type: (default = 'indicator') specify the type of dataset to generate. Valid values are: ['indicator', 'analog'].\n");
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (isArgSet(argc, argv, "-h")) {
        printUsageNetCDFGenerator();
        return 1;
    }
    std::string inputFile = getRequiredArgValue(argc, argv, "-i", "input text file to convert.", &printUsageNetCDFGenerator);
    std::string outputFile = getRequiredArgValue(argc, argv, "-o", "output netcdf file to generate.", &printUsageNetCDFGenerator);
    std::string datasetName = getRequiredArgValue(argc, argv, "-d", "dataset name for the netcdf metadata.", &printUsageNetCDFGenerator);
    std::string featureIndexFile = getRequiredArgValue(argc, argv, "-f", "feature index file.", &printUsageNetCDFGenerator);
    std::string sampleIndexFile = getRequiredArgValue(argc, argv, "-s", "samples index file.", &printUsageNetCDFGenerator);

    bool createFeatureIndex = isArgSet(argc, argv, "-c");
    if (createFeatureIndex) {
        std::cout << "Flag -c is set. Will create a new feature file and overwrite: " << featureIndexFile << std::endl;
    }

    bool mergeFeatureIndex = isArgSet(argc, argv, "-m");
    if (mergeFeatureIndex) {
        std::cout << "Flag -m is set. Will merge with existing feature file and overwrite: " << featureIndexFile << std::endl;
    }

    if (createFeatureIndex && mergeFeatureIndex) {
        std::cout << "Error: Cannot create (-c) and update existing (-u) feature index. Please select only one.";
        printUsageNetCDFGenerator();
        return 1;
    }
    bool updateFeatureIndex = createFeatureIndex || mergeFeatureIndex;

    std::string dataType = getOptionalArgValue(argc, argv, "-t", "indicator");
    if (dataType != DATASET_TYPE_INDICATOR && dataType != DATASET_TYPE_ANALOG) {
        std::cout << "Error: Unknown dataset type [" << dataType << "].";
        std::cout << " Please select one of {" << DATASET_TYPE_INDICATOR << "," << DATASET_TYPE_ANALOG << "}" << std::endl;
        return 1;
    }
    std::cout << "Generating dataset of type: " << dataType << std::endl;

    std::unordered_map<std::string, unsigned int> mFeatureIndex;
    std::unordered_map<std::string, unsigned int> mSampleIndex;

    auto const start = std::chrono::steady_clock::now();

    if (!std::filesystem::exists(sampleIndexFile)) {
        std::cout << "Will create a new samples index file: " << sampleIndexFile << std::endl;
    }
    else {
        std::cout << "Loading sample index from: " << sampleIndexFile << std::endl;
        if (!loadIndexFromFile(mSampleIndex, sampleIndexFile, std::cout)) {
            return 1;
        }
    }

    if (createFeatureIndex) {
        std::cout << "Will create a new features index file: " << featureIndexFile << std::endl;
    }
    else if (!std::filesystem::exists(featureIndexFile)) {
        std::cout << "Error: Cannnot find a valid feature index file: " << featureIndexFile << std::endl;
        return 1;
    }
    else {
        std::cout << "Loading feature index from: " << featureIndexFile << std::endl;
        if (!loadIndexFromFile(mFeatureIndex, featureIndexFile, std::cout)) {
            return 1;
        }
    }

    std::vector<unsigned int> vSparseStart;
    std::vector<unsigned int> vSparseEnd;
    std::vector<unsigned int> vSparseIndex;
    std::vector<float> vSparseData;

    if (!generateNetCDFIndexes(inputFile,
        updateFeatureIndex,
        featureIndexFile,
        sampleIndexFile,
        mFeatureIndex,
        mSampleIndex,
        vSparseStart,
        vSparseEnd,
        vSparseIndex,
        vSparseData,
        std::cout)) {
        return 1;
    }

    if (dataType == DATASET_TYPE_ANALOG) {
        writeNetCDFFile(vSparseStart,
            vSparseEnd,
            vSparseIndex,
            vSparseData,
            outputFile,
            datasetName,
            mFeatureIndex.size());
    }
    else {
        writeNetCDFFile(vSparseStart, vSparseEnd, vSparseIndex, outputFile, datasetName, mFeatureIndex.size());
    }

    auto const end = std::chrono::steady_clock::now();
    std::cout << "Total time for generating NetCDF: " << elapsed_seconds(start, end) << " secs. " << std::endl;

    return 0;
}
