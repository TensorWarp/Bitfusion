#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include "netcdf"
#include "utils.h"
#include "filters.h"
#include "../../src/gpuTypes.h"
#include "../../src/types.h"
#include "recommendations.h"
#include "netCDFhelper.h"

constexpr unsigned int INTERVAL_REPORT_PROGRESS = 1000000;

void extractMapsToVectors(std::vector<std::string>& vVectors, const std::unordered_map<std::string, unsigned int>& mMaps) {
    for (const auto& entry : mMaps) {
        vVectors[entry.second] = entry.first;
    }
}

void convertTextToNetCDF(const std::string& inputTextFile, const std::string& dataSetName, const std::string& outputNCDFFile, std::unordered_map<std::string, unsigned int>& mFeatureIndex,
    std::unordered_map<std::string, unsigned int>& mSignalIndex, const std::string& featureIndexFile, const std::string& sampleIndexFile) {
    std::vector<unsigned int> vSparseStart, vSparseEnd, vSparseIndex;
    std::vector<float> vSparseData;

    if (!generateNetCDFIndexes(inputTextFile, false, featureIndexFile, sampleIndexFile, mFeatureIndex, mSignalIndex, vSparseStart, vSparseEnd, vSparseIndex, vSparseData, std::cout)) {
        exit(1);
    }

    if (getGpu()._id == 0) {
        writeNetCDFFile(vSparseStart, vSparseEnd, vSparseIndex, outputNCDFFile, dataSetName, mFeatureIndex.size());
    }
}

void printUsagePredict() {
    std::cout << "Predict: Generates predictions from a trained neural network given a signals/input dataset." << std::endl;
    std::cout << "Usage: predict -d <dataset_name> -n <network_file> -r <input_text_file> -i <input_feature_index> -o <output_feature_index> -f <filters_json> [-b <batch_size>] [-k <num_recs>] [-l layer] [-s input_signals_index] [-p score_precision]" << std::endl;
    std::cout << "    -b batch_size: (default = 1024) the number records/input rows to process in a batch." << std::endl;
    std::cout << "    -d dataset_name: (required) name for the dataset within the netcdf file." << std::endl;
    std::cout << "    -f samples filterFileName ." << std::endl;
    std::cout << "    -i input_feature_index: (required) path to the feature index file, used to tranform input signals to correct input feature vector." << std::endl;
    std::cout << "    -k num_recs: (default = 100) The number of predictions (sorted by score to generate). Ignored if -l flag is used." << std::endl;
    std::cout << "    -l layer: (default = Output) the network layer to use for predictions. If specified, the raw scores for each node in the layer is output in order." << std::endl;
    std::cout << "    -n network_file: (required) the trained neural network in NetCDF file." << std::endl;
    std::cout << "    -o output_feature_index: (required) path to the feature index file, used to tranform the network output feature vector to appropriate features." << std::endl;
    std::cout << "    -p score_precision: (default = 4.3f) precision of the scores in output" << std::endl;
    std::cout << "    -r input_text_file: (required) path to the file with input signal to use to generate predictions (i.e. recommendations)." << std::endl;
    std::cout << "    -s filename (required) . to put the output recs to." << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (isArgSet(argc, argv, "-h")) {
        printUsagePredict();
        return 1;
    }

    std::string dataSetName = getRequiredArgValue(argc, argv, "-d", "dataset_name is not specified.", &printUsagePredict);
    dataSetName += INPUT_DATASET_SUFFIX;

    std::string filtersFileName = getRequiredArgValue(argc, argv, "-f", "filters_json is not specified.", &printUsagePredict);
    if (!fileExists(filtersFileName)) {
        std::cout << "Error: Cannot read filter file: " << filtersFileName << std::endl;
        return 1;
    }

    std::string inputIndexFileName = getRequiredArgValue(argc, argv, "-i", "input features index file is not specified.", &printUsagePredict);
    if (!fileExists(inputIndexFileName)) {
        std::cout << "Error: Cannot read input feature index file: " << inputIndexFileName << std::endl;
        return 1;
    }

    std::string networkFileName = getRequiredArgValue(argc, argv, "-n", "network file is not specified.", &printUsagePredict);
    if (!fileExists(networkFileName)) {
        std::cout << "Error: Cannot read network file: " << networkFileName << std::endl;
        return 1;
    }

    std::string outputIndexFileName = getRequiredArgValue(argc, argv, "-o", "output features index file is not specified.", &printUsagePredict);
    if (!fileExists(outputIndexFileName)) {
        std::cout << "Error: Cannot read output feature index file: " << outputIndexFileName << std::endl;
        return 1;
    }

    std::string recsFileName = getRequiredArgValue(argc, argv, "-r", "input_text_file is not specified.", &printUsagePredict);
    if (!fileExists(recsFileName)) {
        std::cout << "Error: Cannot read input_text_file: " << recsFileName << std::endl;
        return 1;
    }

    std::string recsOutputFileName = getRequiredArgValue(argc, argv, "-s", "filename to put the output recs to.", &printUsagePredict);

    unsigned int batchSize = stoi(getOptionalArgValue(argc, argv, "-b", "1024"));

    unsigned int topK = stoi(getOptionalArgValue(argc, argv, "-k", "100"));
    if (topK >= 128) {
        std::cout << "Error: Optimized topk Only works for top 128. " << topK << " is greater" << std::endl;
        return 1;
    }

    std::string scoreFormat = getOptionalArgValue(argc, argv, "-p", recommendations::DEFAULT_SCORE_PRECISION);

    getGpu().Startup(argc, argv);
    getGpu().SetRandomSeed(FIXED_SEED);

    auto const preProcessingStart = std::chrono::steady_clock::now();

    std::unordered_map<std::string, unsigned int> mInput;
    std::cout << "Loading input feature index from: " << inputIndexFileName << std::endl;
    if (!loadIndexFromFile(mInput, inputIndexFileName, std::cout)) {
        return 1;
    }

    std::unordered_map<std::string, unsigned int> mSignals;
    std::string inputNetCDFFileName;

    std::string dataSetFilesPrefix = dataSetName + "_predict";
    inputNetCDFFileName.assign(dataSetFilesPrefix + NETCDF_FILE_EXTENTION);

    std::string featureIndexFile = dataSetFilesPrefix + ".featuresIndex";
    std::string sampleIndexFile = dataSetFilesPrefix + ".samplesIndex";
    convertTextToNetCDF(recsFileName, dataSetName, inputNetCDFFileName, mInput, mSignals, featureIndexFile, sampleIndexFile);

    if (getGpu()._id == 0) {
        std::cout << "Number of network input nodes: " << mInput.size() << std::endl;
        std::cout << "Number of entries to generate predictions for: " << mSignals.size() << std::endl;
    }

    std::vector<DataSetBase*> vDataSetInput = LoadNetCDF(inputNetCDFFileName);
    Network* pNetwork = LoadNeuralNetworkNetCDF(networkFileName, batchSize);
    pNetwork->LoadDataSets(vDataSetInput);

    std::vector<std::string> vSignals(mSignals.size());
    extractMapsToVectors(vSignals, mSignals);

    std::unordered_map<std::string, unsigned int> mOutput;
    std::cout << "Loading output feature index from: " << outputIndexFileName << std::endl;
    if (!loadIndexFromFile(mOutput, outputIndexFileName, std::cout)) {
        return 1;
    }

    std::vector<std::string> vOutput(mOutput.size());
    extractMapsToVectors(vOutput, mOutput);
    std::unique_ptr<FilterConfig> filterConfigPtr(loadFilters(filtersFileName, recsOutputFileName, mOutput, mSignals));
    FilterConfig* vFilterSet = filterConfigPtr.get();
    mInput.clear();
    mOutput.clear();
    mSignals.clear();

    auto const preProcessingEnd = std::chrono::steady_clock::now();
    std::cout << "Total time for loading network and data is: " << elapsed_seconds(preProcessingStart, preProcessingEnd) << std::endl;

    std::string recsGenLayerLabel = "Output";
    const Layer* pLayer = pNetwork->GetLayer(recsGenLayerLabel);
    unsigned int lx, ly, lz, lw;
    std::tie(lx, ly, lz, lw) = pLayer->GetDimensions();
    unsigned int lBatch = pNetwork->GetBatch();
    unsigned int outputBufferSize = pNetwork->GetBufferSize(recsGenLayerLabel);

    recommendations* recs = new recommendations(lBatch, topK, outputBufferSize, recsGenLayerLabel, scoreFormat);

    auto const recsGenerationStart = std::chrono::steady_clock::now();

    auto progressReporterStart = std::chrono::steady_clock::now();
    for (unsigned long long int pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch()) {
        std::cout << "Predicting from position " << pos << std::endl;

        pNetwork->SetPosition(pos);
        pNetwork->PredictBatch();
        recs->generateRecs(pNetwork, topK, vFilterSet, vSignals, vOutput);
        if ((pos % INTERVAL_REPORT_PROGRESS) < pNetwork->GetBatch() && (pos / INTERVAL_REPORT_PROGRESS) > 0 && getGpu()._id == 0) {
            auto const progressReporterEnd = std::chrono::steady_clock::now();
            auto const progressReportDuration = elapsed_seconds(progressReporterStart, progressReporterEnd);
            std::cout << "Elapsed time after " << pos << " is " << progressReportDuration << std::endl;
            progressReporterStart = std::chrono::steady_clock::now();
        }
    }
    auto const recsGenerationEnd = std::chrono::steady_clock::now();
    auto const recsGenerationDuration = elapsed_seconds(recsGenerationStart, recsGenerationEnd);
    if (getGpu()._id == 0) {
        std::cout << "Total time for Generating recs for " << pNetwork->GetExamples() << " was " << recsGenerationDuration << std::endl;
    }

    delete recs;
    delete pNetwork;
    getGpu().Shutdown();
    return 0;
}
