#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#include <netcdf>
#include "Utils.h"
#include "Filters.h"
#include "../GpuTypes.h"
#include "../Types.h"
#include "RecsGenerator.h"
#include "NetCDFhelper.h"

constexpr unsigned int INTERVAL_REPORT_PROGRESS = 1000000;

void extractMapsToVectors(std::vector<std::string>& vVectors, const std::unordered_map<std::string, unsigned int>& mMaps) {
    for (const auto& entry : mMaps) {
        vVectors[entry.second] = entry.first;
    }
}

void convertTextToNetCDF(const std::string& inputTextFile, const std::string& dataSetName, const std::string& outputNCDFFile,
    std::unordered_map<std::string, unsigned int>& mFeatureIndex,
    std::unordered_map<std::string, unsigned int>& mSignalIndex,
    const std::string& featureIndexFile, const std::string& sampleIndexFile) {
    std::vector<unsigned int> vSparseStart, vSparseEnd, vSparseIndex;
    std::vector<float> vSparseData;

    if (!generateNetCDFIndexes(inputTextFile, false, featureIndexFile, sampleIndexFile,
        mFeatureIndex, mSignalIndex, vSparseStart, vSparseEnd, vSparseIndex, vSparseData, std::cout)) {
        exit(1);
    }

    if (getGpu()._id == 0) {
        writeNetCDFFile(vSparseStart, vSparseEnd, vSparseIndex, outputNCDFFile, dataSetName, mFeatureIndex.size());
    }
}

void printUsagePredict() {
    std::cout << "Prediction: Generates predictions from a trained neural network using a signals/input dataset." << '\n';
    std::cout << "Usage: predict -d <dataset_name> -n <network_file> -r <input_text_file> -i <input_feature_index> -o <output_feature_index> -f <filters_json> [-b <batch_size>] [-k <num_predictions>] [-l layer] [-s input_signals_index] [-p score_precision]" << '\n';
    std::cout << "    -b batch_size: (default = 1024) the number of records/input rows processed in each batch." << '\n';
    std::cout << "    -d dataset_name: (required) name of the dataset within the NetCDF file." << '\n';
    std::cout << "    -f samples filterFileName ." << '\n';
    std::cout << "    -i input_feature_index: (required) path to the feature index file, used to map input signals to the correct input feature vector." << '\n';
    std::cout << "    -k num_predictions: (default = 100) The number of predictions to generate, sorted by score. Ignored if -l flag is used." << '\n';
    std::cout << "    -l layer: (default = Output) the network layer used for predictions. If specified, the raw scores for each node in the layer are output in order." << '\n';
    std::cout << "    -n network_file: (required) the trained neural network in the NetCDF file." << '\n';
    std::cout << "    -o output_feature_index: (required) path to the feature index file, used to map the network's output feature vector to appropriate features." << '\n';
    std::cout << "    -p score_precision: (default = 4.3f) precision of the scores in the output." << '\n';
    std::cout << "    -r input_text_file: (required) path to the file containing input signals for generating predictions (e.g., recommendations)." << '\n';
    std::cout << "    -s output_filename (required) - specify the filename to store the output recommendations." << '\n';
    std::cout << '\n';
}

int main(int argc, char** argv) {
    if (isArgSet(argc, argv, "-h")) {
        printUsagePredict();
        exit(1);
    }

    std::string dataSetName = getRequiredArgValue(argc, argv, "-d", "dataset_name is not specified.", &printUsagePredict);
    dataSetName += INPUT_DATASET_SUFFIX;

    std::string filtersFileName = getRequiredArgValue(argc, argv, "-f", "filters_json is not specified.", &printUsagePredict);
    if (!fileExists(filtersFileName)) {
        std::cout << "Error: Cannot read filter file: " << filtersFileName << '\n';
        return 1;
    }

    std::string inputIndexFileName = getRequiredArgValue(argc, argv, "-i", "input features index file is not specified.", &printUsagePredict);
    if (!fileExists(inputIndexFileName)) {
        std::cout << "Error: Cannot read input feature index file: " << inputIndexFileName << '\n';
        return 1;
    }

    std::string networkFileName = getRequiredArgValue(argc, argv, "-n", "network file is not specified.", &printUsagePredict);
    if (!fileExists(networkFileName)) {
        std::cout << "Error: Cannot read network file: " << networkFileName << '\n';
        return 1;
    }

    std::string outputIndexFileName = getRequiredArgValue(argc, argv, "-o", "output features index file is not specified.", &printUsagePredict);
    if (!fileExists(outputIndexFileName)) {
        std::cout << "Error: Cannot read output feature index file: " << outputIndexFileName << '\n';
        return 1;
    }

    std::string recsFileName = getRequiredArgValue(argc, argv, "-r", "input_text_file is not specified.", &printUsagePredict);
    if (!fileExists(recsFileName)) {
        std::cout << "Error: Cannot read input_text_file: " << recsFileName << '\n';
        return 1;
    }

    std::string recsOutputFileName = getRequiredArgValue(argc, argv, "-s", "filename to put the output recs to.", &printUsagePredict);

    unsigned int batchSize = std::stoi(getOptionalArgValue(argc, argv, "-b", "1024"));
    unsigned int topK = std::stoi(getOptionalArgValue(argc, argv, "-k", "100"));
    if (topK >= 128) {
        std::cout << "Error: Optimized topk Only works for top 128. " << topK << " is greater" << '\n';
        return 1;
    }

    std::string scoreFormat = getOptionalArgValue(argc, argv, "-p", RecsGenerator::DEFAULT_SCORE_PRECISION);

    getGpu().Startup(argc, argv);
    getGpu().SetRandomSeed(FIXED_SEED);

    auto const preProcessingStart = std::chrono::steady_clock::now();

    std::unordered_map<std::string, unsigned int> mInput, mSignals;
    std::cout << "Loading input feature index from: " << inputIndexFileName << '\n';
    if (!loadIndexFromFile(mInput, inputIndexFileName, std::cout)) {
        exit(1);
    }

    std::string inputNetCDFFileName = dataSetName + "_predict" + NETCDF_FILE_EXTENTION;
    std::string featureIndexFile = dataSetName + ".featuresIndex";
    std::string sampleIndexFile = dataSetName + ".samplesIndex";

    convertTextToNetCDF(recsFileName, dataSetName, inputNetCDFFileName, mInput, mSignals, featureIndexFile, sampleIndexFile);

    if (getGpu()._id == 0) {
        std::cout << "Number of network input nodes: " << mInput.size() << '\n';
        std::cout << "Number of entries to generate predictions for: " << mSignals.size() << '\n';
    }

    std::vector<DataSetBase*> vDataSetInput = LoadNetCDF(inputNetCDFFileName);
    Network* pNetwork = LoadNeuralNetworkNetCDF(networkFileName, batchSize);
    pNetwork->LoadDataSets(vDataSetInput);

    std::vector<std::string> vSignals(mSignals.size());
    extractMapsToVectors(vSignals, mSignals);

    std::unordered_map<std::string, unsigned int> mOutput;
    std::cout << "Loading output feature index from: " << outputIndexFileName << '\n';
    if (!loadIndexFromFile(mOutput, outputIndexFileName, std::cout)) {
        exit(1);
    }

    std::vector<std::string> vOutput(mOutput.size());
    extractMapsToVectors(vOutput, mOutput);

    FilterConfig* vFilterSet = loadFilters(filtersFileName, recsOutputFileName, mOutput, mSignals);
    mInput.clear();
    mOutput.clear();
    mSignals.clear();

    auto const preProcessingEnd = std::chrono::steady_clock::now();
    std::cout << "Total time for loading network and data is: " << elapsed_seconds(preProcessingStart, preProcessingEnd) << '\n';

    std::string recsGenLayerLabel = "Output";
    const Layer* pLayer = pNetwork->GetLayer(recsGenLayerLabel);
    unsigned int lx, ly, lz, lw;
    std::tie(lx, ly, lz, lw) = pLayer->GetDimensions();
    unsigned int lBatch = pNetwork->GetBatch();
    unsigned int outputBufferSize = pNetwork->GetBufferSize(recsGenLayerLabel);

    RecsGenerator* nnRecsGenerator = new RecsGenerator(lBatch, topK, outputBufferSize, recsGenLayerLabel, scoreFormat);

    auto const recsGenerationStart = std::chrono::steady_clock::now();

    auto progressReporterStart = std::chrono::steady_clock::now();
    for (unsigned long long int pos = 0; pos < pNetwork->GetExamples(); pos += pNetwork->GetBatch()) {
        std::cout << "Predicting from position " << pos << '\n';

        pNetwork->SetPosition(pos);
        pNetwork->PredictBatch();
        nnRecsGenerator->generateRecs(pNetwork, topK, vFilterSet, vSignals, vOutput);
        if ((pos % INTERVAL_REPORT_PROGRESS) < pNetwork->GetBatch() && (pos / INTERVAL_REPORT_PROGRESS) > 0 && getGpu()._id == 0) {
            auto const progressReporterEnd = std::chrono::steady_clock::now();
            auto const progressReportDuration = elapsed_seconds(progressReporterStart, progressReporterEnd);
            std::cout << "Elapsed time after " << pos << " is " << progressReportDuration << '\n';
            progressReporterStart = std::chrono::steady_clock::now();
        }
    }

    auto const recsGenerationEnd = std::chrono::steady_clock::now();
    auto const recsGenerationDuration = elapsed_seconds(recsGenerationStart, recsGenerationEnd);
    if (getGpu()._id == 0) {
        std::cout << "Total time for Generating recs for " << pNetwork->GetExamples() << " was " << recsGenerationDuration << '\n';
    }

    delete (nnRecsGenerator);
    delete pNetwork;
    getGpu().Shutdown();
    return 0;
}
