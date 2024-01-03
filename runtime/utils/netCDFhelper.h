#pragma once

#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <netcdf>

bool loadIndex(
    std::unordered_map<std::string, unsigned int>& mLabelToIndex,
    std::istream& inputStream,
    std::ostream& outputStream
);

bool loadIndexFromFile(
    std::unordered_map<std::string, unsigned int>& labelsToIndices,
    const std::string& inputFile,
    std::ostream& outputStream
);

void exportIndex(
    std::unordered_map<std::string, unsigned>& mLabelToIndex,
    std::string indexFileName
);

bool parseSamples(
    std::istream& inputStream,
    const bool enableFeatureIndexUpdates,
    std::unordered_map<std::string, unsigned int>& mFeatureIndex,
    std::unordered_map<std::string, unsigned int>& mSampleIndex,
    bool& featureIndexUpdated,
    bool& sampleIndexUpdated,
    std::map<unsigned int, std::vector<unsigned int>>& mSignals,
    std::map<unsigned int, std::vector<float>>& mSignalValues,
    std::ostream& outputStream
);

bool importSamplesFromPath(
    const std::string& samplesPath,
    const bool enableFeatureIndexUpdates,
    std::unordered_map<std::string, unsigned int>& mFeatureIndex,
    std::unordered_map<std::string, unsigned int>& mSampleIndex,
    bool& featureIndexUpdated,
    bool& sampleIndexUpdated,
    std::vector<unsigned int>& vSparseStart,
    std::vector<unsigned int>& vSparseEnd,
    std::vector<unsigned int>& vSparseIndex,
    std::vector<float>& vSparseData,
    std::ostream& outputStream
);

bool generateNetCDFIndexes(
    const std::string& samplesPath,
    const bool enableFeatureIndexUpdates,
    const std::string& outFeatureIndexFileName,
    const std::string& outSampleIndexFileName,
    std::unordered_map<std::string, unsigned int>& mFeatureIndex,
    std::unordered_map<std::string, unsigned int>& mSampleIndex,
    std::vector<unsigned int>& vSparseStart,
    std::vector<unsigned int>& vSparseEnd,
    std::vector<unsigned int>& vSparseIndex,
    std::vector<float>& vSparseData,
    std::ostream& outputStream
);

void writeNetCDFFile(
    std::vector<unsigned int>& vSparseStart,
    std::vector<unsigned int>& vSparseEnd,
    std::vector<unsigned int>& vSparseIndex,
    std::vector<float>& vSparseValue,
    std::string fileName,
    std::string datasetName,
    unsigned int maxFeatureIndex
);

void writeNetCDFFile(
    std::vector<unsigned int>& vSparseStart,
    std::vector<unsigned int>& vSparseEnd,
    std::vector<unsigned int>& vSparseIndex,
    std::string fileName,
    std::string datasetName,
    unsigned int maxFeatureIndex
);

unsigned int roundUpMaxIndex(unsigned int maxFeatureIndex);

int listFiles(
    const std::string& dirname,
    const bool recursive,
    std::vector<std::string>& files
);

void writeNETCDF(
    const std::string& fileName,
    const std::vector<std::string>& vSamplesName,
    const std::map<std::string, unsigned int>& mInputFeatureNameToIndex,
    std::vector<std::vector<unsigned int>>& vInputSamples,
    const std::vector<std::vector<unsigned int>>& vInputSamplesTime,
    std::vector<std::vector<float>>& vInputSamplesData,
    const std::map<std::string, unsigned int>& mOutputFeatureNameToIndex,
    const std::vector<std::vector<unsigned int>>& vOutputSamples,
    const std::vector<std::vector<unsigned int>>& vOutputSamplesTime,
    const std::vector<std::vector<float>>& vOutputSamplesData,
    int& minInpDate,
    int& maxInpDate,
    int& minOutDate,
    int& maxOutDate,
    const bool alignFeatureDimensionality,
    const int datasetNum
);

void readNetCDFsamplesName(
    const std::string& fname,
    std::vector<std::string>& vSamplesName
);

void readNetCDFindToFeature(
    const std::string& fname,
    const int n,
    std::vector<std::string>& vFeaturesStr
);

unsigned int align(size_t size);

bool addDataToNetCDF(
    netCDF::NcFile& nc,
    const long long dataIndex,
    const std::string& dataName,
    const std::map<std::string, unsigned int>& mFeatureNameToIndex,
    const std::vector<std::vector<unsigned int>>& vInputSamples,
    const std::vector<std::vector<unsigned int>>& vInputSamplesTime,
    const std::vector<std::vector<float>>& vInputSamplesData,
    const bool alignFeatureDimensionality,
    int& minDate,
    int& maxDate,
    const int featureDimensionality = -1
);