
#include <cstdio>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <netcdf>
#include <unordered_map>
#include <stdexcept>

#include "../Enum.h"
#include "Utils.h"
#include "NetCDFhelper.h"

using namespace netCDF;
using namespace netCDF::exceptions;

int gLoggingRate = 10000;

bool loadIndex(std::unordered_map<std::string, unsigned int> &labelsToIndices, std::istream &inputStream,
               std::ostream &outputStream) {
    std::string line;
    unsigned int linesProcessed = 0;
    const size_t initialIndexSize = labelsToIndices.size();
    while (getline(inputStream, line)) {
        std::vector<std::string> vData = split(line, '\t');
        linesProcessed++;
        if (vData.size() == 2 && !vData[0].empty()) {
            labelsToIndices[vData[0]] = atoi(vData[1].c_str());
        } else {
            outputStream << "Error: line " << linesProcessed << " contains invalid data" << std::endl;
            return false;
        }
    }

    const size_t numEntriesAdded = labelsToIndices.size() - initialIndexSize;
    outputStream << "Number of lines processed: " << linesProcessed << std::endl;
    outputStream << "Number of entries added to index: " << numEntriesAdded << std::endl;
    if (linesProcessed != numEntriesAdded) {
        outputStream << "Error: Number of entries added to index not equal to number of lines processed" << std::endl;
        return false;
    }

    if (inputStream.bad()) {
        outputStream << "Error: " << strerror(errno) << std::endl;
        return false;
    }

    return true;
}

bool loadIndexFromFile(std::unordered_map<std::string, unsigned int> &labelsToIndices, const std::string &inputFile,
                       std::ostream &outputStream) {
    std::ifstream inputStream(inputFile);
    if (!inputStream.is_open()) {
        outputStream << "Error: Failed to open index file" << std::endl;
        return false;
    }

    return loadIndex(labelsToIndices, inputStream, outputStream);
}

void exportIndex(std::unordered_map<std::string, unsigned int> &mLabelToIndex, std::string indexFileName) {
    std::ofstream outputIndexStream(indexFileName);
    std::unordered_map<std::string, unsigned int>::iterator indexIterator;
    for (indexIterator = mLabelToIndex.begin(); indexIterator != mLabelToIndex.end(); indexIterator++) {
        outputIndexStream << indexIterator->first << "\t" << indexIterator->second << std::endl;
    }
    outputIndexStream.close();
}

bool parseSamples(std::istream &inputStream,
                  const bool enableFeatureIndexUpdates,
                  std::unordered_map<std::string, unsigned int> &mFeatureIndex,
                  std::unordered_map<std::string, unsigned int> &mSampleIndex,
                  bool &featureIndexUpdated,
                  bool &sampleIndexUpdated,
                  std::map<unsigned int, std::vector<unsigned int>> &mSignals,
                  std::map<unsigned int, std::vector<float>> &mSignalValues,
                  std::ostream &outputStream) {
    auto const start = std::chrono::steady_clock::now();
    auto reported = start;
    std::string line;
    int lineNumber = 0;

    while (getline(inputStream, line)) {
        lineNumber++;
        if (line.empty()) {
            continue;
        }

        int index = line.find('\t');
        if (index < 0) {
            outputStream << "Warning: Skipping over malformed line (" << line << ") at line " << lineNumber << std::endl;
            continue;
        }

        std::string sampleLabel = line.substr(0, index);
        std::string dataString = line.substr(index + 1);

        unsigned int sampleIndex = 0;
        try {
            sampleIndex = mSampleIndex.at(sampleLabel);
        }
        catch (const std::out_of_range &oor) {
            unsigned int index = mSampleIndex.size();
            mSampleIndex[sampleLabel] = index;
            sampleIndex = mSampleIndex[sampleLabel];
            sampleIndexUpdated = true;
        }
        std::vector<unsigned int> signals;
        std::vector<float> signalValue;

        std::vector<std::string> dataPointTuples = split(dataString, ':');
        for (unsigned int i = 0; i < dataPointTuples.size(); i++) {
            std::string dataPoint = dataPointTuples[i];
            std::vector<std::string> dataElems = split(dataPoint, ',');

            if (dataElems.empty() || dataElems[0].length() == 0) {
                continue;
            }

            const size_t numDataElems = dataElems.size();
            if (numDataElems > 2) {
                outputStream << "Warning: Data point [" << dataPoint << "] at line " << lineNumber << " has more "
                             << "than 1 value for feature (actual value: " << numDataElems << "). "
                             << "Keeping the first value and ignoring subsequent values." << std::endl;
            }

            std::string featureName = dataElems[0];
            float featureValue = 0.0;
            if (numDataElems > 1) {
                featureValue = stof(dataElems[1]);
            }

            unsigned int featureIndex = 0;
            try {
                featureIndex = mFeatureIndex.at(featureName);
            }
            catch (const std::out_of_range &oor) {
                if (enableFeatureIndexUpdates) {
                    unsigned int index = mFeatureIndex.size();
                    mFeatureIndex[featureName] = index;
                    featureIndex = index;
                    featureIndexUpdated = true;
                } else {
                    continue;
                }
            }
            signals.push_back(featureIndex);
            signalValue.push_back(featureValue);
        }

        mSignals[sampleIndex] = signals;
        mSignalValues[sampleIndex] = signalValue;
        if (mSampleIndex.size() % gLoggingRate == 0) {
            auto const now = std::chrono::steady_clock::now();
            outputStream << "Progress Parsing (Sample " << mSampleIndex.size() << ", ";
            outputStream << "Time " << elapsed_seconds(reported, now) << ", ";
            outputStream << "Total " << elapsed_seconds(start, now) << ")" << std::endl;
            reported = now;
        }
    }

    if (inputStream.bad()) {
        outputStream << "Error: " << strerror(errno) << std::endl;
        return false;
    }

    return true;
}

bool importSamplesFromPath(const std::string &samplesPath,
                           const bool enableFeatureIndexUpdates,
                           std::unordered_map<std::string, unsigned int> &mFeatureIndex,
                           std::unordered_map<std::string, unsigned int> &mSampleIndex,
                           bool &featureIndexUpdated,
                           bool &sampleIndexUpdated,
                           std::vector<unsigned int> &vSparseStart,
                           std::vector<unsigned int> &vSparseEnd,
                           std::vector<unsigned int> &vSparseIndex,
                           std::vector<float> &vSparseData,
                           std::ostream &outputStream) {

    featureIndexUpdated = false;
    sampleIndexUpdated = false;

    if (!fileExists(samplesPath)) {
        outputStream << "Error: " << samplesPath << " not found." << std::endl;
        return false;
    }

    std::vector<std::string> files;

    std::map<unsigned int, std::vector<unsigned int>> mSignals;
    std::map<unsigned int, std::vector<float>> mSignalValues;

    if (listFiles(samplesPath, false, files) == 0) {
        outputStream << "Indexing " << files.size() << " files" << std::endl;

        for (auto const &file: files) {
            outputStream << "\tIndexing file: " << file << std::endl;

            std::ifstream inputStream(file);
            if (!inputStream.is_open()) {
                outputStream << "Error: Failed to open index file" << std::endl;
                return false;
            }

            if (!parseSamples(inputStream,
                              enableFeatureIndexUpdates,
                              mFeatureIndex,
                              mSampleIndex,
                              featureIndexUpdated,
                              sampleIndexUpdated,
                              mSignals,
                              mSignalValues,
                              outputStream)) {
                return false;
            }
        }
    }

    std::map<unsigned int, std::vector<unsigned int>>::iterator mSignalsIter;
    std::map<unsigned int, std::vector<float>>::iterator mSignalValuesIter;
    for (mSignalsIter = mSignals.begin(); mSignalsIter != mSignals.end(); mSignalsIter++) {
        vSparseStart.push_back(vSparseIndex.size());
        std::vector<unsigned int> &signals = mSignalsIter->second;

        mSignalValuesIter = mSignalValues.find(mSignalsIter->first);
        std::vector<float> &signalValues = mSignalValuesIter->second;

        for (unsigned int i = 0; i < signals.size(); ++i) {
            vSparseIndex.push_back(signals[i]);
            vSparseData.push_back(signalValues[i]);
        }
        vSparseEnd.push_back(vSparseIndex.size());
    }

    return true;
}

bool generateNetCDFIndexes(const std::string &samplesPath,
                           const bool enableFeatureIndexUpdates,
                           const std::string &outFeatureIndexFileName,
                           const std::string &outSampleIndexFileName,
                           std::unordered_map<std::string, unsigned int> &mFeatureIndex,
                           std::unordered_map<std::string, unsigned int> &mSampleIndex,
                           std::vector<unsigned int> &vSparseStart,
                           std::vector<unsigned int> &vSparseEnd,
                           std::vector<unsigned int> &vSparseIndex,
                           std::vector<float> &vSparseData,
                           std::ostream &outputStream) {

    bool featureIndexUpdated;
    bool sampleIndexUpdated;

    if (!importSamplesFromPath(samplesPath,
              enableFeatureIndexUpdates,
              mFeatureIndex,
              mSampleIndex,
              featureIndexUpdated,
              sampleIndexUpdated,
              vSparseStart,
              vSparseEnd,
              vSparseIndex,
              vSparseData,
              std::cout)) {

        return false;
    }

    if (featureIndexUpdated) {
        exportIndex(mFeatureIndex, outFeatureIndexFileName);
        std::cout << "Exported " << outFeatureIndexFileName << " with " << mFeatureIndex.size() << " entries." << std::endl;
    }

    if (sampleIndexUpdated) {
        exportIndex(mSampleIndex, outSampleIndexFileName);
        std::cout << "Exported " << outSampleIndexFileName << " with " << mSampleIndex.size() << " entries." << std::endl;
    }

    return true;
}

unsigned int roundUpMaxIndex(unsigned int maxFeatureIndex) {
    return ((maxFeatureIndex + 127) >> 7) << 7;
}

void writeNetCDFFile(std::vector<unsigned int> &vSparseStart,
                     std::vector<unsigned int> &vSparseEnd,
                     std::vector<unsigned int> &vSparseIndex,
                     std::vector<float> &vSparseData,
                     std::string fileName,
                     std::string datasetName,
                     unsigned int maxFeatureIndex) {

    std::cout << "Raw max index is: " << maxFeatureIndex << std::endl;
    maxFeatureIndex = roundUpMaxIndex(maxFeatureIndex);
    std::cout << "Rounded up max index to: " << maxFeatureIndex << std::endl;

    try {
        netCDF::NcFile nc(fileName, netCDF::NcFile::replace);
        if (nc.isNull()) {
            std::cout << "Error creating output file:" << fileName << std::endl;
            throw std::runtime_error("Error creating NetCDF file.");
        }
        nc.putAtt("datasets", netCDF::ncUint, 1);
        nc.putAtt("name0", datasetName);
        nc.putAtt("attributes0", netCDF::ncUint, DataSetEnums::Sparse);
        nc.putAtt("kind0", netCDF::ncUint, DataSetEnums::Numeric);
        nc.putAtt("dataType0", netCDF::ncUint, DataSetEnums::Float);
        nc.putAtt("dimensions0", netCDF::ncUint, 1);
        nc.putAtt("width0", netCDF::ncUint, maxFeatureIndex);
        netCDF::NcDim examplesDim = nc.addDim("examplesDim0", vSparseStart.size());
        netCDF::NcDim sparseDataDim = nc.addDim("sparseDataDim0", vSparseIndex.size());
        netCDF::NcVar sparseStartVar = nc.addVar("sparseStart0", "uint", "examplesDim0");
        netCDF::NcVar sparseEndVar = nc.addVar("sparseEnd0", "uint", "examplesDim0");
        netCDF::NcVar sparseIndexVar = nc.addVar("sparseIndex0", "uint", "sparseDataDim0");
        netCDF::NcVar sparseDataVar = nc.addVar("sparseData0", ncFloat, sparseDataDim);
        sparseStartVar.putVar(&vSparseStart[0]);
        sparseEndVar.putVar(&vSparseEnd[0]);
        sparseIndexVar.putVar(&vSparseIndex[0]);
        sparseDataVar.putVar(&vSparseData[0]);

        std::cout << "Created NetCDF file " << fileName << " " << "for dataset " << datasetName << std::endl;
    } catch (std::exception &e) {
        std::cout << "Caught exception: " << e.what() << "\n";
        throw std::runtime_error("Error writing to NetCDF file.");
    }
}

void writeNetCDFFile(std::vector<unsigned int> &vSparseStart,
                     std::vector<unsigned int> &vSparseEnd,
                     std::vector<unsigned int> &vSparseIndex,
                     std::string fileName,
                     std::string datasetName,
                     unsigned int maxFeatureIndex) {
    std::cout << "Raw max index is: " << maxFeatureIndex << std::endl;
    maxFeatureIndex = roundUpMaxIndex(maxFeatureIndex);
    std::cout << "Rounded up max index to: " << maxFeatureIndex << std::endl;

    try {
        NcFile nc(fileName, NcFile::replace);
        if (nc.isNull()) {
            std::cout << "Error creating output file:" << fileName << std::endl;
            throw std::runtime_error("Error creating NetCDF file.");
        }
        nc.putAtt("datasets", ncUint, 1);
        nc.putAtt("name0", datasetName);
        nc.putAtt("attributes0", ncUint, (DataSetEnums::Sparse + DataSetEnums::Boolean));
        nc.putAtt("kind0", ncUint, DataSetEnums::Numeric);
        nc.putAtt("dataType0", ncUint, DataSetEnums::UInt);
        nc.putAtt("dimensions0", ncUint, 1);
        nc.putAtt("width0", ncUint, maxFeatureIndex);
        NcDim examplesDim = nc.addDim("examplesDim0", vSparseStart.size());
        NcDim sparseDataDim = nc.addDim("sparseDataDim0", vSparseIndex.size());
        NcVar sparseStartVar = nc.addVar("sparseStart0", "uint", "examplesDim0");
        NcVar sparseEndVar = nc.addVar("sparseEnd0", "uint", "examplesDim0");
        NcVar sparseIndexVar = nc.addVar("sparseIndex0", "uint", "sparseDataDim0");
        sparseStartVar.putVar(&vSparseStart[0]);
        sparseEndVar.putVar(&vSparseEnd[0]);
        sparseIndexVar.putVar(&vSparseIndex[0]);

        std::cout << "Created NetCDF file " << fileName << " " << "for dataset " << datasetName << std::endl;
    } catch (std::exception &e) {
        std::cout << "Caught exception: " << e.what() << "\n";
        throw std::runtime_error("Error writing to NetCDF file.");
    }
}


unsigned int align(size_t size) {
    return (unsigned int) ((size + 127) >> 7) << 7;
}

bool addDataToNetCDF(NcFile& nc, const long long dataIndex, const std::string& dataName,
                const std::map<std::string, unsigned int>& mFeatureNameToIndex,
                const std::vector<std::vector<unsigned int> >& vInputSamples,
                const std::vector<std::vector<unsigned int> >& vInputSamplesTime, const std::vector<std::vector<float> >& vInputSamplesData,
                const bool alignFeatureDimensionality, int& minDate, int& maxDate, const int featureDimensionality) {
    std::vector<std::string> vFeatureName(mFeatureNameToIndex.size());
    std::vector<char*> vFeatureNameC(vFeatureName.size());
    if (mFeatureNameToIndex.size()) {
        for (std::map<std::string, unsigned int>::const_iterator it = mFeatureNameToIndex.begin();
                        it != mFeatureNameToIndex.end(); it++) {
            vFeatureName[it->second] = it->first;
        }

        for (int i = 0; i < vFeatureNameC.size(); i++) {
            vFeatureNameC[i] = &(vFeatureName[i])[0];
        }
    }
    std::string sDataIndex = std::to_string(dataIndex);

    NcDim indToFeatureDim;
    if (vFeatureNameC.size()) {
        indToFeatureDim = nc.addDim((std::string("indToFeatureDim") + sDataIndex).c_str(), vFeatureNameC.size());
    }
    NcVar indToFeatureVar;
    if (vFeatureNameC.size()) {
        indToFeatureVar = nc.addVar((std::string("indToFeature") + sDataIndex).c_str(), "string", (std::string("indToFeatureDim") + sDataIndex).c_str());
    }
    if (vFeatureNameC.size()) {
        indToFeatureVar.putVar(std::vector<size_t>(1, 0), std::vector<size_t>(1, mFeatureNameToIndex.size()),
                        vFeatureNameC.data());
    }

    unsigned long long numberSamples = 0;
    for (int i = 0; i < vInputSamples.size(); i++) {
        numberSamples += vInputSamples[i].size();
    }

    if (numberSamples) {
        std::vector<unsigned int> vSparseInputStart(vInputSamples.size());
        std::vector<unsigned int> vSparseInputEnd(vInputSamples.size());
        std::vector<unsigned int> vSparseInputIndex(0), vSparseInputTime(0);
        for (int i = 0; i < vInputSamples.size(); i++) {
            vSparseInputStart[i] = (unsigned int) vSparseInputIndex.size();
            for (int j = 0; j < vInputSamples[i].size(); j++) {
                vSparseInputIndex.push_back(vInputSamples[i][j]);
                if (vInputSamplesTime.size() && vInputSamplesTime[i].size()) {
                    vSparseInputTime.push_back(vInputSamplesTime[i][j]);
                    minDate = std::min(minDate, (int) vInputSamplesTime[i][j]);
                    maxDate = std::max(maxDate, (int) vInputSamplesTime[i][j]);
                }
            }
            vSparseInputEnd[i] = (unsigned int) vSparseInputIndex.size();
        }

        std::vector<float> vSparseData(vSparseInputIndex.size(), 1.f);
        if (vInputSamplesData.size()) {
            int cnt = 0;
            for (int c = 0; c < vInputSamplesData.size(); c++) {
                const std::vector<float>& inputData = vInputSamplesData[c];
                for (int i = 0; i < inputData.size(); i++) {
                    vSparseData[cnt] = inputData[i];
                    cnt++;
                }
            }
        }
        std::cout << vSparseInputIndex.size() << " total input data points." << std::endl;
        std::cout << "write " << dataName << " " << sDataIndex << std::endl;

        unsigned int width = 0;

        if (featureDimensionality > 0 && mFeatureNameToIndex.size() == 0) {
            width = featureDimensionality;
        } else {
            width = (unsigned int) mFeatureNameToIndex.size();
        }

        width = (alignFeatureDimensionality) ? align(width) : width;
        nc.putAtt((std::string("name") + sDataIndex).c_str(), dataName.c_str());
        if (vInputSamplesData.size()) {
            nc.putAtt((std::string("attributes") + sDataIndex).c_str(), ncUint, 1);
            nc.putAtt((std::string("kind") + sDataIndex).c_str(), ncUint, 0);
            nc.putAtt((std::string("dataType") + sDataIndex).c_str(), ncUint, 4);
        } else {
            nc.putAtt((std::string("attributes") + sDataIndex).c_str(), ncUint, 3);
            nc.putAtt((std::string("kind") + sDataIndex).c_str(), ncUint, 0);
            nc.putAtt((std::string("dataType") + sDataIndex).c_str(), ncUint, 0);
        }

        nc.putAtt((std::string("dimensions") + sDataIndex).c_str(), ncUint, 1);
        nc.putAtt((std::string("width") + sDataIndex).c_str(), ncUint, width);
        NcDim examplesDim = nc.addDim((std::string("examplesDim") + sDataIndex).c_str(), vSparseInputStart.size());
        NcDim sparseDataDim = nc.addDim((std::string("sparseDataDim") + sDataIndex).c_str(), vSparseInputIndex.size());
        NcVar sparseStartVar = nc.addVar((std::string("sparseStart") + sDataIndex).c_str(), "uint", (std::string("examplesDim") + sDataIndex).c_str());
        NcVar sparseEndVar = nc.addVar((std::string("sparseEnd") + sDataIndex).c_str(), "uint", (std::string("examplesDim") + sDataIndex).c_str());
        NcVar sparseIndexVar = nc.addVar((std::string("sparseIndex") + sDataIndex).c_str(), "uint", (std::string("sparseDataDim") + sDataIndex).c_str());
        NcVar sparseTimeVar;
        if (vSparseInputTime.size()) {
            sparseTimeVar = nc.addVar((std::string("sparseTime") + sDataIndex).c_str(), "uint", (std::string("sparseDataDim") + sDataIndex).c_str());
        }

        NcVar sparseDataVar;
        if (vInputSamplesData.size()) {
            sparseDataVar = nc.addVar((std::string("sparseData") + sDataIndex).c_str(), ncFloat, sparseDataDim);
        }

        sparseStartVar.putVar(&vSparseInputStart[0]);
        sparseEndVar.putVar(&vSparseInputEnd[0]);
        sparseIndexVar.putVar(&vSparseInputIndex[0]);
        if (vSparseInputTime.size()) {
            sparseTimeVar.putVar(&vSparseInputTime[0]);
        }
        if (vInputSamplesData.size()) {
            sparseDataVar.putVar(&vSparseData[0]);
        }
        return true;
    } else {
        return false;
    }
}

void readNetCDFindToFeature(const std::string& fname, const int n, std::vector<std::string>& vFeaturesStr) {
    NcFile nc(fname, NcFile::read);
    if (nc.isNull()) {
        std::cout << "Error opening binary output file " << fname << std::endl;
        return;
    }

    std::string nstring = std::to_string((long long) n);
    vFeaturesStr.clear();

    NcDim indToFeatureDim = nc.getDim(std::string("indToFeatureDim") + nstring);
    if (indToFeatureDim.isNull()) {
        std::cout << "reading error indToFeatureDim" << std::endl;
        return;
    }

    NcVar indToFeatureVar = nc.getVar(std::string("indToFeature") + nstring);
    if (indToFeatureVar.isNull()) {
        std::cout << "reading error indToFeature" << std::endl;
        return;
    }

    std::vector<char*> vFeaturesChars;
    vFeaturesChars.resize(indToFeatureDim.getSize());
    indToFeatureVar.getVar(&vFeaturesChars[0]);
    vFeaturesStr.resize(indToFeatureDim.getSize());
    for (int i = 0; i < vFeaturesStr.size(); i++) {
        vFeaturesStr[i] = vFeaturesChars[i];
    }
}

void readNetCDFsamplesName(const std::string& fname, std::vector<std::string>& vSamplesName) {

    netCDF::NcFile nc(fname, NcFile::read);
    if (nc.isNull()) {
        std::cout << "Error opening binary output file " << fname << std::endl;
        return;
    }

    vSamplesName.clear();

    netCDF::NcDim samplesDim = nc.getDim("samplesDim");
    if (samplesDim.isNull()) {
        std::cout << "reading error examplesDim" << std::endl;
        return;
    }
    netCDF::NcVar sparseSamplesVar = nc.getVar("samples");
    if (sparseSamplesVar.isNull()) {
        std::cout << "reading error sparseSamplesVar" << std::endl;
        return;
    }
    std::vector<char*> vSamplesChars;

    vSamplesChars.resize(samplesDim.getSize());
    vSamplesName.resize(samplesDim.getSize());
    sparseSamplesVar.getVar(&vSamplesChars[0]);
    for (int i = 0; i < vSamplesChars.size(); i++) {
        vSamplesName[i] = vSamplesChars[i];
    }
}

void writeNETCDF(const std::string& fileName, const std::vector<std::string>& vSamplesName,
                const std::map<std::string, unsigned int>& mInputFeatureNameToIndex, std::vector<std::vector<unsigned int> >& vInputSamples,
                const std::vector<std::vector<unsigned int> >& vInputSamplesTime, std::vector<std::vector<float> >& vInputSamplesData,
                const std::map<std::string, unsigned int>& mOutputFeatureNameToIndex, const std::vector<std::vector<unsigned int> >& vOutputSamples,
                const std::vector<std::vector<unsigned int> >& vOutputSamplesTime,
                const std::vector<std::vector<float> >& vOutputSamplesData, int& minInpDate, int& maxInpDate, int& minOutDate,
                int& maxOutDate, const bool alignFeatureDimensionality, const int datasetNum) {

    netCDF::NcFile nc(fileName, NcFile::replace);
    if (nc.isNull()) {
        std::cout << "Error opening binary output file" << std::endl;
        std::exit(2);
    }

    int countData = 0;
    if (datasetNum >= 1) {
        if (addDataToNetCDF(nc, 0, "input", mInputFeatureNameToIndex, vInputSamples, vInputSamplesTime, vInputSamplesData,
                        alignFeatureDimensionality, minInpDate, maxInpDate)) {
            countData++;
        } else {
            std::cout << "failed to write input data";
            std::exit(1);
        }
    }
    if (datasetNum >= 2) {
        if (addDataToNetCDF(nc, 1, "output", mOutputFeatureNameToIndex, vOutputSamples, vOutputSamplesTime, vOutputSamplesData,
                        alignFeatureDimensionality, minOutDate, maxOutDate)) {
            countData++;
        } else {
            std::cout << "failed to write output data";
            std::exit(1);
        }
    } else {
        std::cout << "number of data sets datasetNum " << datasetNum << " is not implemented";
        std::exit(1);
    }
    nc.putAtt("datasets", ncUint, countData);

    std::vector<const char*> vSamplesChars(vSamplesName.size());
    for (int i = 0; i < vSamplesChars.size(); i++) {
        vSamplesChars[i] = &(vSamplesName[i])[0];
    }

    netCDF::NcDim samplesDim = nc.addDim("samplesDim", vSamplesName.size());
    netCDF::NcVar sparseSamplesVar = nc.addVar("samples", "string", "samplesDim");
    sparseSamplesVar.putVar(std::vector<size_t>(1, 0), std::vector<size_t>(1, vSamplesChars.size()), vSamplesChars.data());
}
