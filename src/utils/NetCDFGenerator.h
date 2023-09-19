#pragma once

#include <omp.h>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace fs = std::filesystem;

const std::string DATASET_TYPE_ANALOG = "analog";

class NetCDFGenerator {
public:
    NetCDFGenerator(const std::string& inputFile,
        const std::string& featureIndexFile,
        const std::string& sampleIndexFile,
        const std::string& outputFile,
        const std::string& datasetName,
        const std::string& dataType)
        : inputFile_(inputFile), featureIndexFile_(featureIndexFile),
        sampleIndexFile_(sampleIndexFile), outputFile_(outputFile),
        datasetName_(datasetName), dataType_(dataType) {}

    void initializeDataSources() {
        if (!fs::exists(sampleIndexFile_)) {
            std::cout << "Will create a new samples index file: " << sampleIndexFile_ << '\n';
        }
        else {
            std::cout << "Loading sample index from: " << sampleIndexFile_ << '\n';
            if (!loadIndexFromFile(sampleIndex_, sampleIndexFile_, std::cout)) {
                exit(1);
            }
        }

        if (!fs::exists(featureIndexFile_)) {
            std::cout << "Will create a new features index file: " << featureIndexFile_ << '\n';
        }
        else {
            std::cout << "Loading feature index from: " << featureIndexFile_ << '\n';
            if (!loadIndexFromFile(featureIndex_, featureIndexFile_, std::cout)) {
                exit(1);
            }
        }
    }

    void generateNetCDFDataset() {
        ThreadPool pool(std::thread::hardware_concurrency());

        bool updateFeatureIndex = !featureIndexFile_.empty();

        if (dataType_ == DATASET_TYPE_ANALOG) {
#pragma omp parallel
            {
#pragma omp for
                for (int i = 0; i < 1; ++i) {
#pragma omp critical
                    {
                        pool.enqueue([this, &updateFeatureIndex] {
                            writeNetCDFFile(vSparseStart_, vSparseEnd_, vSparseIndex_, vSparseData_, outputFile_, datasetName_, featureIndex_.size());
                            });
                    }
                }
            }
        }
        else {
#pragma omp parallel
            {
#pragma omp for
                for (int i = 0; i < 1; ++i) {
#pragma omp critical
                    {
                        pool.enqueue([this, &updateFeatureIndex] {
                            writeNetCDFFile(vSparseStart_, vSparseEnd_, vSparseIndex_, outputFile_, datasetName_, featureIndex_.size());
                            });
                    }
                }
            }
        }
    }

private:
    std::string inputFile_;
    std::string featureIndexFile_;
    std::string sampleIndexFile_;
    std::string outputFile_;
    std::string datasetName_;
    std::string dataType_;

    std::unordered_map<std::string, unsigned int> featureIndex_;
    std::unordered_map<std::string, unsigned int> sampleIndex_;
    std::vector<unsigned int> vSparseStart_;
    std::vector<unsigned int> vSparseEnd_;
    std::vector<unsigned int> vSparseIndex_;
    std::vector<float> vSparseData_;
};