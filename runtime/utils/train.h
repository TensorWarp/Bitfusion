#pragma once

#include <string>

struct TrainingConfig {
    float alpha = 0.025f;
    float lambda = 0.0001f;
    float lambda1 = 0.0f;
    float mu = 0.5f;
    float mu1 = 0.0f;
    std::string configFileName;
    std::string inputDataFile;
    std::string outputDataFile;
    std::string networkFileName;
    unsigned int batchSize = 1024;
    unsigned int epoch = 40;
};

void printUsageTrain();
bool parseCommandLineArgs(int argc, char** argv, TrainingConfig& config);
bool fileExists(const std::string& filePath);
