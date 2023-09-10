#pragma once
#include "TrainingParameters.h"
#include "ConfigBase.h"

class CDLConfig : public ConfigBase {
public:
    std::string networkFileName;
    std::string dataFileName;
    int randomSeed = 0;
    std::string command;
    int mode = 0;
    TrainingParameters trainingParameters;

    void Load(const Json::Value& root) override {
        networkFileName = root["network"].asString();
        dataFileName = root["data"].asString();
        randomSeed = root["randomseed"].asInt();
        command = root["command"].asString();
        trainingParameters.Load(root);
    }
};