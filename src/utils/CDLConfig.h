#pragma once

#include "TrainingParameters.h"
#include "ConfigBase.h"

/// <summary>
/// Represents a configuration class for a specific application named CDL.
/// </summary>
class CDLConfig : public ConfigBase {
public:
    std::string networkFileName;
    std::string dataFileName;
    int randomSeed = 0;
    std::string command;
    int mode = 0;
    TrainingParameters trainingParameters;

    /// <summary>
    /// Loads CDL configuration data from a JSON value.
    /// </summary>
    /// <param name="root">The JSON value containing CDL configuration data.</param>
    void Load(const Json::Value& root) override {
        networkFileName = root["network"].asString();
        dataFileName = root["data"].asString();
        randomSeed = root["randomseed"].asInt();
        command = root["command"].asString();
        trainingParameters.Load(root);
    }
};