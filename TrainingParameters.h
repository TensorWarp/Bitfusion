#pragma once
#include <string>
#include <json.h>

#include "ConfigBase.h"

class TrainingParameters : public ConfigBase {
public:
    int epochs = 0;
    float alpha = 0.0f;
    float alphaInterval = 0.0f;
    float alphaMultiplier = 0.0f;
    float mu = 0.0f;
    float lambda = 0.0f;
    float checkpointInterval = 0.0f;
    std::string checkpointFileName;
    std::string optimizer;
    std::string resultsFileName;

    void Load(const Json::Value& root) override {
        if (const auto params = root["trainingparameters"]; !params.isNull()) {
            epochs = params["epochs"].asInt();
            alpha = params["alpha"].asFloat();
            alphaInterval = params["alphainterval"].asFloat();
            alphaMultiplier = params["alphamultiplier"].asFloat();
            mu = params["mu"].asFloat();
            lambda = params["lambda"].asFloat();
            checkpointInterval = params["checkpointinterval"].asFloat();
            checkpointFileName = params["checkpointname"].asString();
            optimizer = params["optimizer"].asString();
            resultsFileName = params["results"].asString();
        }
    }
};