#pragma once

#include "../../src/gpuTypes.h"
#include "../../src/types.h"

class CDL {
public:
    enum class Mode {
        Training,
        Prediction,
        Validation
    };

    CDL();

    int Load_JSON(const std::string& fname);

    int _randomSeed;
    float _alphaInterval;
    float _alphaMultiplier;
    int _batch;
    float _checkpointInterval;
    std::string _checkpointFileName;
    bool _shuffleIndexes;
    std::string _resultsFileName;
    float _alpha;
    float _lambda;
    float _mu;
    TrainingMode _optimizer;
    std::string _networkFileName;
    std::string _dataFileName;
    Mode _mode;
    int _epochs;

private:
    std::map<std::string, TrainingMode> sOptimizationMap = {
        {"sgd", TrainingMode::SGD},
        {"nesterov", TrainingMode::Nesterov}
    };

    void SetOptimizer(const std::string& optimizerName);

    void ParseTrainingParameters(const Json::Value& value);
};

CDL::CDL()
    : _randomSeed(time(nullptr)),
    _alphaInterval(0),
    _alphaMultiplier(0.5f),
    _batch(1024),
    _checkpointInterval(1),
    _checkpointFileName("check"),
    _shuffleIndexes(false),
    _resultsFileName("network.nc"),
    _alpha(0.1f),
    _lambda(0.001f),
    _mu(0.9f),
    _optimizer(TrainingMode::SGD),
    _mode(Mode::Training),
    _epochs(0) {
}