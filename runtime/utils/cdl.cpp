#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include "../../src/gpuTypes.h"
#include "../../src/types.h"
#include "cdl.h"
#include <omp.h>
#include <mpi.h>

void CDL::SetOptimizer(const std::string& optimizerName) {
    std::string optimizerNameLower = optimizerName;
    std::transform(optimizerNameLower.begin(), optimizerNameLower.end(), optimizerNameLower.begin(), ::tolower);

    static const std::unordered_map<std::string, TrainingMode> optimizerMap = {
        {"sgd", TrainingMode::SGD},
        {"adam", TrainingMode::Adam},
        {"rmsprop", TrainingMode::RMSProp}
    };

    auto it = optimizerMap.find(optimizerNameLower);
    if (it != optimizerMap.end()) {
        _optimizer = it->second;
    }
    else {
        throw std::runtime_error("CDL::SetOptimizer: Unknown optimizer: " + optimizerName);
    }
}

void CDL::ParseTrainingParameters(const Json::Value& value) {
    static const std::unordered_map<std::string, std::function<void(const Json::Value&)>> parameterActions = {
        {"epochs", [this](const Json::Value& val) { _epochs = val.asInt(); }},
        {"alpha", [this](const Json::Value& val) { _alpha = val.asFloat(); }},
        {"alphainterval", [this](const Json::Value& val) { _alphaInterval = val.asFloat(); }},
        {"alphamultiplier", [this](const Json::Value& val) { _alphaMultiplier = val.asFloat(); }},
        {"mu", [this](const Json::Value& val) { _mu = val.asFloat(); }},
        {"lambda", [this](const Json::Value& val) { _lambda = val.asFloat(); }},
        {"checkpointinterval", [this](const Json::Value& val) { _checkpointInterval = val.asFloat(); }},
        {"checkpointname", [this](const Json::Value& val) {
            if (val.isString()) {
                _checkpointFileName = val.asString();
            }
            else {
                throw std::runtime_error("CDL::ParseTrainingParameters: CheckpointName must be a string.");
            }
        }},
        {"optimizer", [this](const Json::Value& val) {
            if (val.isString()) {
                SetOptimizer(val.asString());
            }
            else {
                throw std::runtime_error("CDL::ParseTrainingParameters: Optimizer must be a string.");
            }
        }},
        {"results", [this](const Json::Value& val) {
            if (val.isString()) {
                _resultsFileName = val.asString();
            }
            else {
                throw std::runtime_error("CDL::ParseTrainingParameters: Results must be a string.");
            }
        }},
    };

#pragma omp parallel for
    for (int i = 0; i < value.getMemberNames().size(); ++i) {
        const std::string& pname = value.getMemberNames()[i];
        const Json::Value& pvalue = value[pname];
        std::string pname_lowercase = pname;
        std::transform(pname_lowercase.begin(), pname_lowercase.end(), pname_lowercase.begin(), ::tolower);

        auto it = parameterActions.find(pname_lowercase);
        if (it != parameterActions.end()) {
            it->second(pvalue);
        }
        else {
            throw std::runtime_error("CDL::Load_JSON: Invalid TrainingParameter: " + pname);
        }
    }
}

int CDL::Load_JSON(const std::string& fname) {
    int localError = 0;
    int globalError = 0;

    Json::Value index;
    Json::CharReaderBuilder builder;
    std::string errs;

    std::ifstream stream(fname, std::ifstream::binary);
    bool parsedSuccess = Json::parseFromStream(builder, stream, &index, &errs);

    if (!parsedSuccess) {
        localError = -1;
    }

    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Allreduce(&localError, &globalError, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (globalError < 0) {
        MPI_Finalize();
        return globalError;
    }

    bool networkSet = false, commandSet = false, dataSet = false, epochsSet = false;

#pragma omp parallel
    {
#pragma omp for reduction(|:localError) nowait
        for (int i = 0; i < index.getMemberNames().size(); ++i) {
            const std::string& name = index.getMemberNames()[i];
            const Json::Value& value = index[name];
            std::string name_lowercase = name;
            std::transform(name_lowercase.begin(), name_lowercase.end(), name_lowercase.begin(), ::tolower);

            std::string vstring = value.isString() ? value.asString() : "";
            std::transform(vstring.begin(), vstring.end(), vstring.begin(), ::tolower);

            if (name_lowercase == "version") {
                float version = value.asFloat();
            }
            else if (name_lowercase == "network") {
#pragma omp critical(network_critical)
                {
                    _networkFileName = value.asString();
                    networkSet = true;
                }
            }
            else if (name_lowercase == "data") {
#pragma omp critical(data_critical)
                {
                    _dataFileName = value.asString();
                    dataSet = true;
                }
            }
            else if (name_lowercase == "randomseed") {
#pragma omp critical(random_seed_critical)
                {
                    _randomSeed = value.asInt();
                }
            }
            else if (name_lowercase == "command") {
#pragma omp critical(command_critical)
                {
                    if (vstring == "train") {
                        _mode = Mode::Training;
                    }
                    else if (vstring == "predict") {
                        _mode = Mode::Prediction;
                    }
                    else if (vstring == "validate") {
                        _mode = Mode::Validation;
                    }
                    else {
                        localError = -1;
                    }
                    commandSet = true;
                }
            }
            else if (name_lowercase == "trainingparameters") {
#pragma omp critical(training_parameters_critical)
                {
                    try {
                        ParseTrainingParameters(value);
                    }
                    catch (const std::runtime_error& e) {
                        std::cerr << "Error in parsing training parameters: " << e.what() << std::endl;
                        localError = -1;
                    }
                }
            }
            else {
                localError = -1;
            }
        }

#pragma omp master
        MPI_Barrier(MPI_COMM_WORLD);

#pragma omp master
        MPI_Allreduce(&localError, &globalError, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }

    if (globalError < 0) {
        MPI_Finalize();
        return globalError;
    }

    if (_alphaInterval == 0) {
        _alphaInterval = 20;
        _alphaMultiplier = 1;
    }

    if (!networkSet || !commandSet || !dataSet || (_mode == Mode::Training && !epochsSet)) {
        globalError = -1;
    }

    MPI_Finalize();

    return globalError;
}