#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <future>
#include <vector>
#include <functional>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <filesystem>
#include <string_view>

#include "json/json.h"
#include "../system/Enum.h"
#include "../system/ThreadPool.h"
#include "Logger.h"
#include "JSONParsingException.h"
#include "ConfigException.h"
#include "ConfigFactory.h"
#include "TrainingParameters.h"
#include "CDLConfig.h"

namespace fs = std::filesystem;

void ValidateConfiguration(const CDLConfig& config) {
    if (config.networkFileName.empty() || config.dataFileName.empty()) {
        throw ConfigException("Network and data file names must be specified.", DataSetEnums::ErrorCode::ConfigurationError);
    }
    if (config.trainingParameters.epochs <= 0) {
        throw ConfigException("Epochs must be a positive integer.", DataSetEnums::ErrorCode::ConfigurationError);
    }
}

void LoadAndProcessPlugins(const std::string& pluginsDir) {
    // TODO: Implement the logic to load and process plugins.
    Logger::Log(DataSetEnums::LogLevel::INFO, "Loading and processing plugins from: ", pluginsDir);
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            throw ConfigException("Usage: " + std::string(argv[0]) + " <config_file>", DataSetEnums::ErrorCode::ConfigurationError);
        }

        std::string configFilePath = argv[1];

        std::future<Json::Value> futureRoot = std::async(std::launch::async, ParseJSONFromFile, configFilePath);

        ThreadPool threadPool(4);

        Logger::Log(DataSetEnums::LogLevel::INFO, "Performing some other work...");
        std::this_thread::sleep_for(std::chrono::seconds(2));
        Logger::Log(DataSetEnums::LogLevel::INFO, "Other work complete.");

        Json::Value root = futureRoot.get();

        auto config = ConfigFactory::CreateConfig<CDLConfig>(root);

        Logger::Log(DataSetEnums::LogLevel::INFO, "Network File Name: ", config->networkFileName);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Data File Name: ", config->dataFileName);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Random Seed: ", config->randomSeed);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Command: ", config->command);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Mode: ", config->mode);

        Logger::Log(DataSetEnums::LogLevel::INFO, "Training Parameters:");
        Logger::Log(DataSetEnums::LogLevel::INFO, "Epochs: ", config->trainingParameters.epochs);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Alpha: ", config->trainingParameters.alpha);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Alpha Interval: ", config->trainingParameters.alphaInterval);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Alpha Multiplier: ", config->trainingParameters.alphaMultiplier);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Mu: ", config->trainingParameters.mu);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Lambda: ", config->trainingParameters.lambda);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Checkpoint Interval: ", config->trainingParameters.checkpointInterval);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Checkpoint File Name: ", config->trainingParameters.checkpointFileName);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Optimizer: ", config->trainingParameters.optimizer);
        Logger::Log(DataSetEnums::LogLevel::INFO, "Results File Name: ", config->trainingParameters.resultsFileName);

        threadPool.Execute([&config]() {
            try {
                ValidateConfiguration(*config);
                Logger::Log(DataSetEnums::LogLevel::INFO, "Configuration validation successful.");
            }
            catch (const ConfigException& e) {
                Logger::Log(DataSetEnums::LogLevel::ERROR, "Configuration Error: ", e.what());
            }
            });

        threadPool.Execute([]() {
            try {
                LoadAndProcessPlugins("plugins_dir");
                Logger::Log(DataSetEnums::LogLevel::INFO, "Plugins loaded and processed.");
            }
            catch (const ConfigException& e) {
                Logger::Log(DataSetEnums::LogLevel::ERROR, "Plugin Error: ", e.what());
            }
            });

        Logger::Log(DataSetEnums::LogLevel::INFO, "Main thread continues working...");
        std::this_thread::sleep_for(std::chrono::seconds(2));
        Logger::Log(DataSetEnums::LogLevel::INFO, "Main thread work complete.");
    }
    catch (const JSONParsingException& e) {
        Logger::Log(DataSetEnums::LogLevel::ERROR, "JSON Parsing Error: ", e.what());
        return static_cast<int>(e.GetErrorCode());
    }
    catch (const ConfigException& e) {
        Logger::Log(DataSetEnums::LogLevel::ERROR, "Configuration Error: ", e.what());
        return static_cast<int>(e.GetErrorCode());
    }
    catch (const std::exception& e) {
        Logger::Log(DataSetEnums::LogLevel::ERROR, "Error: ", e.what());
        return -1;
    }

    return static_cast<int>(DataSetEnums::ErrorCode::Success);
}