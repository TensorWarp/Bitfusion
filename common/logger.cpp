#include "logger.h"
#include "exception.h"
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <cstdio>
#include <cstdlib>

namespace bitfusion::common {

    Logger::Logger() {
        char* isFirstRankOnlyChar = std::getenv("LOG_FIRST_RANK_ONLY");
        bool isFirstRankOnly = (isFirstRankOnlyChar != nullptr && std::string(isFirstRankOnlyChar) == "ON");

        int deviceId;
        cudaGetDevice(&deviceId);

        char* levelName = std::getenv("LOG_LEVEL");
        if (levelName != nullptr) {
            static const std::map<std::string, Level> nameToLevel = {
                {"TRACE", TRACE},
                {"DEBUG", DEBUG},
                {"INFO", INFO},
                {"WARNING", WARNING},
                {"ERROR", ERROR},
            };

            auto level = nameToLevel.find(levelName);
            if (isFirstRankOnly && deviceId != 0) {
                level = nameToLevel.find("ERROR");
            }
            if (level != nameToLevel.end()) {
                setLevel(level->second);
            }
            else {
                std::fprintf(stderr,
                    "[WARNING] Invalid logger level LOG_LEVEL=%s. "
                    "Ignore the environment variable and use a default "
                    "logging level.\n",
                    levelName);
                levelName = nullptr;
            }
        }
    }

    void Logger::log(const std::exception& ex, Level level) {
        log(level, "%s: %s", Exception::demangle(typeid(ex).name()).c_str(), ex.what());
    }

    Logger* Logger::getLogger() {
        thread_local Logger instance;
        return &instance;
    }
}
