#pragma once
#include "Enum.h"

class Logger {
public:
    template <typename... Args>
    static void Log(DataSetEnums::LogLevel level, const Args&... args) {
        std::string levelStr;
        switch (level) {
        case DataSetEnums::LogLevel::INFO:
            levelStr = "[INFO] ";
            break;
        case DataSetEnums::LogLevel::ERROR:
            levelStr = "[ERROR] ";
            break;
        case DataSetEnums::LogLevel::DEBUG:
            levelStr = "[DEBUG] ";
            break;
        }
        std::cout << levelStr;
        (std::cout << ... << args);
        std::cout << std::endl;
    }
};