#pragma once

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "stringUtils.h"

namespace bitfusion::common
{

    class Logger
    {

#if _WIN32
#undef ERROR
#endif

    public:
        enum Level
        {
            TRACE = 0,
            DEBUG = 10,
            INFO = 20,
            WARNING = 30,
            ERROR = 40
        };

        static Logger* getLogger();

        Logger(Logger const&) = delete;
        void operator=(Logger const&) = delete;

#if defined(_MSC_VER)
        template <typename... Args>
        void log(Level level, char const* format, const Args&... args);

        template <typename... Args>
        void log(Level level, int rank, char const* format, const Args&... args);
#else
        template <typename... Args>
        void log(Level level, char const* format, const Args&... args) __attribute__((format(printf, 3, 0)));

        template <typename... Args>
        void log(Level level, int rank, char const* format, const Args&... args) __attribute__((format(printf, 4, 0)));
#endif

        template <typename... Args>
        void log(Level level, std::string const& format, const Args&... args)
        {
            return log(level, format.c_str(), args...);
        }

        template <typename... Args>
        void log(const Level level, const int rank, const std::string& format, const Args&... args)
        {
            return log(level, rank, format.c_str(), args...);
        }

        void log(std::exception const& ex, Level level = Level::ERROR);

        Level getLevel()
        {
            return level_;
        }

        void setLevel(const Level level)
        {
            level_ = level;
            log(INFO, "Set logger level by %s", getLevelName(level).c_str());
        }

    private:
        const std::string PREFIX = "[TensorRT-LLM]";
        std::map<Level, std::string> level_name_
            = { {TRACE, "TRACE"}, {DEBUG, "DEBUG"}, {INFO, "INFO"}, {WARNING, "WARNING"}, {ERROR, "ERROR"} };

#ifndef NDEBUG
        const Level DEFAULT_LOG_LEVEL = DEBUG;
#else
        const Level DEFAULT_LOG_LEVEL = INFO;
#endif
        Level level_ = DEFAULT_LOG_LEVEL;

        Logger();

        inline std::string getLevelName(const Level level)
        {
            return level_name_[level];
        }

        inline std::string getPrefix(const Level level)
        {
            return PREFIX + "[" + getLevelName(level) + "] ";
        }

        inline std::string getPrefix(const Level level, const int rank)
        {
            return PREFIX + "[" + getLevelName(level) + "][" + std::to_string(rank) + "] ";
        }
    };

    template <typename... Args>
    void Logger::log(Logger::Level level, char const* format, Args const&... args)
    {
        if (level_ <= level)
        {
            auto const fmt = getPrefix(level) + format;
            auto& out = level_ < WARNING ? std::cout : std::cerr;
            if constexpr (sizeof...(args) > 0)
            {
                out << fmtstr(fmt.c_str(), args...);
            }
            else
            {
                out << fmt;
            }
            out << std::endl;
        }
    }

    template <typename... Args>
    void Logger::log(const Logger::Level level, const int rank, char const* format, const Args&... args)
    {
        if (level_ <= level)
        {
            auto const fmt = getPrefix(level, rank) + format;
            auto& out = level_ < WARNING ? std::cout : std::cerr;
            if constexpr (sizeof...(args) > 0)
            {
                out << fmtstr(fmt.c_str(), args...);
            }
            else
            {
                out << fmt;
            }
            out << std::endl;
        }
    }

#define LOG(level, ...) bitfusion::common::Logger::getLogger()->log(level, __VA_ARGS__)
#define LOG_TRACE(...) LOG(bitfusion::common::Logger::TRACE, __VA_ARGS__)
#define LOG_DEBUG(...) LOG(bitfusion::common::Logger::DEBUG, __VA_ARGS__)
#define LOG_INFO(...) LOG(bitfusion::common::Logger::INFO, __VA_ARGS__)
#define LOG_WARNING(...) LOG(bitfusion::common::Logger::WARNING, __VA_ARGS__)
#define LOG_ERROR(...) LOG(bitfusion::common::Logger::ERROR, __VA_ARGS__)
#define LOG_EXCEPTION(ex, ...) bitfusion::common::Logger::getLogger()->log(ex, ##__VA_ARGS__)
}