#include "logger.h"
#include "../common/assert.h"
#include "../common/logger.h"
#include <unordered_map>
#include <stdexcept>

using namespace bitfusion::runtime;
namespace tc = bitfusion::common;

namespace {
    /// <summary>
    /// Returns a mapping between nvinfer1::ILogger::Severity and tc::Logger::Level.
    /// </summary>
    const auto& getSeverityMapping() {
        static const std::unordered_map<nvinfer1::ILogger::Severity, tc::Logger::Level> mapping = {
            {nvinfer1::ILogger::Severity::kINTERNAL_ERROR, tc::Logger::Level::ERROR},
            {nvinfer1::ILogger::Severity::kERROR, tc::Logger::Level::ERROR},
            {nvinfer1::ILogger::Severity::kWARNING, tc::Logger::Level::WARNING},
            {nvinfer1::ILogger::Severity::kINFO, tc::Logger::Level::INFO},
            {nvinfer1::ILogger::Severity::kVERBOSE, tc::Logger::Level::TRACE}
        };
        return mapping;
    }
}

/// <summary>
/// Logs a message with the specified severity.
/// </summary>
/// <param name="severity">The severity of the log message.</param>
/// <param name="msg">The log message.</param>
void Logger::log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg) noexcept
{
    auto* const logger = tc::Logger::getLogger();
    const auto& severityMapping = getSeverityMapping();

    auto levelIt = severityMapping.find(severity);
    const tc::Logger::Level level = (levelIt != severityMapping.end()) ? levelIt->second : tc::Logger::Level::TRACE;

    logger->log(level, msg);
}

/// <summary>
/// Gets the current logging level.
/// </summary>
/// <returns>The current logging level.</returns>
nvinfer1::ILogger::Severity Logger::getLevel()
{
    auto* const logger = tc::Logger::getLogger();
    const auto level = logger->getLevel();

    for (const auto [tcLevel, nvLevel] : std::array{
            std::pair{tc::Logger::Level::ERROR, nvinfer1::ILogger::Severity::kERROR},
            std::pair{tc::Logger::Level::WARNING, nvinfer1::ILogger::Severity::kWARNING},
            std::pair{tc::Logger::Level::INFO, nvinfer1::ILogger::Severity::kINFO},
            std::pair{tc::Logger::Level::DEBUG, nvinfer1::ILogger::Severity::kVERBOSE},
            std::pair{tc::Logger::Level::TRACE, nvinfer1::ILogger::Severity::kVERBOSE} }) {
        if (level == tcLevel) {
            return nvLevel;
        }
    }
    return nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
}

/// <summary>
/// Sets the logging level.
/// </summary>
/// <param name="level">The desired logging level.</param>
void Logger::setLevel(nvinfer1::ILogger::Severity level)
{
    auto* const logger = tc::Logger::getLogger();

    for (const auto [tcLevel, nvLevel] : std::array{
            std::pair{tc::Logger::Level::ERROR, nvinfer1::ILogger::Severity::kERROR},
            std::pair{tc::Logger::Level::WARNING, nvinfer1::ILogger::Severity::kWARNING},
            std::pair{tc::Logger::Level::INFO, nvinfer1::ILogger::Severity::kINFO},
            std::pair{tc::Logger::Level::DEBUG, nvinfer1::ILogger::Severity::kVERBOSE},
            std::pair{tc::Logger::Level::TRACE, nvinfer1::ILogger::Severity::kVERBOSE} }) {
        if (level == nvLevel) {
            logger->setLevel(tcLevel);
            return;
        }
    }
    throw std::invalid_argument("Unsupported severity");
}
