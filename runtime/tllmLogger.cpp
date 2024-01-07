#include "tllmLogger.h"
#include "../common/assert.h"
#include "../common/logger.h"

using namespace bitfusion::runtime;
namespace tc = bitfusion::common;

void TllmLogger::log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg) noexcept
{
    switch (severity)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
    case nvinfer1::ILogger::Severity::kERROR: LOG_ERROR(msg); break;
    case nvinfer1::ILogger::Severity::kWARNING: LOG_WARNING(msg); break;
    case nvinfer1::ILogger::Severity::kINFO: LOG_INFO(msg); break;
    case nvinfer1::ILogger::Severity::kVERBOSE: LOG_DEBUG(msg); break;
    default: LOG_TRACE(msg); break;
    }
}

nvinfer1::ILogger::Severity TllmLogger::getLevel()
{
    auto* const logger = tc::Logger::getLogger();
    switch (logger->getLevel())
    {
    case tc::Logger::Level::ERROR: return nvinfer1::ILogger::Severity::kERROR;
    case tc::Logger::Level::WARNING: return nvinfer1::ILogger::Severity::kWARNING;
    case tc::Logger::Level::INFO: return nvinfer1::ILogger::Severity::kINFO;
    case tc::Logger::Level::DEBUG:
    case tc::Logger::Level::TRACE: return nvinfer1::ILogger::Severity::kVERBOSE;
    default: return nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
    }
}

void TllmLogger::setLevel(nvinfer1::ILogger::Severity level)
{
    auto* const logger = tc::Logger::getLogger();
    switch (level)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
    case nvinfer1::ILogger::Severity::kERROR: logger->setLevel(tc::Logger::Level::ERROR); break;
    case nvinfer1::ILogger::Severity::kWARNING: logger->setLevel(tc::Logger::Level::WARNING); break;
    case nvinfer1::ILogger::Severity::kINFO: logger->setLevel(tc::Logger::Level::INFO); break;
    case nvinfer1::ILogger::Severity::kVERBOSE: logger->setLevel(tc::Logger::Level::TRACE); break;
    default: THROW("Unsupported severity");
    }
}
