#pragma once

#include <NvInferRuntime.h>

namespace bitfusion::runtime
{
    /// <summary>
    /// Custom logger class that inherits from nvinfer1::ILogger.
    /// </summary>
    class Logger : public nvinfer1::ILogger
    {
    public:
        /// <summary>
        /// Logs a message with the specified severity level.
        /// </summary>
        /// <param name="severity">The severity level of the log message.</param>
        /// <param name="msg">The log message as an ASCII string.</param>
        void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;

        /// <summary>
        /// Get the current severity level of the logger.
        /// </summary>
        /// <returns>The current severity level of the logger.</returns>
        Severity getLevel();

        /// <summary>
        /// Set the severity level of the logger.
        /// </summary>
        /// <param name="level">The severity level to set.</param>
        void setLevel(Severity level);
    };
}
