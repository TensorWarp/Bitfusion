#pragma once

#include "Enum.h"

/// <summary>
/// Represents an exception that occurs during configuration processing.
/// </summary>
class ConfigException : public std::exception {
public:
    /// <summary>
    /// Initializes a new instance of the ConfigException class.
    /// </summary>
    /// <param name="msg">A description of the exception.</param>
    /// <param name="code">The error code associated with the exception.</param>
    ConfigException(const std::string& msg, DataSetEnums::ErrorCode code) : msg_(msg), code_(code) {}

    /// <summary>
    /// Gets a message that describes the exception.
    /// </summary>
    /// <returns>A C-style string describing the exception.</returns>
    const char* what() const noexcept override {
        return msg_.c_str();
    }

    /// <summary>
    /// Gets the error code associated with the exception.
    /// </summary>
    /// <returns>The error code.</returns>
    DataSetEnums::ErrorCode GetErrorCode() const {
        return code_;
    }

private:
    std::string msg_;
    DataSetEnums::ErrorCode code_;
};