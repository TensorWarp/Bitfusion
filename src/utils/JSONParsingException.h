#pragma once

#include <string>

/// <summary>
/// Exception class for handling JSON parsing errors.
/// </summary>
class JSONParsingException : public ConfigException {
public:
    JSONParsingException(const std::string& msg) : ConfigException(msg, DataSetEnums::ErrorCode::JSONParseError) {}
};