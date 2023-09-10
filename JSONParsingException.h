#pragma once
#include <string>

class JSONParsingException : public ConfigException {
public:
    JSONParsingException(const std::string& msg) : ConfigException(msg, DataSetEnums::ErrorCode::JSONParseError) {}
};