#pragma once
#include "JSONParsingException.h"
#include <json.h>

class ConfigBase {
protected:
    ConfigBase() = default;

public:
    virtual ~ConfigBase() = default;
    virtual void Load(const Json::Value& root) = 0;
};

Json::Value ParseJSONFromFile(const std::string& fileName) {
    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    std::string errs;

    std::ifstream stream(fileName, std::ifstream::binary);
    if (!Json::parseFromStream(readerBuilder, stream, &root, &errs)) {
        throw JSONParsingException("Failed to parse JSON file: " + fileName + ", error: " + errs);
    }

    return root;
}