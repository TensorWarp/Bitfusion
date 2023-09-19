#pragma once

#include "JSONParsingException.h"
#include <json.h>

/// <summary>
/// Represents the base class for configuration objects.
/// </summary>
class ConfigBase {
protected:
    ConfigBase() = default;

public:
    virtual ~ConfigBase() = default;

    /// <summary>
    /// Loads configuration data from a JSON value.
    /// </summary>
    /// <param name="root">The JSON value containing configuration data.</param>
    virtual void Load(const Json::Value& root) = 0;
};

/// <summary>
/// Parses a JSON file and returns the JSON value.
/// </summary>
/// <param name="fileName">The path to the JSON file to parse.</param>
/// <returns>The JSON value parsed from the file.</returns>
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