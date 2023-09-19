#pragma once

#include <memory>
#include <json.h>

/// <summary>
/// A factory class for creating configuration objects from JSON data.
/// </summary>
class ConfigFactory {
public:
    /// <summary>
    /// Creates a configuration object of the specified type and loads it from a JSON value.
    /// </summary>
    /// <typeparam name="T">The type of configuration object to create.</typeparam>
    /// <param name="root">The JSON value containing configuration data.</param>
    /// <returns>A unique pointer to the created configuration object.</returns>
    template <typename T>
    static std::unique_ptr<T> CreateConfig(const Json::Value& root) {
        auto config = std::make_unique<T>();
        config->Load(root);
        return config;
    }
};