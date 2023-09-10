#pragma once
#include <memory>
#include <json.h>

class ConfigFactory {
public:
    template <typename T>
    static std::unique_ptr<T> CreateConfig(const Json::Value& root) {
        auto config = std::make_unique<T>();
        config->Load(root);
        return config;
    }
};