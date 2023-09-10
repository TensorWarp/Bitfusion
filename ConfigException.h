#pragma once
#include "Enum.h"

class ConfigException : public std::exception {
public:
    ConfigException(const std::string& msg, DataSetEnums::ErrorCode code) : msg_(msg), code_(code) {}

    const char* what() const noexcept override {
        return msg_.c_str();
    }

    DataSetEnums::ErrorCode GetErrorCode() const {
        return code_;
    }

private:
    std::string msg_;
    DataSetEnums::ErrorCode code_;
};