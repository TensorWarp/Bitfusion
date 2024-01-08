
#pragma once

#include "../common/quantization.h"
#include "common.h"
#include "ModelConfig.h"
#include "worldConfig.h"

#include <filesystem>
#include <istream>
#include <string>
#include <utility>

namespace bitfusion::runtime
{

class JsonConfig
{
public:
    JsonConfig(std::string name, std::string version, std::string precision, SizeType tensorParallelism,
        SizeType pipelineParallelism, ModelConfig const& modelConfig)
        : mName(std::move(name))
        , mVersion(std::move(version))
        , mPrecision(std::move(precision))
        , mTensorParallelism{tensorParallelism}
        , mPipelineParallelism{pipelineParallelism}
        , mModelConfig(modelConfig)
    {
    }

    static JsonConfig parse(std::string const& json);

    static JsonConfig parse(std::istream& json);

    static JsonConfig parse(std::filesystem::path const& path);

    [[nodiscard]] ModelConfig getModelConfig() const
    {
        return mModelConfig;
    }

    [[nodiscard]] std::string const& getName() const
    {
        return mName;
    }

    [[nodiscard]] std::string const& getVersion() const
    {
        return mVersion;
    }

    [[nodiscard]] std::string const& getPrecision() const
    {
        return mPrecision;
    }

    [[nodiscard]] SizeType constexpr getTensorParallelism() const
    {
        return mTensorParallelism;
    }

    [[nodiscard]] SizeType constexpr getPipelineParallelism() const
    {
        return mPipelineParallelism;
    }

    [[nodiscard]] SizeType constexpr getWorldSize() const
    {
        return mTensorParallelism * mPipelineParallelism;
    }

    [[nodiscard]] std::string engineFilename(WorldConfig const& worldConfig, std::string const& model) const;

    [[nodiscard]] std::string engineFilename(WorldConfig const& worldConfig) const
    {
        return engineFilename(worldConfig, getName());
    }

private:
    std::string const mName;
    std::string const mVersion;
    std::string const mPrecision;
    SizeType const mTensorParallelism;
    SizeType const mPipelineParallelism;
    ModelConfig const mModelConfig;
};

}
