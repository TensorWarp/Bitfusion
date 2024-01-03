

#pragma once

#include <NvInferRuntime.h>
#include <mutex>

namespace bitfusion::plugins::api
{

auto constexpr kDefaultNamespace = "bitfusion";

class LoggerFinder : public nvinfer1::ILoggerFinder
{
public:
    //! Set the logger finder.
    void setLoggerFinder(nvinfer1::ILoggerFinder* finder);

    //! Get the logger.
    nvinfer1::ILogger* findLogger() override;

    static LoggerFinder& getInstance() noexcept;

private:
    LoggerFinder() = default;

    nvinfer1::ILoggerFinder* mLoggerFinder{nullptr};
    std::mutex mMutex;
};

} // namespace bitfusion::plugins::api

extern "C"
{
    // This function is used for explicitly registering the TRT-LLM plugins and the default logger.
    bool initTrtLlmPlugins(void* logger, const char* libNamespace = bitfusion::plugins::api::kDefaultNamespace);

    // The functions below are used by TensorRT to when loading a shared plugin library with automatic registering.
    // see https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#generating-plugin-library
    TENSORRTAPI [[maybe_unused]] void setLoggerFinder([[maybe_unused]] nvinfer1::ILoggerFinder* finder);
    TENSORRTAPI [[maybe_unused]] nvinfer1::IPluginCreator* const* getPluginCreators(int32_t& nbCreators);
}
