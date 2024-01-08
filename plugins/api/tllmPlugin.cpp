
#include "tllmPlugin.h"

#include "../../common/stringUtils.h"
#include "../../runtime/Logger.h"

#include "../../plugins/bertAttentionPlugin/bertAttentionPlugin.h"
#include "../../plugins/gemmPlugin/gemmPlugin.h"
#include "../../plugins/gptAttentionPlugin/gptAttentionPlugin.h"
#include "../../plugins/identityPlugin/identityPlugin.h"
#include "../../plugins/layernormPlugin/layernormPlugin.h"
#include "../../plugins/layernormQuantizationPlugin/layernormQuantizationPlugin.h"
#include "../../plugins/lookupPlugin/lookupPlugin.h"
#include "../../plugins/loraPlugin/loraPlugin.h"
#include "../../plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#if ENABLE_MULTI_DEVICE
#include "../../plugins/ncclPlugin/allgatherPlugin.h"
#include "../../plugins/ncclPlugin/allreducePlugin.h"
#include "../../plugins/ncclPlugin/recvPlugin.h"
#include "../../plugins/ncclPlugin/reduceScatterPlugin.h"
#include "../../plugins/ncclPlugin/sendPlugin.h"
#endif // ENABLE_MULTI_DEVICE
#include "../../plugins/quantizePerTokenPlugin/quantizePerTokenPlugin.h"
#include "../../plugins/quantizeTensorPlugin/quantizeTensorPlugin.h"
#include "../../plugins/rmsnormPlugin/rmsnormPlugin.h"
#include "../../plugins/rmsnormQuantizationPlugin/rmsnormQuantizationPlugin.h"
#include "../../plugins/smoothQuantGemmPlugin/smoothQuantGemmPlugin.h"
#include "../../plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.h"
#include "../../plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"

#include <array>
#include <cstdlib>

#include <NvInferRuntime.h>
#include "../identityPlugin/identityPlugin.h"
#include "../layernormPlugin/layernormPlugin.h"
#include "../layernormQuantizationPlugin/layernormQuantizationPlugin.h"
#include "../lookupPlugin/lookupPlugin.h"
#include "../mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "../loraPlugin/loraPlugin.h"

namespace tc = bitfusion::common;

namespace
{

nvinfer1::IPluginCreator* creatorPtr(nvinfer1::IPluginCreator& creator)
{
    return &creator;
}

auto Logger = bitfusion::runtime::Logger();

nvinfer1::ILogger* gLogger{&Logger};

class GlobalLoggerFinder : public nvinfer1::ILoggerFinder
{
public:
    nvinfer1::ILogger* findLogger() override
    {
        return gLogger;
    }
};

GlobalLoggerFinder gGlobalLoggerFinder{};

#if !defined(_MSC_VER)
__attribute__((constructor))
#endif
void initOnLoad()
{
    auto constexpr kLoadPlugins = "TRT_LLM_LOAD_PLUGINS";
    auto const loadPlugins = std::getenv(kLoadPlugins);
    if (loadPlugins && loadPlugins[0] == '1')
    {
        initTrtLlmPlugins(gLogger);
    }
}

bool pluginsInitialized = false;

} // namespace

// New Plugin APIs

extern "C"
{
    bool initTrtLlmPlugins(void* logger, const char* libNamespace)
    {
        if (pluginsInitialized)
            return true;

        if (logger)
        {
            gLogger = static_cast<nvinfer1::ILogger*>(logger);
        }
        setLoggerFinder(&gGlobalLoggerFinder);

        auto registry = getPluginRegistry();
        std::int32_t nbCreators;
        auto creators = getPluginCreators(nbCreators);

        for (std::int32_t i = 0; i < nbCreators; ++i)
        {
            auto const creator = creators[i];
            creator->setPluginNamespace(libNamespace);
            registry->registerCreator(*creator, libNamespace);
            if (gLogger)
            {
                auto const msg = tc::fmtstr("Registered plugin creator %s version %s in namespace %s",
                    creator->getPluginName(), creator->getPluginVersion(), libNamespace);
                gLogger->log(nvinfer1::ILogger::Severity::kVERBOSE, msg.c_str());
            }
        }

        pluginsInitialized = true;
        return true;
    }

    [[maybe_unused]] void setLoggerFinder([[maybe_unused]] nvinfer1::ILoggerFinder* finder)
    {
        bitfusion::plugins::api::LoggerFinder::getInstance().setLoggerFinder(finder);
    }

    [[maybe_unused]] nvinfer1::IPluginCreator* const* getPluginCreators(std::int32_t& nbCreators)
    {
        static bitfusion::plugins::IdentityPluginCreator identityPluginCreator;
        static bitfusion::plugins::BertAttentionPluginCreator bertAttentionPluginCreator;
        static bitfusion::plugins::GPTAttentionPluginCreator gptAttentionPluginCreator;
        static bitfusion::plugins::GemmPluginCreator gemmPluginCreator;
        static bitfusion::plugins::MixtureOfExpertsPluginCreator moePluginCreator;
#if ENABLE_MULTI_DEVICE
        static bitfusion::plugins::SendPluginCreator sendPluginCreator;
        static bitfusion::plugins::RecvPluginCreator recvPluginCreator;
        static bitfusion::plugins::AllreducePluginCreator allreducePluginCreator;
        static bitfusion::plugins::AllgatherPluginCreator allgatherPluginCreator;
        static bitfusion::plugins::ReduceScatterPluginCreator reduceScatterPluginCreator;
#endif // ENABLE_MULTI_DEVICE
        static bitfusion::plugins::LayernormPluginCreator layernormPluginCreator;
        static bitfusion::plugins::RmsnormPluginCreator rmsnormPluginCreator;
        static bitfusion::plugins::SmoothQuantGemmPluginCreator smoothQuantGemmPluginCreator;
        static bitfusion::plugins::LayernormQuantizationPluginCreator layernormQuantizationPluginCreator;
        static bitfusion::plugins::QuantizePerTokenPluginCreator quantizePerTokenPluginCreator;
        static bitfusion::plugins::QuantizeTensorPluginCreator quantizeTensorPluginCreator;
        static bitfusion::plugins::RmsnormQuantizationPluginCreator rmsnormQuantizationPluginCreator;
        static bitfusion::plugins::WeightOnlyGroupwiseQuantMatmulPluginCreator
            weightOnlyGroupwiseQuantMatmulPluginCreator;
        static bitfusion::plugins::WeightOnlyQuantMatmulPluginCreator weightOnlyQuantMatmulPluginCreator;
        static bitfusion::plugins::LookupPluginCreator lookupPluginCreator;
        static bitfusion::plugins::LoraPluginCreator loraPluginCreator;

        static std::array pluginCreators
            = { creatorPtr(identityPluginCreator),
                  creatorPtr(bertAttentionPluginCreator),
                  creatorPtr(gptAttentionPluginCreator),
                  creatorPtr(gemmPluginCreator),
                  creatorPtr(moePluginCreator),
#if ENABLE_MULTI_DEVICE
                  creatorPtr(sendPluginCreator),
                  creatorPtr(recvPluginCreator),
                  creatorPtr(allreducePluginCreator),
                  creatorPtr(allgatherPluginCreator),
                  creatorPtr(reduceScatterPluginCreator),
#endif // ENABLE_MULTI_DEVICE
                  creatorPtr(layernormPluginCreator),
                  creatorPtr(rmsnormPluginCreator),
                  creatorPtr(smoothQuantGemmPluginCreator),
                  creatorPtr(layernormQuantizationPluginCreator),
                  creatorPtr(quantizePerTokenPluginCreator),
                  creatorPtr(quantizeTensorPluginCreator),
                  creatorPtr(rmsnormQuantizationPluginCreator),
                  creatorPtr(weightOnlyGroupwiseQuantMatmulPluginCreator),
                  creatorPtr(weightOnlyQuantMatmulPluginCreator),
                  creatorPtr(lookupPluginCreator),
                  creatorPtr(loraPluginCreator),
              };
        nbCreators = pluginCreators.size();
        return pluginCreators.data();
    }

} // extern "C"

namespace bitfusion::plugins::api
{
LoggerFinder& bitfusion::plugins::api::LoggerFinder::getInstance() noexcept
{
    static LoggerFinder instance;
    return instance;
}

void LoggerFinder::setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder == nullptr && finder != nullptr)
    {
        mLoggerFinder = finder;
    }
}

nvinfer1::ILogger* LoggerFinder::findLogger()
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder != nullptr)
    {
        return mLoggerFinder->findLogger();
    }
    return nullptr;
}
} // namespace bitfusion::plugins::api
