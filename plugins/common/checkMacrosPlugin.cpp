
#include "checkMacrosPlugin.h"

#include "../../common/logger.h"

namespace bitfusion::plugins
{

void caughtError(const std::exception& e)
{
    TLLM_LOG_EXCEPTION(e);
}

void logError(const char* msg, const char* file, const char* fn, int line)
{
    TLLM_LOG_ERROR("Parameter check failed at: %s::%s::%d, condition: %s", file, fn, line, msg);
}

}
