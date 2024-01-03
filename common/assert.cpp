#include "assert.h"

bool CHECK_DEBUG_ENABLED = false;

namespace
{

#if !defined(_MSC_VER)
    __attribute__((constructor))
#endif
        void initOnLoad()
    {
        auto constexpr kDebugEnabled = "TRT_LLM_DEBUG_MODE";
        auto const debugEnabled = std::getenv(kDebugEnabled);
        if (debugEnabled && debugEnabled[0] == '1')
        {
            CHECK_DEBUG_ENABLED = true;
        }
    }
}