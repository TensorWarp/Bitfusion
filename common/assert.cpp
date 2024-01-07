#include "assert.h"

bool CHECK_DEBUG_ENABLED = false;

namespace
{

#if !defined(_MSC_VER)
    /// <summary>
    /// Constructor function that initializes the CHECK_DEBUG_ENABLED flag.
    /// </summary>
    __attribute__((constructor))
#endif
        void initOnLoad()
    {
        auto constexpr kDebugEnabled = "DEBUG_MODE";
        auto const debugEnabled = std::getenv(kDebugEnabled);

        /// <summary>
        /// Check if the DEBUG_MODE environment variable is set to '1'
        /// and set CHECK_DEBUG_ENABLED to true if it is.
        /// </summary>
        if (debugEnabled && debugEnabled[0] == '1')
        {
            CHECK_DEBUG_ENABLED = true;
        }
    }
}