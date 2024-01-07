
#pragma once

#include "stringUtils.h"
#include "exception.h"

#include <string>

namespace bitfusion::common
{
[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "")
{
    throw Exception(file, line, fmtstr("[ERROR] Assertion failed: %s", info.c_str()));
}

}

extern bool CHECK_DEBUG_ENABLED;

#if defined(_WIN32)
#define LIKELY(x) (__assume((x) == 1), (x))
#else
#define LIKELY(x) __builtin_expect((x), 1)
#endif

#define CHECK(val)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                               \
                                            : bitfusion::common::throwRuntimeError(__FILE__, __LINE__, #val);       \
    } while (0)

#define CHECK_WITH_INFO(val, info, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        LIKELY(static_cast<bool>(val))                                                                            \
        ? ((void) 0)                                                                                                   \
        : bitfusion::common::throwRuntimeError(                                                                     \
            __FILE__, __LINE__, bitfusion::common::fmtstr(info, ##__VA_ARGS__));                                    \
    } while (0)

#define CHECK_DEBUG(val)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (CHECK_DEBUG_ENABLED)                                                                                       \
        {                                                                                                              \
            LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                           \
                                                : bitfusion::common::throwRuntimeError(__FILE__, __LINE__, #val);   \
        }                                                                                                              \
    } while (0)

#define CHECK_DEBUG_WITH_INFO(val, info)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (CHECK_DEBUG_ENABLED)                                                                                       \
        {                                                                                                              \
            LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                           \
                                                : bitfusion::common::throwRuntimeError(__FILE__, __LINE__, info);   \
        }                                                                                                              \
    } while (0)

#define THROW(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        throw NEW_EXCEPTION(__VA_ARGS__);                                                                         \
    } while (0)

#define TLLM_WRAP(ex)                                                                                                  \
    NEW_EXCEPTION("%s: %s", bitfusion::common::TllmException::demangle(typeid(ex).name()).c_str(), ex.what())
