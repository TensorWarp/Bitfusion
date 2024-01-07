
#pragma once

#include "stringUtils.h"
#include "exception.h"

#include <string>

namespace bitfusion::common
{
[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "")
{
    throw Exception(file, line, fmtstr("[TensorRT-LLM][ERROR] Assertion failed: %s", info.c_str()));
}

}

extern bool CHECK_DEBUG_ENABLED;

#if defined(_WIN32)
#define TLLM_LIKELY(x) (__assume((x) == 1), (x))
#else
#define TLLM_LIKELY(x) __builtin_expect((x), 1)
#endif

#define TLLM_CHECK(val)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        TLLM_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                               \
                                            : bitfusion::common::throwRuntimeError(__FILE__, __LINE__, #val);       \
    } while (0)

#define TLLM_CHECK_WITH_INFO(val, info, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        TLLM_LIKELY(static_cast<bool>(val))                                                                            \
        ? ((void) 0)                                                                                                   \
        : bitfusion::common::throwRuntimeError(                                                                     \
            __FILE__, __LINE__, bitfusion::common::fmtstr(info, ##__VA_ARGS__));                                    \
    } while (0)

#define TLLM_CHECK_DEBUG(val)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (CHECK_DEBUG_ENABLED)                                                                                       \
        {                                                                                                              \
            TLLM_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                           \
                                                : bitfusion::common::throwRuntimeError(__FILE__, __LINE__, #val);   \
        }                                                                                                              \
    } while (0)

#define TLLM_CHECK_DEBUG_WITH_INFO(val, info)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (CHECK_DEBUG_ENABLED)                                                                                       \
        {                                                                                                              \
            TLLM_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                           \
                                                : bitfusion::common::throwRuntimeError(__FILE__, __LINE__, info);   \
        }                                                                                                              \
    } while (0)

#define TLLM_THROW(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        throw NEW_TLLM_EXCEPTION(__VA_ARGS__);                                                                         \
    } while (0)

#define TLLM_WRAP(ex)                                                                                                  \
    NEW_EXCEPTION("%s: %s", bitfusion::common::TllmException::demangle(typeid(ex).name()).c_str(), ex.what())
