#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdarg>
#include <cstdio>

namespace bitfusion::common
{
    namespace
    {
        /// <summary>
        /// Format a string using a variable number of arguments, similar to sprintf.
        /// </summary>
        /// <param name="fmt">The format string.</param>
        /// <param name="args">A variable number of arguments.</param>
        /// <returns>The formatted string.</returns>
        std::string vformat(const char* fmt, va_list args)
        {
            va_list args_copy;
            va_copy(args_copy, args);
            const int size = std::vsnprintf(nullptr, 0, fmt, args_copy);
            va_end(args_copy);

            if (size <= 0)
                return "";

            std::vector<char> stringBuf(size + 1);
            const int size2 = std::vsnprintf(stringBuf.data(), size + 1, fmt, args);

            if (size2 != size)
                throw std::runtime_error(std::string("vformat error: ") + strerror(errno));

            return std::string(stringBuf.begin(), stringBuf.begin() + size);
        }
    }

    /// <summary>
    /// Format a string using a variable number of arguments, similar to sprintf.
    /// </summary>
    /// <param name="format">The format string.</param>
    /// <param name="...">A variable number of arguments.</param>
    /// <returns>The formatted string.</returns>
    std::string fmtstr(const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        std::string result = vformat(format, args);
        va_end(args);
        return result;
    }

    /// <summary>
    /// Custom macro for checking conditions with additional info.
    /// </summary>
    /// <param name="condition">The condition to check.</param>
    /// <param name="info">Additional information to include in case of failure.</param>
#define CHECK_WITH_INFO(condition, info) \
        do { \
            if (!(condition)) { \
                std::cerr << "Check failed: " #condition << std::endl; \
                std::cerr << "Additional Info: " << (info) << std::endl; \
                std::terminate(); \
            } \
        } while (false)
}
