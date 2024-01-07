#include "stringUtils.h"
#include "assert.h"

#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <string>

namespace bitfusion::common
{

    namespace
    {
        std::string vformat(char const* fmt, va_list args)
        {
            va_list args0;
            va_copy(args0, args);
            auto const size = vsnprintf(nullptr, 0, fmt, args0);
            if (size <= 0)
                return "";

            std::string stringBuf(size, char{});
            auto const size2 = std::vsnprintf(&stringBuf[0], size + 1, fmt, args);

            CHECK_WITH_INFO(size2 == size, std::string(std::strerror(errno)));

            return stringBuf;
        }

    }

    std::string fmtstr(char const* format, ...)
    {
        va_list args;
        va_start(args, format);
        std::string result = vformat(format, args);
        va_end(args);
        return result;
    };

}