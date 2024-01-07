#include "Exception.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _MSC_VER
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#endif

namespace bitfusion::common
{
    namespace
    {
        constexpr int VOID_PTR_SZ = 2 + sizeof(void*) * 2;
        constexpr int MAX_FRAMES = 64;
    }

    Exception::Exception(const char* file, std::size_t line, const std::string& msg)
        : std::runtime_error{ msg + " (" + file + ":" + std::to_string(line) + ")" }
    {
#ifndef _MSC_VER
        mNbFrames = backtrace(mCallstack.data(), MAX_FRAMES);
#endif
    }

    Exception::~Exception() noexcept = default;

#ifndef _MSC_VER
    std::string Exception::getTrace() const
    {
        auto const trace = backtrace_symbols(mCallstack.data(), mNbFrames);
        std::ostringstream buf;
        for (auto i = 1; i < mNbFrames; ++i)
        {
            Dl_info info;
            if (dladdr(mCallstack[i], &info) && info.dli_sname)
            {
                auto const clearName = demangle(info.dli_sname);
                buf << fmtstr("%-3d %*p %s + %zd", i, VOID_PTR_SZ, mCallstack[i], clearName.c_str(),
                    static_cast<char*>(mCallstack[i]) - static_cast<char*>(info.dli_saddr));
            }
            else
            {
                buf << fmtstr("%-3d %*p %s", i, VOID_PTR_SZ, mCallstack[i], trace[i]);
            }
            if (i < mNbFrames - 1)
                buf << '\n';
        }

        if (mNbFrames == MAX_FRAMES)
            buf << "\n[truncated]";

        std::free(trace);
        return buf.str();
    }

    std::string Exception::demangle(const char* name)
    {
        std::string clearName{ name };
        int status = -1;
        auto const demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
        if (status == 0)
        {
            clearName = demangled;
            std::free(demangled);
        }
        return clearName;
    }
#else
    std::string Exception::getTrace() const
    {
        return "";
    }

    std::string Exception::demangle(const char* name)
    {
        return name;
    }
#endif
}
