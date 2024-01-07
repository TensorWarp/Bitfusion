#pragma once

#include "stringUtils.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#define NEW_EXCEPTION(...)                                                                                        \
    bitfusion::common::Exception(__FILE__, __LINE__, bitfusion::common::fmtstr(__VA_ARGS__))

namespace bitfusion::common
{

    class Exception : public std::runtime_error
    {
    public:
        static auto constexpr MAX_FRAMES = 128;

        explicit Exception(char const* file, std::size_t line, std::string const& msg);

        ~Exception() noexcept override;

        [[nodiscard]] std::string getTrace() const;

        static std::string demangle(char const* name);

    private:
        std::array<void*, MAX_FRAMES> mCallstack{};
        int mNbFrames;
    };

}