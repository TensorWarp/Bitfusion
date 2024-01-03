
#include <gtest/gtest.h>

#include "../../common/stringUtils.h"

#include <sstream>

using namespace bitfusion::common;

namespace
{

auto constexpr SHORT_STRING = "Let's format this string 5 times.";

inline std::string formatShort()
{
    return fmtstr("Let's %s this string %d times.", "format", 5);
}

auto const LONG_STRING = std::string(10000, '?');
auto constexpr LONG_PREFIX = "add me";

std::string formatLong()
{
    return fmtstr("add me%s", LONG_STRING.c_str());
}

std::ostringstream priceFormatStream;

template <typename P>
std::string formatFixed(P price)
{
    priceFormatStream.str("");
    priceFormatStream.clear();
    priceFormatStream << price;
    return priceFormatStream.str();
}

}

TEST(StringUtil, ShortStringFormat)
{
    EXPECT_EQ(SHORT_STRING, formatShort());
}

TEST(StringUtil, LongStringFormat)
{
    EXPECT_EQ(LONG_PREFIX + LONG_STRING, formatLong());
}

TEST(StringUtil, FormatFixedDecimals)
{
    auto num = 0.123456789;

    for (auto d = 1; d <= 9; ++d)
    {
        auto const fmt = std::string("%.") + std::to_string(d) + "f";
        auto prefix = fmtstr(fmt.c_str(), num);
        priceFormatStream.precision(d);
        EXPECT_EQ(prefix, formatFixed(num));
        EXPECT_EQ(prefix, formatFixed(num));
        EXPECT_EQ(prefix, formatFixed(num));
    }
}
