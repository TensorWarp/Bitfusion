#include <gtest/gtest.h>
#include "../../common/stringUtils.h"
#include <sstream>
#include <string>
#include <iomanip>

using namespace bitfusion::common;

namespace {

    ///<summary>
    /// A short string used for formatting.
    ///</summary>
    constexpr auto SHORT_STRING = "Let's format this string 5 times.";

    ///<summary>
    /// Format a short string.
    ///</summary>
    ///<returns>The formatted short string.</returns>
    inline std::string formatShort() {
        return fmtstr("Let's %s this string %d times.", "format", 5);
    }

    ///<summary>
    /// A long string used for formatting.
    ///</summary>
    auto const LONG_STRING = std::string(10000, '?');
    ///<summary>
    /// Prefix for the long string.
    ///</summary>
    auto constexpr LONG_PREFIX = "add me";

    ///<summary>
    /// Format a long string.
    ///</summary>
    ///<returns>The formatted long string.</returns>
    std::string formatLong() {
        return LONG_PREFIX + LONG_STRING;
    }

    std::ostringstream priceFormatStream;

    ///<summary>
    /// Format a price with a given precision.
    ///</summary>
    ///<param name="price">The price to format.</param>
    ///<param name="precision">The precision for formatting.</param>
    ///<returns>The formatted price as a string.</returns>
    template <typename P>
    std::string formatFixed(P price, int precision) {
        priceFormatStream.str("");
        priceFormatStream << std::fixed << std::setprecision(precision) << price;
        return priceFormatStream.str();
    }

} // namespace

///<summary>
/// Test case for formatting a short string.
///</summary>
TEST(StringUtil, ShortStringFormat) {
    EXPECT_EQ(SHORT_STRING, formatShort());
}

///<summary>
/// Test case for formatting a long string.
///</summary>
TEST(StringUtil, LongStringFormat) {
    EXPECT_EQ(LONG_PREFIX + LONG_STRING, formatLong());
}

///<summary>
/// Test case for formatting a price with different precisions.
///</summary>
TEST(StringUtil, FormatFixedDecimals) {
    constexpr auto num = 0.123456789;

    for (auto d = 1; d <= 9; ++d) {
        auto const fmt = fmtstr("%%.%df", d);
        auto prefix = fmtstr(fmt.c_str(), num);
        EXPECT_EQ(prefix, formatFixed(num, d));
        EXPECT_EQ(prefix, formatFixed(num, d));
        EXPECT_EQ(prefix, formatFixed(num, d));
    }
}