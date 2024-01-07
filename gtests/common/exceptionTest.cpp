#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include <stdexcept>
#include <iostream>
#include "../../common/logger.h"

using ::testing::HasSubstr;

/// <summary>
/// Test case for StackTrace function.
/// </summary>
TEST(Exception, StackTrace)
{
    try
    {
        throw std::runtime_error("TestException 1");
    }
    catch (const std::exception& e)
    {
        const std::string what = e.what();
        EXPECT_THAT(what, HasSubstr(std::to_string(__LINE__ - 6)));
        EXPECT_THAT(what, HasSubstr("TestException 1"));
        EXPECT_THAT(what, HasSubstr(__FILE__));
#if !defined(_MSC_VER)
        EXPECT_THAT(what, ::testing::Not(HasSubstr("bitfusion::common::Exception::Exception")));
        EXPECT_THAT(what, HasSubstr("tests/tllmExceptionTest"));
        EXPECT_THAT(what, HasSubstr("main"));
#endif
    }
}

/// <summary>
/// Test case for Logger function.
/// </summary>
TEST(Exception, Logger)
{
    try
    {
        throw std::runtime_error("TestException 1");
    }
    catch (const std::exception& e)
    {
        testing::internal::CaptureStdout();
        LOG_EXCEPTION(e);
        auto const out = testing::internal::GetCapturedStdout();
        EXPECT_THAT(out, HasSubstr(std::to_string(__LINE__ - 8)));
        EXPECT_THAT(out, HasSubstr("TestException 1"));
        EXPECT_THAT(out, HasSubstr(__FILE__));
#if !defined(_MSC_VER)
        EXPECT_THAT(out, ::testing::Not(HasSubstr("bitfusion::common::Exception::Exception")));
        EXPECT_THAT(out, HasSubstr("tests/tllmExceptionTest"));
        EXPECT_THAT(out, HasSubstr("main"));
#endif
    }
}
