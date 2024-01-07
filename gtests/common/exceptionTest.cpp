
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "../../common/assert.h"
#include "../../common/logger.h"
#include "../../common/exception.h"

#include <string>

using ::testing::HasSubstr;

TEST(TllmException, StackTrace)
{
    auto ex = NEW_EXCEPTION("TestException %d", 1);
    std::string const what{ex.what()};
    EXPECT_THAT(what, HasSubstr(std::to_string(__LINE__ - 2)));
    EXPECT_THAT(what, HasSubstr("TestException 1"));
    EXPECT_THAT(what, HasSubstr(__FILE__));
#if !defined(_MSC_VER)
    EXPECT_THAT(what, ::testing::Not(HasSubstr("bitfusion::common::TllmException::TllmException")));
    EXPECT_THAT(what, HasSubstr("tests/tllmExceptionTest"));
    EXPECT_THAT(what, HasSubstr("main"));
#endif
}

TEST(Exception, Logger)
{
    try
    {
        TLLM_THROW("TestException %d", 1);
    }
    catch (const std::exception& e)
    {
        testing::internal::CaptureStdout();
        TLLM_LOG_EXCEPTION(e);
        auto const out = testing::internal::GetCapturedStdout();
        EXPECT_THAT(out, HasSubstr(std::to_string(__LINE__ - 7)));
        EXPECT_THAT(out, HasSubstr("TestException 1"));
        EXPECT_THAT(out, HasSubstr(__FILE__));
#if !defined(_MSC_VER)
        EXPECT_THAT(out, ::testing::Not(HasSubstr("bitfusion::common::TllmException::TllmException")));
        EXPECT_THAT(out, HasSubstr("tests/tllmExceptionTest"));
        EXPECT_THAT(out, HasSubstr("main"));
#endif
    }
}
