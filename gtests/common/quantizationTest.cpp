#include <gtest/gtest.h>
#include "../../common/quantization.h"
#include <memory>

using namespace bitfusion::common;

/// <summary>
/// Test suite for the Quantization class constructors.
/// </summary>
TEST(Quantization, Constructor)
{
    const auto defaultQuantMode = std::make_shared<QuantMode>();
    EXPECT_EQ(*defaultQuantMode, QuantMode::none());

    static_assert(QuantMode{} == QuantMode::none());
    static_assert(QuantMode::int4Weights().hasInt4Weights());
    static_assert(QuantMode::int8Weights().hasInt8Weights());
    static_assert(QuantMode::activations().hasActivations());
    static_assert(QuantMode::perChannelScaling().hasPerChannelScaling());
    static_assert(QuantMode::perTokenScaling().hasPerTokenScaling());
    static_assert(QuantMode::int8KvCache().hasInt8KvCache());
    static_assert(QuantMode::fp8KvCache().hasFp8KvCache());
    static_assert(QuantMode::fp8Qdq().hasFp8Qdq());
}

/// <summary>
/// Test suite for the Plus and Minus operations of the Quantization class.
/// </summary>
TEST(Quantization, PlusMinus)
{
    QuantMode quantMode;
    quantMode += QuantMode::activations() + QuantMode::perChannelScaling();
    EXPECT_TRUE(quantMode.hasActivations());
    EXPECT_TRUE(quantMode.hasPerChannelScaling());

    quantMode -= QuantMode::activations();
    EXPECT_FALSE(quantMode.hasActivations());
    EXPECT_TRUE(quantMode.hasPerChannelScaling());

    quantMode -= QuantMode::perChannelScaling();
    EXPECT_FALSE(quantMode.hasActivations());
    EXPECT_FALSE(quantMode.hasPerChannelScaling());
    EXPECT_EQ(quantMode, QuantMode::none());
}
