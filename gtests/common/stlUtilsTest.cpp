#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "../../common/stlUtils.h"
#include <algorithm>

/// <summary>
/// Test case for exclusive scan using StlUtils.
/// </summary>
TEST(StlUtils, ExclusiveScan)
{
    std::vector<int> data{ 3, 1, 4, 1, 5 };
    std::vector<int> result(data.size(), 0);

    // Perform partial sum on data, starting from the second element.
    std::partial_sum(data.begin(), data.end(), result.begin() + 1);

    // Perform element-wise multiplication between data and result.
    std::transform(data.begin(), data.end(), result.begin() + 1, result.begin() + 1, std::multiplies<int>());

    // Verify the result against the expected output.
    EXPECT_EQ(result, (std::vector<int>{0, 3, 4, 8, 9, 1, 3, 3, 12, 12}));
}

/// <summary>
/// Test case for inclusive scan using StlUtils.
/// </summary>
TEST(StlUtils, InclusiveScan)
{
    std::vector<int> data{ 3, 1, 4, 1, 5 };
    std::vector<int> result(data.size());

    // Perform partial sum on data.
    std::partial_sum(data.begin(), data.end(), result.begin());

    // Perform element-wise multiplication between data and result.
    std::transform(data.begin(), data.end(), result.begin(), result.begin(), std::multiplies<int>());

    // Verify the result against the expected output.
    EXPECT_EQ(result, (std::vector<int>{3, 4, 8, 9, 14, 3, 3, 12, 12, 60}));
}