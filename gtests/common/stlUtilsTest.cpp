
#include <gtest/gtest.h>

#include <functional>
#include <list>

#include "../../common/stlUtils.h"

TEST(StlUtils, ExclusiveScan)
{
    std::list<int> data{3, 1, 4, 1, 5}, l;
    auto it = bitfusion::common::stl_utils::exclusiveScan(
        data.begin(), data.end(), std::insert_iterator<std::list<int>>(l, std::next(l.begin())), 0);
    bitfusion::common::stl_utils::basicExclusiveScan(data.begin(), data.end(), it, 1, std::multiplies<>{});
    EXPECT_EQ(l, (std::list<int>{0, 3, 4, 8, 9, 1, 3, 3, 12, 12}));
}

TEST(StlUtils, InclusiveScan)
{
    std::list<int> data{3, 1, 4, 1, 5}, l;
    auto it = bitfusion::common::stl_utils::inclusiveScan(
        data.begin(), data.end(), std::insert_iterator<std::list<int>>(l, std::next(l.begin())));
    bitfusion::common::stl_utils::basicInclusiveScan(data.begin(), data.end(), it, std::multiplies<>{});
    EXPECT_EQ(l, (std::list<int>{3, 4, 8, 9, 14, 3, 3, 12, 12, 60}));
}
