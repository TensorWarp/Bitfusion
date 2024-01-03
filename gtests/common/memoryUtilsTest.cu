#include "../../common/memoryUtils.h"
#include <array>

#include <gtest/gtest.h>

namespace tc = bitfusion::common;

TEST(memory_utils, flat_index)
{
    std::array<int, 4> constexpr dims{4, 3, 2, 1};
    static_assert(tc::flat_index(dims, 0) == 0);
    static_assert(tc::flat_index(dims, 1) == 1);
    static_assert(tc::flat_index(dims, 2) == 2);
    static_assert(tc::flat_index(dims, 3) == 3);
    static_assert(tc::flat_index(dims, 0, 0) == 0);
    static_assert(tc::flat_index(dims, 0, 1) == 1);
    static_assert(tc::flat_index(dims, 1, 0) == 3);
    static_assert(tc::flat_index(dims, 1, 1) == 4);
    static_assert(tc::flat_index(dims, 1, 2) == 5);
    static_assert(tc::flat_index(dims, 2, 0) == 6);
    static_assert(tc::flat_index(dims, 2, 1) == 7);
    static_assert(tc::flat_index(dims, 2, 2) == 8);
    static_assert(tc::flat_index(dims, 0, 0, 0) == 0);
    static_assert(tc::flat_index(dims, 0, 0, 1) == 1);
    static_assert(tc::flat_index(dims, 0, 1, 0) == 2);
    static_assert(tc::flat_index(dims, 0, 1, 1) == 3);
    static_assert(tc::flat_index(dims, 0, 2, 0) == 4);
    static_assert(tc::flat_index(dims, 0, 2, 1) == 5);
    static_assert(tc::flat_index(dims, 1, 0, 0) == 6);
    static_assert(tc::flat_index(dims, 1, 0, 1) == 7);
    static_assert(tc::flat_index(dims, 1, 1, 0) == 8);
    static_assert(tc::flat_index(dims, 1, 1, 1) == 9);
    static_assert(tc::flat_index(dims, 1, 2, 0) == 10);
    static_assert(tc::flat_index(dims, 1, 2, 1) == 11);
    static_assert(tc::flat_index(0, dims, 1, 2, 1) == 11);
    static_assert(tc::flat_index(1, dims, 1, 2, 1) == 35);
    static_assert(tc::flat_index(2, dims, 1, 2, 1) == 59);
    static_assert(tc::flat_index(dims, 0, 0, 0) == tc::flat_index(0, &dims[1], 0, 0));
    static_assert(tc::flat_index(dims, 0, 0, 1) == tc::flat_index(0, &dims[1], 0, 1));
    static_assert(tc::flat_index(dims, 0, 1, 0) == tc::flat_index(0, &dims[1], 1, 0));
    static_assert(tc::flat_index(dims, 0, 1, 1) == tc::flat_index(0, &dims[1], 1, 1));
    static_assert(tc::flat_index(dims, 0, 2, 0) == tc::flat_index(0, &dims[1], 2, 0));
    static_assert(tc::flat_index(dims, 0, 2, 1) == tc::flat_index(0, &dims[1], 2, 1));
    static_assert(tc::flat_index(dims, 1, 0, 0) == tc::flat_index(1, &dims[1], 0, 0));
    static_assert(tc::flat_index(dims, 1, 0, 1) == tc::flat_index(1, &dims[1], 0, 1));
    static_assert(tc::flat_index(dims, 1, 1, 0) == tc::flat_index(1, &dims[1], 1, 0));
    static_assert(tc::flat_index(dims, 1, 1, 1) == tc::flat_index(1, &dims[1], 1, 1));
    static_assert(tc::flat_index(dims, 1, 2, 0) == tc::flat_index(1, &dims[1], 2, 0));
    static_assert(tc::flat_index(dims, 1, 2, 1) == tc::flat_index(1, &dims[1], 2, 1));

    static_assert(tc::flat_index({4, 3, 2, 1}, 1, 2, 1) == 11);
    static_assert(tc::flat_index({4, 3, 2}, 1, 2, 1) == 11);
    static_assert(tc::flat_index(0, {4, 3, 2, 1}, 1, 2, 1) == 11);
    static_assert(tc::flat_index(1, {4, 3, 2, 1}, 1, 2, 1) == 35);

    int constexpr dim_array[]{4, 3, 2, 1};
    static_assert(tc::flat_index(dim_array, 1, 2, 1) == 11);
    static_assert(tc::flat_index(dim_array, 1, 2, 1) == 11);
    static_assert(tc::flat_index(0, dim_array, 1, 2, 1) == 11);
    static_assert(tc::flat_index(1, dim_array, 1, 2, 1) == 35);

    static_assert(tc::flat_index<1>(dims, 2, 1, 0) == 5);
    static_assert(tc::flat_index<1>(0, dims, 2, 1, 0) == 5);
    static_assert(tc::flat_index<1>(1, dims, 2, 1, 0) == 11);
    static_assert(tc::flat_index<1>({4, 3, 2, 1}, 2, 1, 0) == 5);
    static_assert(tc::flat_index<1>(0, {4, 3, 2, 1}, 2, 1, 0) == 5);
    static_assert(tc::flat_index<1>(1, {4, 3, 2, 1}, 2, 1, 0) == 11);
}

TEST(memory_utils, flat_index_dim)
{
    std::array<int, 4> constexpr dims{4, 3, 2, 1};
    static_assert(tc::flat_index(dims, 0, 0) == tc::flat_index2(0, 0, dims[1]));
    static_assert(tc::flat_index(dims, 0, 1) == tc::flat_index2(0, 1, dims[1]));
    static_assert(tc::flat_index(dims, 1, 0) == tc::flat_index2(1, 0, dims[1]));
    static_assert(tc::flat_index(dims, 1, 1) == tc::flat_index2(1, 1, dims[1]));
    static_assert(tc::flat_index(dims, 1, 2) == tc::flat_index2(1, 2, dims[1]));
    static_assert(tc::flat_index(dims, 2, 0) == tc::flat_index2(2, 0, dims[1]));
    static_assert(tc::flat_index(dims, 2, 1) == tc::flat_index2(2, 1, dims[1]));
    static_assert(tc::flat_index(dims, 2, 2) == tc::flat_index2(2, 2, dims[1]));
    static_assert(tc::flat_index(dims, 0, 0, 0) == tc::flat_index3(0, 0, 0, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 0, 0, 1) == tc::flat_index3(0, 0, 1, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 0, 1, 0) == tc::flat_index3(0, 1, 0, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 0, 1, 1) == tc::flat_index3(0, 1, 1, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 0, 2, 0) == tc::flat_index3(0, 2, 0, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 0, 2, 1) == tc::flat_index3(0, 2, 1, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 1, 0, 0) == tc::flat_index3(1, 0, 0, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 1, 0, 1) == tc::flat_index3(1, 0, 1, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 1, 1, 0) == tc::flat_index3(1, 1, 0, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 1, 1, 1) == tc::flat_index3(1, 1, 1, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 1, 2, 0) == tc::flat_index3(1, 2, 0, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 1, 2, 1) == tc::flat_index3(1, 2, 1, dims[1], dims[2]));
    static_assert(tc::flat_index(dims, 1, 1, 0, 0) == tc::flat_index4(1, 1, 0, 0, dims[1], dims[2], dims[3]));
    static_assert(tc::flat_index(dims, 1, 1, 1, 0) == tc::flat_index4(1, 1, 1, 0, dims[1], dims[2], dims[3]));
    static_assert(tc::flat_index(dims, 1, 2, 0, 0) == tc::flat_index4(1, 2, 0, 0, dims[1], dims[2], dims[3]));
    static_assert(tc::flat_index(dims, 1, 2, 1, 0) == tc::flat_index4(1, 2, 1, 0, dims[1], dims[2], dims[3]));
}

TEST(memory_utils, flat_index_strided)
{
    std::array<int, 4> constexpr dims{4, 3, 2, 1};
    auto constexpr dim_12 = dims[1] * dims[2];
    static_assert(tc::flat_index_strided3(0, 0, 0, dim_12, dims[2]) == tc::flat_index3(0, 0, 0, dims[1], dims[2]));
    static_assert(tc::flat_index_strided3(0, 0, 1, dim_12, dims[2]) == tc::flat_index3(0, 0, 1, dims[1], dims[2]));
    static_assert(tc::flat_index_strided3(0, 1, 0, dim_12, dims[2]) == tc::flat_index3(0, 1, 0, dims[1], dims[2]));
    static_assert(tc::flat_index_strided3(0, 1, 1, dim_12, dims[2]) == tc::flat_index3(0, 1, 1, dims[1], dims[2]));
    static_assert(tc::flat_index_strided3(0, 2, 0, dim_12, dims[2]) == tc::flat_index3(0, 2, 0, dims[1], dims[2]));
    static_assert(tc::flat_index_strided3(0, 2, 1, dim_12, dims[2]) == tc::flat_index3(0, 2, 1, dims[1], dims[2]));
    static_assert(tc::flat_index_strided3(1, 0, 0, dim_12, dims[2]) == tc::flat_index3(1, 0, 0, dims[1], dims[2]));
    static_assert(tc::flat_index_strided3(1, 0, 1, dim_12, dims[2]) == tc::flat_index3(1, 0, 1, dims[1], dims[2]));
    static_assert(tc::flat_index_strided3(1, 1, 0, dim_12, dims[2]) == tc::flat_index3(1, 1, 0, dims[1], dims[2]));
    static_assert(tc::flat_index_strided3(1, 1, 1, dim_12, dims[2]) == tc::flat_index3(1, 1, 1, dims[1], dims[2]));
    static_assert(tc::flat_index_strided3(1, 2, 0, dim_12, dims[2]) == tc::flat_index3(1, 2, 0, dims[1], dims[2]));
    static_assert(tc::flat_index_strided3(1, 2, 1, dim_12, dims[2]) == tc::flat_index3(1, 2, 1, dims[1], dims[2]));

    auto constexpr dims_123 = dims[1] * dims[2] * dims[3];
    auto constexpr dims_23 = dims[2] * dims[3];
    static_assert(tc::flat_index_strided4(1, 1, 0, 0, dims_123, dims_23, dims[3])
        == tc::flat_index4(1, 1, 0, 0, dims[1], dims[2], dims[3]));
    static_assert(tc::flat_index_strided4(1, 1, 1, 0, dims_123, dims_23, dims[3])
        == tc::flat_index4(1, 1, 1, 0, dims[1], dims[2], dims[3]));
    static_assert(tc::flat_index_strided4(1, 2, 1, 0, dims_123, dims_23, dims[3])
        == tc::flat_index4(1, 2, 1, 0, dims[1], dims[2], dims[3]));
    static_assert(tc::flat_index_strided4(3, 2, 1, 0, dims_123, dims_23, dims[3])
        == tc::flat_index4(3, 2, 1, 0, dims[1], dims[2], dims[3]));
}
