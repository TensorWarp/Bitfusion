#include <array>
#include <gtest/gtest.h>
#include "../../common/memoryUtils.h"

namespace common = bitfusion::common;

constexpr std::array<int, 4> dims{ 4, 3, 2, 1 };

/// <summary>
/// Tests the flat_index function for various dimensions.
/// </summary>
TEST(memory_utils, flat_index)
{
    static_assert(common::flat_index(dims, 0) == 0);
    static_assert(common::flat_index(dims, 1) == 1);
    static_assert(common::flat_index(dims, 2) == 2);
    static_assert(common::flat_index(dims, 3) == 3);
    static_assert(common::flat_index(dims, 0, 0) == 0);
    static_assert(common::flat_index(dims, 0, 1) == 1);
    static_assert(common::flat_index(dims, 1, 0) == 3);
    static_assert(common::flat_index(dims, 1, 1) == 4);
    static_assert(common::flat_index(dims, 1, 2) == 5);
    static_assert(common::flat_index(dims, 2, 0) == 6);
    static_assert(common::flat_index(dims, 2, 1) == 7);
    static_assert(common::flat_index(dims, 2, 2) == 8);
    static_assert(common::flat_index(dims, 0, 0, 0) == 0);
    static_assert(common::flat_index(dims, 0, 0, 1) == 1);
    static_assert(common::flat_index(dims, 0, 1, 0) == 2);
    static_assert(common::flat_index(dims, 0, 1, 1) == 3);
    static_assert(common::flat_index(dims, 0, 2, 0) == 4);
    static_assert(common::flat_index(dims, 0, 2, 1) == 5);
    static_assert(common::flat_index(dims, 1, 0, 0) == 6);
    static_assert(common::flat_index(dims, 1, 0, 1) == 7);
    static_assert(common::flat_index(dims, 1, 1, 0) == 8);
    static_assert(common::flat_index(dims, 1, 1, 1) == 9);
    static_assert(common::flat_index(dims, 1, 2, 0) == 10);

    static_assert(common::flat_index({ 4, 3, 2, 1 }, 1, 2, 1) == 11);
    static_assert(common::flat_index({ 4, 3, 2 }, 1, 2, 1) == 11);
    static_assert(common::flat_index(0, { 4, 3, 2, 1 }, 1, 2, 1) == 11);
    static_assert(common::flat_index(1, { 4, 3, 2, 1 }, 1, 2, 1) == 35);

    constexpr std::array<int, 4> dim_array{ 4, 3, 2, 1 };
    static_assert(common::flat_index(dim_array, 1, 2, 1) == 11);
    static_assert(common::flat_index(dim_array, 1, 2, 1) == 11);
    static_assert(common::flat_index(0, dim_array, 1, 2, 1) == 11);
    static_assert(common::flat_index(1, dim_array, 1, 2, 1) == 35);

    static_assert(common::flat_index<1>(dims, 2, 1, 0) == 5);
    static_assert(common::flat_index<1>(0, dims, 2, 1, 0) == 5);
    static_assert(common::flat_index<1>(1, dims, 2, 1, 0) == 11);
    static_assert(common::flat_index<1>({ 4, 3, 2, 1 }, 2, 1, 0) == 5);
    static_assert(common::flat_index<1>(0, { 4, 3, 2, 1 }, 2, 1, 0) == 5);
    static_assert(common::flat_index<1>(1, { 4, 3, 2, 1 }, 2, 1, 0) == 11);
}

/// <summary>
/// Tests the flat_index_dim function for various dimensions.
/// </summary>
TEST(memory_utils, flat_index_dim)
{
    static_assert(common::flat_index(dims, 0, 0) == common::flat_index2(0, 0, dims[1]));
    static_assert(common::flat_index(dims, 0, 1) == common::flat_index2(0, 1, dims[1]));
    static_assert(common::flat_index(dims, 1, 0) == common::flat_index2(1, 0, dims[1]));
    static_assert(common::flat_index(dims, 1, 1) == common::flat_index2(1, 1, dims[1]));
    static_assert(common::flat_index(dims, 1, 2) == common::flat_index2(1, 2, dims[1]));
    static_assert(common::flat_index(dims, 2, 0) == common::flat_index2(2, 0, dims[1]));
    static_assert(common::flat_index(dims, 2, 1) == common::flat_index2(2, 1, dims[1]));
    static_assert(common::flat_index(dims, 2, 2) == common::flat_index2(2, 2, dims[1]));
    static_assert(common::flat_index(dims, 0, 0, 0) == common::flat_index3(0, 0, 0, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 0, 0, 1) == common::flat_index3(0, 0, 1, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 0, 1, 0) == common::flat_index3(0, 1, 0, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 0, 1, 1) == common::flat_index3(0, 1, 1, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 0, 2, 0) == common::flat_index3(0, 2, 0, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 0, 2, 1) == common::flat_index3(0, 2, 1, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 1, 0, 0) == common::flat_index3(1, 0, 0, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 1, 0, 1) == common::flat_index3(1, 0, 1, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 1, 1, 0) == common::flat_index3(1, 1, 0, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 1, 1, 1) == common::flat_index3(1, 1, 1, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 1, 2, 0) == common::flat_index3(1, 2, 0, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 1, 2, 1) == common::flat_index3(1, 2, 1, dims[1], dims[2]));
    static_assert(common::flat_index(dims, 1, 1, 0, 0) == common::flat_index4(1, 1, 0, 0, dims[1], dims[2], dims[3]));
    static_assert(common::flat_index(dims, 1, 1, 1, 0) == common::flat_index4(1, 1, 1, 0, dims[1], dims[2], dims[3]));
    static_assert(common::flat_index(dims, 1, 2, 0, 0) == common::flat_index4(1, 2, 0, 0, dims[1], dims[2], dims[3]));
    static_assert(common::flat_index(dims, 1, 2, 1, 0) == common::flat_index4(1, 2, 1, 0, dims[1], dims[2], dims[3]));
}

/// <summary>
/// Tests the flat_index_strided function for various dimensions.
/// </summary>
TEST(memory_utils, flat_index_strided)
{
    constexpr auto dim_12 = dims[1] * dims[2];
    static_assert(common::flat_index_strided3(0, 0, 0, dim_12, dims[2]) == common::flat_index3(0, 0, 0, dims[1], dims[2]));
    static_assert(common::flat_index_strided3(0, 0, 1, dim_12, dims[2]) == common::flat_index3(0, 0, 1, dims[1], dims[2]));
    static_assert(common::flat_index_strided3(0, 1, 0, dim_12, dims[2]) == common::flat_index3(0, 1, 0, dims[1], dims[2]));
    static_assert(common::flat_index_strided3(0, 1, 1, dim_12, dims[2]) == common::flat_index3(0, 1, 1, dims[1], dims[2]));
    static_assert(common::flat_index_strided3(0, 2, 0, dim_12, dims[2]) == common::flat_index3(0, 2, 0, dims[1], dims[2]));
    static_assert(common::flat_index_strided3(0, 2, 1, dim_12, dims[2]) == common::flat_index3(0, 2, 1, dims[1], dims[2]));
    static_assert(common::flat_index_strided3(1, 0, 0, dim_12, dims[2]) == common::flat_index3(1, 0, 0, dims[1], dims[2]));
    static_assert(common::flat_index_strided3(1, 0, 1, dim_12, dims[2]) == common::flat_index3(1, 0, 1, dims[1], dims[2]));
    static_assert(common::flat_index_strided3(1, 1, 0, dim_12, dims[2]) == common::flat_index3(1, 1, 0, dims[1], dims[2]));
    static_assert(common::flat_index_strided3(1, 1, 1, dim_12, dims[2]) == common::flat_index3(1, 1, 1, dims[1], dims[2]));
    static_assert(common::flat_index_strided3(1, 2, 0, dim_12, dims[2]) == common::flat_index3(1, 2, 0, dims[1], dims[2]));
    static_assert(common::flat_index_strided3(1, 2, 1, dim_12, dims[2]) == common::flat_index3(1, 2, 1, dims[1], dims[2]));

    constexpr auto dims_123 = dims[1] * dims[2] * dims[3];
    constexpr auto dims_23 = dims[2] * dims[3];
    static_assert(common::flat_index_strided4(1, 1, 0, 0, dims_123, dims_23, dims[3])
        == common::flat_index4(1, 1, 0, 0, dims[1], dims[2], dims[3]));
    static_assert(common::flat_index_strided4(1, 1, 1, 0, dims_123, dims_23, dims[3])
        == common::flat_index4(1, 1, 1, 0, dims[1], dims[2], dims[3]));
    static_assert(common::flat_index_strided4(1, 2, 1, 0, dims_123, dims_23, dims[3])
        == common::flat_index4(1, 2, 1, 0, dims[1], dims[2], dims[3]));
    static_assert(common::flat_index_strided4(3, 2, 1, 0, dims_123, dims_23, dims[3])
        == common::flat_index4(3, 2, 1, 0, dims[1], dims[2], dims[3]));
}
