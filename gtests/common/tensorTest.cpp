#include <unordered_map>
#include <vector>
#include <memory>
#include <gtest/gtest.h>

#include "../../common/tensor.h"

using namespace bitfusion::common;

namespace
{

#define EXPECT_EQUAL_TENSORS(t1, t2)                                               \
    do                                                                             \
    {                                                                              \
        EXPECT_EQ(t1.where, t2.where);                                             \
        EXPECT_EQ(t1.type, t2.type);                                               \
        EXPECT_EQ(t1.shape, t2.shape);                                             \
        EXPECT_EQ(t1.data, t2.data);                                               \
    } while (false)

    /// <summary>
    /// Test case to check the correctness of the HasKey method in TensorMap.
    /// </summary>
    TEST(TensorMapTest, HasKeyCorrectness)
    {
        std::unique_ptr<bool> v1 = std::make_unique<bool>(true);
        std::unique_ptr<float[]> v2 = std::make_unique<float[]>(6);
        std::copy_n(std::array<float, 6>{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f}.begin(), 6, v2.get());

        Tensor t1 = Tensor{ MEMORY_CPU, TYPE_BOOL, {1}, v1.get() };
        Tensor t2 = Tensor{ MEMORY_CPU, TYPE_FP32, {3, 2}, v2.get() };

        TensorMap map({ {"t1", t1}, {"t2", t2} });
        EXPECT_TRUE(map.contains("t1"));
        EXPECT_TRUE(map.contains("t2"));
        EXPECT_FALSE(map.contains("t3"));
    }

    /// <summary>
    /// Test case to check the correctness of the Insert method in TensorMap.
    /// </summary>
    TEST(TensorMapTest, InsertCorrectness)
    {
        std::unique_ptr<int[]> v1 = std::make_unique<int[]>(4);
        std::copy_n(std::array<int, 4>{1, 10, 20, 30}.begin(), 4, v1.get());

        std::unique_ptr<float[]> v2 = std::make_unique<float[]>(2);
        std::copy_n(std::array<float, 2>{1.0f, 2.0f}.begin(), 2, v2.get());

        Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, { 4 }, v1.get());
        Tensor t2 = Tensor(MEMORY_CPU, TYPE_INT32, { 2 }, v2.get());

        TensorMap map({ {"t1", t1} });
        EXPECT_EQ(map.size(), 1);
        EXPECT_TRUE(map.contains("t1"));
        EXPECT_EQUAL_TENSORS(map.at("t1"), t1);
        EXPECT_FALSE(map.contains("t2"));
    }

    /// <summary>
    /// Test case to check that Insert method does not allow inserting a NoneTensor.
    /// </summary>
    TEST(TensorMapTest, InsertDoesNotAllowNoneTensor)
    {
        TensorMap map;
        EXPECT_EQ(map.size(), 0);
        EXPECT_THROW(map.insert("none", {}), std::runtime_error);

        Tensor none_data_tensor = Tensor(MEMORY_CPU, TYPE_INT32, {}, nullptr);
        EXPECT_THROW(map.insert("empty", none_data_tensor), std::runtime_error);
    }

    /// <summary>
    /// Test case to check that Insert method does not allow inserting a duplicated key.
    /// </summary>
    TEST(TensorMapTest, InsertDoesNotAllowDuplicatedKey)
    {
        std::unique_ptr<int[]> v1 = std::make_unique<int[]>(4);
        std::copy_n(std::array<int, 4>{1, 10, 20, 30}.begin(), 4, v1.get());

        Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, { 4 }, v1.get());
        Tensor t2 = Tensor(MEMORY_CPU, TYPE_INT32, { 2 }, v1.get());

        TensorMap map({ {"t1", t1} });
        EXPECT_EQ(map.size(), 1);
        EXPECT_THROW(map.insert("t1", t2), std::runtime_error);
    }

    /// <summary>
    /// Test case to check the correctness of GetVal method in TensorMap.
    /// </summary>
    TEST(TensorMapTest, GetValCorrectness)
    {
        std::unique_ptr<int[]> v1 = std::make_unique<int[]>(4);
        std::copy_n(std::array<int, 4>{1, 10, 20, 30}.begin(), 4, v1.get());

        Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, { 4 }, v1.get());

        TensorMap map({ {"t1", t1} });
        EXPECT_EQ(map.size(), 1);
        EXPECT_THROW(map.getVal<int>("t3"), std::runtime_error);
        EXPECT_EQ(map.getVal<int>("t1"), 1);
        EXPECT_EQ(map.getVal<int>("t1", 3), 1);

        EXPECT_EQ(map.getVal<int>("t2", 3), 3);

        v1[0] += 1;
        EXPECT_EQ(map.getVal<int>("t1"), 2);
        EXPECT_EQ(map.getVal<int>("t1", 3), 2);

        size_t index = 2;
        EXPECT_EQ(map.getValWithOffset<int>("t1", index), 20);
        EXPECT_EQ(map.getValWithOffset<int>("t1", index, 3), 20);
        EXPECT_EQ(map.getValWithOffset<int>("t2", index, 3), 3);
    }

    /// <summary>
    /// Test case to check the correctness of GetTensor method in TensorMap.
    /// </summary>
    TEST(TensorMapTest, GetTensorCorrectness)
    {
        std::unique_ptr<bool> t1_val = std::make_unique<bool>(true);
        std::unique_ptr<float[]> t2_val = std::make_unique<float[]>(6);
        std::copy_n(std::array<float, 6>{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f}.begin(), 6, t2_val.get());

        Tensor t1 = Tensor{ MEMORY_CPU, TYPE_BOOL, {1}, t1_val.get() };
        Tensor t2 = Tensor{ MEMORY_CPU, TYPE_FP32, {3, 2}, t2_val.get() };

        std::unique_ptr<int[]> default_val = std::make_unique<int[]>(4);
        std::copy_n(std::array<int, 4>{0, 1, 2, 3}.begin(), 4, default_val.get());

        Tensor default_tensor = Tensor{ MEMORY_CPU, TYPE_INT32, {4}, default_val.get() };

        TensorMap map({ {"t1", t1}, {"t2", t2} });
        EXPECT_THROW(map.at("t3"), std::runtime_error);
        EXPECT_EQUAL_TENSORS(map.at("t1", default_tensor), t1);
        EXPECT_EQUAL_TENSORS(map.at("t2", default_tensor), t2);
        EXPECT_EQUAL_TENSORS(map.at("t3", default_tensor), default_tensor);
        EXPECT_EQUAL_TENSORS(map.at("t3", {}), Tensor());
    }

    /// <summary>
    /// Test case to check the correctness of GetTensor method in a const TensorMap.
    /// </summary>
    TEST(TensorMapTest, GetTensorCorrectnessAtConstTensorMap)
    {
        std::unique_ptr<bool> t1_val = std::make_unique<bool>(true);
        std::unique_ptr<float[]> t2_val = std::make_unique<float[]>(6);
        std::copy_n(std::array<float, 6>{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f}.begin(), 6, t2_val.get());

        Tensor t1 = Tensor{ MEMORY_CPU, TYPE_BOOL, {1}, t1_val.get() };
        Tensor t2 = Tensor{ MEMORY_CPU, TYPE_FP32, {3, 2}, t2_val.get() };

        std::unique_ptr<int[]> default_val = std::make_unique<int[]>(4);
        std::copy_n(std::array<int, 4>{0, 1, 2, 3}.begin(), 4, default_val.get());

        Tensor default_tensor = Tensor{ MEMORY_CPU, TYPE_INT32, {4}, default_val.get() };

        const TensorMap map({ {"t1", t1}, {"t2", t2} });
        EXPECT_THROW(map.at("t3"), std::runtime_error);
        EXPECT_EQUAL_TENSORS(map.at("t1", default_tensor), t1);
        EXPECT_EQUAL_TENSORS(map.at("t2", default_tensor), t2);
        EXPECT_EQUAL_TENSORS(map.at("t3", default_tensor), default_tensor);
        EXPECT_EQUAL_TENSORS(map.at("t3", {}), Tensor());
    }

    /// <summary>
    /// Test case to check that Min and Max methods raise errors for empty tensors.
    /// </summary>
    TEST(TensorTest, EmptyTensorMinMaxRaiseError)
    {
        Tensor t1;
        EXPECT_THROW(t1.min<int>(), std::runtime_error);
        EXPECT_THROW(t1.max<int>(), std::runtime_error);

        Tensor t2 = Tensor{ MEMORY_CPU, TYPE_INT32, {}, nullptr };
        EXPECT_THROW(t2.min<int>(), std::runtime_error);
        EXPECT_THROW(t2.max<int>(), std::runtime_error);
    }

    using TensorTypes = testing::Types<int8_t, int, float>;

    template <typename T>
    class TensorFuncTest : public testing::Test
    {
    };

    TYPED_TEST_SUITE(TensorFuncTest, TensorTypes);

    /// <summary>
    /// Test case to check the correctness of the Max method for tensors of different types.
    /// </summary>
    TYPED_TEST(TensorFuncTest, MaxCorrectness)
    {
        using T = TypeParam;

        size_t size = 4;

        std::unique_ptr<T[]> v1 = std::make_unique<T[]>(size);
        std::copy_n(std::array<T, 4>{T(1), T(2), T(3), T(4)}.begin(), size, v1.get());

        std::unique_ptr<T[]> v2 = std::make_unique<T[]>(size);
        std::copy_n(std::array<T, 4>{T(4), T(3), T(2), T(1)}.begin(), size, v2.get());

        std::unique_ptr<T[]> v3 = std::make_unique<T[]>(size);
        std::copy_n(std::array<T, 4>{T(1), T(2), T(4), T(3)}.begin(), size, v3.get());

        Tensor t1 = Tensor(MEMORY_CPU, getTensorType<T>(), { size }, v1.get());
        Tensor t2 = Tensor(MEMORY_CPU, getTensorType<T>(), { size }, v2.get());
        Tensor t3 = Tensor(MEMORY_CPU, getTensorType<T>(), { size }, v3.get());

        EXPECT_EQ(t1.max<T>(), T(4));
        EXPECT_EQ(t2.max<T>(), T(4));
        EXPECT_EQ(t3.max<T>(), T(4));
    }

    /// <summary>
    /// Test case to check the correctness of the Min method for tensors of different types.
    /// </summary>
    TYPED_TEST(TensorFuncTest, MinCorrectness)
    {
        using T = TypeParam;

        size_t size = 4;

        std::unique_ptr<T[]> v1 = std::make_unique<T[]>(size);
        std::copy_n(std::array<T, 4>{T(1), T(2), T(3), T(4)}.begin(), size, v1.get());

        std::unique_ptr<T[]> v2 = std::make_unique<T[]>(size);
        std::copy_n(std::array<T, 4>{T(4), T(3), T(2), T(1)}.begin(), size, v2.get());

        std::unique_ptr<T[]> v3 = std::make_unique<T[]>(size);
        std::copy_n(std::array<T, 4>{T(1), T(2), T(4), T(3)}.begin(), size, v3.get());

        Tensor t1 = Tensor(MEMORY_CPU, getTensorType<T>(), { size }, v1.get());
        Tensor t2 = Tensor(MEMORY_CPU, getTensorType<T>(), { size }, v2.get());
        Tensor t3 = Tensor(MEMORY_CPU, getTensorType<T>(), { size }, v3.get());

        EXPECT_EQ(t1.min<T>(), T(1));
        EXPECT_EQ(t2.min<T>(), T(1));
        EXPECT_EQ(t3.min<T>(), T(1));
    }

    /// <summary>
    /// Test case to check the correctness of the Any method for tensors of different types.
    /// </summary>
    TYPED_TEST(TensorFuncTest, AnyCorrectness)
    {
        using T = TypeParam;

        std::unique_ptr<T[]> v = std::make_unique<T[]>(4);
        std::copy_n(std::array<T, 4>{T(1), T(2), T(3), T(4)}.begin(), 4, v.get());

        Tensor t = Tensor{ MEMORY_CPU, getTensorType<T>(), {4}, v.get() };
        EXPECT_TRUE(t.any<T>(T(1)));
        EXPECT_FALSE(t.any<T>(T(5)));
    }

    /// <summary>
    /// Test case to check the correctness of the All method for tensors of different types.
    /// </summary>
    /// <typeparam name="T">The data type of the tensor elements.</typeparam>
    TYPED_TEST(TensorFuncTest, AllCorrectness)
    {
        using T = TypeParam;

        constexpr size_t size = 4;

        std::unique_ptr<T[]> v1 = std::make_unique<T[]>(size);
        std::copy_n(std::array<T, 4>{T(1), T(1), T(1), T(1)}.begin(), size, v1.get());

        std::unique_ptr<T[]> v2 = std::make_unique<T[]>(size);
        std::copy_n(std::array<T, 4>{T(1), T(1), T(1), T(2)}.begin(), size, v2.get());

        Tensor t1 = Tensor{ MEMORY_CPU, getTensorType<T>(), {size}, v1.get() };
        Tensor t2 = Tensor{ MEMORY_CPU, getTensorType<T>(), {size}, v2.get() };

        // Assert that all elements in t1 are equal to 1.
        EXPECT_TRUE(t1.all<T>(T(1)));

        // Assert that not all elements in t2 are equal to 2.
        EXPECT_FALSE(t2.all<T>(T(2)));
    }

    /// <summary>
    /// Test case to check the correctness of the Slice method for tensors of different types.
    /// </summary>
    /// <typeparam name="T">The data type of the tensor elements.</typeparam>
    TYPED_TEST(TensorFuncTest, SliceCorrectness)
    {
        using T = TypeParam;

        constexpr int size = 12;

        std::unique_ptr<T[]> v = std::make_unique<T[]>(size);
        for (int i = 0; i < size; ++i)
        {
            v[i] = i;
        }

        DataType dtype = getTensorType<T>();
        Tensor t1 = Tensor(MEMORY_CPU, dtype, { 3, 4 }, v.get());
        Tensor t2 = t1.slice({ 2, 4 }, 4);

        // Assert that t2 is a slice of t1 with the expected values.
        EXPECT_EQUAL_TENSORS(t2, Tensor(MEMORY_CPU, dtype, { 2, 4 }, &v[4]));

        // Assert that attempting to slice t1 with an invalid size throws a runtime error.
        EXPECT_THROW(t1.slice({ 2, 4 }, 5), std::runtime_error);
    }
}
