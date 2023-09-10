#pragma once

/// <summary>
/// Perform a bitonic warp exchange operation for 64-bit keys and values.
/// </summary>
/// <param name="mask">Bitonic mask for comparison.</param>
#define BITONICWARPEXCHANGE_64(mask) \
    key1 = k0; \
    value1 = v0; \
    otgx = tgx ^ mask; \
    key2 = __shfl_sync(0xFFFFFFFF, k0, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v0, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k1, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v1, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2;

/// <summary>
/// Perform a 32-bit to 64-bit bitonic sorting operation.
/// </summary>
/// <remarks>
/// This macro uses bitonic warp exchange operations to sort the keys and values in a 32-bit to 64-bit conversion.
/// </remarks>
#define BITONICSORT32_64() \
    BITONICWARPEXCHANGE_64(1) \
    BITONICWARPEXCHANGE_64(3) \
    BITONICWARPEXCHANGE_64(1) \
    BITONICWARPEXCHANGE_64(7) \
    BITONICWARPEXCHANGE_64(2) \
    BITONICWARPEXCHANGE_64(1) \
    BITONICWARPEXCHANGE_64(15) \
    BITONICWARPEXCHANGE_64(4) \
    BITONICWARPEXCHANGE_64(2) \
    BITONICWARPEXCHANGE_64(1) \
    BITONICWARPEXCHANGE_64(31) \
    BITONICWARPEXCHANGE_64(8) \
    BITONICWARPEXCHANGE_64(4) \
    BITONICWARPEXCHANGE_64(2) \
    BITONICWARPEXCHANGE_64(1)


/// <summary>
/// Perform a 64-bit bitonic merge operation.
/// </summary>
/// <remarks>
/// This macro merges two sets of keys and values in a bitonic manner, resulting in a sorted set of keys and values.
/// </remarks>
#define BITONICMERGE64_64() \
    otgx = 31 - tgx; \
    key1 = k0; \
    value1 = v0; \
    key2 = __shfl_sync(0xFFFFFFFF, k1, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v1, otgx); \
    flag = (key1 > key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k1 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v1 = __shfl_sync(0xFFFFFFFF, value1, otgx);

/// <summary>
/// Perform a 64-bit bitonic sorting operation.
/// </summary>
/// <remarks>
/// This macro uses bitonic warp exchange operations to sort the keys and values in a 64-bit bitonic manner.
/// </remarks>
#define BITONICSORT64_64() \
    BITONICSORT32_64() \
    BITONICMERGE64_64() \
    BITONICWARPEXCHANGE_64(16) \
    BITONICWARPEXCHANGE_64(8) \
    BITONICWARPEXCHANGE_64(4) \
    BITONICWARPEXCHANGE_64(2) \
    BITONICWARPEXCHANGE_64(1)

/// <summary>
/// Perform a bitonic warp exchange operation for 128-bit keys and values.
/// </summary>
/// <param name="mask">Bitonic mask for comparison.</param>
#define BITONICWARPEXCHANGE_128(mask) \
    key1 = k0; \
    value1 = v0; \
    otgx = tgx ^ mask; \
    key2 = __shfl_sync(0xFFFFFFFF, k0, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v0, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k1, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v1, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2; \
    key1 = k2; \
    value1 = v2; \
    key2 = __shfl_sync(0xFFFFFFFF, k2, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v2, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k2 = flag ? key1 : key2; \
    v2 = flag ? value1 : value2; \
    key1 = k3; \
    value1 = v3; \
    key2 = __shfl_sync(0xFFFFFFFF, k3, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v3, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k3 = flag ? key1 : key2; \
    v3 = flag ? value1 : value2;

/// <summary>
/// Perform a 32-bit to 128-bit bitonic sorting operation.
/// </summary>
/// <remarks>
/// This macro uses bitonic warp exchange operations to sort the keys and values in a 32-bit to 128-bit conversion.
/// </remarks>
#define BITONICSORT32_128() \
    BITONICWARPEXCHANGE_128(1) \
    BITONICWARPEXCHANGE_128(3) \
    BITONICWARPEXCHANGE_128(1) \
    BITONICWARPEXCHANGE_128(7) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1) \
    BITONICWARPEXCHANGE_128(15) \
    BITONICWARPEXCHANGE_128(4) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1) \
    BITONICWARPEXCHANGE_128(31) \
    BITONICWARPEXCHANGE_128(8) \
    BITONICWARPEXCHANGE_128(4) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1)

/// <summary>
/// Perform a 128-bit bitonic merge operation.
/// </summary>
/// <remarks>
/// This macro merges two sets of keys and values in a bitonic manner, resulting in a sorted set of keys and values.
/// </remarks>
#define BITONICMERGE64_128() \
    otgx = 31 - tgx; \
    key1 = k0; \
    value1 = v0; \
    key2 = __shfl_sync(0xFFFFFFFF, k1, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v1, otgx); \
    flag = (key1 > key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k1 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v1 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k2; \
    value1 = v2; \
    key2 = __shfl_sync(0xFFFFFFFF, k3, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v3, otgx); \
    flag = (key1 > key2); \
    k2 = flag ? key1 : key2; \
    v2 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k3 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v3 = __shfl_sync(0xFFFFFFFF, value1, otgx);

/// <summary>
/// Perform a 64-bit to 128-bit bitonic sorting operation.
/// </summary>
/// <remarks>
/// This macro uses bitonic warp exchange operations to sort the keys and values in a 64-bit to 128-bit conversion.
/// </remarks>
#define BITONICSORT64_128() \
    BITONICSORT32_128() \
    BITONICMERGE64_128() \
    BITONICWARPEXCHANGE_128(16) \
    BITONICWARPEXCHANGE_128(8) \
    BITONICWARPEXCHANGE_128(4) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1)

/// <summary>
/// Perform a 128-bit bitonic merge operation.
/// </summary>
/// <remarks>
/// This macro merges two sets of keys and values in a bitonic manner, resulting in a sorted set of keys and values.
/// </remarks>
#define BITONICMERGE128_128() \
    otgx = 31 - tgx; \
    key1 = k0; \
    value1 = v0; \
    key2 = __shfl_sync(0xFFFFFFFF, k3, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v3, otgx); \
    flag = (key1 > key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k3 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v3 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k2, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v2, otgx); \
    flag = (key1 > key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k2 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v2 = __shfl_sync(0xFFFFFFFF, value1, otgx);

/// <summary>
/// Perform a 32-bit to 128-bit bitonic exchange operation.
/// </summary>
/// <remarks>
/// This macro compares and exchanges pairs of keys and values in a bitonic manner between four elements, resulting in sorted elements.
/// </remarks>
#define BITONICEXCHANGE32_128() \
    if (k0 < k1) { \
        key1 = k0; \
        value1 = v0; \
        k0 = k1; \
        v0 = v1; \
        k1 = key1; \
        v1 = value1; \
    } \
    if (k2 < k3) { \
        key1 = k2; \
        value1 = v2; \
        k2 = k3; \
        v2 = v3; \
        k3 = key1; \
        v3 = value1; \
    }

/// <summary>
/// Perform a 64-bit to 128-bit bitonic exchange operation.
/// </summary>
/// <remarks>
/// This macro compares and exchanges pairs of keys and values in a bitonic manner between four elements, resulting in sorted elements.
/// </remarks>
#define BITONICEXCHANGE64_128() \
    if (k0 < k2) { \
        key1 = k0; \
        value1 = v0; \
        k0 = k2; \
        v0 = v2; \
        k2 = key1; \
        v2 = value1; \
    } \
    if (k1 < k3) { \
        key1 = k1; \
        value1 = v1; \
        k1 = k3; \
        v1 = v3; \
        k3 = key1; \
        v3 = value1; \
    }

/// <summary>
/// Perform a 128-bit bitonic sorting operation.
/// </summary>
/// <remarks>
/// This macro uses bitonic warp exchange operations and bitonic merge operations to sort the keys and values in a 128-bit conversion.
/// </remarks>
#define BITONICSORT128_128() \
    BITONICSORT64_128() \
    BITONICMERGE128_128() \
    BITONICEXCHANGE32_128() \
    BITONICWARPEXCHANGE_128(16) \
    BITONICWARPEXCHANGE_128(8) \
    BITONICWARPEXCHANGE_128(4) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1)

/// <summary>
/// Perform a 256-bit bitonic warp exchange operation.
/// </summary>
/// <param name="mask">The mask value used to determine exchange pairs.</param>
/// <remarks>
/// This macro compares and exchanges pairs of keys and values in a bitonic manner between eight elements, resulting in sorted elements.
/// </remarks>
#define BITONICWARPEXCHANGE_256(mask) \
    key1 = k0; \
    value1 = v0; \
    otgx = tgx ^ mask; \
    key2 = __shfl_sync(0xFFFFFFFF, k0, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v0, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k1, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v1, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2; \
    key1 = k2; \
    value1 = v2; \
    key2 = __shfl_sync(0xFFFFFFFF, k2, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v2, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k2 = flag ? key1 : key2; \
    v2 = flag ? value1 : value2; \
    key1 = k3; \
    value1 = v3; \
    key2 = __shfl_sync(0xFFFFFFFF, k3, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v3, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k3 = flag ? key1 : key2; \
    v3 = flag ? value1 : value2; \
    key1 = k4; \
    value1 = v4; \
    key2 = __shfl_sync(0xFFFFFFFF, k4, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v4, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k4 = flag ? key1 : key2; \
    v4 = flag ? value1 : value2; \
    key1 = k5; \
    value1 = v5; \
    key2 = __shfl_sync(0xFFFFFFFF, k5, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v5, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k5 = flag ? key1 : key2; \
    v5 = flag ? value1 : value2; \
    key1 = k6; \
    value1 = v6; \
    key2 = __shfl_sync(0xFFFFFFFF, k6, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v6, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k6 = flag ? key1 : key2; \
    v6 = flag ? value1 : value2; \
    key1 = k7; \
    value1 = v7; \
    key2 = __shfl_sync(0xFFFFFFFF, k7, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v7, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k7 = flag ? key1 : key2; \
    v7 = flag ? value1 : value2;

/// <summary>
/// Perform a 256-bit bitonic sort operation using warp exchange for 32 elements.
/// </summary>
/// <remarks>
/// This macro applies a series of bitonic warp exchange steps with varying masks to sort an array of 32 elements in a bitonic order.
/// Each warp exchange step compares and exchanges elements within a warp, resulting in a bitonic sequence.
/// </remarks>
#define BITONICSORT32_256() \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(3) \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(7) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(15) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(31) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

/// <summary>
/// Perform a 256-bit bitonic merge operation for 64 elements.
/// </summary>
/// <remarks>
/// This macro merges two sets of 32 elements in bitonic order into a single set of 64 elements in bitonic order.
/// It utilizes a series of shuffling operations to compare and merge elements from two sets, ensuring they remain in bitonic order.
/// </remarks>
#define BITONICMERGE64_256() \
    otgx = 31 - tgx; \
    key1 = k0; \
    value1 = v0; \
    key2 = __shfl_sync(0xFFFFFFFF, k1, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v1, otgx); \
    flag = (key1 > key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k1 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v1 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k2; \
    value1 = v2; \
    key2 = __shfl_sync(0xFFFFFFFF, k3, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v3, otgx); \
    flag = (key1 > key2); \
    k2 = flag ? key1 : key2; \
    v2 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k3 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v3 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k4; \
    value1 = v4; \
    key2 = __shfl_sync(0xFFFFFFFF, k5, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v5, otgx); \
    flag = (key1 > key2); \
    k4 = flag ? key1 : key2; \
    v4 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k5 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v5 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k6; \
    value1 = v6; \
    key2 = __shfl_sync(0xFFFFFFFF, k7, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v7, otgx); \
    flag = (key1 > key2); \
    k6 = flag ? key1 : key2; \
    v6 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k7 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v7 = __shfl_sync(0xFFFFFFFF, value1, otgx);

/// <summary>
/// Perform a 256-bit bitonic sort operation for 64 elements.
/// </summary>
/// <remarks>
/// This macro sorts a set of 64 elements in bitonic order using a series of bitonic merge and warp exchange operations.
/// It combines 32 elements at a time into 64 elements in bitonic order, and then performs warp exchange operations to finalize the sorting.
/// </remarks>
#define BITONICSORT64_256() \
    BITONICSORT32_256() \
    BITONICMERGE64_256() \
    BITONICWARPEXCHANGE_256(16) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

/// <summary>
/// Perform a 256-bit bitonic merge operation for 128 elements.
/// </summary>
/// <remarks>
/// This macro merges two sets of 128 elements in bitonic order into a single set of 256 elements in bitonic order.
/// It compares and rearranges elements within the sets to ensure the merged set remains in bitonic order.
/// </remarks>
#define BITONICMERGE128_256() \
    otgx = 31 - tgx; \
    key1 = k0; \
    value1 = v0; \
    key2 = __shfl_sync(0xFFFFFFFF, k3, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v3, otgx); \
    flag = (key1 > key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k3 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v3 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k2, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v2, otgx); \
    flag = (key1 > key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k2 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v2 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k4; \
    value1 = v4; \
    key2 = __shfl_sync(0xFFFFFFFF, k7, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v7, otgx); \
    flag = (key1 > key2); \
    k4 = flag ? key1 : key2; \
    v4 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k7 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v7 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k5; \
    value1 = v5; \
    key2 = __shfl_sync(0xFFFFFFFF, k6, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v6, otgx); \
    flag = (key1 > key2); \
    k5 = flag ? key1 : key2; \
    v5 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k6 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v6 = __shfl_sync(0xFFFFFFFF, value1, otgx);

/// <summary>
/// Perform a 256-bit bitonic merge operation for 256 elements.
/// </summary>
/// <remarks>
/// This macro merges two sets of 256 elements in bitonic order into a single set of 256 elements in bitonic order.
/// It compares and rearranges elements within the sets to ensure the merged set remains in bitonic order.
/// </remarks>
#define BITONICMERGE256_256() \
    otgx = 31 - tgx; \
    key1 = k0; \
    value1 = v0; \
    key2 = __shfl_sync(0xFFFFFFFF, k7, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v7, otgx); \
    flag = (key1 > key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k7 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v7 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k6, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v6, otgx); \
    flag = (key1 > key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k6 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v6 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k2; \
    value1 = v2; \
    key2 = __shfl_sync(0xFFFFFFFF, k5, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v5, otgx); \
    flag = (key1 > key2); \
    k2 = flag ? key1 : key2; \
    v2 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k5 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v5 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k3; \
    value1 = v3; \
    key2 = __shfl_sync(0xFFFFFFFF, k4, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v4, otgx); \
    flag = (key1 > key2); \
    k3 = flag ? key1 : key2; \
    v3 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k4 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v4 = __shfl_sync(0xFFFFFFFF, value1, otgx);

/// <summary>
/// Perform a 32-bit bitonic exchange operation for 256 elements.
/// </summary>
/// <remarks>
/// This macro performs bitonic exchange operations on eight pairs of 32-bit key-value elements within a set of 256 elements.
/// Each pair of elements is compared, and if they are out of order, their positions are swapped to ensure bitonic order is maintained.
/// </remarks>
#define BITONICEXCHANGE32_256() \
if (k0 < k1) \
{ \
    key1 = k0; \
    value1 = v0; \
    k0 = k1; \
    v0 = v1; \
    k1 = key1; \
    v1 = value1; \
} \
if (k2 < k3) \
{ \
    key1 = k2; \
    value1 = v2; \
    k2 = k3; \
    v2 = v3; \
    k3 = key1; \
    v3 = value1; \
} \
if (k4 < k5) \
{ \
    key1 = k4; \
    value1 = v4; \
    k4 = k5; \
    v4 = v5; \
    k5 = key1; \
    v5 = value1; \
} \
if (k6 < k7) \
{ \
    key1 = k6; \
    value1 = v6; \
    k6 = k7; \
    v6 = v7; \
    k7 = key1; \
    v7 = value1; \
}

/// <summary>
/// Perform a 64-bit bitonic exchange operation for 256 elements.
/// </summary>
/// <remarks>
/// This macro performs bitonic exchange operations on four pairs of 64-bit key-value elements within a set of 256 elements.
/// Each pair of elements is compared, and if they are out of order, their positions are swapped to ensure bitonic order is maintained.
/// </remarks>
#define BITONICEXCHANGE64_256() \
if (k0 < k2) \
{ \
    key1 = k0; \
    value1 = v0; \
    k0 = k2; \
    v0 = v2; \
    k2 = key1; \
    v2 = value1; \
} \
if (k1 < k3) \
{ \
    key1 = k1; \
    value1 = v1; \
    k1 = k3; \
    v1 = v3; \
    k3 = key1; \
    v3 = value1; \
} \
if (k4 < k6) \
{ \
    key1 = k4; \
    value1 = v4; \
    k4 = k6; \
    v4 = v6; \
    k6 = key1; \
    v6 = value1; \
} \
if (k5 < k7) \
{ \
    key1 = k5; \
    value1 = v5; \
    k5 = k7; \
    v5 = v7; \
    k7 = key1; \
    v7 = value1; \
}

/// <summary>
/// Perform a bitonic sort for 128 elements within a 256-element array.
/// </summary>
#define BITONICSORT128_256() \
    BITONICSORT64_256() \
    BITONICMERGE128_256() \
    BITONICEXCHANGE32_256() \
    BITONICWARPEXCHANGE_256(16) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

/// <summary>
/// Perform a bitonic sort for 256 elements within a 256-element array.
/// </summary>
#define BITONICSORT256_256() \
    BITONICSORT128_256() \
    BITONICMERGE256_256() \
    BITONICEXCHANGE64_256() \
    BITONICEXCHANGE32_256() \
    BITONICWARPEXCHANGE_256(16) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

/// <summary>
/// Perform a bitonic warp exchange operation on a 512-element array.
/// </summary>
/// <param name="mask">The warp exchange mask.</param>
#define BITONICWARPEXCHANGE_512(mask) \
    key1 = k0; \
    value1 = v0; \
    otgx = tgx ^ mask; \
    key2 = __shfl_sync(0xFFFFFFFF, k0, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v0, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k1, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v1, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2; \
    key1 = k2; \
    value1 = v2; \
    key2 = __shfl_sync(0xFFFFFFFF, k2, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v2, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k2 = flag ? key1 : key2; \
    v2 = flag ? value1 : value2; \
    key1 = k3; \
    value1 = v3; \
    key2 = __shfl_sync(0xFFFFFFFF, k3, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v3, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k3 = flag ? key1 : key2; \
    v3 = flag ? value1 : value2; \
    key1 = k4; \
    value1 = v4; \
    key2 = __shfl_sync(0xFFFFFFFF, k4, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v4, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k4 = flag ? key1 : key2; \
    v4 = flag ? value1 : value2; \
    key1 = k5; \
    value1 = v5; \
    key2 = __shfl_sync(0xFFFFFFFF, k5, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v5, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k5 = flag ? key1 : key2; \
    v5 = flag ? value1 : value2; \
    key1 = k6; \
    value1 = v6; \
    key2 = __shfl_sync(0xFFFFFFFF, k6, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v6, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k6 = flag ? key1 : key2; \
    v6 = flag ? value1 : value2; \
    key1 = k7; \
    value1 = v7; \
    key2 = __shfl_sync(0xFFFFFFFF, k7, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v7, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k7 = flag ? key1 : key2; \
    v7 = flag ? value1 : value2; \
    key1 = k8; \
    value1 = v8; \
    key2 = __shfl_sync(0xFFFFFFFF, k8, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v8, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k8 = flag ? key1 : key2; \
    v8 = flag ? value1 : value2; \
    key1 = k9; \
    value1 = v9; \
    key2 = __shfl_sync(0xFFFFFFFF, k9, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v9, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k9 = flag ? key1 : key2; \
    v9 = flag ? value1 : value2; \
    key1 = k10; \
    value1 = v10; \
    key2 = __shfl_sync(0xFFFFFFFF, k10, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v10, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k10 = flag ? key1 : key2; \
    v10 = flag ? value1 : value2; \
    key1 = k11; \
    value1 = v11; \
    key2 = __shfl_sync(0xFFFFFFFF, k11, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v11, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k11 = flag ? key1 : key2; \
    v11 = flag ? value1 : value2; \
    key1 = k12; \
    value1 = v12; \
    key2 = __shfl_sync(0xFFFFFFFF, k12, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v12, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k12 = flag ? key1 : key2; \
    v12 = flag ? value1 : value2; \
    key1 = k13; \
    value1 = v13; \
    key2 = __shfl_sync(0xFFFFFFFF, k13, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v13, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k13 = flag ? key1 : key2; \
    v13 = flag ? value1 : value2; \
    key1 = k14; \
    value1 = v14; \
    key2 = __shfl_sync(0xFFFFFFFF, k14, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v14, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k14 = flag ? key1 : key2; \
    v14 = flag ? value1 : value2; \
    key1 = k15; \
    value1 = v15; \
    key2 = __shfl_sync(0xFFFFFFFF, k15, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v15, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k15 = flag ? key1 : key2; \
    v15 = flag ? value1 : value2;

/// <summary>
/// Perform a 32-element bitonic sort on a 512-element array using warp exchange operations.
/// </summary>
#define BITONICSORT32_512() \
    BITONICWARPEXCHANGE_512(1) \
    BITONICWARPEXCHANGE_512(3) \
    BITONICWARPEXCHANGE_512(1) \
    BITONICWARPEXCHANGE_512(7) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1) \
    BITONICWARPEXCHANGE_512(15) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1) \
    BITONICWARPEXCHANGE_512(31) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1) 

/// <summary>
/// Perform a 64-element bitonic merge on a 512-element array using warp exchange operations.
/// </summary>
#define BITONICMERGE64_512() \
    otgx = 31 - tgx; \
    key1 = k0; \
    value1 = v0; \
    key2 = __shfl_sync(0xFFFFFFFF, k1, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v1, otgx); \
    flag = (key1 > key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k1 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v1 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k2; \
    value1 = v2; \
    key2 = __shfl_sync(0xFFFFFFFF, k3, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v3, otgx); \
    flag = (key1 > key2); \
    k2 = flag ? key1 : key2; \
    v2 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k3 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v3 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k4; \
    value1 = v4; \
    key2 = __shfl_sync(0xFFFFFFFF, k5, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v5, otgx); \
    flag = (key1 > key2); \
    k4 = flag ? key1 : key2; \
    v4 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k5 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v5 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k6; \
    value1 = v6; \
    key2 = __shfl_sync(0xFFFFFFFF, k7, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v7, otgx); \
    flag = (key1 > key2); \
    k6 = flag ? key1 : key2; \
    v6 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k7 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v7 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k8; \
    value1 = v8; \
    key2 = __shfl_sync(0xFFFFFFFF, k9, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v9, otgx); \
    flag = (key1 > key2); \
    k8 = flag ? key1 : key2; \
    v8 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k9 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v9 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k10; \
    value1 = v10; \
    key2 = __shfl_sync(0xFFFFFFFF, k11, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v11, otgx); \
    flag = (key1 > key2); \
    k10 = flag ? key1 : key2; \
    v10 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k11 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v11 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k12; \
    value1 = v12; \
    key2 = __shfl_sync(0xFFFFFFFF, k13, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v13, otgx); \
    flag = (key1 > key2); \
    k12 = flag ? key1 : key2; \
    v12 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k13 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v13 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k14; \
    value1 = v14; \
    key2 = __shfl_sync(0xFFFFFFFF, k15, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v15, otgx); \
    flag = (key1 > key2); \
    k14 = flag ? key1 : key2; \
    v14 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k15 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v15 = __shfl_sync(0xFFFFFFFF, value1, otgx);


/// <summary>
/// Performs a bitonic sort operation for 64 elements in a 512-bit vector.
/// </summary>
#define BITONICSORT64_512() \
    BITONICSORT32_512() \
    BITONICMERGE64_512() \
    BITONICWARPEXCHANGE_512(16) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1)

/// <summary>
/// Performs a bitonic merge operation for 128 elements in a 512-bit vector.
/// </summary>
#define BITONICMERGE128_512() \
    otgx = 31 - tgx; \
    key1 = k0; \
    value1 = v0; \
    key2 = __shfl_sync(0xFFFFFFFF, k3, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v3, otgx); \
    flag = (key1 > key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k3 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v3 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k2, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v2, otgx); \
    flag = (key1 > key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k2 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v2 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k4; \
    value1 = v4; \
    key2 = __shfl_sync(0xFFFFFFFF, k7, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v7, otgx); \
    flag = (key1 > key2); \
    k4 = flag ? key1 : key2; \
    v4 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k7 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v7 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k5; \
    value1 = v5; \
    key2 = __shfl_sync(0xFFFFFFFF, k6, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v6, otgx); \
    flag = (key1 > key2); \
    k5 = flag ? key1 : key2; \
    v5 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k6 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v6 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k8; \
    value1 = v8; \
    key2 = __shfl_sync(0xFFFFFFFF, k11, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v11, otgx); \
    flag = (key1 > key2); \
    k8 = flag ? key1 : key2; \
    v8 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k11 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v11 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k9; \
    value1 = v9; \
    key2 = __shfl_sync(0xFFFFFFFF, k10, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v10, otgx); \
    flag = (key1 > key2); \
    k9 = flag ? key1 : key2; \
    v9 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k10 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v10 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k12; \
    value1 = v12; \
    key2 = __shfl_sync(0xFFFFFFFF, k15, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v15, otgx); \
    flag = (key1 > key2); \
    k12 = flag ? key1 : key2; \
    v12 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k15 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v15 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k13; \
    value1 = v13; \
    key2 = __shfl_sync(0xFFFFFFFF, k14, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v14, otgx); \
    flag = (key1 > key2); \
    k13 = flag ? key1 : key2; \
    v13 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k14 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v14 = __shfl_sync(0xFFFFFFFF, value1, otgx);

/// <summary>
/// Performs a bitonic merge operation for 256 elements in a 512-bit vector.
/// </summary>
#define BITONICMERGE256_512() \
    otgx = 31 - tgx; \
    key1 = k0; \
    value1 = v0; \
    key2 = __shfl_sync(0xFFFFFFFF, k7, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v7, otgx); \
    flag = (key1 > key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k7 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v7 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k6, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v6, otgx); \
    flag = (key1 > key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k6 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v6 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k2; \
    value1 = v2; \
    key2 = __shfl_sync(0xFFFFFFFF, k5, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v5, otgx); \
    flag = (key1 > key2); \
    k2 = flag ? key1 : key2; \
    v2 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k5 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v5 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k3; \
    value1 = v3; \
    key2 = __shfl_sync(0xFFFFFFFF, k4, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v4, otgx); \
    flag = (key1 > key2); \
    k3 = flag ? key1 : key2; \
    v3 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k4 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v4 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k8; \
    value1 = v8; \
    key2 = __shfl_sync(0xFFFFFFFF, k15, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v15, otgx); \
    flag = (key1 > key2); \
    k8 = flag ? key1 : key2; \
    v8 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k15 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v15 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k9; \
    value1 = v9; \
    key2 = __shfl_sync(0xFFFFFFFF, k14, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v14, otgx); \
    flag = (key1 > key2); \
    k9 = flag ? key1 : key2; \
    v9 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k14 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v14 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k10; \
    value1 = v10; \
    key2 = __shfl_sync(0xFFFFFFFF, k13, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v13, otgx); \
    flag = (key1 > key2); \
    k10 = flag ? key1 : key2; \
    v10 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k13 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v13 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k11; \
    value1 = v11; \
    key2 = __shfl_sync(0xFFFFFFFF, k12, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v12, otgx); \
    flag = (key1 > key2); \
    k11 = flag ? key1 : key2; \
    v11 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k12 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v12 = __shfl_sync(0xFFFFFFFF, value1, otgx);


/// <summary>
/// Performs a bitonic merge operation for a 512x512 array.
/// </summary>
#define BITONICMERGE512_512() \
    otgx = 31 - tgx; \
    key1 = k0; \
    value1 = v0; \
    key2 = __shfl_sync(0xFFFFFFFF, k15, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v15, otgx); \
    flag = (key1 > key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k15 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v15 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k14, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v14, otgx); \
    flag = (key1 > key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k14 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v14 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k2; \
    value1 = v2; \
    key2 = __shfl_sync(0xFFFFFFFF, k13, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v13, otgx); \
    flag = (key1 > key2); \
    k2 = flag ? key1 : key2; \
    v2 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k13 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v13 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k3; \
    value1 = v3; \
    key2 = __shfl_sync(0xFFFFFFFF, k12, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v12, otgx); \
    flag = (key1 > key2); \
    k3 = flag ? key1 : key2; \
    v3 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k12 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v12 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k4; \
    value1 = v4; \
    key2 = __shfl_sync(0xFFFFFFFF, k11, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v11, otgx); \
    flag = (key1 > key2); \
    k4 = flag ? key1 : key2; \
    v4 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k11 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v11 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k5; \
    value1 = v5; \
    key2 = __shfl_sync(0xFFFFFFFF, k10, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v10, otgx); \
    flag = (key1 > key2); \
    k5 = flag ? key1 : key2; \
    v5 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k10 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v10 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k6; \
    value1 = v6; \
    key2 = __shfl_sync(0xFFFFFFFF, k9, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v9, otgx); \
    flag = (key1 > key2); \
    k6 = flag ? key1 : key2; \
    v6 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k9 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v9 = __shfl_sync(0xFFFFFFFF, value1, otgx); \
    key1 = k7; \
    value1 = v7; \
    key2 = __shfl_sync(0xFFFFFFFF, k8, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v8, otgx); \
    flag = (key1 > key2); \
    k7 = flag ? key1 : key2; \
    v7 = flag ? value1 : value2; \
    key1 = flag ? key2 : key1; \
    value1 = flag ? value2 : value1; \
    k8 = __shfl_sync(0xFFFFFFFF, key1, otgx); \
    v8 = __shfl_sync(0xFFFFFFFF, value1, otgx);

/// <summary>
/// Performs a bitonic exchange operation for 32 elements in a 512-bit vector.
/// </summary>
#define BITONICEXCHANGE32_512() \
if (k0 < k1) \
{ \
    key1 = k0; \
    value1 = v0; \
    k0 = k1; \
    v0 = v1; \
    k1 = key1; \
    v1 = value1; \
} \
if (k2 < k3) \
{ \
    key1 = k2; \
    value1 = v2; \
    k2 = k3; \
    v2 = v3; \
    k3 = key1; \
    v3 = value1; \
} \
if (k4 < k5) \
{ \
    key1 = k4; \
    value1 = v4; \
    k4 = k5; \
    v4 = v5; \
    k5 = key1; \
    v5 = value1; \
} \
if (k6 < k7) \
{ \
    key1 = k6; \
    value1 = v6; \
    k6 = k7; \
    v6 = v7; \
    k7 = key1; \
    v7 = value1; \
} \
if (k8 < k9) \
{ \
    key1 = k8; \
    value1 = v8; \
    k8 = k9; \
    v8 = v9; \
    k9 = key1; \
    v9 = value1; \
} \
if (k10 < k11) \
{ \
    key1 = k10; \
    value1 = v10; \
    k10 = k11; \
    v10 = v11; \
    k11 = key1; \
    v11 = value1; \
} \
if (k12 < k13) \
{ \
    key1 = k12; \
    value1 = v12; \
    k12 = k13; \
    v12 = v13; \
    k13 = key1; \
    v13 = value1; \
} \
if (k14 < k15) \
{ \
    key1 = k14; \
    value1 = v14; \
    k14 = k15; \
    v14 = v15; \
    k15 = key1; \
    v15 = value1; \
}

/// <summary>
/// Performs a bitonic exchange operation for 64 elements in a 512-bit vector.
/// </summary>
#define BITONICEXCHANGE64_512() \
if (k0 < k2) \
{ \
    key1 = k0; \
    value1 = v0; \
    k0 = k2; \
    v0 = v2; \
    k2 = key1; \
    v2 = value1; \
} \
if (k1 < k3) \
{ \
    key1 = k1; \
    value1 = v1; \
    k1 = k3; \
    v1 = v3; \
    k3 = key1; \
    v3 = value1; \
} \
if (k4 < k6) \
{ \
    key1 = k4; \
    value1 = v4; \
    k4 = k6; \
    v4 = v6; \
    k6 = key1; \
    v6 = value1; \
} \
if (k5 < k7) \
{ \
    key1 = k5; \
    value1 = v5; \
    k5 = k7; \
    v5 = v7; \
    k7 = key1; \
    v7 = value1; \
} \
if (k8 < k10) \
{ \
    key1 = k8; \
    value1 = v8; \
    k8 = k10; \
    v8 = v10; \
    k10 = key1; \
    v10 = value1; \
} \
if (k9 < k11) \
{ \
    key1 = k9; \
    value1 = v9; \
    k9 = k11; \
    v9 = v11; \
    k11 = key1; \
    v11 = value1; \
} \
if (k12 < k14) \
{ \
    key1 = k12; \
    value1 = v12; \
    k12 = k14; \
    v12 = v14; \
    k14 = key1; \
    v14 = value1; \
} \
if (k13 < k15) \
{ \
    key1 = k13; \
    value1 = v13; \
    k13 = k15; \
    v13 = v15; \
    k15 = key1; \
    v13 = value1; \
}

/// <summary>
/// Performs a bitonic exchange operation for 128 elements in a 512-bit vector.
/// </summary>
#define BITONICEXCHANGE128_512() \
if (k0 < k4) \
{ \
    key1 = k0; \
    value1 = v0; \
    k0 = k4; \
    v0 = v4; \
    k4 = key1; \
    v4 = value1; \
} \
if (k1 < k5) \
{ \
    key1 = k1; \
    value1 = v1; \
    k1 = k5; \
    v1 = v5; \
    k5 = key1; \
    v5 = value1; \
} \
if (k2 < k6) \
{ \
    key1 = k2; \
    value1 = v2; \
    k2 = k6; \
    v2 = v6; \
    k6 = key1; \
    v6 = value1; \
} \
if (k3 < k7) \
{ \
    key1 = k3; \
    value1 = v3; \
    k3 = k7; \
    v3 = v7; \
    k7 = key1; \
    v7 = value1; \
} \
if (k8 < k12) \
{ \
    key1 = k8; \
    value1 = v8; \
    k8 = k12; \
    v8 = v12; \
    k12 = key1; \
    v12 = value1; \
} \
if (k9 < k13) \
{ \
    key1 = k9; \
    value1 = v9; \
    k9 = k13; \
    v9 = v13; \
    k13 = key1; \
    v13 = value1; \
} \
if (k10 < k14) \
{ \
    key1 = k10; \
    value1 = v10; \
    k10 = k14; \
    v10 = v14; \
    k14 = key1; \
    v14 = value1; \
} \
if (k11 < k15) \
{ \
    key1 = k11; \
    value1 = v11; \
    k11 = k15; \
    v11 = v15; \
    k15 = key1; \
    v15 = value1; \
}

/// <summary>
/// Performs a bitonic sort operation for 128 elements in a 512-bit vector.
/// </summary>
#define BITONICSORT128_512() \
    BITONICSORT64_512() \
    BITONICMERGE128_512() \
    BITONICEXCHANGE32_512() \
    BITONICWARPEXCHANGE_512(16) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1)

/// <summary>
/// Performs a bitonic sort operation for 256 elements in a 512-bit vector.
/// </summary>
#define BITONICSORT256_512() \
    BITONICSORT128_512() \
    BITONICMERGE256_512() \
    BITONICEXCHANGE64_512() \
    BITONICEXCHANGE32_512() \
    BITONICWARPEXCHANGE_512(16) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1)

/// <summary>
/// Performs a bitonic sort operation for 512 elements in a 512-bit vector.
/// </summary>
#define BITONICSORT512_512() \
    BITONICSORT256_512() \
    BITONICMERGE512_512() \
    BITONICEXCHANGE128_512() \
    BITONICEXCHANGE64_512() \
    BITONICEXCHANGE32_512() \
    BITONICWARPEXCHANGE_512(16) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1)