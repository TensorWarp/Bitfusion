#pragma once

#define BITONIC_WARP_EXCHANGE_64(mask) \
    do { \
        otgx = tgx ^ mask; \
        key2 = __shfl_xor_sync(0xFFFFFFFF, k0, mask); \
        value2 = __shfl_xor_sync(0xFFFFFFFF, v0, mask); \
        key1 = __shfl_xor_sync(0xFFFFFFFF, k1, mask);  \
        value1 = __shfl_xor_sync(0xFFFFFFFF, v1, mask); \
        flag = ((k0 > key2) ^ (tgx > otgx)) && (k0 != key2); \
        k0 = flag ? k0 : key2; \
        v0 = flag ? v0 : value2; \
        k1 = flag ? k1 : key1; \
        v1 = flag ? v1 : value1; \
    } while (0)

#define BITONIC_SORT_64() \
    do { \
        BITONIC_WARP_EXCHANGE_64(1); \
        BITONIC_WARP_EXCHANGE_64(2); \
        BITONIC_WARP_EXCHANGE_64(4); \
        BITONIC_WARP_EXCHANGE_64(8); \
        BITONIC_WARP_EXCHANGE_64(16); \
        BITONIC_WARP_EXCHANGE_64(32); \
        BITONIC_WARP_EXCHANGE_64(64); \
        BITONIC_WARP_EXCHANGE_64(128); \
        BITONIC_WARP_EXCHANGE_64(256); \
        BITONIC_WARP_EXCHANGE_64(512); \
        BITONIC_WARP_EXCHANGE_64(1024); \
        BITONIC_WARP_EXCHANGE_64(2048); \
        BITONIC_WARP_EXCHANGE_64(4096); \
        BITONIC_WARP_EXCHANGE_64(8192); \
        BITONIC_WARP_EXCHANGE_64(16384); \
        BITONIC_WARP_EXCHANGE_64(32768); \
        BITONIC_WARP_EXCHANGE_64(65536); \
        BITONIC_WARP_EXCHANGE_64(131072); \
        BITONIC_WARP_EXCHANGE_64(262144); \
        BITONIC_WARP_EXCHANGE_64(524288); \
        BITONIC_WARP_EXCHANGE_64(1048576); \
        BITONIC_WARP_EXCHANGE_64(2097152); \
        for (int i = 0; i < 6; ++i) { \
            int step = 1 << (i + 18); \
            for (int j = 0; j < step; ++j) { \
                BITONIC_WARP_EXCHANGE_64(step); \
            } \
        } \
    } while (0)

#define BITONIC_MERGE_64() \
    do { \
        otgx = 31 - tgx; \
        key2 = __shfl_xor_sync(0xFFFFFFFF, k1, 1); \
        value2 = __shfl_xor_sync(0xFFFFFFFF, v1, 1); \
        flag = (k0 > key2); \
        k0 = flag ? k0 : key2; \
        v0 = flag ? v0 : value2; \
        k1 = flag ? key2 : k1; \
        v1 = flag ? value2 : v1; \
        k1 = __shfl_xor_sync(0xFFFFFFFF, k1, 1); \
        v1 = __shfl_xor_sync(0xFFFFFFFF, v1, 1); \
        otgx = tgx ^ 1; \
        key2 = __shfl_xor_sync(0xFFFFFFFF, k0, 2); \
        value2 = __shfl_xor_sync(0xFFFFFFFF, v0, 2); \
        flag = ((k0 > key2) ^ (tgx > otgx)) && (k0 != key2); \
        k0 = flag ? k0 : key2; \
        v0 = flag ? v0 : value2; \
        k1 = __shfl_xor_sync(0xFFFFFFFF, k1, 2); \
        v1 = __shfl_xor_sync(0xFFFFFFFF, v1, 2); \
        otgx = tgx ^ 2; \
        key2 = __shfl_xor_sync(0xFFFFFFFF, k0, 4); \
        value2 = __shfl_xor_sync(0xFFFFFFFF, v0, 4); \
        flag = ((k0 > key2) ^ (tgx > otgx)) && (k0 != key2); \
        k0 = flag ? k0 : key2; \
        v0 = flag ? v0 : value2; \
        k1 = __shfl_xor_sync(0xFFFFFFFF, k1, 4); \
        v1 = __shfl_xor_sync(0xFFFFFFFF, v1, 4); \
    } while (0)

#define BITONIC_SORT_64_LARGE() \
    do { \
        BITONIC_SORT_64(); \
        BITONIC_MERGE_64(); \
        for (int i = 0; i < 6; ++i) { \
            int step = 1 << (i + 18); \
            for (int j = 0; j < step; ++j) { \
                BITONIC_SORT_64(); \
                BITONIC_MERGE_64(); \
            } \
        } \
    } while (0)


#define BITONICWARPEXCHANGE_128(mask) \
    key1 = k0; \
    value1 = v0; \
    otgx = tgx ^ mask; \
    key2 = __shfl_sync(0xFFFFFFFF, k0, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v0, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k1, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v1, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2; \
    \
    key1 = k2; \
    value1 = v2; \
    key2 = __shfl_sync(0xFFFFFFFF, k2, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v2, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k2 = flag ? key1 : key2; \
    v2 = flag ? value1 : value2; \
    \
    key1 = k3; \
    value1 = v3; \
    key2 = __shfl_sync(0xFFFFFFFF, k3, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v3, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k3 = flag ? key1 : key2; \
    v3 = flag ? value1 : value2;

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
    \
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

#define BITONICSORT64_128() \
    BITONICSORT32_128() \
    BITONICMERGE64_128() \
    BITONICWARPEXCHANGE_128(16) \
    BITONICWARPEXCHANGE_128(8) \
    BITONICWARPEXCHANGE_128(4) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1)

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
    \
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

#define BITONICEXCHANGE32_128() \
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
    }

#define BITONICEXCHANGE64_128() \
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
    }

#define BITONICSORT128_128() \
    BITONICSORT64_128() \
    BITONICMERGE128_128() \
    BITONICEXCHANGE32_128() \
    BITONICWARPEXCHANGE_128(16) \
    BITONICWARPEXCHANGE_128(8) \
    BITONICWARPEXCHANGE_128(4) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1)

#define BITONICWARPEXCHANGE_256(mask) \
    key1 = k0; \
    value1 = v0; \
    otgx = tgx ^ mask; \
    key2 = __shfl_sync(0xFFFFFFFF, k0, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v0, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k0 = flag ? key1 : key2; \
    v0 = flag ? value1 : value2; \
    \
    key1 = k1; \
    value1 = v1; \
    key2 = __shfl_sync(0xFFFFFFFF, k1, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v1, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k1 = flag ? key1 : key2; \
    v1 = flag ? value1 : value2; \
    \
    key1 = k2; \
    value1 = v2; \
    key2 = __shfl_sync(0xFFFFFFFF, k2, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v2, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k2 = flag ? key1 : key2; \
    v2 = flag ? value1 : value2; \
    \
    key1 = k3; \
    value1 = v3; \
    key2 = __shfl_sync(0xFFFFFFFF, k3, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v3, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k3 = flag ? key1 : key2; \
    v3 = flag ? value1 : value2; \
    \
    key1 = k4; \
    value1 = v4; \
    key2 = __shfl_sync(0xFFFFFFFF, k4, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v4, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k4 = flag ? key1 : key2; \
    v4 = flag ? value1 : value2; \
    \
    key1 = k5; \
    value1 = v5; \
    key2 = __shfl_sync(0xFFFFFFFF, k5, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v5, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k5 = flag ? key1 : key2; \
    v5 = flag ? value1 : value2; \
    \
    key1 = k6; \
    value1 = v6; \
    key2 = __shfl_sync(0xFFFFFFFF, k6, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v6, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k6 = flag ? key1 : key2; \
    v6 = flag ? value1 : value2; \
    \
    key1 = k7; \
    value1 = v7; \
    key2 = __shfl_sync(0xFFFFFFFF, k7, otgx); \
    value2 = __shfl_sync(0xFFFFFFFF, v7, otgx); \
    flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
    k7 = flag ? key1 : key2; \
    v7 = flag ? value1 : value2;

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
    \
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
    \
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
    \
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

#define BITONICSORT64_256() \
    BITONICSORT32_256() \
    BITONICMERGE64_256() \
    BITONICWARPEXCHANGE_256(16) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

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
    \
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
    \
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
    \
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
    \
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
    \
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
    \
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

#define BITONICSORT128_256() \
    BITONICSORT64_256() \
    BITONICMERGE128_256() \
    BITONICEXCHANGE32_256() \
    BITONICWARPEXCHANGE_256(16) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

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

#define BITONICSORT64_512() \
    BITONICSORT32_512() \
    BITONICMERGE64_512() \
    BITONICWARPEXCHANGE_512(16) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1)

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
        v15 = value1; \
    }

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

#define BITONICSORT128_512() \
    BITONICSORT64_512() \
    BITONICMERGE128_512() \
    BITONICEXCHANGE32_512() \
    BITONICWARPEXCHANGE_512(16) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1)

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