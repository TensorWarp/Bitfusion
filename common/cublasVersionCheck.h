#pragma once


#define CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH) (MAJOR * 10000 + MINOR * 100 + PATCH)
#define CUBLAS_VER_LE(MAJOR, MINOR, PATCH)                                                                        \
    CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH)                                     \
    <= CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)
#define CUBLAS_VER_LT(MAJOR, MINOR, PATCH)                                                                        \
    CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH)                                     \
        < CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)
#define CUBLAS_VER_GE(MAJOR, MINOR, PATCH)                                                                        \
    CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH)                                     \
    >= CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)
#define CUBLAS_VER_GT(MAJOR, MINOR, PATCH)                                                                        \
    CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH)                                     \
        > CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)