
#pragma once

#include "../../common/assert.h"
#include "../../common/stringUtils.h"

#include <mpi.h>

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif

#define TLLM_MPI_CHECK(cmd)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        auto e = cmd;                                                                                                  \
        TLLM_CHECK_WITH_INFO(e == MPI_SUCCESS, "Failed: MPI error %s:%d '%d'", __FILE__, __LINE__, e);                 \
    } while (0)

#if ENABLE_MULTI_DEVICE
#define TLLM_NCCL_CHECK(cmd)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t r = cmd;                                                                                          \
        TLLM_CHECK_WITH_INFO(                                                                                          \
            r == ncclSuccess, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));           \
    } while (0)
#endif
