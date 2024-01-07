#pragma once

#include "../runtime/utils/multiDeviceUtils.h"

#include <cstdlib>
#include <memory>
#include <mpi.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>

#define COMM_WORLD MpiComm(MPI_COMM_WORLD)
#define MPICHECK(cmd) MPI_CHECK(cmd)

namespace bitfusion::mpi
{

    enum MpiType
    {
        MPI_TYPE_BYTE,
        MPI_TYPE_CHAR,
        MPI_TYPE_INT,
        MPI_TYPE_FLOAT,
        MPI_TYPE_DOUBLE,
        MPI_TYPE_INT64_T,
        MPI_TYPE_INT32_T,
        MPI_TYPE_UINT64_T,
        MPI_TYPE_UINT32_T,
        MPI_TYPE_UNSIGNED_LONG_LONG,
        MPI_TYPE_SIZETYPE,
    };

    enum MpiOp
    {
        MPI_OP_NULLOP,
        MPI_OP_MAX,
        MPI_OP_MIN,
        MPI_OP_SUM,
        MPI_OP_PROD,
        MPI_OP_LAND,
        MPI_OP_BAND,
        MPI_OP_LOR,
        MPI_OP_BOR,
        MPI_OP_LXOR,
        MPI_OP_BXOR,
        MPI_OP_MINLOC,
        MPI_OP_MAXLOC,
        MPI_OP_REPLACE,
    };

    enum MpiThreadSupport
    {
        THREAD_SINGLE,
        THREAD_FUNNELED,
        THREAD_SERIALIZED,
        THREAD_MULTIPLE
    };

    struct MpiComm
    {
        MPI_Comm group;
        MpiComm() {};
        MpiComm(MPI_Comm g)
            : group(g) {};
    };

    class MpiRequest
    {
    public:
        MpiRequest() {}

        ~MpiRequest() {}

        void wait()
        {
            MPI_Wait(&mRequest, MPI_STATUS_IGNORE);
        }

        MPI_Request mRequest;
    };

    MPI_Datatype getMpiDtype(MpiType dtype);

    void initialize(int* argc, char*** argv);
    void initThread(int* argc, char*** argv, MpiThreadSupport required, int* provided);
    void finalize();
    bool isInitialized();
    void barrier(MpiComm comm);
    void barrier();

    int getCommWorldRank();
    int getCommWorldSize();

    std::shared_ptr<MpiRequest> bcast_async(void* buffer, size_t size, MpiType dtype, int root, MpiComm comm);
    void bcast(void* buffer, size_t size, MpiType dtype, int root, MpiComm comm);
    void bcast(std::vector<int64_t>& packed, int root, MpiComm comm);
    void comm_split(MpiComm comm, int color, int key, MpiComm* newcomm);
    void allreduce(const void* sendbuf, void* recvbuf, int count, MpiType dtype, MpiOp op, MpiComm comm);
    void allgather(const void* sendbuf, void* recvbuf, int count, MpiType dtype, MpiComm comm);

}