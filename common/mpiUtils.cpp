#include "mpiUtils.h"
#include "mpi.h"
#include "../runtime/common.h"
#include <type_traits>
#include <unordered_map>
#include <memory>
#include <vector>

static_assert(std::is_same_v<bitfusion::runtime::SizeType, std::int32_t>);

namespace bitfusion::mpi
{
    /// <summary>
    /// Mapping of MPI datatypes to corresponding MPI_Datatype.
    /// </summary>
    const std::unordered_map<MpiType, MPI_Datatype> dtype_map{
        {MPI_TYPE_BYTE, MPI_BYTE},
        {MPI_TYPE_CHAR, MPI_CHAR},
        {MPI_TYPE_INT, MPI_INT},
        {MPI_TYPE_FLOAT, MPI_FLOAT},
        {MPI_TYPE_DOUBLE, MPI_DOUBLE},
        {MPI_TYPE_INT64_T, MPI_INT64_T},
        {MPI_TYPE_INT32_T, MPI_INT32_T},
        {MPI_TYPE_UINT64_T, MPI_UINT64_T},
        {MPI_TYPE_UINT32_T, MPI_UINT32_T},
        {MPI_TYPE_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG},
        {MPI_TYPE_SIZETYPE, MPI_INT32_T},
    };

    /// <summary>
    /// Mapping of MPI reduction operations to corresponding MPI_Op.
    /// </summary>
    const std::unordered_map<MpiOp, MPI_Op> op_map{
        {MPI_OP_NULLOP, MPI_OP_NULL},
        {MPI_OP_MAX, MPI_MAX},
        {MPI_OP_MIN, MPI_MIN},
        {MPI_OP_SUM, MPI_SUM},
        {MPI_OP_PROD, MPI_PROD},
        {MPI_OP_LAND, MPI_LAND},
        {MPI_OP_BAND, MPI_BAND},
        {MPI_OP_LOR, MPI_LOR},
        {MPI_OP_BOR, MPI_BOR},
        {MPI_OP_LXOR, MPI_LXOR},
        {MPI_OP_BXOR, MPI_BXOR},
        {MPI_OP_MINLOC, MPI_MINLOC},
        {MPI_OP_MAXLOC, MPI_MAXLOC},
        {MPI_OP_REPLACE, MPI_REPLACE},
    };

    /// <summary>
    /// Initialize MPI with the given command line arguments.
    /// </summary>
    /// <param name="argc">Pointer to the number of command line arguments.</param>
    /// <param name="argv">Pointer to the command line argument array.</param>
    void initialize(int* argc, char*** argv)
    {
        MPICHECK(MPI_Init(argc, argv));
    }

    /// <summary>
    /// Finalize MPI and clean up resources.
    /// </summary>
    void finalize()
    {
        MPICHECK(MPI_Finalize());
    }

    /// <summary>
    /// Check if MPI has been initialized.
    /// </summary>
    /// <returns>True if MPI is initialized, false otherwise.</returns>
    bool isInitialized()
    {
        int mpi_initialized = 0;
        MPICHECK(MPI_Initialized(&mpi_initialized));
        return static_cast<bool>(mpi_initialized);
    }

    /// <summary>
    /// Initialize MPI with thread support.
    /// </summary>
    /// <param name="argc">Pointer to the number of command line arguments.</param>
    /// <param name="argv">Pointer to the command line argument array.</param>
    /// <param name="required">The desired level of thread support.</param>
    /// <param name="provided">Pointer to store the provided level of thread support.</param>
    void initThread(int* argc, char*** argv, MpiThreadSupport required, int* provided)
    {
        MPI_Init_thread(argc, argv, static_cast<int>(required), provided);
    }

    /// <summary>
    /// Get the rank of the current process in the MPI_COMM_WORLD communicator.
    /// </summary>
    /// <returns>The rank of the current process.</returns>
    [[nodiscard]] int getCommWorldRank()
    {
        int rank = 0;
        MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        return rank;
    }

    /// <summary>
    /// Get the size of the MPI_COMM_WORLD communicator.
    /// </summary>
    /// <returns>The size of the MPI_COMM_WORLD communicator.</returns>
    [[nodiscard]] int getCommWorldSize()
    {
        int world_size = 1;
        MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
        return world_size;
    }

    /// <summary>
    /// Perform a barrier synchronization operation on the specified communicator.
    /// </summary>
    /// <param name="comm">The communicator on which to perform the barrier.</param>
    void barrier(MpiComm comm)
    {
        MPICHECK(MPI_Barrier(comm.group));
    }

    /// <summary>
    /// Perform a barrier synchronization operation on the MPI_COMM_WORLD communicator.
    /// </summary>
    void barrier()
    {
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    /// <summary>
    /// Asynchronously broadcast data from the root process to all processes in the communicator.
    /// </summary>
    /// <param name="buffer">Pointer to the data buffer to be broadcast.</param>
    /// <param name="size">The size of the data buffer.</param>
    /// <param name="dtype">The datatype of the data.</param>
    /// <param name="root">The rank of the root process.</param>
    /// <param name="comm">The communicator for the broadcast.</param>
    /// <returns>A shared pointer to an MpiRequest representing the broadcast operation.</returns>
    [[nodiscard]] std::shared_ptr<MpiRequest> bcast_async(void* buffer, size_t size, MpiType dtype, int root, MpiComm comm)
    {
        auto r = std::make_shared<MpiRequest>();
        MPICHECK(MPI_Ibcast(buffer, size, dtype_map.at(dtype), root, comm.group, &r->mRequest));
        return r;
    }

    /// <summary>
    /// Broadcast data from the root process to all processes in the communicator.
    /// </summary>
    /// <param name="buffer">Pointer to the data buffer to be broadcast.</param>
    /// <param name="size">The size of the data buffer.</param>
    /// <param name="dtype">The datatype of the data.</param>
    /// <param name="root">The rank of the root process.</param>
    /// <param name="comm">The communicator for the broadcast.</param>
    void bcast(void* buffer, size_t size, MpiType dtype, int root, MpiComm comm)
    {
        MPICHECK(MPI_Bcast(buffer, size, dtype_map.at(dtype), root, comm.group));
    }

    /// <summary>
    /// Broadcast a vector of int64_t values from the root process to all processes in the communicator.
    /// </summary>
    /// <param name="packed">The vector of int64_t values to be broadcast.</param>
    /// <param name="root">The rank of the root process.</param>
    /// <param name="comm">The communicator for the broadcast.</param>
    void bcast(std::vector<int64_t>& packed, int root, MpiComm comm)
    {
        int64_t nWords1;
        if (getCommWorldRank() == root)
        {
            nWords1 = static_cast<int64_t>(packed.size());
        }
        bcast(&nWords1, 1, MPI_TYPE_INT64_T, root, comm);
        if (getCommWorldRank() != root)
        {
            packed.resize(nWords1);
        }
        bcast(packed.data(), packed.size(), MPI_TYPE_INT64_T, root, comm);
    }

    /// <summary>
    /// Split an existing communicator into multiple new communicators.
    /// </summary>
    /// <param name="comm">The original communicator to be split.</param>
    /// <param name="color">The color value for the new communicator.</param>
    /// <param name="key">The key value for ordering within the new communicator.</param>
    /// <param name="newcomm">Pointer to store the new communicator.</param>
    void comm_split(MpiComm comm, int color, int key, MpiComm* newcomm)
    {
        MPICHECK(MPI_Comm_split(comm.group, color, key, &newcomm->group));
    }

    /// <summary>
    /// Perform an allreduce operation on data across all processes in the communicator.
    /// </summary>
    /// <param name="sendbuf">Pointer to the send buffer.</param>
    /// <param name="recvbuf">Pointer to the receive buffer.</param>
    /// <param name="count">The number of elements in the buffer.</param>
    /// <param name="dtype">The datatype of the data.</param>
    /// <param name="op">The reduction operation to be performed.</param>
    /// <param name="comm">The communicator for the allreduce operation.</param>
    void allreduce(const void* sendbuf, void* recvbuf, int count, MpiType dtype, MpiOp op, MpiComm comm)
    {
        MPICHECK(MPI_Allreduce(sendbuf, recvbuf, count, dtype_map.at(dtype), op_map.at(op), comm.group));
    }

    /// <summary>
    /// Perform an allgather operation to gather data from all processes in the communicator to all processes.
    /// </summary>
    /// <param name="sendbuf">Pointer to the send buffer.</param>
    /// <param name="recvbuf">Pointer to the receive buffer.</param>
    /// <param name="count">The number of elements in the buffer.</param>
    /// <param name="dtype">The datatype of the data.</param>
    /// <param name="comm">The communicator for the allgather operation.</param>
    void allgather(const void* sendbuf, void* recvbuf, int count, MpiType dtype, MpiComm comm)
    {
        MPICHECK(MPI_Allgather(sendbuf, count, dtype_map.at(dtype), recvbuf, count, dtype_map.at(dtype), comm.group));
    }
}
