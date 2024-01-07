#include "ncclCommunicator.h"
#include "../runtime/utils/multiDeviceUtils.h"
#include <NvInferRuntime.h>
#include <mpi.h>
#include <type_traits>

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif

using namespace bitfusion::runtime;

namespace
{
#if ENABLE_MULTI_DEVICE
    /// <summary>
    /// Template class for mapping C++ types to NCCL data types.
    /// </summary>
    template <typename T>
    constexpr auto NcclDataType_v = [] {
        if constexpr (std::is_same_v<T, half>) {
            return ncclDataType_t::ncclHalf;
        }
        else if constexpr (std::is_same_v<T, float>) {
            return ncclDataType_t::ncclFloat;
        }
        else if constexpr (std::is_same_v<T, std::uint8_t>) {
            return ncclDataType_t::ncclUint8;
        }
        else if constexpr (std::is_same_v<T, std::int32_t>) {
            return ncclDataType_t::ncclInt32;
        }
        else {
            return ncclDataType_t::ncclInt32; // Default to ncclInt32
        }
        };
#endif
}

/// <summary>
/// Send data using NCCL.
/// </summary>
/// <typeparam name="T">The data type.</typeparam>
/// <param name="sendbuff">Pointer to the data to send.</param>
/// <param name="count">The number of elements to send.</param>
/// <param name="peer">The peer rank to send data to.</param>
/// <param name="stream">The CUDA stream.</param>
template <typename T>
void NcclCommunicator::send(T* sendbuff, size_t count, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    auto datatype = NcclDataType_v<T>();
    NCCL_CHECK(ncclSend(sendbuff, count, datatype, peer, mComm, stream.get()));
#else
    THROW("Multi-device support is disabled.");
#endif
}

template void NcclCommunicator::send(std::uint8_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(std::int32_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(std::uint8_t const*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(std::int32_t const*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(float const*, size_t, int, CudaStream const&) const;

/// <summary>
/// Receive data using NCCL.
/// </summary>
/// <typeparam name="T">The data type.</typeparam>
/// <param name="sendbuff">Pointer to the receive buffer.</param>
/// <param name="count">The number of elements to receive.</param>
/// <param name="peer">The peer rank to receive data from.</param>
/// <param name="stream">The CUDA stream.</param>
template <typename T>
void NcclCommunicator::receive(T* sendbuff, size_t count, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    auto datatype = NcclDataType_v<T>();
    NCCL_CHECK(ncclRecv(sendbuff, count, datatype, peer, mComm, stream.get()));
#else
    THROW("Multi-device support is disabled.");
#endif
}

template void NcclCommunicator::receive(std::uint8_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::receive(std::int32_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::receive(float*, size_t, int, CudaStream const&) const;

/// <summary>
/// Create an NCCL communicator for pipeline communication.
/// </summary>
/// <param name="worldConfig">The world configuration.</param>
/// <returns>The created NCCL communicator.</returns>
std::shared_ptr<NcclCommunicator> NcclCommunicator::createPipelineComm(WorldConfig const& worldConfig)
{
#if ENABLE_MULTI_DEVICE
    int const myRank = worldConfig.getRank();
    int const worldSize = worldConfig.getSize();

    ncclUniqueId id;
    if (myRank == 0)
    {
        ncclGetUniqueId(&id);
        for (auto peer = 1; peer < worldSize; ++peer)
        {
            MPI_CHECK(MPI_Send(&id, sizeof(id), MPI_BYTE, peer, 0, MPI_COMM_WORLD));
        }
    }
    else
    {
        auto constexpr peer = 0;
        MPI_Status status;
        MPI_CHECK(MPI_Recv(&id, sizeof(id), MPI_BYTE, peer, 0, MPI_COMM_WORLD, &status));
    }

    auto pipelineComm = std::make_shared<NcclCommunicator>();
    NCCL_CHECK(ncclCommInitRank(&pipelineComm->mComm, worldSize, id, myRank));

    return pipelineComm;
#else
    return nullptr;
#endif
}