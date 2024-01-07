
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
template <typename T>
struct NcclDataType
{
};

template <>
struct NcclDataType<half>
{
    static constexpr auto value = ncclDataType_t::ncclHalf;
};

template <>
struct NcclDataType<float>
{
    static constexpr auto value = ncclDataType_t::ncclFloat;
};

template <>
struct NcclDataType<std::uint8_t>
{
    static constexpr auto value = ncclDataType_t::ncclUint8;
};

template <>
struct NcclDataType<std::int32_t>
{
    static constexpr auto value = ncclDataType_t::ncclInt32;
};
#endif
}

template <typename T>
void NcclCommunicator::send(T* sendbuff, size_t count, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    auto datatype = NcclDataType<std::remove_cv_t<T>>::value;
    NCCL_CHECK(ncclSend(sendbuff, count, datatype, peer, mComm, stream.get()));
#else
    THROW("Multi device support is disabled.");
#endif
}

template void NcclCommunicator::send(std::uint8_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(std::int32_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(std::uint8_t const*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(std::int32_t const*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::send(float const*, size_t, int, CudaStream const&) const;

template <typename T>
void NcclCommunicator::receive(T* sendbuff, size_t count, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    auto datatype = NcclDataType<std::remove_cv_t<T>>::value;
    NCCL_CHECK(ncclRecv(sendbuff, count, datatype, peer, mComm, stream.get()));
#else
    THROW("Multi device support is disabled.");
#endif
}

template void NcclCommunicator::receive(std::uint8_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::receive(std::int32_t*, size_t, int, CudaStream const&) const;
template void NcclCommunicator::receive(float*, size_t, int, CudaStream const&) const;

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
