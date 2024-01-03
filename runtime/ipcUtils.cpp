#include "ipcUtils.h"
#include "../common/cudaUtils.h"
#include "../common/mpiUtils.h"

namespace bitfusion::runtime
{

void setPeerAccess(WorldConfig worldConfig, bool enable)
{
    const auto srcNode = worldConfig.getTensorParallelRank();

    for (SizeType destNode = 0; destNode < worldConfig.getTensorParallelism(); destNode++)
    {
        if (destNode == srcNode)
        {
            continue;
        }

        int canAccessPeer;
        TLLM_CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, srcNode, destNode));

        if (enable)
        {
            cudaDeviceEnablePeerAccess(destNode, 0);
        }
        else
        {
            cudaDeviceDisablePeerAccess(destNode);
        }
        const auto error = cudaGetLastError();
        if (error != cudaErrorPeerAccessAlreadyEnabled && error != cudaErrorPeerAccessNotEnabled)
        {
            TLLM_CUDA_CHECK(error);
        }
    }
}

IpcMemory::IpcMemory(WorldConfig worldConfig, std::size_t bufferSize)
    : mWorldConfig(worldConfig)
    , mCommPtrs(worldConfig.getTensorParallelism())
    , mBufferSize(bufferSize)
{
    allocateIpcMemory();
}

void IpcMemory::allocateIpcMemory()
{
    TLLM_CUDA_CHECK(cudaMalloc(&mBufferPtr, mBufferSize));
    TLLM_CUDA_CHECK(cudaMemset(mBufferPtr, 0, mBufferSize));

    cudaIpcMemHandle_t localHandle;
    TLLM_CUDA_CHECK(cudaIpcGetMemHandle(&localHandle, mBufferPtr));

    const auto tpRank = mWorldConfig.getTensorParallelRank();
    const auto ppRank = mWorldConfig.getPipelineParallelRank();
    mpi::MpiComm comm;
    mpi::comm_split(MPI_COMM_WORLD, ppRank, tpRank, &comm);
    std::vector<char> serialHandles(CUDA_IPC_HANDLE_SIZE * mWorldConfig.getTensorParallelism(), 0);
    mpi::allgather(&localHandle.reserved, serialHandles.data(), CUDA_IPC_HANDLE_SIZE, mpi::MPI_TYPE_BYTE, comm);

    std::vector<cudaIpcMemHandle_t> handles(mWorldConfig.getTensorParallelism());
    for (size_t i = 0; i < handles.size(); ++i)
    {
        memcpy(handles[i].reserved, &serialHandles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
    }

    for (size_t nodeId = 0; nodeId < handles.size(); nodeId++)
    {
        if ((int) nodeId == mWorldConfig.getTensorParallelRank())
        {
            mCommPtrs[nodeId] = mBufferPtr;
        }
        else
        {
            uint8_t* foreignBuffer;
            TLLM_CUDA_CHECK(cudaIpcOpenMemHandle(
                reinterpret_cast<void**>(&foreignBuffer), handles[nodeId], cudaIpcMemLazyEnablePeerAccess));
            mCommPtrs[nodeId] = foreignBuffer;
        }
    }
}

IpcMemory::~IpcMemory()
{
    destroyIpcMemory();
}

void IpcMemory::destroyIpcMemory()
{
    for (SizeType nodeId = 0; nodeId < mWorldConfig.getTensorParallelism(); ++nodeId)
    {
        if ((int) nodeId == mWorldConfig.getTensorParallelRank())
        {
            TLLM_CUDA_CHECK(cudaFree(mCommPtrs[nodeId]));
        }
        else
        {
            TLLM_CUDA_CHECK(cudaIpcCloseMemHandle(mCommPtrs[nodeId]));
        }
    }
    cudaFree(mBufferPtr);
}

}
