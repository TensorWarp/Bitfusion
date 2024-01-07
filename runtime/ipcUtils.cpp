#include "ipcUtils.h"
#include "../common/cudaUtils.h"
#include "../common/mpiUtils.h"
#include <span>
#include <vector>

namespace bitfusion::runtime
{
    /// <summary>
    /// Enables or disables peer access between CUDA devices based on the provided WorldConfig.
    /// </summary>
    /// <param name="worldConfig">The WorldConfig object containing configuration information.</param>
    /// <param name="enable">If true, enable peer access; otherwise, disable it.</param>
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
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, srcNode, destNode));

            if (enable)
            {
                CUDA_CHECK(cudaDeviceEnablePeerAccess(destNode, 0));
            }
            else
            {
                CUDA_CHECK(cudaDeviceDisablePeerAccess(destNode));
            }

            const auto error = cudaGetLastError();
            if (error != cudaErrorPeerAccessAlreadyEnabled && error != cudaErrorPeerAccessNotEnabled)
            {
                CUDA_CHECK(error);
            }
        }
    }

    /// <summary>
    /// Constructs an IpcMemory object with the given WorldConfig and buffer size.
    /// </summary>
    /// <param name="worldConfig">The WorldConfig object containing configuration information.</param>
    /// <param name="bufferSize">The size of the IPC memory buffer to allocate.</param>
    IpcMemory::IpcMemory(WorldConfig worldConfig, std::size_t bufferSize)
        : mWorldConfig(worldConfig),
        mCommPtrs(worldConfig.getTensorParallelism()),
        mBufferSize(bufferSize)
    {
        allocateIpcMemory();
    }

    /// <summary>
    /// Allocates IPC memory and initializes the necessary data structures.
    /// </summary>
    void IpcMemory::allocateIpcMemory()
    {
        CUDA_CHECK(cudaMalloc(&mBufferPtr, mBufferSize));
        CUDA_CHECK(cudaMemset(mBufferPtr, 0, mBufferSize));

        cudaIpcMemHandle_t localHandle;
        CUDA_CHECK(cudaIpcGetMemHandle(&localHandle, mBufferPtr));

        const auto tpRank = mWorldConfig.getTensorParallelRank();
        const auto ppRank = mWorldConfig.getPipelineParallelRank();
        mpi::MpiComm comm;
        mpi::comm_split(MPI_COMM_WORLD, ppRank, tpRank, &comm);

        std::vector<char> serialHandles(CUDA_IPC_HANDLE_SIZE * mWorldConfig.getTensorParallelism(), 0);
        mpi::allgather(&localHandle.reserved, serialHandles.data(), CUDA_IPC_HANDLE_SIZE, mpi::MPI_TYPE_BYTE, comm);

        std::vector<cudaIpcMemHandle_t> handles;
        handles.reserve(mWorldConfig.getTensorParallelism());
        for (const auto& serialHandle : serialHandles)
        {
            cudaIpcMemHandle_t handle;
            std::memcpy(handle.reserved, &serialHandle, CUDA_IPC_HANDLE_SIZE);
            handles.push_back(handle);
        }

        for (size_t nodeId = 0; nodeId < handles.size(); nodeId++)
        {
            if (nodeId == static_cast<size_t>(mWorldConfig.getTensorParallelRank()))
            {
                mCommPtrs[nodeId] = mBufferPtr;
            }
            else
            {
                uint8_t* foreignBuffer;
                CUDA_CHECK(cudaIpcOpenMemHandle(
                    reinterpret_cast<void**>(&foreignBuffer), handles[nodeId], cudaIpcMemLazyEnablePeerAccess));
                mCommPtrs[nodeId] = foreignBuffer;
            }
        }
    }

    /// <summary>
    /// Destroys the IpcMemory object and releases allocated IPC memory.
    /// </summary>
    IpcMemory::~IpcMemory()
    {
        destroyIpcMemory();
    }

    /// <summary>
    /// Releases the allocated IPC memory.
    /// </summary>
    void IpcMemory::destroyIpcMemory()
    {
        for (SizeType nodeId = 0; nodeId < mWorldConfig.getTensorParallelism(); ++nodeId)
        {
            if (nodeId == static_cast<size_t>(mWorldConfig.getTensorParallelRank()))
            {
                CUDA_CHECK(cudaFree(mCommPtrs[nodeId]));
            }
            else
            {
                CUDA_CHECK(cudaIpcCloseMemHandle(mCommPtrs[nodeId]));
            }
        }
        cudaFree(mBufferPtr);
    }
}
