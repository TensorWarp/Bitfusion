#pragma once

#include "cudaStream.h"
#include "iBuffer.h"
#include "worldConfig.h"

struct ncclComm;
typedef struct ncclComm* ncclComm_t;

namespace bitfusion::runtime
{

    /// <summary>
    /// Class for handling communication using NCCL (NVIDIA Collective Communications Library).
    /// </summary>
    class NcclCommunicator
    {
    public:
        /// <summary>
        /// Send data of type T to a specified peer using a CUDA stream.
        /// </summary>
        /// <typeparam name="T">Type of data to send.</typeparam>
        /// <param name="sendbuff">Pointer to the data to send.</param>
        /// <param name="count">Number of elements to send.</param>
        /// <param name="peer">Peer identifier for communication.</param>
        /// <param name="stream">CUDA stream for the operation.</param>
        template <typename T>
        void send(T* sendbuff, size_t count, int peer, CudaStream const& stream) const;

        /// <summary>
        /// Send data from an IBuffer to a specified peer using a CUDA stream.
        /// </summary>
        /// <typeparam name="T">Type of data to send.</typeparam>
        /// <param name="buf">Reference to the IBuffer containing the data to send.</param>
        /// <param name="peer">Peer identifier for communication.</param>
        /// <param name="stream">CUDA stream for the operation.</param>
        template <typename T>
        void send(IBuffer const& buf, int peer, CudaStream const& stream) const
        {
            send(bufferCast<T>(buf), buf.getSize(), peer, stream);
        }

        /// <summary>
        /// Receive data of type T from a specified peer using a CUDA stream.
        /// </summary>
        /// <typeparam name="T">Type of data to receive.</typeparam>
        /// <param name="sendbuff">Pointer to the data to receive into.</param>
        /// <param name="count">Number of elements to receive.</param>
        /// <param name="peer">Peer identifier for communication.</param>
        /// <param name="stream">CUDA stream for the operation.</param>
        template <typename T>
        void receive(T* sendbuff, size_t count, int peer, CudaStream const& stream) const;

        /// <summary>
        /// Receive data from an IBuffer from a specified peer using a CUDA stream.
        /// </summary>
        /// <typeparam name="T">Type of data to receive.</typeparam>
        /// <param name="buf">Reference to the IBuffer where the data will be received.</param>
        /// <param name="peer">Peer identifier for communication.</param>
        /// <param name="stream">CUDA stream for the operation.</param>
        template <typename T>
        void receive(IBuffer& buf, int peer, CudaStream const& stream) const
        {
            receive(bufferCast<T>(buf), buf.getSize(), peer, stream);
        }

        /// <summary>
        /// Create an instance of NcclCommunicator for pipeline communication based on the provided WorldConfig.
        /// </summary>
        /// <param name="worldConfig">Reference to the WorldConfig object.</param>
        /// <returns>Shared pointer to the created NcclCommunicator.</returns>
        static std::shared_ptr<NcclCommunicator> createPipelineComm(WorldConfig const& worldConfig);

    private:
        ncclComm_t mComm;
    };
}