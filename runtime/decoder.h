#pragma once

#include "../common/cudaAllocator.h"
#include "../runtime/bufferManager.h"
#include "decodingInput.h"
#include "decodingOutput.h"
#include "samplingConfig.h"
#include <curand_kernel.h>

#include <cstdint>
#include <memory>

#include <NvInferRuntime.h>

namespace bitfusion
{
    namespace layers
    {
        template <typename T>
        class DynamicDecodeLayer;
    }

    namespace runtime
    {
        /// <summary>
        /// Interface for decoding operations.
        /// </summary>
        class IDecoder
        {
        public:
            virtual ~IDecoder() = default;

            /// <summary>
            /// Set up the decoder with the provided sampling configuration, batch size, and maximum sequence length.
            /// </summary>
            /// <param name="samplingConfig">Sampling configuration for decoding.</param>
            /// <param name="batchSize">Batch size for decoding.</param>
            /// <param name="maxSequenceLength">Maximum sequence length for decoding.</param>
            virtual void setup(const SamplingConfig& samplingConfig, size_t batchSize, SizeType maxSequenceLength) = 0;

            /// <summary>
            /// Perform forward decoding and update the output.
            /// </summary>
            /// <param name="output">Decoding output to be updated.</param>
            /// <param name="input">Decoding input.</param>
            /// <returns>True if the decoding was successful, false otherwise.</returns>
            virtual bool forward(DecodingOutput& output, const DecodingInput& input) = 0;

            /// <summary>
            /// Perform forward decoding asynchronously and update the output.
            /// </summary>
            /// <param name="output">Decoding output to be updated.</param>
            /// <param name="input">Decoding input.</param>
            virtual void forwardAsync(DecodingOutput& output, const DecodingInput& input) = 0;

            /// <summary>
            /// Gather the final output of the decoding operation.
            /// </summary>
            /// <param name="finalOutputIds">Final output tensor for IDs.</param>
            /// <param name="decodingOutput">Decoding output.</param>
            /// <param name="decodingInput">Decoding input.</param>
            /// <param name="manager">Buffer manager for resource management.</param>
            virtual void gatherTree(ITensor& finalOutputIds, const DecodingOutput& decodingOutput,
                const DecodingInput& decodingInput, const BufferManager& manager) = 0;

            /// <summary>
            /// Get the sampling configuration for decoding.
            /// </summary>
            /// <returns>Sampling configuration for decoding.</returns>
            virtual const SamplingConfig& getSamplingConfig() = 0;

            /// <summary>
            /// Create an instance of IDecoder based on the data type, vocabulary size, and stream.
            /// </summary>
            /// <param name="dtype">Data type for decoding.</param>
            /// <param name="vocabSize">Vocabulary size.</param>
            /// <param name="vocabSizePadded">Padded vocabulary size.</param>
            /// <param name="stream">CUDA stream for decoding.</param>
            /// <returns>A unique pointer to the created IDecoder instance.</returns>
            static std::unique_ptr<IDecoder> create(nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded,
                const BufferManager::CudaStreamPtr& stream);
        };

        template <typename T>
        class Decoder final : public IDecoder
        {
        public:
            using CudaStreamPtr = BufferManager::CudaStreamPtr;
            using TensorPtr = std::shared_ptr<ITensor>;

            /// <summary>
            /// Constructor for the Decoder class.
            /// </summary>
            /// <param name="vocabSize">Vocabulary size.</param>
            /// <param name="vocabSizePadded">Padded vocabulary size.</param>
            /// <param name="stream">CUDA stream for decoding.</param>
            Decoder(size_t vocabSize, size_t vocabSizePadded, const CudaStreamPtr& stream);

            void setup(const SamplingConfig& samplingConfig, size_t batchSize, SizeType maxSequenceLength) override;

            bool forward(DecodingOutput& output, const DecodingInput& input) override;

            void forwardAsync(DecodingOutput& output, const DecodingInput& input) override;

            void gatherTree(ITensor& finalOutputIds, const DecodingOutput& decodingOutput, const DecodingInput& decodingInput,
                const BufferManager& manager) override;

            const SamplingConfig& getSamplingConfig() override { return mSamplingConfig; }

        private:
            BufferManager mManager;
            common::CudaAllocator mAllocator;
            std::shared_ptr<bitfusion::layers::DynamicDecodeLayer<T>> mDynamicDecodeLayer;
            TensorPtr mLogProbsTiled;
            SamplingConfig mSamplingConfig;
        };

        /// <summary>
        /// Create an instance of IDecoder based on the data type, vocabulary size, and stream.
        /// </summary>
        /// <param name="dtype">Data type for decoding.</param>
        /// <param name="vocabSize">Vocabulary size.</param>
        /// <param name="vocabSizePadded">Padded vocabulary size.</param>
        /// <param name="stream">CUDA stream for decoding.</param>
        /// <returns>A unique pointer to the created IDecoder instance.</returns>
        static std::unique_ptr<IDecoder> create(nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded,
            const BufferManager::CudaStreamPtr& stream)
        {
            switch (dtype)
            {
            case nvinfer1::DataType::kFLOAT:
                return std::make_unique<Decoder<float>>(vocabSize, vocabSizePadded, stream);
            case nvinfer1::DataType::kHALF:
                return std::make_unique<Decoder<half>>(vocabSize, vocabSizePadded, stream);
            default:
                return nullptr;
            }
        }
    }
}