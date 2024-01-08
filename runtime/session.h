#pragma once

#include "../batch_manager/kvCacheConfig.h"
#include "bufferManager.h"
#include "common.h"
#include "cudaEvent.h"
#include "generationInput.h"
#include "generationOutput.h"
#include "ModelConfig.h"
#include "iTensor.h"
#include "samplingConfig.h"
#include "worldConfig.h"
#include <NvInferRuntime.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/// <summary>
/// Namespace containing classes related to batch management.
/// </summary>
namespace bitfusion::batch_manager
{
    /// <summary>
    /// Forward declaration of the TrtGptModelV1 class.
    /// </summary>
    class TrtGptModelV1;
}

/// <summary>
/// Namespace containing classes related to key-value cache management.
/// </summary>
namespace bitfusion::batch_manager::kv_cache_manager
{
    /// <summary>
    /// Forward declaration of the KVCacheManager class.
    /// </summary>
    class KVCacheManager;
}

/// <summary>
/// Namespace containing runtime-related classes and utilities.
/// </summary>
namespace bitfusion::runtime
{
    /// <summary>
    /// Namespace containing utility functions for runtime operations.
    /// </summary>
    namespace utils
    {
        /// <summary>
        /// Load an engine from the specified engine file path and return it as a vector of bytes.
        /// </summary>
        /// <param name="enginePath">The path to the engine file.</param>
        /// <returns>A vector of bytes representing the loaded engine.</returns>
        std::vector<uint8_t> loadEngine(std::string const& enginePath);
    }

    /// <summary>
    /// Forward declaration of the IpcMemory class.
    /// </summary>
    class IpcMemory;

    /// <summary>
    /// Forward declaration of the IStatefulDecoder class.
    /// </summary>
    class IStatefulDecoder;

    /// <summary>
    /// Forward declaration of the NcclCommunicator class.
    /// </summary>
    class NcclCommunicator;

    /// <summary>
    /// Forward declaration of the RuntimeBuffers class.
    /// </summary>
    class RuntimeBuffers;

    /// <summary>
    /// Forward declaration of the Runtime class.
    /// </summary>
    class Runtime;

    /// <summary>
    /// Main class representing an inference session for a neural network model.
    /// </summary>
    class Session
    {
        using KvCacheManager = batch_manager::kv_cache_manager::KVCacheManager;
        using KvCacheConfig = batch_manager::kv_cache_manager::KvCacheConfig;
        using TensorPtr = runtime::ITensor::SharedPtr;
        using TokenGeneratedCallback = std::function<void(SizeType step, bool finished)>;

    public:
        using LoggerPtr = std::shared_ptr<nvinfer1::ILogger>;

        /// <summary>
        /// Configuration class for a Session.
        /// </summary>
        class Config
        {
        public:
            /// <summary>
            /// Constructor to initialize session configuration.
            /// </summary>
            /// <param name="maxBatchSize">Maximum batch size for the session.</param>
            /// <param name="maxBeamWidth">Maximum beam width for the session.</param>
            /// <param name="maxSequenceLength">Maximum sequence length for the session.</param>
            Config(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength)
                : maxBatchSize{ maxBatchSize }
                , maxBeamWidth{ maxBeamWidth }
                , maxSequenceLength{ maxSequenceLength }
            {
            }

            SizeType maxBatchSize;
            SizeType maxBeamWidth;
            SizeType maxSequenceLength;
            bool decoderPerRequest{ false };
            bool cudaGraphMode{ false };
            KvCacheConfig kvCacheConfig{};
            std::optional<SizeType> ctxMicroBatchSize = std::nullopt;
            std::optional<SizeType> genMicroBatchSize = std::nullopt;
        };

        /// <summary>
        /// Constructor for the Session class using a pre-built engine buffer.
        /// </summary>
        /// <param name="sessionConfig">Configuration for the session.</param>
        /// <param name="modelConfig">Configuration for the model.</param>
        /// <param name="worldConfig">Configuration for the world.</param>
        /// <param name="engineBuffer">Pointer to the engine buffer.</param>
        /// <param name="engineSize">Size of the engine buffer in bytes.</param>
        /// <param name="logger">Optional logger for the session.</param>
        Session(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
            void const* engineBuffer, std::size_t engineSize, LoggerPtr logger = nullptr);

        /// <summary>
        /// Constructor for the Session class using an engine buffer provided as a vector of bytes.
        /// </summary>
        /// <param name="sessionConfig">Configuration for the session.</param>
        /// <param name="modelConfig">Configuration for the model.</param>
        /// <param name="worldConfig">Configuration for the world.</param>
        /// <param name="engineBuffer">Vector of bytes containing the engine buffer.</param>
        /// <param name="logger">Optional logger for the session.</param>
        Session(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
            std::vector<uint8_t> const& engineBuffer, LoggerPtr logger = nullptr)
            : Session(
                sessionConfig, modelConfig, worldConfig, engineBuffer.data(), engineBuffer.size(), std::move(logger))
        {
        }

        /// <summary>
        /// Constructor for the Session class using an engine file.
        /// </summary>
        /// <param name="sessionConfig">Configuration for the session.</param>
        /// <param name="modelConfig">Configuration for the model.</param>
        /// <param name="worldConfig">Configuration for the world.</param>
        /// <param name="engineFile">Path to the engine file.</param>
        /// <param name="logger">Optional logger for the session.</param>
        Session(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
            std::string const& engineFile, LoggerPtr logger = nullptr)
            : Session(sessionConfig, modelConfig, worldConfig, utils::loadEngine(engineFile), std::move(logger))
        {
        }

        /// <summary>
        /// Get the logger associated with this session.
        /// </summary>
        /// <returns>Reference to the session's logger.</returns>
        [[nodiscard]] nvinfer1::ILogger& getLogger() const;

        /// <summary>
        /// Get the buffer manager associated with this session.
        /// </summary>
        /// <returns>Reference to the session's buffer manager.</returns>
        [[nodiscard]] BufferManager const& getBufferManager() const;

        /// <summary>
        /// Get the model configuration used in this session.
        /// </summary>
        /// <returns>Reference to the session's model configuration.</returns>
        [[nodiscard]] ModelConfig const& getModelConfig() const
        {
            return mModelConfig;
        }

        /// <summary>
        /// Get the world configuration used in this session.
        /// </summary>
        /// <returns>Reference to the session's world configuration.</returns>
        [[nodiscard]] WorldConfig const& getWorldConfig() const
        {
            return mWorldConfig;
        }

        /// <summary>
        /// Get the device ID associated with this session.
        /// </summary>
        /// <returns>The device ID used for inference.</returns>
        [[nodiscard]] int getDevice() const noexcept
        {
            return mDevice;
        }

        /// <summary>
        /// Generate sequences using the configured model and inputs.
        /// </summary>
        /// <param name="outputs">Output container for generated sequences.</param>
        /// <param name="inputs">Input configuration for generation.</param>
        /// <param name="samplingConfig">Sampling configuration for generation.</param>
        void generate(GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig);

    private:
        /// <summary>
        /// Check if CUDA graphs are enabled in the session.
        /// </summary>
        /// <returns>True if CUDA graphs are enabled, false otherwise.</returns>
        [[nodiscard]] bool useCudaGraphs()
        {
            return !mCudaGraphInstances.empty();
        }

        /// <summary>
        /// Generate sequences for a batch of micro-batches.
        /// </summary>
        /// <param name="microBatchesOutputs">Output container for generated sequences.</param>
        /// <param name="microBatchesInputs">Input configurations for generation.</param>
        /// <param name="samplingConfig">Sampling configuration for generation.</param>
        /// <param name="onTokenGenerated">Callback function to track token generation progress.</param>
        void generateBatched(std::vector<GenerationOutput>& microBatchesOutputs,
            std::vector<GenerationInput> const& microBatchesInputs, SamplingConfig const& samplingConfig,
            TokenGeneratedCallback const& onTokenGenerated);

        /// <summary>
        /// Setup the session using the provided configuration.
        /// </summary>
        /// <param name="sessionConfig">Configuration for the session.</param>
        void setup(Config const& sessionConfig);

        /// <summary>
        /// Create execution contexts for inference.
        /// </summary>
        /// <param name="numBatchesCtx">Number of context batches.</param>
        /// <param name="numBatchesGen">Number of generation batches.</param>
        /// <param name="useCudaGraphs">Flag indicating whether CUDA graphs are used.</param>
        void createContexts(SizeType numBatchesCtx, SizeType numBatchesGen, bool useCudaGraphs);

        /// <summary>
        /// Create buffer manager for input and output data.
        /// </summary>
        /// <param name="numMicroBatches">Number of micro-batches.</param>
        void createBuffers(SizeType numMicroBatches);

        /// <summary>
        /// Create decoder instances for generation.
        /// </summary>
        /// <param name="batchSize">Batch size for generation.</param>
        /// <param name="beamWidth">Beam width for generation.</param>
        /// <param name="maxAttentionWindow">Maximum attention window size.</param>
        /// <param name="maxSequenceLength">Maximum sequence length.</param>
        /// <param name="logitsType">Data type for logits.</param>
        /// <param name="decoderPerRequest">Flag indicating if a decoder is created per request.</param>
        /// <param name="numMicroBatches">Number of micro-batches.</param>
        void createDecoders(SizeType batchSize, SizeType beamWidth, SizeType maxAttentionWindow, SizeType maxSequenceLength,
            nvinfer1::DataType logitsType, bool decoderPerRequest, SizeType numMicroBatches);

        /// <summary>
        /// Create a key-value cache manager for generation.
        /// </summary>
        /// <param name="batchSize">Batch size for generation.</param>
        /// <param name="beamWidth">Beam width for generation.</param>
        /// <param name="maxAttentionWindow">Maximum attention window size.</param>
        /// <param name="maxSequenceLength">Maximum sequence length.</param>
        /// <param name="config">Configuration for the cache manager.</param>
        void createKvCacheManager(SizeType batchSize, SizeType beamWidth, SizeType maxAttentionWindow,
            SizeType maxSequenceLength, KvCacheConfig const& config);

        /// <summary>
        /// Create a custom all-reduce workspace for generation.
        /// </summary>
        /// <param name="batchSize">Batch size for generation.</param>
        /// <param name="beamWidth">Beam width for generation.</param>
        /// <param name="maxSequenceLength">Maximum sequence length.</param>
        void createCustomAllReduceWorkspace(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength);

        /// <summary>
        /// Execute a step within the inference context for a batch of micro-batches.
        /// </summary>
        /// <param name="microBatches">Input configurations for the micro-batches.</param>
        /// <param name="microBatchOffsets">Offsets for the micro-batches.</param>
        /// <param name="kvCacheManager">Pointer to the key-value cache manager.</param>
        void executeContextStep(std::vector<GenerationInput> const& microBatches,
            std::vector<SizeType> const& microBatchOffsets, KvCacheManager const* kvCacheManager);

        /// <summary>
        /// Execute a generation step for a batch of micro-batches.
        /// </summary>
        /// <param name="step">Current step in generation.</param>
        /// <param name="microBatchesInputs">Input configurations for the micro-batches.</param>
        /// <param name="microBatchesOutputs">Output container for the micro-batches.</param>
        /// <param name="microBatchOffsets">Offsets for the micro-batches.</param>
        /// <param name="kvCacheManager">Pointer to the key-value cache manager.</param>
        /// <param name="microBatchesFinished">Flags indicating if micro-batches are finished.</param>
        /// <returns>The number of finished micro-batches.</returns>
        SizeType executeGenerationStep(SizeType step, std::vector<GenerationInput> const& microBatchesInputs,
            std::vector<GenerationOutput>& microBatchesOutputs, std::vector<SizeType> const& microBatchOffsets,
            KvCacheManager* kvCacheManager, std::vector<bool>& microBatchesFinished);

        /// <summary>
        /// Asynchronously perform a decoder step for a micro-batch.
        /// </summary>
        /// <param name="decoderStep">Current step in the decoder.</param>
        /// <param name="microBatchId">Identifier for the micro-batch.</param>
        void decoderStepAsync(SizeType decoderStep, SizeType microBatchId);

        /// <summary>
        /// Check if inference should stop synchronously for a micro-batch.
        /// </summary>
        /// <param name="batchSize">Batch size for generation.</param>
        /// <param name="beamWidth">Beam width for generation.</param>
        /// <param name="microBatchId">Identifier for the micro-batch.</param>
        /// <returns>True if inference should stop synchronously, false otherwise.</returns>
        bool shouldStopSync(SizeType batchSize, SizeType beamWidth, SizeType microBatchId);

        /// <summary>
        /// Finalize the generation process for a micro-batch.
        /// </summary>
        /// <param name="microBatchId">Identifier for the micro-batch.</param>
        void finalize(SizeType microBatchId);

        /// <summary>
        /// Add sequences to the key-value cache for a micro-batch.
        /// </summary>
        /// <param name="beamWidth">Beam width for generation.</param>
        /// <param name="microBatchId">Identifier for the micro-batch.</param>
        /// <param name="firstBatchIdx">Index of the first batch in the micro-batch.</param>
        void kvCacheAddSequences(SizeType beamWidth, SizeType microBatchId, SizeType firstBatchIdx);

        /// <summary>
        /// Initialize the decoder for generation.
        /// </summary>
        /// <param name="outputIds">Output tensor for generated IDs.</param>
        /// <param name="inputs">Input configuration for generation.</param>
        /// <param name="outputs">Output configuration for generation.</param>
        /// <param name="samplingConfig">Sampling configuration for generation.</param>
        /// <param name="microBatchId">Identifier for the micro-batch.</param>
        /// <returns>Shared pointer to the initialized decoder.</returns>
        ITensor::SharedPtr initDecoder(ITensor& outputIds, GenerationInput const& inputs, GenerationOutput const& outputs,
            SamplingConfig const& samplingConfig, SizeType microBatchId) const;

        /// <summary>
        /// Create a callback function to track token generation progress.
        /// </summary>
        /// <param name="outputs">Output container for generated sequences.</param>
        /// <returns>TokenGeneratedCallback function for tracking token generation.</returns>
        TokenGeneratedCallback createOnTokenGeneratedCallback(GenerationOutput& outputs);

        /// <summary>
        /// Class for managing CUDA graph execution.
        /// </summary>
        class CudaGraphExecutor
        {
        public:
            /// <summary>
            /// Default constructor for CudaGraphExecutor.
            /// </summary>
            CudaGraphExecutor() = default;

            /// <summary>
            /// Destructor for CudaGraphExecutor, clears the instance and handles exceptions.
            /// </summary>
            ~CudaGraphExecutor()
            {
                try
                {
                    clear();
                }
                catch (std::exception& e)
                {
                    LOG_EXCEPTION(e);
                }
            }

            /// <summary>
            /// Check if a CUDA graph execution instance exists.
            /// </summary>
            /// <returns>True if an instance exists, false otherwise.</returns>
            bool hasInstance()
            {
                return mInstance != nullptr;
            }

            /// <summary>
            /// Clear the CUDA graph execution instance.
            /// </summary>
            void clear();

            /// <summary>
            /// Prepare the next CUDA graph for execution.
            /// </summary>
            /// <param name="runtime">Reference to the runtime.</param>
            /// <param name="nextContextId">The next context ID.</param>
            void prepareNextGraph(Runtime const& runtime, SizeType nextContextId);

            /// <summary>
            /// Launch the CUDA graph on a specified CUDA stream.
            /// </summary>
            /// <param name="stream">Reference to the CUDA stream.</param>
            void launch(CudaStream const& stream);

        private:
            /// <summary>
            /// Create a CUDA graph execution instance.
            /// </summary>
            /// <param name="graph">Reference to the CUDA graph.</param>
            void create(cudaGraph_t const& graph);

            /// <summary>
            /// Update the CUDA graph execution instance with a new graph.
            /// </summary>
            /// <param name="graph">Reference to the new CUDA graph.</param>
            /// <returns>True if the update was successful, false otherwise.</returns>
            bool update(cudaGraph_t const& graph);

            /// <summary>
            /// Upload the CUDA graph execution instance to a CUDA stream.
            /// </summary>
            /// <param name="stream">Reference to the CUDA stream.</param>
            void uploadToStream(CudaStream const& stream);

            /// <summary>
            /// Pointer to the CUDA graph execution instance.
            /// </summary>
            cudaGraphExec_t mInstance;
        };

        /// <summary>
        /// Configuration class for micro-batching in a neural network model.
        /// </summary>
        class MicroBatchConfig
        {
        public:
            /// <summary>
            /// Default constructor initializing default values.
            /// </summary>
            MicroBatchConfig()
                : numCtxBatches{ 1 }
                , numGenBatches{ 1 }
                , ctxBatchSize{ 0 }
                , genBatchSize{ 0 }
            {
            }

            /// <summary>
            /// Constructor for initializing micro-batch configuration.
            /// </summary>
            /// <param name="maxBatchSize">Maximum batch size for the model.</param>
            /// <param name="pipelineParallelism">Number of pipeline stages.</param>
            /// <param name="genMicroBatchSize">Optional generation micro-batch size.</param>
            /// <param name="ctxMicroBatchSize">Optional context micro-batch size.</param>
            explicit MicroBatchConfig(SizeType maxBatchSize, SizeType pipelineParallelism,
                std::optional<SizeType> genMicroBatchSize, std::optional<SizeType> ctxMicroBatchSize);

            /// <summary>
            /// Calculate the number of context batches per generation batch.
            /// </summary>
            /// <returns>The number of context batches per generation batch.</returns>
            constexpr SizeType numCtxPerGen() const
            {
                return numCtxBatches / numGenBatches;
            }

            /// <summary>
            /// Get the context context ID given generation batch ID and context batch ID.
            /// </summary>
            /// <param name="generationBatchId">Generation batch ID.</param>
            /// <param name="contextBatchId">Context batch ID.</param>
            /// <returns>The context context ID.</returns>
            constexpr SizeType getCtxContextId(SizeType generationBatchId, SizeType contextBatchId) const
            {
                return 2 * numGenBatches + generationBatchId * numCtxPerGen() + contextBatchId;
            }

            /// <summary>
            /// Get the generation context ID given flip-flop ID and generation batch ID.
            /// </summary>
            /// <param name="flipFlopId">Flip-flop ID.</param>
            /// <param name="generationBatchId">Generation batch ID.</param>
            /// <returns>The generation context ID.</returns>
            constexpr SizeType getGenContextId(SizeType flipFlopId, SizeType generationBatchId) const
            {
                return flipFlopId * numGenBatches + generationBatchId;
            }

            /// <summary>
            /// Number of context batches.
            /// </summary>
            SizeType numCtxBatches;

            /// <summary>
            /// Number of generation batches.
            /// </summary>
            SizeType numGenBatches;

            /// <summary>
            /// Context batch size.
            /// </summary>
            SizeType ctxBatchSize;

            /// <summary>
            /// Generation batch size.
            /// </summary>
            SizeType genBatchSize;
        };

        /// <summary>
        /// Friend class declaration for access to the MicroBatchConfig class.
        /// </summary>
        friend class batch_manager::TrtGptModelV1;

    private:
        /// <summary>
        /// Configuration for the model.
        /// </summary>
        ModelConfig const mModelConfig;

        /// <summary>
        /// Configuration for the world.
        /// </summary>
        WorldConfig const mWorldConfig;

        /// <summary>
        /// The device ID.
        /// </summary>
        int mDevice{ -1 };

        /// <summary>
        /// The communicator for Nccl.
        /// </summary>
        std::shared_ptr<NcclCommunicator> mPipelineComm;

        /// <summary>
        /// The CUDA stream for communication.
        /// </summary>
        std::shared_ptr<CudaStream> mCommStream;

        /// <summary>
        /// CUDA event for communication.
        /// </summary>
        CudaEvent mCommEvent{};

        /// <summary>
        /// Shared pointer to an ITensor.
        /// </summary>
        ITensor::SharedPtr mCommPtrs;

        /// <summary>
        /// List of shared pointers to IpcMemory.
        /// </summary>
        std::vector<std::shared_ptr<IpcMemory>> mIpcMemoryHandles;

        /// <summary>
        /// Maximum sequence length for the decoder.
        /// </summary>
        SizeType mDecoderMaxSequenceLength{};

        /// <summary>
        /// Maximum attention window for the decoder.
        /// </summary>
        SizeType mDecoderMaxAttentionWindow{};

        /// <summary>
        /// Logger instance.
        /// </summary>
        LoggerPtr mLogger;

        /// <summary>
        /// Runtime instance.
        /// </summary>
        std::shared_ptr<Runtime> mRuntime;

        /// <summary>
        /// Key-Value cache manager.
        /// </summary>
        std::shared_ptr<KvCacheManager> mKvCacheManager;

        /// <summary>
        /// Configuration for micro-batch processing.
        /// </summary>
        MicroBatchConfig mMicroBatchConfig;

        /// <summary>
        /// List of shared pointers to stateful decoders.
        /// </summary>
        std::vector<std::shared_ptr<IStatefulDecoder>> mDecoders;

        /// <summary>
        /// List of shared pointers to runtime buffers.
        /// </summary>
        std::vector<std::shared_ptr<RuntimeBuffers>> mBuffers;

        /// <summary>
        /// List of CUDA events for received data.
        /// </summary>
        std::vector<CudaEvent> mReceivedEvents;

        /// <summary>
        /// Flag indicating CUDA graph mode.
        /// </summary>
        bool mCudaGraphMode{ false };

        /// <summary>
        /// List of CUDA graph executor instances.
        /// </summary>
        std::vector<CudaGraphExecutor> mCudaGraphInstances;
    };
}