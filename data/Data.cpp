#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "CudaUtil.h"
#include "Data.h"
#include "MathUtil.cuh"

namespace
{
    static const int ROW_PADDING = 8;

    // Mapping from string to DataType enumeration
    static const std::unordered_map<std::string, astdl::knn::DataType> STRING_TO_DATA_TYPE = {
        { "fp32", astdl::knn::DataType::FP32 },
        { "fp16", astdl::knn::DataType::FP16 }
    };
}

namespace astdl
{
    namespace knn
    {
        /// <summary>
        /// Default constructor for the Matrix class. Initializes with default values.
        /// </summary>
        Matrix::Matrix() :
            data(nullptr),
            numRows(0),
            numColumns(0),
            elementSize(0),
            memoryType(cudaMemoryTypeHost)
        {
        }

        /// <summary>
        /// Constructor for the Matrix class with specified parameters.
        /// </summary>
        /// <param name="data">Pointer to the data.</param>
        /// <param name="numRows">Number of rows in the matrix.</param>
        /// <param name="numColumns">Number of columns in the matrix.</param>
        /// <param name="elementSize">Size (in bytes) of each matrix element.</param>
        /// <param name="memoryType">The memory type (host or device).</param>
        Matrix::Matrix(void *data, uint32_t numRows, int numColumns, size_t elementSize, cudaMemoryType memoryType) :
            data(data),
            numRows(numRows),
            numColumns(numColumns),
            elementSize(elementSize),
            memoryType(memoryType)
        {
        }

        /// <summary>
        /// Get the size of the matrix data in bytes.
        /// </summary>
        /// <returns>The size in bytes.</returns>
        size_t Matrix::getSizeInBytes()
        {
            size_t sizeInBytes = numRows * numColumns * elementSize;
            return sizeInBytes;
        }

        /// <summary>
        /// Get the total number of elements in the matrix.
        /// </summary>
        /// <returns>The total number of elements.</returns>
        size_t Matrix::getLength()
        {
            size_t length = numRows * numColumns;
            return length;
        }

        /// <summary>
        /// Get the string representation of the DataType enumeration.
        /// </summary>
        /// <param name="dataType">The DataType enumeration value.</param>
        /// <returns>The string representation of the DataType.</returns>
        std::string getDataTypeString(DataType dataType)
        {
            switch (dataType) {
                case DataType::FP32:
                    return "fp32";
                case DataType::FP16:
                    return "fp16";
                default:
                    return "unknown";
            }
        }

        /// <summary>
        /// Converts a string representation of DataType to its corresponding enum value.
        /// </summary>
        /// <param name="dataTypeLiteral">The string representation of the DataType.</param>
        /// <returns>The DataType enum value corresponding to the input string.</returns>
        DataType getDataTypeFromString(const std::string& dataTypeLiteral)
        {
            // Find the DataType enum value in the mapping
            auto entry = STRING_TO_DATA_TYPE.find(dataTypeLiteral);

            // Check if the input string is valid
            if (entry == STRING_TO_DATA_TYPE.end())
            {
                std::stringstream msg;
                msg << "Unknown DataType " << dataTypeLiteral;
                throw std::invalid_argument(msg.str());
            }

            // Return the corresponding DataType enum value
            return entry->second;
        }

        /// <summary>
        /// Loads data from a DataReader into a Matrix object allocated on the host.
        /// </summary>
        /// <param name="dataReader">A pointer to the DataReader for reading data.</param>
        /// <returns>A Matrix object containing the loaded data on the host.</returns>
        Matrix loadDataOnHost(DataReader* dataReader)
        {
            // Get the number of rows, columns, and element size from the DataReader
            uint32_t rows = dataReader->getRows();
            int columns = dataReader->getColumns();
            size_t elementSize = sizeof(float);

            // Allocate a host-side Matrix for the loaded data
            Matrix matrix = allocateMatrixOnHost(rows, columns, elementSize);

            // Read data from the DataReader and populate the Matrix
            std::string ignored;
            for (int rowNum = 0; dataReader->readRow(&ignored, ((float*)matrix.data) + (rowNum * columns)); ++rowNum)
            {
                // Data is read and placed into the Matrix
            }

            return matrix;
        }

        /// <summary>
        /// Initializes a KnnData object with specified configuration parameters.
        /// </summary>
        /// <param name="numGpus">The number of GPUs to use for computation.</param>
        /// <param name="batchSize">The batch size for processing data.</param>
        /// <param name="maxK">The maximum number of neighbors (K) to find.</param>
        /// <param name="dataType">The data type used for computation (e.g., FP16 or float).</param>
        KnnData::KnnData(int numGpus, int batchSize, int maxK, DataType dataType) :
            numGpus(numGpus),
            batchSize(batchSize),
            maxK(maxK),
            dataType(dataType),
            dCollectionPartitions(numGpus),
            dInputBatches(numGpus),
            dProducts(numGpus),
            dResultScores(numGpus),
            dResultIndexes(numGpus),
            hResultScores(numGpus),
            hResultIndexes(numGpus),
            dInputBatchTmpBuffers(numGpus),
            collectionRowsPadded(numGpus),
            hKeys(numGpus),
            elapsedSgemm(numGpus),
            elapsedTopK(numGpus)
        {
            // Get the number of available GPU devices
            int deviceCount = astdl::cuda_util::getDeviceCount();
            if (deviceCount < 1)
            {
                std::stringstream msg;
                msg << "No GPU device found on host. Device count is " << deviceCount;
                throw std::runtime_error(msg.str());
            }

            // Check if there are enough GPUs available for the specified configuration
            if (deviceCount < numGpus)
            {
                std::stringstream msg;
                msg << "Not enough GPUs on host. Required " << numGpus << ", found " << deviceCount;
                throw std::runtime_error(msg.str());
            }

            // Print initialization information
            fprintf(stderr, "INFO: Initializing KnnData with numGpus = %d, batchSize = %d, maxK = %d, dataType = %s\n", numGpus,
                batchSize, maxK, getDataTypeString(dataType).c_str());

            // Check if the data type is FP16 and if it's supported on the GPU devices
            if (dataType == DataType::FP16)
            {
                cudaDeviceProp deviceProp;
                CHECK_ERR(cudaGetDeviceProperties(&deviceProp, 0));
                int smMajor = deviceProp.major;
                int smMinor = deviceProp.minor;

                if (smMajor < 7)
                {
                    fprintf(stderr, "WARNING: fp16 compute is not supported in sm %d.%d < 7. Only storing data in fp16.\n",
                        smMajor, smMinor);
                }
            }

            // Initialize CUDA and cuBLAS handles for each GPU
            for (int i = 0; i < numGpus; ++i)
            {
                CHECK_ERR(cudaSetDevice(i));
                cublasHandle_t handle;
                STATUS_ERR(cublasCreate(&handle));

                if (dataType == DataType::FP16)
                {
                    fprintf(stderr, "INFO: On device %d, setting cublas mode to CUBLAS_TENSOR_OP_MATH\n", i);
                    STATUS_ERR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
                }

                cublasHandles.push_back(handle);
            }
        }

        /// <summary>
        /// Gets the number of features (columns) in the data collection partitions.
        /// </summary>
        /// <returns>The number of features (columns) in the data collection.</returns>
        int KnnData::getFeatureSize() const
        {
            // Return the number of columns in the first data collection partition
            return dCollectionPartitions[0].numColumns;
        }

        /// <summary>
        /// Loads data for a specific device from a DataReader into the KnnData object.
        /// </summary>
        /// <param name="device">The ID of the device to load data onto.</param>
        /// <param name="dataReader">A pointer to the DataReader for reading data.</param>
        void KnnData::load(int device, DataReader* dataReader)
        {
            // Set the CUDA device to the specified device
            CHECK_ERR(cudaSetDevice(device));

            // Calculate dimensions and allocate memory for the host-side temporary matrix
            uint32_t actualRows = dataReader->getRows();
            uint32_t rows = ((actualRows + (ROW_PADDING - 1)) / ROW_PADDING) * ROW_PADDING;
            uint32_t rowsPadded = rows - actualRows;
            int columns = dataReader->getColumns();

            // Allocate a host-side temporary matrix for data loading
            Matrix hTmpMatrix = allocateMatrixOnHost(rows, columns, sizeof(float));
            size_t hTmpDataBytes = hTmpMatrix.getSizeInBytes();
            float* hTmpData = (float*)hTmpMatrix.data;

            // Store the number of padded rows for the current device
            collectionRowsPadded[device] = rowsPadded;

            // Load data from the DataReader into the temporary matrix
            std::string key;
            std::vector<float> vector(columns);
            for (int rowNum = 0; dataReader->readRow(&key, vector.data()); ++rowNum)
            {
                hKeys[device].push_back(key);
                for (int j = 0; j < columns; ++j) {
                    hTmpData[rowNum * columns + j] = vector[j];
                }
            }

            // Depending on data type, allocate and copy data to device-side matrices
            if (dataType == DataType::FP16)
            {
                dCollectionPartitions[device] = allocateMatrixOnDevice(rows, columns, sizeof(half));
                astdl::math::kFloatToHalf(hTmpData, hTmpDataBytes, (half*)dCollectionPartitions[device].data);

                dInputBatches[device] = allocateMatrixOnDevice(batchSize, columns, sizeof(half));
                dInputBatchTmpBuffers[device] = allocateMatrixOnDevice(batchSize, columns, sizeof(float));
            }
            else
            {
                dCollectionPartitions[device] = allocateMatrixOnDevice(rows, columns, sizeof(float));
                CHECK_ERR(cudaMemcpy(dCollectionPartitions[device].data, hTmpData, hTmpDataBytes, cudaMemcpyHostToDevice));
                dInputBatches[device] = allocateMatrixOnDevice(batchSize, columns, sizeof(float));
            }

            // Free host-side temporary data
            free(hTmpData);

            // Allocate device-side matrices for processing
            dProducts[device] = allocateMatrixOnDevice(batchSize, rows, sizeof(float));
            dResultScores[device] = allocateMatrixOnDevice(batchSize, maxK, sizeof(float));
            dResultIndexes[device] = allocateMatrixOnDevice(batchSize, maxK, sizeof(uint32_t));

            // Allocate host-side result matrices
            hResultScores[device] = allocateMatrixOnHost(batchSize, maxK, sizeof(float));
            hResultIndexes[device] = allocateMatrixOnHost(batchSize, maxK, sizeof(uint32_t));

            // Get and print device memory information
            size_t totalMemory;
            size_t freeMemory;
            astdl::cuda_util::getDeviceMemoryInfoInMb(device, &totalMemory, &freeMemory);

            fprintf(
                stderr,
                "INFO: loaded %zu (%zu padded) rows and %d columns into device %d. Used: %zu MB, Free: %zu MB, Total: %zu MB\n",
                actualRows, rows, columns, device, totalMemory - freeMemory, freeMemory, totalMemory);
        }

        /// <summary>
        /// Load data from DataReader instances into the KnnData object for each device.
        /// </summary>
        /// <param name="deviceToData">A map that associates device IDs with DataReader instances.</param>
        void KnnData::load(const std::map<int, DataReader*>& deviceToData)
        {
            // Set the number of OpenMP threads to the specified number of GPUs
            omp_set_num_threads(numGpus);

            // Use OpenMP parallelism to load data for each device in parallel
#pragma omp parallel
            {
                // Get the device ID for the current thread
                int device = omp_get_thread_num();

                // Find the DataReader instance for the current device
                auto dataReader = deviceToData.find(device);

                // Check if a DataReader is found for the current device
                if (dataReader == deviceToData.end())
                {
                    // Generate an error message if no DataReader is found
                    std::stringstream msg;
                    msg << "Data reader for device " << device << " not specified. Must specify readers for all " << numGpus
                        << " devices";
                    throw std::runtime_error(msg.str());
                }

                // Load data for the current device using the associated DataReader
                load(device, dataReader->second);
            }
        }

        /// <summary>
        /// Loads data from text files into the KnnData object for multiple devices.
        /// </summary>
        /// <param name="deviceToFile">A map that associates device IDs with file paths.</param>
        /// <param name="keyValDelim">The delimiter used to separate keys and values in the text files.</param>
        /// <param name="vecDelim">The delimiter used to separate vector elements in the text files.</param>
        void KnnData::load(const std::map<int, std::string>& deviceToFile, char keyValDelim, char vecDelim)
        {
            // Create a map to associate devices with DataReader instances
            std::map<int, DataReader*> deviceToData;

            // Iterate through the provided device-to-file mapping
            for (const auto& entry : deviceToFile)
            {
                int device = entry.first;
                std::string file = entry.second;

                // Create a DataReader for the current file
                DataReader* dataReader = new TextFileDataReader(file, keyValDelim, vecDelim);

                // Insert the DataReader into the map
                deviceToData.insert({ device, dataReader });
            }

            // Load data for each device using the map of DataReaders
            load(deviceToData);

            // Clean up the allocated DataReader instances
            for (const auto& entry : deviceToData)
            {
                delete entry.second;
            }

            // Clear the deviceToData map
            deviceToData.clear();
        }

        /// <summary>
        /// Destructor for the KnnData class. Cleans up resources and data structures.
        /// </summary>
        KnnData::~KnnData()
        {
            for (auto handle : cublasHandles)
            {
                cublasDestroy(handle);
            }

            for (auto dCollection : dCollectionPartitions)
            {
                freeMatrix(dCollection);
            }

            for (auto dInputBatch : dInputBatches)
            {
                freeMatrix(dInputBatch);
            }

            for (auto dProduct : dProducts)
            {
                freeMatrix(dProduct);
            }

            for (auto dResultScore : dResultScores)
            {
                freeMatrix(dResultScore);
            }

            for (auto dResultIndex : dResultIndexes)
            {
                freeMatrix(dResultIndex);
            }

            for (auto hResultScore : hResultScores)
            {
                freeMatrix(hResultScore);
            }

            for (auto hResultIndex : hResultIndexes)
            {
                freeMatrix(hResultIndex);
            }

            for (auto hKey : hKeys)
            {
                hKey.clear();
            }

            for (auto dInputBatchTmpBuffer : dInputBatchTmpBuffers)
            {
                freeMatrix(dInputBatchTmpBuffer);
            }

            // Clear all data structures and containers
            cublasHandles.clear();
            dCollectionPartitions.clear();
            dInputBatches.clear();
            dProducts.clear();
            dResultScores.clear();
            dResultIndexes.clear();
            hKeys.clear();
            elapsedSgemm.clear();
            elapsedTopK.clear();
        }

        /// <summary>
        /// Allocates memory for a Matrix object on the host.
        /// </summary>
        /// <param name="numRows">The number of rows in the matrix.</param>
        /// <param name="numColumns">The number of columns in the matrix.</param>
        /// <param name="elementSize">The size (in bytes) of each matrix element.</param>
        /// <returns>A Matrix object allocated on the host.</returns>
        Matrix allocateMatrixOnHost(uint32_t numRows, int numColumns, size_t elementSize)
        {
            void* data = malloc(numRows * numColumns * elementSize);
            return Matrix(data, numRows, numColumns, elementSize, cudaMemoryTypeHost);
        }

        /// <summary>
        /// Allocates memory for a Matrix object on the CUDA device.
        /// </summary>
        /// <param name="numRows">The number of rows in the matrix.</param>
        /// <param name="numColumns">The number of columns in the matrix.</param>
        /// <param name="elementSize">The size (in bytes) of each matrix element.</param>
        /// <returns>A Matrix object allocated on the CUDA device.</returns>
        Matrix allocateMatrixOnDevice(uint32_t numRows, int numColumns, size_t elementSize)
        {
            void* data;
            CHECK_ERR(cudaMalloc(&data, numRows * numColumns * elementSize));
            return Matrix(data, numRows, numColumns, elementSize, cudaMemoryTypeDevice);
        }

        /// <summary>
        /// Frees the memory allocated for a Matrix object.
        /// </summary>
        /// <param name="matrix">The Matrix object to be freed.</param>
        void freeMatrix(const Matrix& matrix)
        {
            if (matrix.data != nullptr)
            {
                switch (matrix.memoryType) {
                case cudaMemoryTypeDevice:
                    CHECK_ERR(cudaFree(matrix.data))
                        break;
                case cudaMemoryTypeHost:
                    free(matrix.data);
                    break;
                default:
                    std::stringstream msg;
                    msg << "Unknown memory type " << matrix.memoryType;
                    throw std::invalid_argument(msg.str());
                }
            }
        }
    } // namespace knn
} // namespace astdl