#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <cublas_v2.h>

#include "DataReader.h"

namespace astdl
{
    namespace knn
    {

        /// <summary>
        /// Enumeration representing data types.
        /// </summary>
        enum class DataType
        {
            FP32 = 0, FP16 = 1
        };

        /// <summary>
        /// Converts a DataType enum value to its string representation.
        /// </summary>
        /// <param name="dataType">The DataType enum value.</param>
        /// <returns>The string representation of the DataType.</returns>
        std::string getDataTypeString(DataType dataType);

        /// <summary>
        /// Converts a string representation of a DataType to the corresponding enum value.
        /// </summary>
        /// <param name="dataTypeLiteral">The string representation of the DataType.</param>
        /// <returns>The DataType enum value.</returns>
        DataType getDataTypeFromString(const std::string& dataTypeLiteral);

        /// <summary>
        /// Structure representing a matrix.
        /// </summary>
        struct Matrix
        {
            void* data;
            uint32_t numRows;
            int numColumns;
            size_t elementSize;
            cudaMemoryType memoryType;

            /// <summary>
            /// Default constructor for Matrix.
            /// </summary>
            Matrix();

            /// <summary>
            /// Constructor for Matrix with specified parameters.
            /// </summary>
            Matrix(void* data, uint32_t numRows, int numColumns, size_t elementSize, cudaMemoryType memoryType);

            /// <summary>
            /// Get the size of the matrix in bytes.
            /// </summary>
            /// <returns>The size of the matrix in bytes.</returns>
            size_t getSizeInBytes();

            /// <summary>
            /// Get the length of the matrix.
            /// </summary>
            /// <returns>The length of the matrix.</returns>
            size_t getLength();
        };

        /// <summary>
        /// Load data from a DataReader onto the host.
        /// </summary>
        /// <param name="dataReader">The DataReader object.</param>
        /// <returns>A Matrix containing the loaded data.</returns>
        Matrix loadDataOnHost(DataReader* dataReader);

        /// <summary>
        /// Structure representing K-Nearest Neighbors (KNN) data.
        /// </summary>
        struct KnnData
        {
            const int numGpus;
            const int batchSize;
            const int maxK;

            std::vector<cublasHandle_t> cublasHandles;

            std::vector<Matrix> dCollectionPartitions;
            std::vector<Matrix> dInputBatches;
            std::vector<Matrix> dProducts;

            std::vector<Matrix> dResultScores;
            std::vector<Matrix> dResultIndexes;

            std::vector<Matrix> hResultScores;
            std::vector<Matrix> hResultIndexes;
            std::vector<std::vector<std::string>> hKeys;

            std::vector<Matrix> dInputBatchTmpBuffers;

            std::vector<uint32_t> collectionRowsPadded;

            std::vector<float> elapsedSgemm;
            std::vector<float> elapsedTopK;

            const DataType dataType;

            /// <summary>
            /// Constructor for KnnData with specified parameters.
            /// </summary>
            KnnData(int numGpus, int batchSize, int maxK, DataType dataType);

            /// <summary>
            /// Load KNN data on a specific device from a DataReader.
            /// </summary>
            /// <param name="device">The device number.</param>
            /// <param name="dataReader">The DataReader object.</param>
            void load(int device, DataReader* dataReader);

            /// <summary>
            /// Load KNN data on multiple devices from a map of device-to-DataReader.
            /// </summary>
            /// <param name="deviceToData">A map of device number to DataReader.</param>
            void load(const std::map<int, DataReader*>& deviceToData);

            /// <summary>
            /// Load KNN data on multiple devices from a map of device-to-file mappings.
            /// </summary>
            /// <param name="deviceToFile">A map of device number to file name.</param>
            /// <param name="keyValDelim">The delimiter for key-value pairs in the file.</param>
            /// <param name="vecDelim">The delimiter for vectors in the file.</param>
            void load(const std::map<int, std::string>& deviceToFile, char keyValDelim, char vecDelim);

            /// <summary>
            /// Get the feature size of the KNN data.
            /// </summary>
            /// <returns>The feature size.</returns>
            int getFeatureSize() const;

            /// <summary>
            /// Destructor for KnnData.
            /// </summary>
            ~KnnData();
        };

        /// <summary>
        /// Allocate a matrix on the host.
        /// </summary>
        /// <param name="numRows">The number of rows in the matrix.</param>
        /// <param name="numColumns">The number of columns in the matrix.</param>
        /// <param name="elementSize">The size of each element in bytes.</param>
        /// <returns>A Matrix allocated on the host.</returns>
        Matrix allocateMatrixOnHost(uint32_t numRows, int numColumns, size_t elementSize);

        /// <summary>
        /// Allocate a matrix on the device.
        /// </summary>
        /// <param name="numRows">The number of rows in the matrix.</param>
        /// <param name="numColumns">The number of columns in the matrix.</param>
        /// <param name="elementSize">The size of each element in bytes.</param>
        /// <returns>A Matrix allocated on the device.</returns>
        Matrix allocateMatrixOnDevice(uint32_t numRows, int numColumns, size_t elementSize);

        /// <summary>
        /// Free the memory associated with a matrix.
        /// </summary>
        /// <param name="matrix">The Matrix to free.</param>
        void freeMatrix(const Matrix& matrix);

    }
}