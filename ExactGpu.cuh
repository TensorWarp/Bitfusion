#pragma once

#include "Data.h"
#include <string>
#include <vector>

namespace astdl
{
    namespace knn
    {
        /// <summary>
        /// Abstract class for performing k-Nearest Neighbors (KNN) search.
        /// </summary>
        class Knn
        {
        public:

            /// <summary>
            /// Performs a KNN search on the data.
            /// </summary>
            /// <param name="k">The number of nearest neighbors to find.</param>
            /// <param name="inputs">The input vectors for which nearest neighbors are to be found.</param>
            /// <param name="keys">A pointer to store the keys of the nearest neighbors found.</param>
            /// <param name="scores">A pointer to store the scores (distances) of the nearest neighbors found.</param>
            virtual void search(int k, const float* inputs, std::string* keys, float* scores) = 0;

            /// <summary>
            /// Virtual destructor for the Knn class.
            /// </summary>
            virtual ~Knn()
            {

            }

        protected:
            KnnData* data; ///< A pointer to the data used for KNN search.

            /// <summary>
            /// Constructor for the Knn class.
            /// </summary>
            /// <param name="data">A pointer to the data used for KNN search.</param>
            Knn(KnnData* data);
        };

        /// <summary>
        /// Class for performing exact KNN search on GPU.
        /// </summary>
        class ExactGpu : public Knn
        {
        public:

            /// <summary>
            /// Constructor for ExactGpu.
            /// </summary>
            /// <param name="data">A pointer to the data used for KNN search.</param>
            ExactGpu(KnnData* data);

            /// <summary>
            /// Performs a KNN search on GPU.
            /// </summary>
            /// <param name="k">The number of nearest neighbors to find.</param>
            /// <param name="inputs">The input vectors for which nearest neighbors are to be found.</param>
            /// <param name="size">The size of the input vectors.</param>
            /// <param name="keys">A pointer to store the keys of the nearest neighbors found.</param>
            /// <param name="scores">A pointer to store the scores (distances) of the nearest neighbors found.</param>
            void search(int k, const float* inputs, int size, std::string* keys, float* scores);

            /// <summary>
            /// Performs a KNN search on GPU with a default batch size.
            /// </summary>
            /// <param name="k">The number of nearest neighbors to find.</param>
            /// <param name="inputs">The input vectors for which nearest neighbors are to be found.</param>
            /// <param name="keys">A pointer to store the keys of the nearest neighbors found.</param>
            /// <param name="scores">A pointer to store the scores (distances) of the nearest neighbors found.</param>
            void search(int k, const float* inputs, std::string* keys, float* scores)
            {
                search(k, inputs, data->batchSize, keys, scores);
            }

        };

        /// <summary>
        /// Merges results from multiple GPUs in a multi-GPU KNN search.
        /// </summary>
        /// <param name="k">The number of nearest neighbors to find.</param>
        /// <param name="batchSize">The batch size used in the multi-GPU KNN search.</param>
        /// <param name="width">The width of the search results from each GPU.</param>
        /// <param name="numGpus">The number of GPUs used in the multi-GPU KNN search.</param>
        /// <param name="allScores">A vector of pointers to arrays containing scores from each GPU.</param>
        /// <param name="allIndexes">A vector of pointers to arrays containing indexes from each GPU.</param>
        /// <param name="allKeys">A vector of vectors containing keys from each GPU.</param>
        /// <param name="scores">A pointer to store the merged scores (distances) of the nearest neighbors found.</param>
        /// <param name="keys">A pointer to store the merged keys of the nearest neighbors found.</param>
        void mergeKnn(int k, int batchSize, int width, int numGpus, const std::vector<float*>& allScores,
            const std::vector<uint32_t*>& allIndexes, const std::vector<std::vector<std::string>>& allKeys, float* scores,
            std::string* keys);
    }
}