#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

#include "Utils.h"

#include "../ThreadPool.h"

namespace fs = std::filesystem;

/// <summary>
/// Get the value of a command-line option.
/// </summary>
/// <param name="begin">A pointer to the beginning of the command-line arguments array.</param>
/// <param name="end">A pointer to the end of the command-line arguments array.</param>
/// <param name="option">The command-line option to search for.</param>
/// <returns>A pointer to the value of the option if found, nullptr otherwise.</returns>
char* getCmdOption(char** begin, char** end, const std::string& option) {
    auto itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return nullptr;
}

/// <summary>
/// Check if a command-line option exists.
/// </summary>
/// <param name="begin">A pointer to the beginning of the command-line arguments array.</param>
/// <param name="end">A pointer to the end of the command-line arguments array.</param>
/// <param name="option">The command-line option to check for.</param>
/// <returns>True if the option exists, false otherwise.</returns>
bool cmdOptionExists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

/// <summary>
/// Get the value of a required command-line argument or display an error message and exit.
/// </summary>
/// <param name="argc">The number of command-line arguments.</param>
/// <param name="argv">The array of command-line argument strings.</param>
/// <param name="flag">The name of the required argument.</param>
/// <param name="message">A message describing the required argument.</param>
/// <param name="usage">A function pointer to the usage function to display usage information.</param>
/// <returns>The value of the required argument.</returns>
std::string getRequiredArgValue(int argc, char** argv, const std::string& flag, const std::string& message, void (*usage)()) {
    if (!cmdOptionExists(argv, argv + argc, flag)) {
        std::string errorMessage = "Error: Missing required argument: " + flag + ": " + message;
        usage();
        throw std::runtime_error(errorMessage);
    }
    else {
        return std::string(getCmdOption(argv, argv + argc, flag));
    }
}

/// <summary>
/// Get the value of an optional command-line argument or return a default value.
/// </summary>
/// <param name="argc">The number of command-line arguments.</param>
/// <param name="argv">The array of command-line argument strings.</param>
/// <param name="flag">The name of the optional argument.</param>
/// <param name="defaultValue">The default value to return if the argument is not provided.</param>
/// <returns>The value of the optional argument or the default value.</returns>
std::string getOptionalArgValue(int argc, char** argv, const std::string& flag, const std::string& defaultValue) {
    if (!cmdOptionExists(argv, argv + argc, flag)) {
        return defaultValue;
    }
    else {
        return std::string(getCmdOption(argv, argv + argc, flag));
    }
}

/// <summary>
/// Check if a specific command-line flag is set in the given command-line arguments.
/// </summary>
/// <param name="argc">The number of command-line arguments.</param>
/// <param name="argv">The array of command-line argument strings.</param>
/// <param name="flag">The flag to check for.</param>
/// <returns>True if the flag is set, false otherwise.</returns>
bool isArgSet(int argc, char** argv, const std::string& flag) {
    return cmdOptionExists(argv, argv + argc, flag);
}

/// <summary>
/// Check if a file with the given filename exists.
/// </summary>
/// <param name="fileName">The name of the file to check.</param>
/// <returns>True if the file exists, false otherwise.</returns>
bool fileExists(const std::string& fileName) {
    std::ifstream stream(fileName);
    return stream.good();
}

/// <summary>
/// Check if a file with the given filename has the NetCDF file extension.
/// </summary>
/// <param name="filename">The name of the file to check.</param>
/// <returns>True if the file has the NetCDF file extension, false otherwise.</returns>
bool isNetCDFfile(const std::string& filename) {
    size_t extIndex = filename.find_last_of(".");
    if (extIndex == std::string::npos) {
        return false;
    }

    std::string ext = filename.substr(extIndex);
    return (ext.compare(NETCDF_FILE_EXTENTION) == 0);
}

/// <summary>
/// Split a string into a vector of substrings based on a delimiter character.
/// </summary>
/// <param name="s">The input string to split.</param>
/// <param name="delim">The delimiter character.</param>
/// <param name="elems">A reference to the vector to store the split substrings.</param>
/// <returns>A reference to the vector containing the split substrings.</returns>
std::vector<std::string>& split(const std::string& s, char delim, std::vector<std::string>& elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

/// <summary>
/// Split a string into a vector of substrings based on a delimiter character.
/// </summary>
/// <param name="s">The input string to split.</param>
/// <param name="delim">The delimiter character.</param>
/// <returns>A vector containing the split substrings.</returns>
std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

/// <summary>
/// Check if a path corresponds to a directory.
/// </summary>
/// <param name="dirname">The path to check.</param>
/// <returns>True if the path is a directory, false otherwise.</returns>
bool isDirectory(const std::string& dirname) {
    return fs::is_directory(dirname);
}

/// <summary>
/// Check if a path corresponds to a regular file.
/// </summary>
/// <param name="filename">The path to check.</param>
/// <returns>True if the path is a regular file, false otherwise.</returns>
bool isFile(const std::string& filename) {
    return fs::is_regular_file(filename);
}

/// <summary>
/// List files in a directory, optionally recursively, and store them in a vector.
/// </summary>
/// <param name="dirname">The directory name to start listing files from.</param>
/// <param name="recursive">True to list files recursively within subdirectories.</param>
/// <param name="files">A vector to store the list of file paths.</param>
/// <returns>0 on success, 1 on error.</returns>
int listFiles(const std::string& dirname, const bool recursive, std::vector<std::string>& files) {
    try {
        // Check if dirname is a file, and if so, add it to the files vector.
        if (isFile(dirname)) {
            files.push_back(dirname);
        }
        // Check if dirname is a directory.
        else if (isDirectory(dirname)) {
#pragma omp parallel
            {
                // Create a local vector to store file paths.
                std::vector<std::string> localFiles;

#pragma omp for
                // Iterate over files and directories in dirname.
                for (const auto& entry : fs::recursive_directory_iterator(dirname)) {
                    // If it's a directory and not in recursive mode, skip it.
                    if (entry.is_directory() && !recursive) {
                        continue;
                    }
                    // Add the file path to the local vector.
                    localFiles.push_back(entry.path().string());
                }

#pragma omp critical
                {
                    // Safely merge the local vector into the files vector.
                    files.insert(files.end(), localFiles.begin(), localFiles.end());
                }
            }
        }
        else {
            // If dirname is neither a file nor a directory, return an error code.
            return 1;
        }

        // Sort the file paths in the files vector.
        std::sort(files.begin(), files.end());
        return 0;
    }
    catch (const std::filesystem::filesystem_error& e) {
        // Handle filesystem errors and print an error message.
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

/// <summary>
/// Compare two pairs based on their first element (key).
/// </summary>
/// <typeparam name="Tkey">The data type of the first element (key).</typeparam>
/// <typeparam name="Tval">The data type of the second element (value).</typeparam>
/// <param name="left">The left pair to compare.</param>
/// <param name="right">The right pair to compare.</param>
/// <returns>True if the first element of the left pair is greater than the right pair.</returns>
template <typename Tkey, typename Tval>
bool cmpFirst(const std::pair<Tkey, Tval>& left, const std::pair<Tkey, Tval>& right) {
    return left.first > right.first;
}

/// <summary>
/// Compare two pairs based on their second element (value).
/// </summary>
/// <typeparam name="Tkey">The data type of the first element (key).</typeparam>
/// <typeparam name="Tval">The data type of the second element (value).</typeparam>
/// <param name="left">The left pair to compare.</param>
/// <param name="right">The right pair to compare.</param>
/// <returns>True if the second element of the left pair is greater than the right pair.</returns>
template <typename Tkey, typename Tval>
bool cmpSecond(const std::pair<Tkey, Tval>& left, const std::pair<Tkey, Tval>& right) {
    return left.second > right.second;
}

/// <summary>
/// Perform a parallel topological sort on an array of keys and values.
/// </summary>
/// <typeparam name="Tkey">The data type of keys.</typeparam>
/// <typeparam name="Tval">The data type of values.</typeparam>
/// <param name="keys">The input array of keys.</param>
/// <param name="vals">The input array of values (optional).</param>
/// <param name="size">The size of the input arrays.</param>
/// <param name="topKkeys">An array to store the top K keys.</param>
/// <param name="topKvals">An array to store the top K values.</param>
/// <param name="topK">The number of top elements to retrieve.</param>
/// <param name="sortByKey">True to sort by keys, false to sort by values.</param>
template <typename Tkey, typename Tval>
void topSort(Tkey* keys, Tval* vals, const int size, Tkey* topKkeys, Tval* topKvals, const int topK, const bool sortByKey) {
    int rank, numProcs;
    // Get the rank and number of processes in MPI_COMM_WORLD
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (!keys || !topKkeys || !topKvals) {
        if (rank == 0) {
            // If any of the required arrays are null, display an error message.
            std::cerr << "Error: Null input array" << std::endl;
        }
        // Finalize MPI and throw a runtime error.
        MPI_Finalize();
        throw std::runtime_error("Null input array");
    }

    int localSize = size / numProcs;
    // Create a local vector to store a portion of the input data.
    std::vector<std::pair<Tkey, Tval>> localData(localSize);
    // Scatter the input keys to the localData vector on each process.
    MPI_Scatter(keys, localSize, MPI_DOUBLE, &localData[0], localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (vals) {
        // If values are provided, scatter them to the localData vector as well.
        MPI_Scatter(vals, localSize, MPI_UNSIGNED, &localData[0], localSize, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    else {
        // If values are not provided, generate them based on local indices.
        for (int i = 0; i < localSize; i++) {
            localData[i].second = i + rank * localSize;
        }
    }

    // Create a local vector to store the local results.
    std::vector<std::pair<Tkey, Tval>> localResult(localSize * topK);

#pragma omp parallel
    {
        int numThreads = omp_get_num_threads();
        int threadId = omp_get_thread_num();

        auto worker = [&](int i) {
            if (sortByKey) {
                // Sort the localData portion by keys.
                std::nth_element(localData.begin() + i, localData.begin() + i + topK, localData.end(), cmpFirst<Tkey, Tval>);
                std::sort(localData.begin() + i, localData.begin() + i + topK, cmpFirst<Tkey, Tval>);
            }
            else {
                // Sort the localData portion by values.
                std::nth_element(localData.begin() + i, localData.begin() + i + topK, localData.end(), cmpSecond<Tkey, Tval>);
                std::sort(localData.begin() + i, localData.begin() + i + topK, cmpSecond<Tkey, Tval>);
            }
            };

        ThreadPool pool(numThreads);

#pragma omp for
        for (int i = 0; i < localSize; i++) {
            // Execute the worker function for each element in the localData portion.
            worker(i);
        }

        // Shutdown the thread pool.
        pool.shutdown();
    }

    // Gather the localData results from all processes into localResult.
    MPI_Gather(&localData[0], localSize * topK, MPI_DOUBLE, &localResult[0], localSize * topK, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // If the current process is rank 0, aggregate the final results.
        std::vector<std::pair<Tkey, Tval>> result(size * topK);
        // Reduce the localResult data from all processes to the final result.
        MPI_Reduce(&localResult[0], &result[0], size * topK, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        for (int i = 0; i < topK; i++) {
            // Extract the top K keys and values from the final result.
            topKkeys[i] = result[i].first;
            topKvals[i] = result[i].second;
        }
    }

    // Synchronize all processes.
    MPI_Barrier(MPI_COMM_WORLD);
    // Finalize MPI.
    MPI_Finalize();
}

/// <summary>
/// Explicit template instantiation for float keys and unsigned int values.
/// </summary>
template void topSort<float, unsigned int>(float*, unsigned int*, const int, float*, unsigned int*, const int, const bool);

/// <summary>
/// Explicit template instantiation for float keys and float values.
/// </summary>
template void topSort<float, float>(float*, float*, const int, float*, float*, const int, const bool);
