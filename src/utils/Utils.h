#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <ratio>
#include <string>
#include <utility>

const std::string NETCDF_FILE_EXTENTION = ".nc";
const std::string INPUT_DATASET_SUFFIX = "_input";
const std::string OUTPUT_DATASET_SUFFIX = "_output";
const unsigned long FIXED_SEED = 12134ull;

/// <summary>
/// Get the value of a command-line option.
/// </summary>
/// <param name="argc">The number of command-line arguments.</param>
/// <param name="argv">The array of command-line arguments.</param>
/// <param name="flag">The option flag.</param>
/// <returns>A pointer to the value of the option, or nullptr if not found.</returns>
char* getCmdOption(char**, char**, const std::string&);

/// <summary>
/// Check if a command-line option exists.
/// </summary>
/// <param name="argc">The number of command-line arguments.</param>
/// <param name="argv">The array of command-line arguments.</param>
/// <param name="flag">The option flag to check.</param>
/// <returns>true if the option exists, false otherwise.</returns>
bool cmdOptionExists(char**, char**, const std::string&);

/// <summary>
/// Get the value of a required command-line argument.
/// </summary>
/// <param name="argc">The number of command-line arguments.</param>
/// <param name="argv">The array of command-line arguments.</param>
/// <param name="flag">The argument flag.</param>
/// <param name="message">The error message to display if the argument is missing.</param>
/// <param name="usage">A function pointer to the usage display function.</param>
/// <returns>The value of the required argument.</returns>
std::string getRequiredArgValue(int argc, char** argv, std::string flag, std::string message, void (*usage)());

/// <summary>
/// Get the value of an optional command-line argument.
/// </summary>
/// <param name="argc">The number of command-line arguments.</param>
/// <param name="argv">The array of command-line arguments.</param>
/// <param name="flag">The argument flag.</param>
/// <param name="defaultValue">The default value to use if the argument is missing.</param>
/// <returns>The value of the optional argument or the default value.</returns>
std::string getOptionalArgValue(int argc, char** argv, std::string flag, std::string defaultValue);

/// <summary>
/// Check if a command-line argument is set.
/// </summary>
/// <param name="argc">The number of command-line arguments.</param>
/// <param name="argv">The array of command-line arguments.</param>
/// <param name="flag">The argument flag to check.</param>
/// <returns>true if the argument is set, false otherwise.</returns>
bool isArgSet(int argc, char** argv, std::string flag);

/// <summary>
/// Check if a file exists.
/// </summary>
/// <param name="filename">The name of the file to check.</param>
/// <returns>true if the file exists, false otherwise.</returns>
bool fileExists(const std::string&);

/// <summary>
/// Check if a file is a NetCDF file.
/// </summary>
/// <param name="filename">The name of the file to check.</param>
/// <returns>true if the file is a NetCDF file, false otherwise.</returns>
bool isNetCDFfile(const std::string& filename);

/// <summary>
/// Split a string into a vector of substrings using a delimiter.
/// </summary>
/// <param name="s">The input string to split.</param>
/// <param name="delim">The delimiter character.</param>
/// <param name="elems">The vector to store the resulting substrings.</param>
/// <returns>A reference to the vector of substrings.</returns>
std::vector<std::string>& split(const std::string& s, char delim, std::vector<std::string>& elems);

/// <summary>
/// Split a string into a vector of substrings using a delimiter.
/// </summary>
/// <param name="s">The input string to split.</param>
/// <param name="delim">The delimiter character.</param>
/// <returns>A vector of substrings.</returns>
std::vector<std::string> split(const std::string& s, char delim);

/// <summary>
/// Calculate the elapsed time in seconds between two time points.
/// </summary>
/// <param name="start">The start time point.</param>
/// <param name="end">The end time point.</param>
/// <returns>The elapsed time in seconds.</returns>
template <typename Clock, typename Duration1, typename Duration2>
double elapsed_seconds(std::chrono::time_point<Clock, Duration1> start,
    std::chrono::time_point<Clock, Duration2> end);

/// <summary>
/// Check if a directory exists.
/// </summary>
/// <param name="dirname">The name of the directory to check.</param>
/// <returns>true if the directory exists, false otherwise.</returns>
bool isDirectory(const std::string& dirname);

/// <summary>
/// Check if a file exists.
/// </summary>
/// <param name="filename">The name of the file to check.</param>
/// <returns>true if the file exists, false otherwise.</returns>
bool isFile(const std::string& filename);

/// <summary>
/// List files in a directory.
/// </summary>
/// <param name="dirname">The name of the directory.</param>
/// <param name="recursive">Whether to list files recursively.</param>
/// <param name="files">The vector to store the list of files.</param>
/// <returns>The number of files listed.</returns>
int listFiles(const std::string& dirname, const bool recursive, std::vector<std::string>& files);

/// <summary>
/// Perform a topological sort.
/// </summary>
/// <param name="keys">The array of keys to be sorted.</param>
/// <param name="vals">The array of values corresponding to the keys.</param>
/// <param name="size">The size of the arrays.</param>
/// <param name="topKkeys">The array to store the top K sorted keys.</param>
/// <param name="topKvals">The array to store the top K sorted values.</param>
/// <param name="topK">The number of top elements to retrieve.</param>
/// <param name="sortByKey">Whether to sort by key (true) or value (false).</param>
template<typename Tkey, typename Tval>
void topsort(Tkey* keys, Tval* vals, const int size, Tkey* topKkeys, Tval* topKvals, const int topK, const bool sortByKey = true);

/// <summary>
/// Generate a random integer between min and max (inclusive).
/// </summary>
/// <param name="min">The minimum value.</param>
/// <param name="max">The maximum value.</param>
/// <returns>A random integer between min and max.</returns>
inline int rand(int min, int max) {
    return rand() % (max - min + 1) + min;
}

/// <summary>
/// Generate a random floating-point number between min and max.
/// </summary>
/// <param name="min">The minimum value.</param>
/// <param name="max">The maximum value.</param>
/// <returns>A random floating-point number between min and max.</returns>
inline float rand(float min, float max) {
    float r = (float)rand() / (float)RAND_MAX;
    return min + r * (max - min);
}
