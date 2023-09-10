#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <ratio>
#include <string>
#include <utility>

const std::string INPUT_DATASET_SUFFIX = "_input";
const std::string OUTPUT_DATASET_SUFFIX = "_output";
const std::string NETCDF_FILE_EXTENTION = ".nc";
const unsigned long FIXED_SEED = 12134ull;

char* getCmdOption(char ** , char **, const std::string & );

bool cmdOptionExists(char** , char**, const std::string& );

std::string getRequiredArgValue(int argc, char** argv, std::string flag, std::string message, void (*usage)());

std::string getOptionalArgValue(int argc, char** argv, std::string flag, std::string defaultValue);

bool isArgSet(int argc, char** argv, std::string flag);

bool fileExists(const std::string &);

bool isNetCDFfile(const std::string &filename);

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

std::vector<std::string> split(const std::string &s, char delim);

template <typename Clock, typename Duration1, typename Duration2>
double elapsed_seconds(std::chrono::time_point<Clock, Duration1> start,
                       std::chrono::time_point<Clock, Duration2> end)
{
  using FloatingPointSeconds = std::chrono::duration<double, std::ratio<1>>;
  return std::chrono::duration_cast<FloatingPointSeconds>(end - start).count();
}

bool isDirectory(const std::string &dirname);

bool isFile(const std::string &filename);

int listFiles(const std::string &dirname, const bool recursive, std::vector<std::string> &files);

template<typename Tkey, typename Tval>
void topsort(Tkey* keys, Tval* vals, const int size, Tkey* topKkeys, Tval* topKvals, const int topK, const bool sortByKey = true);


inline int rand(int min, int max) {
  return rand() % (max - min + 1) + min;
}

inline float rand(float min, float max) {
  float r = (float)rand() / (float)RAND_MAX;
  return min + r * (max - min);
}
