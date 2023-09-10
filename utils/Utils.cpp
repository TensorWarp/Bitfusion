#include <cerrno>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <cstring>

#include "Utils.h"

#include <filesystem>
namespace fs = std::filesystem;

using namespace std;

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return find(begin, end, option) != end;
}

string getRequiredArgValue(int argc, char** argv, string flag, string message, void (*usage)()) 
{
    if(!cmdOptionExists(argv, argv+argc, flag))
    {
        std::cout << "Error: Missing required argument: " << flag << ": " << message << std::endl;
        usage();
        exit(1);
    } 
    else 
    {
        return string(getCmdOption(argv, argv + argc, flag));
    } 
}

string getOptionalArgValue(int argc, char** argv, string flag, string defaultValue) 
{
    if(!cmdOptionExists(argv, argv+argc, flag))
    {
        return defaultValue;
    } 
    else 
    {
        return string(getCmdOption(argv, argv + argc, flag));
    } 
}

bool isArgSet(int argc, char** argv, string flag) {
    return cmdOptionExists(argv, argv+argc, flag);
}


bool fileExists(const std::string& fileName)
{
    ifstream stream(fileName.c_str());
    if(stream.good()) {
        return true;
    } else {
        return false;
    }
}

bool isNetCDFfile(const string &filename) 
{
    size_t extIndex = filename.find_last_of(".");
    if (extIndex == string::npos) {
        return false;
    }

    string ext = filename.substr(extIndex);
    return (ext.compare(NETCDF_FILE_EXTENTION) == 0);
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

bool isDirectory(const std::string& dirname) {
    return fs::is_directory(dirname);
}

bool isFile(const std::string& filename) {
    return fs::is_regular_file(filename);
}

int listFiles(const std::string& dirname, const bool recursive, std::vector<std::string>& files) {
    try {
        if (isFile(dirname)) {
            files.push_back(dirname);
        }
        else if (isDirectory(dirname)) {
            for (const auto& entry : fs::directory_iterator(dirname)) {
                if (entry.is_directory() && recursive) {
                    listFiles(entry.path().string(), recursive, files);
                }
                else {
                    files.push_back(entry.path().string());
                }
            }
        }
        else {
            return 1;
        }

        std::sort(files.begin(), files.end());
        return 0;
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


template<typename Tkey, typename Tval>
bool cmpFirst(const pair<Tkey, Tval>& left, const pair<Tkey, Tval>& right) {
  if (left.first > right.first) {
    return true;
  } else {
    return false;
  }
}

template<typename Tkey, typename Tval>
bool cmpSecond(const pair<Tkey, Tval>& left, const pair<Tkey, Tval>& right) {
  if (left.second > right.second) {
    return true;
  } else {
    return false;
  }
}

template<typename Tkey, typename Tval>
void topSort(Tkey* keys, Tval* vals, const int size, Tkey* topKkeys, Tval* topKvals, const int topK, const bool sortByKey) {
  if (!keys || !topKkeys || !topKvals) {
    cout << "null input array" << endl;
    exit(0);
  }
  vector<pair<Tkey, Tval> > data(size);
  if (vals) {
    for (int i = 0; i < size; i++) {
      data[i].first = keys[i];
      data[i].second = vals[i];
    }
  } else {
    for (int i = 0; i < size; i++) {
      data[i].first = keys[i];
      data[i].second = i;
    }
  }

  if (sortByKey) {
    std::nth_element(data.begin(), data.begin() + topK, data.end(), cmpFirst<Tkey, Tval>);
    std::sort(data.begin(), data.begin() + topK, cmpFirst<Tkey, Tval>);
  } else {
    std::nth_element(data.begin(), data.begin() + topK, data.end(), cmpSecond<Tkey, Tval>);
    std::sort(data.begin(), data.begin() + topK, cmpSecond<Tkey, Tval>);
  }
  for (int i = 0; i < topK; i++) {
    topKkeys[i] = data[i].first;
    topKvals[i] = data[i].second;
  }
}

template
void topSort<float, unsigned int>(float*, unsigned int*, const int, float*, unsigned int*, const int, const bool);

template
void topSort<float, float>(float*, float*, const int, float*, float*, const int, const bool);

