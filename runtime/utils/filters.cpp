#include <json/json.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <chrono>
#include <omp.h>

#include "Filters.h"
#include "Utils.h"

using Json::Value;
using Json::Reader;
namespace fs = std::filesystem;

constexpr int gSamplesLoggingInterval = 10000;

void AbstractFilter::updateRecords(float* xArray, const std::unordered_map<int, float>* xFilter) const
{
    if (xFilter && !xFilter->empty())
    {
#pragma omp parallel for
        for (int i = 0; i < xFilter->size(); ++i)
        {
            auto it = std::next(xFilter->begin(), i);
            xArray[it->first] *= it->second;
        }
    }
}

void AbstractFilter::updateRecords(float* xArray, const std::unordered_map<int, float>* xFilter, int offset, int width) const
{
    if (xFilter && !xFilter->empty())
    {
#pragma omp parallel for
        for (int i = 0; i < xFilter->size(); ++i)
        {
            auto it = std::next(xFilter->begin(), i);
            int index = it->first;
            if (index >= offset && index < offset + width)
            {
                xArray[index - offset] *= it->second;
            }
        }
    }
}

void SamplesFilter::loadSingleFilter(std::unordered_map<string, unsigned int>& xMInput,
    std::unordered_map<std::string, unsigned int>& xMSamples,
    std::vector<std::unique_ptr<std::unordered_map<int, float>>>& sampleFilters,
    const std::string& filePath)
{
    std::ifstream samplesFile(filePath);
    auto start = std::chrono::steady_clock::now();
    std::unordered_map<int, float>* sampleFilter = nullptr;
    int samplesFilterCount = 0;
    vector<string> filters;

    if (samplesFile.good())
    {
        string line;
        int sample = -1;

        while (getline(samplesFile, line))
        {
            filters = split(line, ':');

            if (filters.size() > 0)
            {
                vector<string> vals = split(filters[0], '\t');

                if (vals.size() > 0)
                {
                    try
                    {
                        sample = xMSamples.at(vals[0]);

                        if (vals.size() > 1)
                        {
                            filters[0] = vals[1];
                        }
                    }
                    catch (const std::out_of_range& oor)
                    {
                        continue;
                    }
                }
            }

            sampleFilter = new std::unordered_map<int, float>();

#pragma omp parallel for
            for (int i = 0; i < filters.size(); ++i)
            {
                vector<string> vals = split(filters[i], ',');

                if (vals.size() > 0)
                {
                    try
                    {
                        int key = xMInput.at(vals[0]);
                        float value = 0.0f;

                        if (vals.size() > 1)
                        {
                            value = stof(vals[1]);
                        }

#pragma omp critical
                        {
                            (*sampleFilter)[key] = value;
                        }
                    }
                    catch (const std::out_of_range& oor)
                    {
                        continue;
                    }
                }
            }

            if (sample != -1)
            {
#pragma omp critical
                {
                    sampleFilters[sample] = std::unique_ptr<std::unordered_map<int, float>>(sampleFilter);
                    ++samplesFilterCount;

                    if (samplesFilterCount % gSamplesLoggingInterval == 0)
                    {
                        auto const end = std::chrono::steady_clock::now();
                        std::cout << "Progress Parsing Filter " << samplesFilterCount;
                        std::cout << "Time " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << std::endl;
                        start = std::chrono::steady_clock::now();
                    }
                }
            }
        }
    }
    else
    {
        std::cout << "Unable to read the file " << filePath << std::endl;
        throw std::invalid_argument("invalid sample filters " + filePath + ", exiting...");
    }
}

void SamplesFilter::loadFilter(std::unordered_map<std::string, unsigned int>& xMInput,
    std::unordered_map<std::string, unsigned int>& xMSamples,
    const std::string& filePath)
{
    sampleFilters.clear();

    std::ifstream file(filePath);

    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open filter file: " << filePath << std::endl;
        return;
    }

    std::cout << "Loading filter from file: " << filePath << std::endl;

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        int key;
        float value;

        if (!(iss >> key >> value))
        {
            std::cerr << "Error: Invalid filter data in file: " << filePath << std::endl;
            continue;
        }

        auto filter = std::make_unique<std::unordered_map<int, float>>();
        (*filter)[key] = value;
        sampleFilters.emplace_back(std::move(filter));
    }

    std::cout << "Info: SamplesFilter " << sampleFilters.size() << " filters loaded." << std::endl;
}

void SamplesFilter::applyFilter(float* xArray, int xSamplesIndex, int offset, int width) const
{
    const auto& filter = sampleFilters.at(xSamplesIndex);
    updateRecords(xArray, filter.get(), offset, width);
}

void SamplesFilter::applyFilter(float* xArray, int xSamplesIndex) const
{
    const auto& filter = sampleFilters.at(xSamplesIndex);
    updateRecords(xArray, filter.get());
}

std::unique_ptr<FilterConfig> loadFilters(const string& samplesFilterFileName,
    const string& outputFileName,
    std::unordered_map<std::string, unsigned int>& xMInput,
    std::unordered_map<std::string, unsigned int>& xMSamples)
{
    Value index;
    Reader reader;
    auto filterConfig = std::make_unique<FilterConfig>();
    auto samplesFilter = std::make_unique<SamplesFilter>();

    samplesFilter->loadFilter(xMInput, xMSamples, samplesFilterFileName);
    filterConfig->setSamplesFilter(move(samplesFilter));
    filterConfig->setOutputFileName(outputFileName);

    std::ofstream outputFile(outputFileName);
    if (!outputFile.is_open()) {
        throw std::runtime_error("Unable to create the file " + outputFileName);
    }

    return filterConfig;
}