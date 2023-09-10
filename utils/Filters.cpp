#include <json/json.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <chrono>

#include "Filters.h"
#include "Utils.h"

using namespace Json;

const static int gSamplesLoggingInterval = 10000;

void AbstractFilter::updateRecords(float *xArray, const std::unordered_map<int, float> *xFilter) const
{
    if (xFilter && xFilter->size() > 0)
    {
        std::unordered_map<int, float>::const_iterator filterIter;
        for (filterIter = xFilter->begin(); filterIter != xFilter->end(); ++filterIter)
        {
            int index = filterIter->first;
            float value = filterIter->second;
            xArray[index] = value * xArray[index];
        }
    }
}

void AbstractFilter::updateRecords(float *xArray, const std::unordered_map<int, float> *xFilter, int offset, int width) const
{
    if (xFilter && xFilter->size() > 0)
    {
        std::unordered_map<int, float>::const_iterator filterIter;
        for (filterIter = xFilter->begin(); filterIter != xFilter->end(); ++filterIter)
        {
            int index = filterIter->first;
            float value = filterIter->second;
            if (index >= offset && index < offset + width)
            { 
                xArray[index - offset] = value * xArray[index - offset];
            }
        }
    }
}

void SamplesFilter::loadSingleFilter(std::unordered_map<std::string, unsigned int> &xMInput,
                                     std::unordered_map<std::string, unsigned int> &xMSamples,
                                     std::vector<std::unique_ptr<std::unordered_map<int, float>>> &sampleFilters,
                                     const std::string &filePath)
{
    std::ifstream samplesFile(filePath);
    auto start = std::chrono::steady_clock::now();
    std::unordered_map<int, float> *sampleFilter = nullptr;
    int samplesFilterCount = 0;
    std::vector<std::string> filters;
    if (samplesFile.good())
    {
        std::string line;
        int sample = -1;
        while (getline(samplesFile, line))
        {
            filters = split(line, ':');
            if (filters.size() > 0)
            {
                std::vector<std::string> vals = split(filters[0], '\t');
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
            for (int i = 0; i < filters.size(); ++i)
            {
                std::vector<std::string> vals = split(filters[i], ',');
                if (vals.size() > 0)
                {
                    try
                    {
                        int key = xMInput.at(vals[0]);
                        float value = 0.0f;
                        if (vals.size() > 1)
                        {
                            value = atof(vals[1].c_str());
                        }
                        (*sampleFilter)[key] = value;
                    }
                    catch (const std::out_of_range& oor)
                    {
                        continue;
                    }
                }
            }
            if (sample != -1)
            {
                sampleFilters[sample].reset(sampleFilter);
                ++samplesFilterCount;
                if (samplesFilterCount % gSamplesLoggingInterval == 0)
                {
                    auto const end = std::chrono::steady_clock::now();
                    std::cout << "Progress Parsing Filter " << samplesFilterCount;
                    std::cout << "Time " << elapsed_seconds(start, end) << std::endl;
                    start = std::chrono::steady_clock::now();
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
                               const std::string &filterFilePath)
{

    samplefilters.reset(new std::vector<std::unique_ptr<std::unordered_map<int, float>>>(xMSamples.size()));

    std::vector<std::string> files;
    if (listFiles(filterFilePath, false, files) == 0)
    {
        std::cout << "Loading " << files.size() << " filter files" << std::endl;

        for (auto const &file : files)
        {
            std::cout << "\tLoading filter: " << file << std::endl;
            loadSingleFilter(xMInput, xMSamples, *samplefilters.get(), file);
        }
    }

    std::cout << "Info:SamplesFilter " << samplefilters->size() << std::endl;
}

void SamplesFilter::applyFilter(float *xArray, int xSamplesIndex, int offset, int width) const
{
    std::unordered_map<int, float> *filter = (*samplefilters)[xSamplesIndex].get();
    updateRecords(xArray, filter, offset, width);
}

void SamplesFilter::applyFilter(float *xArray, int xSamplesIndex) const
{
    std::unordered_map<int, float> *filter = (*samplefilters)[xSamplesIndex].get();
    updateRecords(xArray, filter);
}

FilterConfig* loadFilters(const std::string &samplesFilterFileName,
                          const std::string &outputFileName,
                          std::unordered_map<std::string, unsigned int>& xMInput,
                          std::unordered_map<std::string, unsigned int>& xMSamples)
{
    Value index;
    Reader reader;
    FilterConfig *filterConfig  = new FilterConfig();
    SamplesFilter *samplesFilter = new SamplesFilter() ;
    samplesFilter->loadFilter(xMInput, xMSamples, samplesFilterFileName);
    filterConfig->setSamplesFilter(samplesFilter);
    filterConfig->setOutputFileName(outputFileName);
    FILE *fp = fopen(outputFileName.c_str(), "w");
    fclose(fp);
    return filterConfig;
}

