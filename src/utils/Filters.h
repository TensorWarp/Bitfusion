#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

/// <summary>
/// Abstract class representing a filter.
/// </summary>
class AbstractFilter
{
public:
    virtual ~AbstractFilter() = default;

    /// <summary>
    /// Load filter parameters from a file.
    /// </summary>
    /// <param name="xMInput">A reference to the input map.</param>
    /// <param name="xMSamples">A reference to the samples map.</param>
    /// <param name="filePath">The path to the filter configuration file.</param>
    virtual void loadFilter(std::unordered_map<std::string, unsigned int>& xMInput,
        std::unordered_map<std::string, unsigned int>& xMSamples,
        const std::string& filePath) = 0;

    /// <summary>
    /// Apply the filter to an array of data.
    /// </summary>
    /// <param name="xArray">The data array to apply the filter to.</param>
    /// <param name="xSamplesIndex">The index of the samples in the map.</param>
    virtual void applyFilter(float* xArray, int xSamplesIndex) const = 0;

    /// <summary>
    /// Apply the filter to a specific portion of an array of data.
    /// </summary>
    /// <param name="xArray">The data array to apply the filter to.</param>
    /// <param name="xSamplesIndex">The index of the samples in the map.</param>
    /// <param name="offset">The offset within the array to apply the filter.</param>
    /// <param name="width">The width of the portion to apply the filter.</param>
    virtual void applyFilter(float* xArray, int xSamplesIndex, int offset, int width) const = 0;

    /// <summary>
    /// Get the filter type.
    /// </summary>
    /// <returns>The filter type as a string.</returns>
    virtual std::string getFilterType() const = 0;

protected:
    /// <summary>
    /// Update records in the data array based on the filter.
    /// </summary>
    /// <param name="xArray">The data array to update.</param>
    /// <param name="xFilter">The filter to apply.</param>
    void updateRecords(float* xArray, const std::unordered_map<int, float>* xFilter) const;

    /// <summary>
    /// Update records in a portion of the data array based on the filter.
    /// </summary>
    /// <param name="xArray">The data array to update.</param>
    /// <param name="xFilter">The filter to apply.</param>
    /// <param name="offset">The offset within the array to start updating.</param>
    /// <param name="width">The width of the portion to update.</param>
    void updateRecords(float* xArray, const std::unordered_map<int, float>* xFilter, int offset, int width) const;
};

/// <summary>
/// Class representing a filter for processing samples.
/// </summary>
class SamplesFilter : public AbstractFilter
{
    std::unique_ptr<std::vector<std::unique_ptr<std::unordered_map<int, float>>>> samplefilters;

    void loadSingleFilter(std::unordered_map<std::string, unsigned int>& xMInput,
        std::unordered_map<std::string, unsigned int>& xMSamples,
        std::vector<std::unique_ptr<std::unordered_map<int, float>>>& sampleFilters,
        const std::string& filePath);

public:
    void loadFilter(std::unordered_map<std::string, unsigned int>& xMInput,
        std::unordered_map<std::string, unsigned int>& xMSamples,
        const std::string& filePath);

    void applyFilter(float* xArray, int xSamplesIndex) const;
    void applyFilter(float* xArray, int xSamplesIndex, int offset, int width) const;

    std::string getFilterType() const
    {
        return "samplesFilterType";
    }
};

/// <summary>
/// Class representing filter configuration.
/// </summary>
class FilterConfig
{
    std::unique_ptr<SamplesFilter> sampleFilter;
    std::string outputFileName;

public:
    /// <summary>
    /// Set the output file name for filtered data.
    /// </summary>
    /// <param name="xOutputFileName">The output file name.</param>
    void setOutputFileName(const std::string& xOutputFileName)
    {
        outputFileName = xOutputFileName;
    }

    /// <summary>
    /// Get the output file name for filtered data.
    /// </summary>
    /// <returns>The output file name.</returns>
    std::string getOutputFileName() const
    {
        return outputFileName;
    }

    /// <summary>
    /// Set the samples filter for filtering data.
    /// </summary>
    /// <param name="xSampleFilter">A pointer to the SamplesFilter instance.</param>
    void setSamplesFilter(SamplesFilter* xSampleFilter)
    {
        sampleFilter.reset(xSampleFilter);
    }

    /// <summary>
    /// Apply the samples filter to input data.
    /// </summary>
    /// <param name="xInput">The input data array.</param>
    /// <param name="xSampleIndex">The index of the sample.</param>
    /// <param name="offset">The offset within the array to apply the filter.</param>
    /// <param name="width">The width of the portion to apply the filter.</param>
    void applySamplesFilter(float* xInput, int xSampleIndex, int offset, int width) const
    {
        if (sampleFilter)
        {
            sampleFilter->applyFilter(xInput, xSampleIndex, offset, width);
        }
    }
};

/// <summary>
/// Load filter configuration from a file.
/// </summary>
/// <param name="samplesFilterFileName">The file name for samples filter configuration.</param>
/// <param name="outputFileName">The output file name for filtered data.</param>
/// <param name="xMInput">A reference to the input map.</param>
/// <param name="xMSamples">A reference to the samples map.</param>
/// <returns>A pointer to the loaded FilterConfig instance.</returns>
FilterConfig* loadFilters(const std::string& samplesFilterFileName,
    const std::string& outputFileName,
    std::unordered_map<std::string, unsigned int>& xMInput,
    std::unordered_map<std::string, unsigned int>& xMSamples);