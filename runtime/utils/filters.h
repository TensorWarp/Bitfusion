#ifndef FILTERS_H
#define FILTERS_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

class AbstractFilter
{
public:
    virtual ~AbstractFilter() = default;

    virtual void loadFilter(std::unordered_map<std::string, unsigned int>& xMInput,
        std::unordered_map<std::string, unsigned int>& xMSamples,
        const std::string& filePath) = 0;

    virtual void applyFilter(float* xArray, int xSamplesIndex) const = 0;
    virtual void applyFilter(float* xArray, int xSamplesIndex, int offset, int width) const = 0;

    [[nodiscard]] virtual std::string getFilterType() const = 0;

protected:
    void updateRecords(float* xArray, const std::unordered_map<int, float>* xFilter) const;
    void updateRecords(float* xArray, const std::unordered_map<int, float>* xFilter, int offset, int width) const;
};

class SamplesFilter : public AbstractFilter
{
    std::vector<std::unique_ptr<std::unordered_map<int, float>>> sampleFilters;

public:
    
    void loadSingleFilter(std::unordered_map<std::string, unsigned int>& xMInput, std::unordered_map<std::string, unsigned int>& xMSamples, std::vector<std::unique_ptr<std::unordered_map<int, float>>>& sampleFilters, const std::string& filePath);

    void loadFilter(std::unordered_map<std::string, unsigned int>& xMInput,
        std::unordered_map<std::string, unsigned int>& xMSamples,
        const std::string& filePath) override;

    void applyFilter(float* xArray, int xSamplesIndex) const override;
    void applyFilter(float* xArray, int xSamplesIndex, int offset, int width) const override;

    [[nodiscard]] std::string getFilterType() const override
    {
        return "samplesFilterType";
    }
};

class FilterConfig
{
    std::unique_ptr<SamplesFilter> sampleFilter;
    std::string outputFileName;

public:
    void setOutputFileName(const std::string& xOutputFileName)
    {
        outputFileName = xOutputFileName;
    }

    [[nodiscard]] const std::string& getOutputFileName() const
    {
        return outputFileName;
    }

    void setSamplesFilter(std::unique_ptr<SamplesFilter> xSampleFilter)
    {
        sampleFilter = std::move(xSampleFilter);
    }

    void applySamplesFilter(float* xInput, int xSampleIndex, int offset, int width) const
    {
        if (sampleFilter)
        {
            sampleFilter->applyFilter(xInput, xSampleIndex, offset, width);
        }
    }
};

std::unique_ptr<FilterConfig> loadFilters(const std::string& samplesFilterFileName,
    const std::string& outputFileName,
    std::unordered_map<std::string, unsigned int>& xMInput,
    std::unordered_map<std::string, unsigned int>& xMSamples);

#endif
