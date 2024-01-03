#pragma once

#include "../../common/mpiUtils.h"
#include "../../kernels/customAllReduceKernels.h"
#include "../../plugins/common/plugin.h"
#include <cassert>
#include <memory>
#include <mpi.h>
#include <nccl.h>
#include <set>
#include <string>
#include <vector>

namespace bitfusion::plugins
{

class AllreducePlugin : public BasePlugin
{
public:
    AllreducePlugin(
        std::set<int> group, nvinfer1::DataType type, kernels::AllReduceStrategyType strategy, int32_t counter);

    AllreducePlugin(const void* data, size_t length);

    ~AllreducePlugin() override = default;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

    bool isCustomAllReduceSuported(int ranks_per_node) const noexcept;

private:
    kernels::AllReduceStrategyType selectImplementation(size_t messageSize, int worldSize) const noexcept;
    const std::string mLayerName;
    std::set<int> mGroup;
    nvinfer1::DataType mType;
    kernels::AllReduceStrategyType mStrategy;
    int32_t mCounter;
};

class AllreducePluginCreator : public BaseCreator
{
public:
    AllreducePluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}
