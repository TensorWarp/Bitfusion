#include "reduceScatterPlugin.h"

using namespace nvinfer1;
using bitfusion::plugins::ReduceScatterPluginCreator;
using bitfusion::plugins::ReduceScatterPlugin;

static const char* REDUCE_SCATTER_PLUGIN_VERSION{"1"};
static const char* REDUCE_SCATTER_PLUGIN_NAME{"ReduceScatter"};
PluginFieldCollection ReduceScatterPluginCreator::mFC{};
std::vector<PluginField> ReduceScatterPluginCreator::mPluginAttributes;

ReduceScatterPlugin::ReduceScatterPlugin(std::set<int> group, nvinfer1::DataType type)
    : mGroup(group)
    , mType(type)
{
}

ReduceScatterPlugin::ReduceScatterPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mType);
    mGroup.clear();
    int groupItem = 0;
    while (d != a + length)
    {
        read(d, groupItem);
        mGroup.insert(groupItem);
    }
    TLLM_CHECK(d == a + length);
}

nvinfer1::IPluginV2DynamicExt* ReduceScatterPlugin::clone() const noexcept
{
    auto* plugin = new ReduceScatterPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs ReduceScatterPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    auto output = inputs[0];
    output.d[0]
        = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *output.d[0], *exprBuilder.constant(mGroup.size()));
    return output;
}

bool ReduceScatterPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void ReduceScatterPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t ReduceScatterPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int ReduceScatterPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    int size = 1;
    for (int i = 0; i < outputDesc[0].dims.nbDims; ++i)
    {
        size *= outputDesc[0].dims.d[i];
    }

    NCCLCHECK(ncclReduceScatter(
        inputs[0], outputs[0], size, (*getDtypeMap())[inputDesc[0].type], ncclSum, (*getCommMap())[mGroup], stream));

    return 0;
}

nvinfer1::DataType ReduceScatterPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}


const char* ReduceScatterPlugin::getPluginType() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_NAME;
}

const char* ReduceScatterPlugin::getPluginVersion() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_VERSION;
}

int ReduceScatterPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int ReduceScatterPlugin::initialize() noexcept
{
    auto* commMap = getCommMap();
    if (isBuilding() || (*commMap)[mGroup] != nullptr)
    {
        return 0;
    }
    int myRank, nRanks;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    int groupRank = 0;
    for (auto it = mGroup.begin(); it != mGroup.end(); ++it)
    {
        if (*it == myRank)
        {
            break;
        }
        ++groupRank;
    }

    ncclUniqueId id;
    if (myRank == *mGroup.begin())
    {
        ncclGetUniqueId(&id);
        for (auto it = std::next(std::begin(mGroup), 1); it != mGroup.end(); ++it)
        {
            MPICHECK(MPI_Send(&id, sizeof(id), MPI_BYTE, *it, 0, MPI_COMM_WORLD));
        }
    }
    else
    {
        MPI_Status status;
        MPICHECK(MPI_Recv(&id, sizeof(id), MPI_BYTE, *mGroup.begin(), 0, MPI_COMM_WORLD, &status));
    }
    NCCLCHECK(ncclCommInitRank(&((*commMap)[mGroup]), mGroup.size(), id, groupRank));
    return 0;
}

void ReduceScatterPlugin::terminate() noexcept
{
    auto* commMap = getCommMap();
    if (isBuilding() || (*commMap)[mGroup] == nullptr)
    {
        return;
    }
    NCCLCHECK(ncclCommDestroy((*commMap)[mGroup]));
    (*commMap)[mGroup] = nullptr;
}

size_t ReduceScatterPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * mGroup.size() + sizeof(mType);
}

void ReduceScatterPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    for (auto it = mGroup.begin(); it != mGroup.end(); ++it)
    {
        write(d, *it);
    }
    assert(d == a + getSerializationSize());
}

void ReduceScatterPlugin::destroy() noexcept
{
    delete this;
}

void ReduceScatterPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* ReduceScatterPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


ReduceScatterPluginCreator::ReduceScatterPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ReduceScatterPluginCreator::getPluginName() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_NAME;
}

const char* ReduceScatterPluginCreator::getPluginVersion() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_VERSION;
}

const PluginFieldCollection* ReduceScatterPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* ReduceScatterPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    std::set<int> group;
    nvinfer1::DataType type;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "group"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            const auto* r = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < fields[i].length; ++j)
            {
                group.insert(*r);
                ++r;
            }
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }

    try
    {
        auto* obj = new ReduceScatterPlugin(group, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* ReduceScatterPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new ReduceScatterPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
