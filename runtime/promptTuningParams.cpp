
#include "promptTuningParams.h"

namespace bitfusion::runtime
{

void PromptTuningParams::fillTasksTensor(TensorPtr tasksHost, const SizeType batchSize,
    const SizeType numContextRequests, const std::vector<SizeType>& reqBeamWidths,
    const std::vector<SizeType>& reqPromptLengths, BufferManager const& manager, bool packedInput)
{
    auto const& tasksHostShape = tasksHost->getShape();
    CHECK_WITH_INFO(tasksHostShape.nbDims == 1, "tasksHost expected to have dimension [batchSize]");
    CHECK_WITH_INFO(tasksHostShape.d[0] == batchSize, "tasksHost expected to have dimension [batchSize]");

    auto const tasksHostPtr = bufferCast<SizeType const>(*tasksHost);

    bool validInput = packedInput || numContextRequests == batchSize || numContextRequests == 0;
    CHECK_WITH_INFO(validInput,
        "fillTasksTensor function with packed inputs must be called with only context requests or only generation "
        "requests.");

    bool validShapes = (static_cast<SizeType>(reqBeamWidths.size()) == batchSize
        && static_cast<SizeType>(reqPromptLengths.size()) == numContextRequests
        && static_cast<SizeType>(promptTuningEnabled.size()) == batchSize);
    CHECK_WITH_INFO(validShapes,
        "Invalid inputs to fillTasksTensor function. reqBeamWidths and reqPtuningEnabled size must be batchSize and "
        "propmtLenghts size must be numContextRequests");

    SizeType totalInputSize = 0;
    std::vector<SizeType> promptTasksHost;
    for (SizeType bid = 0; bid < batchSize; bid++)
    {
        SizeType taskId = promptTuningEnabled[bid] ? tasksHostPtr[bid] : 0;
        if (packedInput)
        {
            if (bid < numContextRequests)
            {
                totalInputSize += reqPromptLengths[bid];
                promptTasksHost.insert(promptTasksHost.end(), reqPromptLengths[bid], taskId);
            }
            else
            {
                for (SizeType beam = 0; beam < reqBeamWidths[bid]; ++beam)
                {
                    promptTasksHost.insert(promptTasksHost.end(), 1, taskId);
                    totalInputSize++;
                }
            }
        }
        else
        {
            if (bid < numContextRequests)
            {
                promptTasksHost.push_back(taskId);
                ++totalInputSize;
            }
            else
            {
                promptTasksHost.insert(promptTasksHost.end(), reqBeamWidths[bid], taskId);
                totalInputSize += reqBeamWidths[bid];
            }
        }
    }

    if (packedInput)
    {
        tasks = manager.copyFrom(
            promptTasksHost, runtime::ITensor::makeShape({1, totalInputSize}), runtime::MemoryType::kGPU);
    }
    else
    {
        tasks = manager.copyFrom(
            promptTasksHost, runtime::ITensor::makeShape({totalInputSize, 1}), runtime::MemoryType::kGPU);
    }
}

}
