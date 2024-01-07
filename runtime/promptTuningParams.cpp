#include "promptTuningParams.h"
#include <span>

namespace bitfusion::runtime
{

    /// <summary>
    /// Fills the tasks tensor based on input parameters.
    /// </summary>
    /// <param name="tasksHost">The tasks host tensor.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="numContextRequests">The number of context requests.</param>
    /// <param name="reqBeamWidths">The vector of requested beam widths.</param>
    /// <param name="reqPromptLengths">The vector of requested prompt lengths.</param>
    /// <param name="manager">The buffer manager.</param>
    /// <param name="packedInput">Flag indicating packed input.</param>
    void PromptTuningParams::fillTasksTensor(TensorPtr tasksHost, const SizeType batchSize,
        const SizeType numContextRequests, const std::vector<SizeType>& reqBeamWidths,
        const std::vector<SizeType>& reqPromptLengths, BufferManager const& manager, bool packedInput)
    {
        const auto& tasksHostShape = tasksHost->getShape();
        CHECK_WITH_INFO(tasksHostShape.nbDims == 1, "tasksHost expected to have dimension [batchSize]");
        CHECK_WITH_INFO(tasksHostShape.d[0] == batchSize, "tasksHost expected to have dimension [batchSize]");

        const auto tasksHostPtr = bufferCast<SizeType const>(*tasksHost);

        const bool validInput = packedInput || numContextRequests == batchSize || numContextRequests == 0;
        CHECK_WITH_INFO(validInput,
            "fillTasksTensor function with packed inputs must be called with only context requests or only generation "
            "requests.");

        const bool validShapes = (reqBeamWidths.size() == batchSize
            && reqPromptLengths.size() == numContextRequests
            && promptTuningEnabled.size() == batchSize);
        CHECK_WITH_INFO(validShapes,
            "Invalid inputs to fillTasksTensor function. reqBeamWidths and reqPromptLengths size must be batchSize and "
            "promptTuningEnabled size must be batchSize");

        SizeType totalInputSize = 0;
        std::vector<SizeType> promptTasksHost;

        for (SizeType bid = 0; bid < batchSize; bid++)
        {
            const SizeType taskId = promptTuningEnabled[bid] ? tasksHostPtr[bid] : 0;

            if (packedInput)
            {
                if (bid < numContextRequests)
                {
                    totalInputSize += reqPromptLengths[bid];
                    promptTasksHost.insert(promptTasksHost.end(), reqPromptLengths[bid], taskId);
                }
                else
                {
                    totalInputSize += reqBeamWidths[bid];
                    promptTasksHost.resize(totalInputSize, taskId);
                }
            }
            else
            {
                if (bid < numContextRequests)
                {
                    totalInputSize += 1;
                    promptTasksHost.push_back(taskId);
                }
                else
                {
                    totalInputSize += reqBeamWidths[bid];
                    promptTasksHost.resize(totalInputSize, taskId);
                }
            }
        }

        const auto shape = packedInput ? runtime::ITensor::makeShape({ 1, totalInputSize })
            : runtime::ITensor::makeShape({ totalInputSize, 1 });
        tasks = manager.copyFrom(promptTasksHost, shape, runtime::MemoryType::kGPU);
    }
}