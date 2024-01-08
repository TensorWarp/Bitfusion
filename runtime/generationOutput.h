#pragma once

#include "../common/cudaUtils.h"
#include "common.h"
#include "iTensor.h"

#include <functional>
#include <utility>

namespace bitfusion::runtime
{

    /// <summary>
    /// Template class for handling generic generation output data.
    /// </summary>
    /// <typeparam name="TTensor">Type of the tensor used for generation output.</typeparam>
    template <typename TTensor>
    class GenericGenerationOutput
    {
    public:
        using TensorPtr = TTensor;
        using Callback = std::function<void(TensorPtr const& ids, SizeType step, bool finished)>;

        /// <summary>
        /// Constructor for GenericGenerationOutput.
        /// </summary>
        /// <param name="ids">Pointer to the output tensor for IDs.</param>
        /// <param name="lengths">Pointer to the output tensor for lengths.</param>
        explicit GenericGenerationOutput(TensorPtr ids, TensorPtr lengths)
            : ids{ std::move(ids) }
            , lengths{ std::move(lengths) }
        {
            CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
            CHECK_WITH_INFO(static_cast<bool>(this->lengths), "Invalid lengths tensor");
        }

        TensorPtr ids;
        TensorPtr lengths;

        TensorPtr cumLogProbs;
        TensorPtr logProbs;
        TensorPtr contextLogits;
        TensorPtr generationLogits;

        Callback onTokenGenerated;
    };

    /// <summary>
    /// Class for handling generation output data.
    /// </summary>
    class GenerationOutput : public GenericGenerationOutput<ITensor::SharedPtr>
    {
    public:
        using Base = GenericGenerationOutput<ITensor::SharedPtr>;
        using TensorPtr = Base::TensorPtr;

        /// <summary>
        /// Constructor for GenerationOutput.
        /// </summary>
        /// <param name="ids">Pointer to the output tensor for IDs.</param>
        /// <param name="lengths">Pointer to the output tensor for lengths.</param>
        explicit GenerationOutput(TensorPtr ids, TensorPtr lengths)
            : GenericGenerationOutput(std::move(ids), std::move(lengths))
        {
        }
    };
}