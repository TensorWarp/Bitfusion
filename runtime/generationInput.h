#pragma once

#include "../common/cudaUtils.h"
#include "common.h"
#include "iTensor.h"
#include "promptTuningParams.h"

#include <optional>
#include <utility>

namespace bitfusion::runtime
{

    /// <summary>
    /// Template class for handling generic generation input data.
    /// </summary>
    /// <typeparam name="TTensor">Type of the tensor used for generation input.</typeparam>
    /// <typeparam name="PromptTuningParams">Type of prompt tuning parameters.</typeparam>
    template <typename TTensor, typename PromptTuningParams>
    class GenericGenerationInput
    {
    public:
        using TensorPtr = TTensor;

        /// <summary>
        /// Constructor for GenericGenerationInput.
        /// </summary>
        /// <param name="endId">End identifier for generation input.</param>
        /// <param name="padId">Padding identifier for generation input.</param>
        /// <param name="ids">Pointer to the input tensor for IDs.</param>
        /// <param name="lengths">Pointer to the input tensor for lengths.</param>
        /// <param name="packed">Boolean indicating if input is packed.</param>
        explicit GenericGenerationInput(
            SizeType const endId, SizeType const padId, TensorPtr ids, TensorPtr lengths, bool packed = false)
            : endId{ endId }
            , padId{ padId }
            , ids{ std::move(ids) }
            , lengths{ std::move(lengths) }
            , packed{ packed }
            , maxNewTokens(std::nullopt)
        {
            CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
            CHECK_WITH_INFO(static_cast<bool>(this->lengths), "Invalid lengths tensor");
        }

        SizeType endId;
        SizeType padId;
        TensorPtr ids;
        TensorPtr lengths;
        bool packed;

        TensorPtr embeddingBias;
        TensorPtr badWordsList;
        TensorPtr stopWordsList;
        std::optional<SizeType> maxNewTokens;

        PromptTuningParams promptTuningParams;
    };

    /// <summary>
    /// Class for handling generation input data.
    /// </summary>
    class GenerationInput : public GenericGenerationInput<ITensor::SharedPtr, PromptTuningParams>
    {
    public:
        using Base = GenericGenerationInput<ITensor::SharedPtr, PromptTuningParams>;
        using TensorPtr = Base::TensorPtr;

        /// <summary>
        /// Constructor for GenerationInput.
        /// </summary>
        /// <param name="endId">End identifier for generation input.</param>
        /// <param name="padId">Padding identifier for generation input.</param>
        /// <param name="ids">Pointer to the input tensor for IDs.</param>
        /// <param name="lengths">Pointer to the input tensor for lengths.</param>
        /// <param name="packed">Boolean indicating if input is packed.</param>
        explicit GenerationInput(
            SizeType const endId, SizeType const padId, TensorPtr ids, TensorPtr lengths, bool packed = false)
            : GenericGenerationInput(endId, padId, std::move(ids), std::move(lengths), packed)
        {
        }
    };
}