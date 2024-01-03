#pragma once

#include "kvCacheConfig.h"
#include "../runtime/common.h"

#include <optional>

namespace bitfusion::batch_manager
{

    class TrtGptModelOptionalParams
    {
        using KvCacheConfig = kv_cache_manager::KvCacheConfig;

    public:
        using SizeType = bitfusion::runtime::SizeType;

        explicit TrtGptModelOptionalParams(KvCacheConfig const& kvCacheConfig = KvCacheConfig{},
            std::optional<SizeType> maxNumSequences = std::nullopt, bool enableTrtOverlap = true,
            std::optional<std::vector<SizeType>> const& deviceIds = std::nullopt)
            : kvCacheConfig{ kvCacheConfig }
            , maxNumSequences{ maxNumSequences }
            , enableTrtOverlap{ enableTrtOverlap }
            , deviceIds(deviceIds)
        {
        }

        KvCacheConfig kvCacheConfig;
        std::optional<SizeType> maxNumSequences;
        bool enableTrtOverlap;
        std::optional<std::vector<SizeType>> deviceIds;
    };

}