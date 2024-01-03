#pragma once

#include "../runtime/common.h"

#include <optional>

namespace bitfusion::batch_manager::kv_cache_manager
{

    class KvCacheConfig
    {
    public:
        using SizeType = bitfusion::runtime::SizeType;

        explicit KvCacheConfig(std::optional<SizeType> maxTokens = std::nullopt,
            std::optional<SizeType> maxAttentionWindow = std::nullopt,
            std::optional<float> freeGpuMemoryFraction = std::nullopt, bool enableBlockReuse = false)
            : maxTokens{ maxTokens }
            , maxAttentionWindow{ maxAttentionWindow }
            , freeGpuMemoryFraction{ freeGpuMemoryFraction }
            , enableBlockReuse(enableBlockReuse)
        {
        }

        std::optional<SizeType> maxTokens;
        std::optional<SizeType> maxAttentionWindow;
        std::optional<float> freeGpuMemoryFraction;
        bool enableBlockReuse;

        static constexpr auto kDefaultGpuMemFraction = 0.85f;
    };
}