#pragma once

namespace bitfusion
{
    namespace cutlass_extensions
    {
        enum class CutlassTileConfig
        {
            Undefined,

            ChooseWithHeuristic,

            CtaShape128x128x8_WarpShape64x64x8,

            CtaShape32x128x64_WarpShape32x32x64,

            CtaShape64x128x64_WarpShape32x64x64,
            CtaShape64x64x128_WarpShape32x64x64,
            CtaShape64x128x64_WarpShape64x32x64,

            CtaShape128x64x64_WarpShape64x32x64,
            CtaShape128x128x64_WarpShape64x32x64,
            CtaShape128x128x64_WarpShape64x64x64,
            CtaShape128x128x64_WarpShape128x32x64,
            CtaShape128x256x64_WarpShape64x64x64,

            CtaShape256x128x64_WarpShape64x64x64
        };

        enum class SplitKStyle
        {
            NO_SPLIT_K,
            SPLIT_K_SERIAL,
        };

        struct CutlassGemmConfig
        {
            CutlassTileConfig tile_config = CutlassTileConfig::ChooseWithHeuristic;
            SplitKStyle split_k_style = SplitKStyle::NO_SPLIT_K;
            int split_k_factor = -1;
            int stages = -1;
        };

    }
}