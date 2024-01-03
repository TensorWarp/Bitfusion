#pragma once

#include <nvtx3/nvtx3.hpp>

#include <array>

namespace bitfusion::common::nvtx
{
    inline nvtx3::color nextColor()
    {
#if !defined(NVTX_DISABLE)
        constexpr std::array kColors{ nvtx3::color{0xff00ff00}, nvtx3::color{0xff0000ff}, nvtx3::color{0xffffff00},
            nvtx3::color{0xffff00ff}, nvtx3::color{0xff00ffff}, nvtx3::color{0xffff0000}, nvtx3::color{0xffffffff} };
        constexpr auto numColors = kColors.size();

        static thread_local std::size_t colorId = 0;
        auto const color = kColors[colorId];
        colorId = colorId + 1 >= numColors ? 0 : colorId + 1;
        return color;
#else
        return nvtx3::color{ 0 };
#endif
    }

}

#define NVTX3_SCOPED_RANGE(range) ::nvtx3::scoped_range range##_range(::bitfusion::common::nvtx::nextColor(), #range)
