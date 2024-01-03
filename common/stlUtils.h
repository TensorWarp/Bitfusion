#pragma once
#include "assert.h"
#include <functional>
#include <numeric>

namespace bitfusion::common::stl_utils
{

    template <typename TInputIt, typename TOutputIt, typename TBinOp>
    constexpr TOutputIt basicInclusiveScan(TInputIt first, TInputIt last, TOutputIt dFirst, TBinOp op)
    {
        if (first != last)
        {
            auto val = *first;
            while (true)
            {
                *dFirst = val;
                ++dFirst;
                ++first;
                if (first == last)
                {
                    break;
                }
                val = op(std::move(val), *first);
            }
        }
        return dFirst;
    }

    template <typename TInputIt, typename TOutputIt>
    constexpr TOutputIt inclusiveScan(TInputIt first, TInputIt last, TOutputIt dFirst)
    {
#if defined(__GNUC__) && __GNUC__ <= 8
        return basicInclusiveScan(first, last, dFirst, std::plus<>{});
#else
        return std::inclusive_scan(first, last, dFirst);
#endif
    }

    template <typename TInputIt, typename TOutputIt, typename T, typename TBinOp>
    constexpr TOutputIt basicExclusiveScan(TInputIt first, TInputIt last, TOutputIt dFirst, T init, TBinOp op)
    {
        if (first != last)
        {
            while (true)
            {
                T tmp{ op(init, *first) };
                *dFirst = init;
                ++dFirst;
                ++first;
                if (first == last)
                {
                    break;
                }
                init = std::move(tmp);
            }
        }
        return dFirst;
    }

    template <typename TInputIt, typename TOutputIt, typename T>
    constexpr TOutputIt exclusiveScan(TInputIt first, TInputIt last, TOutputIt dFirst, T init)
    {
#if defined(__GNUC__) && __GNUC__ <= 8
        return basicExclusiveScan(first, last, dFirst, std::move(init), std::plus<>{});
#else
        return std::exclusive_scan(first, last, dFirst, std::move(init));
#endif
    }

}