#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#if defined(ENABLE_BF16)
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace bitfusion
{
    namespace kernels
    {
        enum class WeightOnlyQuantType
        {
            Int4b,
            Int8b
        };
        enum class WeightOnlyType
        {
            PerChannel,
            GroupWise
        };

        struct WeightOnlyPerChannel;
        template <int GS>
        struct WeightOnlyGroupWise;

        enum class WeightOnlyActivationFunctionType
        {
            Gelu,
            Relu,
            Identity,
            InvalidType
        };

        enum class WeightOnlyActivationType
        {
            FP16,
            BF16
        };

        struct WeightOnlyParams
        {
            using ActType = void;
            using WeiType = uint8_t;

            const uint8_t* qweight;
            const ActType* scales;
            const ActType* zeros;
            const ActType* in;
            const ActType* act_scale;
            const ActType* bias;
            ActType* out;
            const int m;
            const int n;
            const int k;
            const int group_size;
            WeightOnlyQuantType quant_type;
            WeightOnlyType weight_only_type;
            WeightOnlyActivationFunctionType act_func_type;
            WeightOnlyActivationType act_type;

            WeightOnlyParams(const uint8_t* _qweight, const ActType* _scales, const ActType* _zeros, const ActType* _in,
                const ActType* _act_scale, const ActType* _bias, ActType* _out, const int _m, const int _n, const int _k,
                const int _group_size, const WeightOnlyQuantType _quant_type, const WeightOnlyType _weight_only_type,
                const WeightOnlyActivationFunctionType _act_func_type, const WeightOnlyActivationType _act_type)
                : qweight(_qweight)
                , scales(_scales)
                , zeros(_zeros)
                , in(_in)
                , act_scale(_act_scale)
                , bias(_bias)
                , out(_out)
                , m(_m)
                , n(_n)
                , k(_k)
                , group_size(_group_size)
                , quant_type(_quant_type)
                , weight_only_type(_weight_only_type)
                , act_func_type(_act_func_type)
                , act_type(_act_type)
            {
            }
        };
    }
}