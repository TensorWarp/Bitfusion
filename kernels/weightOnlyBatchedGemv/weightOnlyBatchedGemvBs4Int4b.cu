#include "kernel.h"

namespace bitfusion
{
    namespace kernels
    {

        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel,
            IdentityActivation, false, false, 4, 4, 256>;

        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
            IdentityActivation, true, true, 2, 4, 128>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
            IdentityActivation, true, false, 2, 4, 128>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
            IdentityActivation, false, true, 2, 4, 128>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
            IdentityActivation, false, false, 2, 4, 128>;

        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
            IdentityActivation, true, true, 2, 4, 128>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
            IdentityActivation, true, false, 2, 4, 128>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
            IdentityActivation, false, true, 2, 4, 128>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
            IdentityActivation, false, false, 2, 4, 128>;

    }
}