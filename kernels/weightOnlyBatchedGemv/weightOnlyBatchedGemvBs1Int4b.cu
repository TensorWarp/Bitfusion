#include "kernel.h"

namespace bitfusion
{
    namespace kernels
    {

        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel,
            IdentityActivation, false, false, 1, 1, 192>;

        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
            IdentityActivation, true, true, 2, 1, 256>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
            IdentityActivation, true, false, 2, 1, 256>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
            IdentityActivation, false, true, 2, 1, 256>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>,
            IdentityActivation, false, false, 2, 1, 256>;

        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
            IdentityActivation, true, true, 2, 1, 256>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
            IdentityActivation, true, false, 2, 1, 256>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
            IdentityActivation, false, true, 2, 1, 256>;
        template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<128>,
            IdentityActivation, false, false, 2, 1, 256>;

    }
}