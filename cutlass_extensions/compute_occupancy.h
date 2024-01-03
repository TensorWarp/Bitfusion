#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/device_kernel.h"
#include "../common/cudaUtils.h"

namespace bitfusion
{
    namespace cutlass_extensions
    {

        template <typename GemmKernel>
        inline int compute_occupancy_for_kernel()
        {

            int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

            if (smem_size > (48 << 10))
            {
                cudaFuncAttributes attr;
                int device = 0;
                int max_smem_per_block = 0;
                bitfusion::common::check_cuda_error(cudaGetDevice(&device));
                bitfusion::common::check_cuda_error(
                    cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
                bitfusion::common::check_cuda_error(cudaFuncGetAttributes(&attr, cutlass::Kernel<GemmKernel>));
                if (smem_size + attr.sharedSizeBytes >= static_cast<size_t>(max_smem_per_block))
                {
                    return 0;
                }
            }

            int max_active_blocks = -1;
            bitfusion::common::check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_active_blocks, cutlass::Kernel<GemmKernel>, GemmKernel::kThreadCount, smem_size));

            return max_active_blocks;
        }

    }
}