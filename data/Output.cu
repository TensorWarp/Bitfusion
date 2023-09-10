#ifdef __CUDACC__

#include "../system/Bitonic.cuh"
#include "Output.cuh"
#include <limits>

/// <summary>
/// CUDA/HIP kernel for calculating output.
/// </summary>
/// <param name="pOutputBuffer">Pointer to the output buffer.</param>
/// <param name="pKeyBuffer">Pointer to the key buffer.</param>
/// <param name="pValueBuffer">Pointer to the value buffer.</param>
/// <param name="batch">Batch size.</param>
/// <param name="width">Width of the data.</param>
/// <param name="widthPadding">Width padding.</param>
/// <param name="k">Value of k.</param>
static __global__ void
LAUNCH_BOUNDS()
CalculateOutput_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch,
    uint32_t width, uint32_t widthPadding, uint32_t k)
{
    __shared__ volatile float sKey[160 * 4];
    __shared__ volatile uint32_t sValue[160 * 4];

    uint32_t dataWidth = width - widthPadding;
    uint32_t pos = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    uint32_t tgx = threadIdx.x & 31;

    if (pos < batch)
    {
        float* pOutput = pOutputBuffer + pos * width;
        uint32_t offset = threadIdx.x >> 5;
        volatile float* psKey = &sKey[160 * offset];
        volatile uint32_t* psValue = &sValue[160 * offset];

        float k0 = -FLT_MAX;
        float k1 = -FLT_MAX;
        float k2 = -FLT_MAX;
        float k3 = -FLT_MAX;
        float k4 = -FLT_MAX;
        float k5 = -FLT_MAX;
        float k6 = -FLT_MAX;
        float k7 = -FLT_MAX;
        uint32_t v0 = 0;
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        uint32_t v3 = 0;
        uint32_t v4 = 0;
        uint32_t v5 = 0;
        uint32_t v6 = 0;
        uint32_t v7 = 0;

        uint32_t wpos = tgx;
        if (wpos < dataWidth)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += 32;
        if (wpos < dataWidth)
        {
            k1 = pOutput[wpos];
            v1 = wpos;
        }
        wpos += 32;
        if (wpos < dataWidth)
        {
            k2 = pOutput[wpos];
            v2 = wpos;
        }
        wpos += 32;
        if (wpos < dataWidth)
        {
            k3 = pOutput[wpos];
            v3 = wpos;
        }

        float minValue = -FLT_MAX;
        uint32_t rpos = 128;
        uint32_t bufferSize = 0;
        float key1, key2;
        uint32_t value1, value2;
        uint32_t otgx;
        bool flag;
        while (rpos < dataWidth)
        {
            unsigned wpos = rpos + tgx;
            float key = -FLT_MAX;
            uint32_t value = wpos;
            if (wpos < dataWidth)
            {
                key = pOutput[wpos];
            }

            uint32_t count = __ballot_sync(0xFFFFFFFF, key > minValue);
            if (key > minValue)
            {
                uint32_t mask = 0xffffffff >> (32 - tgx);
                uint32_t offset = __popc(count & mask);
                offset += bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }
            bufferSize += __popc(count);

            if (bufferSize >= 128)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
                k5 = psKey[tgx + 32];
                v5 = psValue[tgx + 32];
                k6 = psKey[tgx + 2 * 32];
                v6 = psValue[tgx + 2 * 32];
                k7 = psKey[tgx + 3 * 32];
                v7 = psValue[tgx + 3 * 32];
                bool flag;
                BITONICSORT256_256();

                minValue = __shfl_sync(0xFFFFFFFF, k3, 31);

                bufferSize -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 128];
                    psValue[tgx] = psValue[tgx + 128];
                }
            }

            rpos += 32;
        }

        if ((bufferSize > 0) || (dataWidth <= 128))
        {
            k4 = -FLT_MAX;
            k5 = -FLT_MAX;
            k6 = -FLT_MAX;
            k7 = -FLT_MAX;
            v4 = 0;
            v5 = 0;
            v6 = 0;
            v7 = 0;

            if (tgx < bufferSize)
            {
                k4 = psKey[tgx];
                v4 = psValue[tgx];
            }
            if (tgx + 32 < bufferSize)
            {
                k5 = psKey[tgx + 32];
                v5 = psValue[tgx + 32];
            }
            if (tgx + 2 * 32 < bufferSize)
            {
                k6 = psKey[tgx + 2 * 32];
                v6 = psValue[tgx + 2 * 32];
            }
            if (tgx + 3 * 32 < bufferSize)
            {
                k7 = psKey[tgx + 3 * 32];
                v7 = psValue[tgx + 3 * 32];
            }

            BITONICSORT256_256();
        }

        float* pKey = pKeyBuffer + pos * k;
        uint32_t* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += 32;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += 32;
        if (wpos < k)
        {
            pKey[wpos] = k2;
            pValue[wpos] = v2;
        }
        wpos += 32;
        if (wpos < k)
        {
            pKey[wpos] = k3;
            pValue[wpos] = v3;
        }
    }
}

/// <summary>
/// Calculate the output using CUDA kernel.
/// </summary>
/// <param name="pOutput">Pointer to the output buffer.</param>
/// <param name="pKey">Pointer to the key buffer.</param>
/// <param name="pValue">Pointer to the value buffer.</param>
/// <param name="batch">Batch size.</param>
/// <param name="width">Width of the data.</param>
/// <param name="widthPadding">Width padding.</param>
/// <param name="k">Value of k.</param>
void CalculateOutput(float* pOutput, float* pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t widthPadding, uint32_t k)
{
    void* args[] = { &pOutput, &pKey, &pValue, &batch, &width, &widthPadding, &k };

    if (batch % 4 != 0 || widthPadding % 4 != 0) {
        throw std::invalid_argument("Batch and widthPadding must be multiples of 4.");
    }

    cudaLaunchKernel((void*)CalculateOutput_kernel, gridDim, blockDim, args, 0, nullptr);
}

#endif