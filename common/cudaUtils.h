#pragma once

#include "cudaBf16Wrapper.h"
#include "cudaFp8Utils.h"
#include "logger.h"
#include "Exception.h"
#include <cinttypes>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace bitfusion::common
{

#define CUBLAS_WORKSPACE_SIZE 33554432

    typedef struct __align__(4)
    {
        half x, y, z, w;
    }

    half4;


    enum CublasDataType
    {
        FLOAT_DATATYPE = 0,
        HALF_DATATYPE = 1,
        BFLOAT16_DATATYPE = 2,
        INT8_DATATYPE = 3,
        FP8_DATATYPE = 4
    };

    enum TRTLLMCudaDataType
    {
        FP32 = 0,
        FP16 = 1,
        BF16 = 2,
        INT8 = 3,
        FP8 = 4
    };

    enum class OperationType
    {
        FP32,
        FP16,
        BF16,
        INT8,
        FP8
    };

    static const char* _cudaGetErrorEnum(cudaError_t error)
    {
        return cudaGetErrorString(error);
    }

    static const char* _cudaGetErrorEnum(cublasStatus_t error)
    {
        switch (error)
        {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        }
        return "<unknown>";
    }

    template <typename T>
    void check(T result, char const* const func, const char* const file, int const line)
    {
        if (result)
        {
            throw Exception(
                file, line, fmtstr("[TensorRT-LLM][ERROR] CUDA runtime error in %s: %s", func, _cudaGetErrorEnum(result)));
        }
    }

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)

    inline bool isCudaLaunchBlocking()
    {
        static bool firstCall = true;
        static bool result = false;

        if (firstCall)
        {
            const char* env = std::getenv("CUDA_LAUNCH_BLOCKING");
            result = env != nullptr && std::string(env) == "1";
            firstCall = false;
        }

        return result;
    }

    inline void syncAndCheck(const char* const file, int const line)
    {
#ifndef NDEBUG
        const bool checkError = true;
#else
        const bool checkError = isCudaLaunchBlocking();
#endif

        if (checkError)
        {
            cudaError_t result = cudaDeviceSynchronize();
            check(result, "cudaDeviceSynchronize", file, line);
        }
    }

#define sync_check_cuda_error() bitfusion::common::syncAndCheck(__FILE__, __LINE__)

#define PRINT_FUNC_NAME_()                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        std::cout << "[TensorRT-LLM][CALL] " << __FUNCTION__ << " " << std::endl;                                      \
    } while (0)

    template<typename T> struct packed_type;
    template <>          struct packed_type<float> { using type = float; };
    template <>          struct packed_type<half> { using type = half2; };

#ifdef ENABLE_BF16
    template<>
    struct packed_type<__nv_bfloat16> {
        using type = __nv_bfloat162;
    };
#endif

#ifdef ENABLE_FP8
    template<>
    struct packed_type<__nv_fp8_e4m3> {
        using type = __nv_fp8x2_e4m3;
    };
#endif

    template<typename T> struct num_elems;
    template <>          struct num_elems<float> { static constexpr int value = 1; };
    template <>          struct num_elems<float2> { static constexpr int value = 2; };
    template <>          struct num_elems<float4> { static constexpr int value = 4; };
    template <>          struct num_elems<half> { static constexpr int value = 1; };
    template <>          struct num_elems<half2> { static constexpr int value = 2; };
#ifdef ENABLE_BF16
    template <>          struct num_elems<__nv_bfloat16> { static constexpr int value = 1; };
    template <>          struct num_elems<__nv_bfloat162> { static constexpr int value = 2; };
#endif
#ifdef ENABLE_FP8
    template <>          struct num_elems<__nv_fp8_e4m3> { static constexpr int value = 1; };
    template <>          struct num_elems<__nv_fp8x2_e4m3> { static constexpr int value = 2; };
#endif

    template<typename T, int num> struct packed_as;
    template<typename T>          struct packed_as<T, 1> { using type = T; };
    template<>                    struct packed_as<half, 2> { using type = half2; };
    template<>                    struct packed_as<float, 2> { using type = float2; };
    template<>                    struct packed_as<int8_t, 2> { using type = int16_t; };
    template<>                    struct packed_as<int32_t, 2> { using type = int2; };
    template<>                    struct packed_as<half2, 1> { using type = half; };
    template<>                    struct packed_as<float2, 1> { using type = float; };
#ifdef ENABLE_BF16
    template<> struct packed_as<__nv_bfloat16, 2> { using type = __nv_bfloat162; };
    template<> struct packed_as<__nv_bfloat162, 1> { using type = __nv_bfloat16; };
#endif
#ifdef ENABLE_FP8
    template<> struct packed_as<__nv_fp8_e4m3, 2> { using type = __nv_fp8x2_e4m3; };
    template<> struct packed_as<__nv_fp8x2_e4m3, 1> { using type = __nv_fp8_e4m3; };
    template<> struct packed_as<__nv_fp8_e5m2, 2> { using type = __nv_fp8x2_e5m2; };
    template<> struct packed_as<__nv_fp8x2_e5m2, 1> { using type = __nv_fp8_e5m2; };
#endif

    inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
    inline __device__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
    inline __device__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }

    inline __device__ float2 operator*(float2 a, float  b) { return make_float2(a.x * b, a.y * b); }
    inline __device__ float2 operator+(float2 a, float  b) { return make_float2(a.x + b, a.y + b); }
    inline __device__ float2 operator-(float2 a, float  b) { return make_float2(a.x - b, a.y - b); }


    template <typename T>
    struct CudaDataType
    {
    };

    template <>
    struct CudaDataType<float>
    {
        static constexpr cudaDataType_t value = cudaDataType::CUDA_R_32F;
    };

    template <>
    struct CudaDataType<half>
    {
        static constexpr cudaDataType_t value = cudaDataType::CUDA_R_16F;
    };

#ifdef ENABLE_BF16
    template <>
    struct CudaDataType<__nv_bfloat16>
    {
        static constexpr cudaDataType_t value = cudaDataType::CUDA_R_16BF;
    };
#endif

    inline int getSMVersion()
    {
        int device{ -1 };
        check_cuda_error(cudaGetDevice(&device));
        int sm_major = 0;
        int sm_minor = 0;
        check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
        check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
        return sm_major * 10 + sm_minor;
    }

    inline int getDevice()
    {
        int current_dev_id = 0;
        check_cuda_error(cudaGetDevice(&current_dev_id));
        return current_dev_id;
    }

    inline int getDeviceCount()
    {
        int count = 0;
        check_cuda_error(cudaGetDeviceCount(&count));
        return count;
    }

    inline std::tuple<size_t, size_t> getDeviceMemoryInfo()
    {
        size_t free, total;
        check_cuda_error(cudaMemGetInfo(&free, &total));
        return { free, total };
    }

    inline int getMultiProcessorCount()
    {
        int device_id;
        int multi_processor_count;
        check_cuda_error(cudaGetDevice(&device_id));
        check_cuda_error(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, device_id));
        return multi_processor_count;
    }

    inline int getMaxSharedMemoryPerBlockOptin()
    {
        int device_id;
        int max_shared_memory_per_block;
        check_cuda_error(cudaGetDevice(&device_id));
        check_cuda_error(
            cudaDeviceGetAttribute(&max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
        return max_shared_memory_per_block;
    }

    template <typename T1, typename T2>
    inline size_t divUp(const T1& a, const T2& n)
    {
        size_t tmp_a = static_cast<size_t>(a);
        size_t tmp_n = static_cast<size_t>(n);
        return (tmp_a + tmp_n - 1) / tmp_n;
    }

    template <typename T, typename U, typename = std::enable_if_t<std::is_integral<T>::value>,
        typename = std::enable_if_t<std::is_integral<U>::value>>
        auto constexpr ceilDiv(T numerator, U denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }

    template <typename T>
    void printAbsMean(const T* buf, uint64_t size, cudaStream_t stream, std::string name = "")
    {
        if (buf == nullptr)
        {
            TLLM_LOG_WARNING("%s is an nullptr, skip!", name.c_str());
            return;
        }
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
        T* h_tmp = new T[size];
        cudaMemcpyAsync(h_tmp, buf, sizeof(T) * size, cudaMemcpyDeviceToHost, stream);
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
        double sum = 0.0f;
        uint64_t zero_count = 0;
        float max_val = -1e10;
        bool find_inf = false;
        for (uint64_t i = 0; i < size; i++)
        {
            if (std::isinf((float)(h_tmp[i])))
            {
                find_inf = true;
                continue;
            }
            sum += abs((double)h_tmp[i]);
            if ((float)h_tmp[i] == 0.0f)
            {
                zero_count++;
            }
            max_val = max_val > abs(float(h_tmp[i])) ? max_val : abs(float(h_tmp[i]));
        }
        TLLM_LOG_INFO("%20s size: %u, abs mean: %f, abs sum: %f, abs max: %f, find inf: %s", name.c_str(), size, sum / size,
            sum, max_val, find_inf ? "true" : "false");
        delete[] h_tmp;
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
    }

    template <typename T>
    void printToStream(const T* result, const int size, FILE* strm)
    {
        const bool split_rows = (strm == stdout);
        if (result == nullptr)
        {
            TLLM_LOG_WARNING("It is an nullptr, skip! \n");
            return;
        }
        T* tmp = reinterpret_cast<T*>(malloc(sizeof(T) * size));
        check_cuda_error(cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost));
        for (int i = 0; i < size; ++i)
        {
            fprintf(strm, "%f, ", static_cast<float>(tmp[i]));
            if (split_rows && ((i + 1) % 10) == 0)
                fprintf(strm, "\n");
        }
        if (!split_rows || (size % 10) != 0)
        {
            fprintf(strm, "\n");
        }
        free(tmp);
    }

    template <typename T>
    void printToScreen(const T* result, const int size)
    {
        printToStream(result, size, stdout);
    }

    template <typename T>
    void print2dToStream(const T* result, const int r, const int c, const int stride, FILE* strm)
    {
        if (result == nullptr)
        {
            TLLM_LOG_WARNING("It is an nullptr, skip! \n");
            return;
        }
        for (int ri = 0; ri < r; ++ri)
        {
            const T* ptr = result + ri * stride;
            printToStream(ptr, c, strm);
        }
        fprintf(strm, "\n");
    }

    template <typename T>
    void print2dToScreen(const T* result, const int r, const int c, const int stride)
    {
        print2dToStream(result, r, c, stride, stdout);
    }

    template <typename T>
    void print2dToFile(std::string fname, const T* result, const int r, const int c, const int stride)
    {
        FILE* fp = fopen(fname.c_str(), "wt");
        if (fp != nullptr)
        {
            print2dToStream(result, r, c, stride, fp);
            fclose(fp);
        }
    }

    inline void print_float_(float x)
    {
        printf("%7.3f ", x);
    }

    inline void print_element_(float x)
    {
        print_float_(x);
    }

    inline void print_element_(half x)
    {
        print_float_((float)x);
    }
#ifdef ENABLE_BF16
    inline void print_element_(__nv_bfloat16 x)
    {
        print_float_((float)x);
    }
#endif
    inline void print_element_(uint32_t ul)
    {
        printf("%7" PRIu32, ul);
    }

    inline void print_element_(uint64_t ull)
    {
        printf("%7" PRIu64, ull);
    }

    inline void print_element_(int32_t il)
    {
        printf("%7" PRId32, il);
    }

    inline void print_element_(int64_t ill)
    {
        printf("%7" PRId64, ill);
    }

    template <typename T>
    inline void printMatrix(const T* ptr, int m, int k, int stride, bool is_device_ptr)
    {
        T* tmp;
        if (is_device_ptr)
        {
            tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
            check_cuda_error(cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
        }
        else
        {
            tmp = const_cast<T*>(ptr);
        }

        for (int ii = -1; ii < m; ++ii)
        {
            if (ii >= 0)
            {
                printf("%07d ", ii);
            }
            else
            {
                printf("   ");
            }

            for (int jj = 0; jj < k; jj += 1)
            {
                if (ii >= 0)
                {
                    print_element_(tmp[ii * stride + jj]);
                }
                else
                {
                    printf("%7d ", jj);
                }
            }
            printf("\n");
        }
        if (is_device_ptr)
        {
            free(tmp);
        }
    }

    template void printMatrix(const float* ptr, int m, int k, int stride, bool is_device_ptr);
    template void printMatrix(const half* ptr, int m, int k, int stride, bool is_device_ptr);
#ifdef ENABLE_BF16
    template void printMatrix(const __nv_bfloat16* ptr, int m, int k, int stride, bool is_device_ptr);
#endif
    template void printMatrix(const uint32_t* ptr, int m, int k, int stride, bool is_device_ptr);
    template void printMatrix(const uint64_t* ptr, int m, int k, int stride, bool is_device_ptr);
    template void printMatrix(const int* ptr, int m, int k, int stride, bool is_device_ptr);

}

#define TLLM_CUDA_CHECK(stat)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        bitfusion::common::check((stat), #stat, __FILE__, __LINE__);                                                \
    } while (0)
