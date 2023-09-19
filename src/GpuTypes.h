#pragma once

#include <stdio.h>
#ifdef _MSC_VER
#else

#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#endif
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include <vector_functions.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cstring>
#include <cstdint>
#include <assert.h>
#include <mpi.h>
#include <memory>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>

void handleError(const char* errorMessage, cudaError_t cudaStatus) {
    if (cudaStatus != cudaSuccess) {
        std::cerr << errorMessage << " (CUDA error: " << cudaGetErrorString(cudaStatus) << ")" << '\n';
        exit(1);
    }
}

#define VALIDATION

#if defined(CUDA_VERSION) && (CUDA_VERSION < 12000)
#error "CUDA support requires the use of a 12.0 or later CUDA toolkit. Aborting compilation."
#endif

#define use_SPFP

#if !(defined(use_DPFP) && !defined(use_HPFP) && !defined(use_SPFP)) && \
    !(defined(use_HPFP) && !defined(use_DPFP) && !defined(use_SPFP)) && \
    !(defined(use_SPFP) && !defined(use_DPFP) && !defined(use_HPFP))
#error "You must define one and only one precision mode (use_SPFP, use_HPFP, or use_DPFP). Aborting compilation."
#endif

constexpr long long int ESCALE = (1ll << 30);
constexpr double ERRORSCALE = static_cast<double>(ESCALE);
constexpr float ERRORSCALEF = static_cast<float>(ESCALE);
constexpr double ONEOVERERRORSCALE = 1.0 / ERRORSCALE;
constexpr float ONEOVERERRORSCALEF = static_cast<float>(1.0 / ERRORSCALE);

template <typename T, size_t Alignment>
#ifdef _MSC_VER
using AlignedType = __declspec(align(Alignment)) T;
#else
using AlignedType = T __attribute__((aligned(Alignment)));
#endif

using AlignedDouble = AlignedType<double, 8>;
using AlignedULI = AlignedType<unsigned long int, 8>;
using AlignedLLI = AlignedType<long long int, 8>;
using UInt64 = unsigned long long int;

#ifdef use_DPFP
using NNAccumulator = AlignedDouble;
using NNDouble = AlignedDouble;
using FLOAT = AlignedDouble;
using NNDouble2 = AlignedType<double2, 16>;
using NNDouble4 = AlignedType<double4, 32>;
using Float2 = AlignedType<double2, 16>;
using Float4 = AlignedType<double4, 16>;

constexpr MPI_Datatype MPI_NNDOUBLE = MPI_DOUBLE_PRECISION;
constexpr MPI_Datatype MPI_Float = MPI_DOUBLE_PRECISION;
constexpr MPI_Datatype MPI_NNACCUMULATOR = MPI_FLOAT;

#elif defined(use_SPFP)
using NNAccumulator = float;
using NNDouble = AlignedDouble;
using FLOAT = float;
using NNDouble2 = AlignedType<double2, 16>;
using NNDouble4 = AlignedType<double4, 32>;
using Float2 = AlignedType<float2, 8>;
using Float4 = AlignedType<float4, 16>;

constexpr MPI_Datatype MPI_NNDOUBLE = MPI_DOUBLE_PRECISION;
constexpr MPI_Datatype MPI_Float = MPI_FLOAT;
constexpr MPI_Datatype MPI_NNACCUMULATOR = MPI_LONG_LONG_INT;

#else
using NNAccumulator = float;
using NNDouble = AlignedDouble;
using FLOAT = float;
using NNDouble2 = AlignedType<double2, 16>;
using NNDouble4 = AlignedType<double4, 32>;
using Float2 = AlignedType<float2, 8>;
using Float4 = AlignedType<float4, 16>;

constexpr MPI_Datatype MPI_NNDOUBLE = MPI_DOUBLE_PRECISION;
constexpr MPI_Datatype MPI_Float = MPI_FLOAT;
constexpr MPI_Datatype MPI_NNACCUMULATOR = MPI_LONG_LONG_INT;
#endif

static const int SM_3X_THREADS_PER_BLOCK = 128;
static const int SM_5X_THREADS_PER_BLOCK = 128;
static const int SM_6X_THREADS_PER_BLOCK = 128;

#if (__CUDA_ARCH__ >= 600)
#define LAUNCH_BOUNDS() __launch_bounds__(SM_6X_THREADS_PER_BLOCK, 8)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#elif (__CUDA_ARCH__ >= 500)
#define LAUNCH_BOUNDS() __launch_bounds__(SM_5X_THREADS_PER_BLOCK, 8)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
#else
#define LAUNCH_BOUNDS() __launch_bounds__(SM_3X_THREADS_PER_BLOCK, 10)
#define LAUNCH_BOUNDS256() __launch_bounds__(256, 4)
#endif

#define LAUNCH_BOUNDS512() __launch_bounds__(512, 2)
#define LAUNCH_BOUNDS1024() __launch_bounds__(1024, 1)

constexpr uint32_t SM_6X_MAXSPARSE = 4608;
constexpr uint32_t SM_6X_MAXSPARSEANALOG = 2304;
constexpr uint32_t SM_5X_MAXSPARSE = 4608;
constexpr uint32_t SM_5X_MAXSPARSEANALOG = 2304;
constexpr uint32_t SM_3X_MAXSPARSE = 2304;
constexpr uint32_t SM_3X_MAXSPARSEANALOG = 1152;

constexpr bool bShadowedOutputBuffers = false;

constexpr long long int FPSCALE = (1ll << 40);
constexpr long long int DFSCALE = (1ll << 44);



#ifdef GVERBOSE
#ifndef MEMTRACKING
#define MEMTRACKING
#endif

#ifdef SYNCHRONOUS
#define LAUNCHERROR(s) \
    { \
        std::cout << ("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::string errMsg = std::format("Error: %s launching kernel %s", cudaGetErrorString(status), s); \
            std::cerr << errMsg << ".\n"; \
            getGpu().Shutdown(); \
            throw std::runtime_error(errMsg); \
        } \
        cudaDeviceSynchronize(); \
    }
#else
#define LAUNCHERROR(s) \
    { \
        std::cout << ("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::string errMsg = std::format("Error: %s launching kernel %s", cudaGetErrorString(status), s); \
            std::cerr << errMsg << ".\n"; \
            getGpu().Shutdown(); \
            throw std::runtime_error(errMsg); \
        } \
    }
#endif

#define LAUNCHERROR_BLOCKING(s) \
    { \
        std::cout << ("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::string errMsg = std::format("Error: %s launching kernel %s", cudaGetErrorString(status), s); \
            std::cerr << errMsg << ".\n"; \
            getGpu().Shutdown(); \
            throw std::runtime_error(errMsg); \
        } \
        cudaDeviceSynchronize(); \
    }

#define LAUNCHERROR_NONBLOCKING(s) \
    { \
        std::cout << ("Launched %s on node %d\n", s, getGpu()._id); \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::string errMsg = ("Error: %s launching kernel %s", cudaGetErrorString(status), s); \
            std::cerr << errMsg << ".\n"; \
            getGpu().Shutdown(); \
            throw std::runtime_error(errMsg); \
        } \
    }

#else

#ifdef SYNCHRONOUS
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            char errMsg[1024]; \
            std::snprintf(errMsg, sizeof(errMsg), "Error: %s launching kernel %s", cudaGetErrorString(status), s); \
            std::cerr << errMsg << ".\n"; \
            getGpu().Shutdown(); \
            throw std::runtime_error(errMsg); \
        } \
    }
#else
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::cout << ("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            getGpu().Shutdown(); \
            std::terminate(); \
        } \
    }
#endif

#define LAUNCHERROR_BLOCKING(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::string errMsg = ("Error: %s launching kernel %s", cudaGetErrorString(status), s); \
            std::cerr << errMsg << ".\n"; \
            getGpu().Shutdown(); \
            throw std::runtime_error(errMsg); \
        } \
        cudaDeviceSynchronize(); \
    }

#define LAUNCHERROR_NONBLOCKING(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            std::string errMsg = ("Error: %s launching kernel %s", cudaGetErrorString(status), s); \
            std::cerr << errMsg << ".\n"; \
            getGpu().Shutdown(); \
            throw std::runtime_error(errMsg); \
        } \
    }

#endif

#define RTERROR(status, s) \
    if (status != cudaSuccess) { \
        std::string errMsg = ("%s %s", s, cudaGetErrorString(status)); \
        std::cerr << errMsg << ".\n"; \
        cudaDeviceReset(); \
        throw std::runtime_error(errMsg); \
    }

#define CUDNNERROR(status, s) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::string errMsg = ("%s %s", s, cudnnGetErrorString(status)); \
        std::cerr << errMsg << ".\n"; \
        cudaDeviceReset(); \
        throw std::runtime_error(errMsg); \
    }

struct GpuData {

    unsigned int _warpSize;

    unsigned int _warpBits;

    unsigned int _warpMask;

    unsigned long long int* _pAccumulator;

    float _LRN_k;

    int _LRN_n;

    float _LRN_alpha;

    float _LRN_beta;

    int _maxout_k;

    float _deltaBoost_one;

    float _deltaBoost_zero;

    float _SMCE_oneTarget;

    float _SMCE_zeroTarget;

    float _SMCE_oneScale;

    float _SMCE_zeroScale;

    bool _bSparsenessPenalty;

    float _sparsenessPenalty_p;

    float _sparsenessPenalty_beta;

    bool _bDenoising;

    float _denoising_p;

    float _denoising_q;

    bool _bShuffleIndices;

    unsigned int* _pShuffleIndex;

    AlignedULI _deviceMemory;

    uint32_t _maxUint32_t;

    int32_t _maxInt32_t;

    uint64_t _maxUint64_t;

    int64_t _maxInt64_t;

    float _maxFloat;

    float _minFloat;
};

template <typename T> struct GpuBuffer;
template <typename T> struct MultiGpuBuffer;
class Network;

struct GpuContext {

    enum SM_VERSION
    {
        SM_3X,
        SM_5X,
        SM_6X
    };

    enum {
        PADDING = 32,
        PADDINGBITS = 5,
        PADDINGMASK = 0xffffffff - (PADDING - 1)
    };

    GpuData _data;

    bool _bECCSupport;

    bool _bCanMapHostMemory;

    AlignedULI _totalMemory;

    AlignedULI _totalCPUMemory;

    AlignedULI _totalGPUMemory;

    bool _bUnifiedMemory;

    SM_VERSION _sm_version;

    unsigned int _sm_major;

    unsigned int _threadsPerBlock;

    unsigned int _warpSize;

    unsigned int _warpBits;

    unsigned int _warpMask;

    int _numprocs;

    int _id;

    int _device;

    uint32_t _maxSparse;

    uint32_t _maxSparseAnalog;

    cublasHandle_t _cuBLASHandle;

    curandGenerator_t _RNG;

    cudnnHandle_t _cuDNNHandle;

    Network* _pNetwork;

    std::unique_ptr<GpuBuffer<unsigned long long int>> _pbAccumulator;

    bool _bCPUValidate;

    float _acceptableError;

    bool _bSingleNode;

    bool _bP2P;

    GpuContext();

    ~GpuContext();

    void GetMemoryUsage(int* gpuMemory, int* cpuMemory) const;

    void SetRandomSeed(unsigned long seed);

    unsigned long ModifySeed(unsigned long seed) const;

    bool TrySetCuRandSeed(unsigned long seed) const;

    bool TrySetStdRandSeed(unsigned long seed) const;

    void HandleSeedError() const;

    void LogSeedSet(unsigned long seed) const;

    void SetNeuralNetwork(Network* pNetwork);

    void SetFastMath(bool flag);

    void Startup(int argc, char** argv);

    void CopyConstants();

    void Shutdown();

    void SetCPUValidate(bool bCPUValidate);

    static unsigned int Pad(unsigned int x) { return (x + PADDING - 1) & PADDINGMASK; }
};

extern struct GpuContext& getGpu();

template <typename T>
struct GpuBuffer
{
    size_t _length;
    bool _bSysMem;
    bool _bManaged;
    T* _pSysData;
    T* _pDevData;

    GpuBuffer(size_t length, bool bSysMem = false, bool bManaged = false);

    virtual ~GpuBuffer();

    void Allocate();
    void Resize(size_t length);
    void Deallocate();
    void Upload(const T* pBuff = nullptr) const;
    void Download(T* pBuff = nullptr);
    void Copy(T* pBuff);
    size_t GetLength();
    size_t GetSize();
};

template <typename T>
GpuBuffer<T>::GpuBuffer(size_t length, bool bSysMem, bool bManaged)
    : _length(length), _bSysMem(bSysMem), _bManaged(bManaged), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();
}

template <typename T>
GpuBuffer<T>::~GpuBuffer()
{
    Deallocate();
}

template <typename T>
void GpuBuffer<T>::Allocate()
{
    cudaError_t status;

    if (_bManaged)
        _bSysMem = true;

#ifdef MEMTRACKING
    printf("Allocating %llu bytes of GPU memory", _length * sizeof(T));
    if (!_bSysMem)
    {
        std::cout << (", unshadowed");
    }
    else if (_bManaged)
    {
        std::cout << (", managed");
    }
    std::cout << ("\n");
#endif

    if (_bManaged) {
        status = cudaMallocManaged((void**)&_pDevData, _length * sizeof(T), cudaMemAttachGlobal);
        getGpu()._totalGPUMemory += _length * sizeof(T);
        _pSysData = _pDevData;
        handleError("GpuBuffer::Allocate failed (cudaMallocManaged)", status);
        memset(_pSysData, 0, _length * sizeof(T));
    }
    else {
        status = cudaMalloc((void**)&_pDevData, _length * sizeof(T));
        getGpu()._totalGPUMemory += static_cast<AlignedULI>(_length * sizeof(T));
        handleError("GpuBuffer::Allocate failed (cudaMalloc)", status);

        status = cudaMemset((void*)_pDevData, 0, _length * sizeof(T));
        handleError("GpuBuffer::Allocate failed (cudaMemset)", status);

        if (_bSysMem) {
            _pSysData = new T[_length];
            getGpu()._totalCPUMemory += static_cast<AlignedULI>(_length * sizeof(T));
            memset(_pSysData, 0, _length * sizeof(T));
        }
    }

#ifdef MEMTRACKING
    std::printf("Mem++: %llu %llu\n", getGpu()._totalGPUMemory, getGpu()._totalCPUMemory);
#endif
}

template<typename T>
void GpuBuffer<T>::Resize(size_t length)
{
    if (length > _length)
    {
        Deallocate();
        _length = length;
        Allocate();
    }
}

template <typename T>
void GpuBuffer<T>::Deallocate()
{
    cudaError_t status;

    status = cudaFree(_pDevData);
    if (status != cudaSuccess) {
        std::cerr << "GpuBuffer::Deallocate failed (cudaFree) (CUDA error: " << cudaGetErrorString(status) << ")" << '\n';
        std::exit(1);
    }
    getGpu()._totalGPUMemory -= static_cast<AlignedULI>(_length * sizeof(T));

    if (_bSysMem && !_bManaged)
    {
        delete[] _pSysData;
        getGpu()._totalCPUMemory -= static_cast<AlignedULI>(_length * sizeof(T));
    }

    _pSysData = nullptr;
    _pDevData = nullptr;
    _length = 0;

#ifdef MEMTRACKING
    std::cout << "Mem--: " << getGpu()._totalGPUMemory << " " << getGpu()._totalCPUMemory << '\n';
#endif
}

template <typename T>
void GpuBuffer<T>::Copy(T* pBuff)
{
    cudaError_t status;
    status = cudaMemcpy(_pDevData, pBuff, _length * sizeof(T), cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy GpuBuffer::Copy failed (CUDA error: " << cudaGetErrorString(status) << ")" << '\n';
        std::exit(1);
    }
}

template <typename T>
void GpuBuffer<T>::Upload(const T* pBuff) const
{
    if (pBuff)
    {
        cudaError_t status;
        status = cudaMemcpy(_pDevData, pBuff, _length * sizeof(T), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            std::cerr << "cudaMemcpy GpuBuffer::Upload failed (CUDA error: " << cudaGetErrorString(status) << ")" << '\n';
            std::exit(1);
        }
    }
    else if (_bSysMem && !_bManaged)
    {
        cudaError_t status;
        status = cudaMemcpy(_pDevData, _pSysData, _length * sizeof(T), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            std::cerr << "cudaMemcpy GpuBuffer::Upload failed (CUDA error: " << cudaGetErrorString(status) << ")" << '\n';
            std::exit(1);
        }
    }
}

template <typename T>
void GpuBuffer<T>::Download(T* pBuff)
{
    if (pBuff)
    {
        cudaError_t status;
        status = cudaMemcpy(pBuff, _pDevData, _length * sizeof(T), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            std::cerr << "cudaMemcpy GpuBuffer::Download failed (CUDA error: " << cudaGetErrorString(status) << ")" << '\n';
            std::exit(1);
        }
    }
    else if (_bSysMem && !_bManaged)
    {
        cudaError_t status;
        status = cudaMemcpy(_pSysData, _pDevData, _length * sizeof(T), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            std::cerr << "cudaMemcpy GpuBuffer::Download failed (CUDA error: " << cudaGetErrorString(status) << ")" << '\n';
            std::exit(1);
        }
    }
}

template<typename T>
size_t GpuBuffer<T>::GetLength()
{
    return _length;
}

template<typename T>
size_t GpuBuffer<T>::GetSize()
{
    return _length * sizeof(T);
}

#define SGEMM(A,B,C,m,n,k,alpha,beta,transf_A,transf_B) \
        cublasSgemm(getGpu()._cuBLASHandle, transf_B, transf_A, n, m, k, alpha, B, n, A, k, beta, C, n)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)

#define std::printf(f,...)
#endif