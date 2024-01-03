#include "GpuTypes.h"
#include "Types.h"
#include "Kernels.cuh"
#include <omp.h>
#include <mpi.h>
#include "Constants.h"
#include "ThreadPool.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <stdexcept>
#include <format>
#include <latch>
#include <ranges>
#include <random>

#ifdef USE_HIGHWAY
#include <highway/highway.h>
using namespace highway;
#endif

inline constexpr unsigned long PRIME_A = 2654435761ul;
inline constexpr unsigned long PRIME_B = 63689ul;
inline constexpr unsigned long PRIME_C = 378551ul;
inline constexpr unsigned long PRIME_D = 6367ul;
inline constexpr unsigned long XOR_MASK_A = 0x5A17A17Aul;
inline constexpr unsigned long XOR_MASK_B = 0xC3A5C3A5ul;
inline constexpr unsigned long XOR_MASK_C = 0x81958195ul;
inline constexpr unsigned long SHIFT_BITS = 7;

template<std::integral T>
T Mix(T x) {
    x ^= (x >> 17);
    x += 0xABCD1234u;
    x ^= (x << 9);
    x ^= (x >> 27);
    return x;
}

static const float cAcceptableError = 0.00001f;

static GpuContext gpu;

GpuContext& getGpu() { return gpu; }


#if defined(_MSC_VER)

static __forceinline int fls(int x)
{
    if (x == 0) return 0;
    unsigned long index;
    _BitScanReverse(&index, static_cast<unsigned long>(x));
    return static_cast<int>(index) + 1;
}

#elif defined(__GNUC__)

static __inline int fls(int x)
{
    return x ? sizeof(x) * 8 - __builtin_clz(x) : 0;
}

#else
#error Unsupported compiler
#endif

GpuContext::GpuContext() :
    _bECCSupport(false),
    _bCanMapHostMemory(false),
    _bCPUValidate(false),
    _bUnifiedMemory(false),
    _acceptableError(cAcceptableError),
    _numprocs(1),
    _id(0),
    _sm_version(SM_3X),
    _sm_major(0),
    _warpSize(32),
    _maxSparse(SM_3X_MAXSPARSE),
    _maxSparseAnalog(SM_3X_MAXSPARSEANALOG),
    _cuBLASHandle(0),
    _cuDNNHandle(0),
    _pbAccumulator()
{
    std::cout << "Initializing GpuContext...\n";
    std::cout << "_bECCSupport: " << _bECCSupport << "\n";
    std::cout << "_bCanMapHostMemory: " << _bCanMapHostMemory << "\n";
    std::cout << "_bCPUValidate: " << _bCPUValidate << "\n";
    std::cout << "_bUnifiedMemory: " << _bUnifiedMemory << "\n";
    std::cout << "_acceptableError: " << _acceptableError << "\n";
    std::cout << "_numprocs: " << _numprocs << "\n";
    std::cout << "_id: " << _id << "\n";
    std::cout << "_sm_version: " << _sm_version << "\n";
    std::cout << "_sm_major: " << _sm_major << "\n";
    std::cout << "_warpSize: " << _warpSize << "\n";
    std::cout << "_maxSparse: " << _maxSparse << "\n";
    std::cout << "_maxSparseAnalog: " << _maxSparseAnalog << "\n";
    std::cout << "_cuBLASHandle: " << _cuBLASHandle << "\n";
    std::cout << "_cuDNNHandle: " << _cuDNNHandle << "\n";
    std::cout << "_pbAccumulator: {}\n";
    std::cout << "GpuContext initialized.";
}

GpuContext::~GpuContext()
{

}

void GpuContext::SetCPUValidate(bool bValidate)
{
    _bCPUValidate = bValidate;
}

void GpuContext::Startup(int argc, char** argv)
{

    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) {
        MPI_Init(&argc, &argv);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &_numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &_id);

    std::cout << "GpuContext::Startup: Process " << _id << " out of " << _numprocs << " initialized.";

    char* cudaProfile = nullptr;
#ifdef _WIN32
    if (_dupenv_s(&cudaProfile, nullptr, "CUDA_PROFILE") == 0 && cudaProfile != nullptr) {
#else
    cudaProfile = getenv("CUDA_PROFILE");
    if (cudaProfile != nullptr) {
#endif
        char profile_log[512];
        char* cudaProfileLog = nullptr;
#ifdef _WIN32
        if (_dupenv_s(&cudaProfileLog, nullptr, "CUDA_PROFILE_LOG") == 0 && cudaProfileLog != nullptr) {
#else
        cudaProfileLog = getenv("CUDA_PROFILE_LOG");
        if (cudaProfileLog != nullptr) {
#endif
            snprintf(profile_log, sizeof(profile_log), "%s%d", cudaProfileLog, _id);
#ifdef _WIN32
            free((void*)cudaProfileLog);
#else
            free((void*)const_cast<char*>(cudaProfileLog));
#endif
        }
        else {
            snprintf(profile_log, sizeof(profile_log), "cu%d.csv", _id);
        }

#ifdef _WIN32
        _putenv_s("CUDA_PROFILE_LOG", profile_log);
#else
        setenv("CUDA_PROFILE_LOG", profile_log, 1);
        setenv("CUDA_PROFILE_CSV", "1", 1);
#endif

#ifdef _WIN32
        free(cudaProfile);
#else
        free(cudaProfile);
#endif
    }

    int device = -1;

    int gpuCount = 0;

    cudaError_t status;

    cudaDeviceProp deviceProp;

    status = cudaGetDeviceCount(&gpuCount);

    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed with status: " << cudaGetErrorString(status);
    }

    if (gpuCount == 0) {
        std::cerr << "GpuContext::Startup: No CUDA-capable devices found, exiting.";
        cudaDeviceReset();
        Shutdown();
        exit(EXIT_FAILURE);
    }

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int length;

    char myName[MPI_MAX_PROCESSOR_NAME + 1];

    std::vector<char> pName(world_size * (MPI_MAX_PROCESSOR_NAME + 1));

    std::vector<int> pNameCount(world_size);
    std::vector<int> pNameDisp(world_size);

    MPI_Get_processor_name(myName, &length);

    strcpy_s(&pName[static_cast<std::vector<char, std::allocator<char>>::size_type>(world_rank) * (MPI_MAX_PROCESSOR_NAME + 1)], MPI_MAX_PROCESSOR_NAME + 1, myName);

    for (int i = 0; i < world_size; i++) {
        pNameCount[i] = MPI_MAX_PROCESSOR_NAME + 1;
        pNameDisp[i] = i * (MPI_MAX_PROCESSOR_NAME + 1);
    }

    MPI_Allgatherv(myName, MPI_MAX_PROCESSOR_NAME + 1, MPI_CHAR, pName.data(), pNameCount.data(), pNameDisp.data(),
        MPI_CHAR, MPI_COMM_WORLD);

    bool bSingleNode = true;
    bool bP2P = false;

    for (int i = 0; i < world_size; i++) {
        if (std::string(&pName[i * (MPI_MAX_PROCESSOR_NAME + 1)]) != myName)
            bSingleNode = false;
    }

    cudaSetDeviceFlags(cudaDeviceMapHost);

    int localCount = 0;
    int offset = 1;

    for (int i = 0; i < world_size; i++) {
        if (!strcmp(&pName[static_cast<std::vector<char, std::allocator<char>>::size_type>(i) * (MPI_MAX_PROCESSOR_NAME + 1)], myName)) {
            localCount++;
            if (i < world_rank)
                offset++;
        }
    }

    if (localCount > 1) {
        int pos = 0;
        int device = -1;

        while (offset > 0) {
#ifdef _WIN32
#else
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, pos);

            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3)) {
                device = pos;
                offset--;
            }
#endif
            pos++;
            if (pos == gpuCount)
                pos = 0;
        }

        char hostname[128]{};

#ifdef _WIN32
#else
        gethostname(hostname, sizeof(hostname) - 1);
#endif

        std::cout << "GpuContext::Startup: Process " << _id << " running on device " << device << " out of " << gpuCount << " GPUs on " << hostname;
    }
    else {
        std::vector<int> pGPUList(gpuCount);
        std::vector<unsigned int> pGPUScore(gpuCount);
        int gpus = 0;

        for (int i = 0; i < gpuCount; i++) {
            cudaGetDeviceProperties(&deviceProp, i);

            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3)) {
                pGPUList[gpus] = i;
                pGPUScore[gpus] = (static_cast<unsigned long long>(deviceProp.major) << 24) + (deviceProp.totalGlobalMem >> 20);
                gpus++;
            }
        }

        if (gpus > 0) {
            bool done = true;
            do {
                done = true;
                for (int i = 0; i < gpus - 1; i++) {
                    if (pGPUScore[i] < pGPUScore[static_cast<std::vector<uint32_t, std::allocator<uint32_t>>::size_type>(i) + 1]) {
                        done = false;
                        int gpu = pGPUList[i];
                        unsigned int score = pGPUScore[i];
                        pGPUList[i] = pGPUList[static_cast<std::vector<int, std::allocator<int>>::size_type>(i) + 1];
                        pGPUScore[i] = pGPUScore[static_cast<std::vector<uint32_t, std::allocator<uint32_t>>::size_type>(i) + 1];
                        pGPUList[static_cast<std::vector<int, std::allocator<int>>::size_type>(i) + 1] = gpu;
                        pGPUScore[static_cast<std::vector<uint32_t, std::allocator<uint32_t>>::size_type>(i) + 1] = score;
                    }
                }
            } while (!done);
        }

        status = cudaSetValidDevices(pGPUList.data(), gpus);

        if (status != cudaSuccess) {
            std::cerr << "GpuContext::Startup: Error searching for compatible GPU";
        }

        status = cudaFree(0);

        if (status != cudaSuccess) {
            std::cerr << "GpuContext::Startup: Error selecting compatible GPU";
        }

        status = cudaGetDevice(&device);

        if (status != cudaSuccess) {
            std::cerr << "GpuContext::Startup: Error fetching current GPU";
        }

        if (device == -1) {
            std::cerr << "GpuContext::Startup: No Kepler or later GPU located, exiting.";
            cudaDeviceReset();
            Shutdown();
            exit(EXIT_FAILURE);
        }

        status = cudaSetDevice(device);

        if (status != cudaSuccess) {
            std::cerr << "GpuContext::Startup: Error setting CUDA device";
        }

        cudaDeviceSynchronize();

        _pbAccumulator.reset(new GpuBuffer<unsigned long long int>((unsigned int)1, true));
        _data._pAccumulator = _pbAccumulator->_pDevData;

        cudaGetDeviceProperties(&deviceProp, _device);

        if (deviceProp.major == 3) {
            _sm_version = SM_3X;
            _threadsPerBlock = SM_3X_THREADS_PER_BLOCK;
            _maxSparse = SM_3X_MAXSPARSE;
            _maxSparseAnalog = SM_3X_MAXSPARSEANALOG;
        }
        else if (deviceProp.major == 5) {
            _sm_version = SM_5X;
            _threadsPerBlock = SM_5X_THREADS_PER_BLOCK;
            _maxSparse = SM_5X_MAXSPARSE;
            _maxSparseAnalog = SM_5X_MAXSPARSEANALOG;
        }
        else {
            _sm_version = SM_6X;
            _threadsPerBlock = SM_6X_THREADS_PER_BLOCK;
            _maxSparse = SM_6X_MAXSPARSE;
            _maxSparseAnalog = SM_6X_MAXSPARSEANALOG;
        }
        _sm_major = deviceProp.major;
        _warpSize = deviceProp.warpSize;
        _warpBits = fls(_warpSize) - 1;
        _warpMask = _warpSize - 1;
        _data._warpSize = _warpSize;
        _data._warpBits = _warpBits;
        _data._warpMask = _warpMask;
        _bUnifiedMemory = (deviceProp.managedMemory != 0);

        _data._maxUint32_t = 0xFFFFFFFF;
        _data._maxInt32_t = 0x7FFFFFFF;
        _data._maxUint64_t = 0xFFFFFFFFFFFFFFFF;
        _data._maxInt64_t = 0x7FFFFFFFFFFFFFFF;

        if (getGpu()._id == 0)
            std::cout << "GpuContext::Startup: Enumerating GPUs in use.";

        for (size_t i = 0; i < getGpu()._numprocs; i++) {
            if (static_cast<size_t>(getGpu()._id) == i)
                std::cout << "Process: " << i << ", GPU: " << deviceProp.name << ", running SM " << deviceProp.major << "." << deviceProp.minor;
            MPI_Barrier(MPI_COMM_WORLD);
        }

        std::cout << "GpuContext::Startup: Single node flag on GPU for process " << _device << " is " << bSingleNode << "\n";

        if (bSingleNode) {
            bP2P = true;
            std::vector<int> pDevice(_numprocs);
            pDevice[_id] = device;

            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pDevice.data(), 1, MPI_INT, MPI_COMM_WORLD);

            std::vector<int> pUnifiedAddressing(_numprocs);
            cudaGetDeviceProperties(&deviceProp, device);
            pUnifiedAddressing[_id] = deviceProp.unifiedAddressing;

            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pUnifiedAddressing.data(), 1, MPI_INT, MPI_COMM_WORLD);

            for (int i = 0; i < _numprocs; i++) {
                if (pDevice[i] != device) {
                    int canAccessPeer;
                    std::cout << "GpuContext::Startup: Testing P2P for processes " << device << " and " << pDevice[i];
                    cudaError_t status = cudaDeviceCanAccessPeer(&canAccessPeer, device, pDevice[i]);

                    if (status != cudaSuccess) {
                        std::cerr << "cudaDeviceCanAccessPeer";
                    }

                    if (canAccessPeer == 0) {
                        bP2P = false;
                    }
                    else {
                        status = cudaDeviceEnablePeerAccess(pDevice[i], 0);

                        if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
                            std::cerr << "cudaDeviceEnablePeerAccess";
                        }
                        else if (status == cudaErrorPeerAccessAlreadyEnabled) {
                            cudaGetLastError();
                        }
                    }
                }
                if (!pUnifiedAddressing[i])
                    bSingleNode = false;
            }
        }

        _bSingleNode = bSingleNode;
        _bP2P = bP2P;

        std::cout << "GpuContext::Startup: P2P support flags on GPU for process " << _device << " are " << _bP2P << " " << _bSingleNode;

        MPI_Allreduce(MPI_IN_PLACE, &_bP2P, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

        if (!_bP2P) {
            if (_id == 0)
                std::cout << "GpuContext::Startup: Not all GPUs can P2P between each other, falling back to MPI.";
        }

        MPI_Allreduce(MPI_IN_PLACE, &_bSingleNode, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

        if (!_bSingleNode) {
            if (_id == 0)
                std::cout << "GpuContext::Startup: P2P support only works within a single node, falling back to MPI.";
        }

        cudaGetDeviceProperties(&deviceProp, device);
        _bECCSupport = deviceProp.ECCEnabled || deviceProp.tccDriver;

        std::string deviceNameLower = deviceProp.name;
        std::transform(deviceNameLower.begin(), deviceNameLower.end(), deviceNameLower.begin(), ::tolower);

        if (deviceNameLower.find("tesla") != std::string::npos) {
            _bECCSupport = true;
        }

        _bCanMapHostMemory = deviceProp.canMapHostMemory;

#ifdef GVERBOSE
        double memsize = (double)deviceProp.totalGlobalMem / (1024.0 * 1024.0);
        std::cout << "GpuContext::Startup: Using GPU " << device << ", " << deviceProp.name << ", SM " << deviceProp.major << "." << deviceProp.minor << ", " << memsize << " MBytes of memory";
#endif

        cublasStatus_t cstatus = cublasCreate(&_cuBLASHandle);

        if (cstatus != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "GpuContext::Startup: Failed to initialize cuBLAS on GPU for process " << _device << ", exiting.";
            Shutdown();
            std::exit(EXIT_FAILURE);
        }

        cudnnStatus_t cdstatus = cudnnCreate(&_cuDNNHandle);

        if (cdstatus != CUDNN_STATUS_SUCCESS) {
            std::cerr << "GpuContext::Startup: Failed to initialize cuDNN on GPU for process " << _device << ", exiting.";
            Shutdown();
            std::exit(EXIT_FAILURE);
        }

        curandStatus_t crstatus = curandCreateGenerator(&_RNG, CURAND_RNG_PSEUDO_DEFAULT);

        if (crstatus != CURAND_STATUS_SUCCESS) {
            std::cerr << "GpuContext::Startup: Failed to initialize cuRand on GPU for process " << _device << ", exiting.";
            Shutdown();
            std::exit(EXIT_FAILURE);
        }

        std::cout << "GpuContext::Startup: GPU for process " << device << " initialized.";
    }
}

void GpuContext::CopyConstants() {
    struct GpuDataLambda {
        std::function<void(GpuContext&, const GpuData&)> setGpuData;
        const char* name;
    };

    std::vector<GpuDataLambda> gpuDataLambdas = {
        {&GpuContext::SetsortingGpuData, "sortingGpuData"},
        {&GpuContext::SetsparseGpuData, "sparseGpuData"},
        {&GpuContext::SetactivationGpuData, "activationGpuData"},
        {&GpuContext::SetdeltaGpuData, "deltaGpuData"},
        {&GpuContext::SetbanBadWordsGpuData, "banBadWordsGpuData"},
        {&GpuContext::SetbanRepeatNgramGpuData, "banRepeatNgramGpuData"},
        {&GpuContext::SetdecodingGpuData, "decodingGpuData"},
        {&GpuContext::SetbeamSearchPenaltyGpuData, "beamSearchPenaltyGpuData"},
        {&GpuContext::SetonlineSoftmaxBeamsearchGpuData, "onlineSoftmaxBeamsearchGpuData"},
        {&GpuContext::SetstopCriteriaGpuData, "stopCriteriaGpuData"},
        {&GpuContext::SetrmsnormGpuData, "rmsnormGpuData"},
        {&GpuContext::SetquantizationGpuData, "quantizationGpuData"},
        {&GpuContext::SetpreQuantScaleGpuData, "preQuantScaleGpuData"},
        {&GpuContext::SetbeamSearchTopkGpuData, "beamSearchTopkGpuData"},
        {&GpuContext::SetcomputeSeqOffsetsGpuData, "computeSeqOffsetsGpuData"},
        {&GpuContext::SetlayernormalisationGpuData, "layernormalisationGpuData"},
        {&GpuContext::SetlookupGpuData, "lookupGpuData"}
    };

    GpuContext gpuContext;
    GpuData gpuData{};

    for (const auto& lambda : gpuDataLambdas) {
        try {
            lambda.setGpuData(gpuContext, gpuData);
        }
        catch (const std::exception& e) {
            std::cerr << "Exception caught for copying GPU constants " << lambda.name << ": " << e.what() << "\n";
        }
    }
}

void GpuContext::SetFastMath(bool flag)
{
    try
    {
        int deviceCount;
        cudaError_t cudaError = cudaGetDeviceCount(&deviceCount);

        if (cudaError != cudaSuccess || deviceCount == 0)
        {
            throw std::runtime_error("No CUDA-compatible GPU found: " + std::string(cudaGetErrorString(cudaError)));
        }

        if (NUM_GPUS > deviceCount)
        {
            throw std::runtime_error("Requested number of GPUs (" + std::to_string(NUM_GPUS) +
                ") exceeds available GPUs (" + std::to_string(deviceCount) + ")");
        }

        std::vector<std::jthread> gpuThreads;
        gpuThreads.reserve(NUM_GPUS);

        std::vector<cublasHandle_t> cuBLASHandles(NUM_GPUS);

        std::latch latch(NUM_GPUS);

        for (int deviceId = 0; deviceId < NUM_GPUS; ++deviceId)
        {
            gpuThreads.emplace_back([&cuBLASHandles, &latch, deviceId, flag] {
                try
                {
                    cudaSetDevice(deviceId);

                    int sm_major, sm_minor;
                    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, deviceId);
                    cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, deviceId);

                    if (sm_major < 8 || (sm_major == 8 && sm_minor < 0))
                    {
                        throw std::runtime_error("GPU SM revision is < 8.0");
                    }

                    cublasHandle_t cuBLASHandle;
                    cublasCreate(&cuBLASHandle);

                    cublasMath_t mathMode = flag ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
                    cublasSetMathMode(cuBLASHandle, mathMode);

                    cuBLASHandles[deviceId] = cuBLASHandle;
                }
                catch (const std::exception& e)
                {
                    std::cerr << "GPU " << deviceId << " setup exception: " << e.what() << '\n';
                }
                latch.count_down();
                });
        }

        latch.wait();
        _cuBLASHandles = std::move(cuBLASHandles);
    }
    catch (const std::exception& e)
    {
        std::cerr << "GpuContext::SetFastMath: " << e.what() << '\n';
    }
}

void GpuContext::Shutdown() {
    try {
        auto shutdownLibrary = [this](const char* libraryName, auto destroyFunc, auto& handle, auto successStatus) -> std::future<void> {
            return std::async(std::launch::async, [this, libraryName, destroyFunc, &handle, successStatus]() {
                auto message = std::format("Shutting down {} on GPU for process {}\n", libraryName, _device);
                std::cout << message;

                auto status = std::invoke(destroyFunc, handle);

                if (status != successStatus) {
                    throw std::runtime_error(std::format("Failed to shut down {} on GPU for process {}\n", libraryName, _device));
                }

                message = std::format("{} shut down on GPU for process {}\n", libraryName, _device);
                std::cout << message;
                });
            };

        auto cuBLASShutdown = shutdownLibrary("cuBLAS", cublasDestroy, _cuBLASHandle, CUBLAS_STATUS_SUCCESS);
        auto cuDNNShutdown = shutdownLibrary("cuDNN", cudnnDestroy, _cuDNNHandle, CUDNN_STATUS_SUCCESS);
        auto cuRandShutdown = shutdownLibrary("cuRand", curandDestroyGenerator, _RNG, CURAND_STATUS_SUCCESS);

        cuBLASShutdown.wait();
        cuDNNShutdown.wait();
        cuRandShutdown.wait();

        cudaDeviceReset();

        MPI_Finalize();

        auto finalMessage = std::format("Process {} out of {} finalized.\n", _id, _numprocs);
        std::cout << finalMessage;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during GPU context shutdown: " << e.what();
    }
}

void GpuContext::SetNeuralNetwork(Network* pNetwork)
{
    std::cout << "Setting Neural Network parameters in GpuContext";

    if (!pNetwork) {
        std::cerr << "Invalid Network pointer provided.";
        return;
    }

    auto& data = _data;
    const auto& network = *pNetwork;

    try {
        data._LRN_k = network._LRN_k;
        data._LRN_n = network._LRN_n;
        data._LRN_alpha = network._LRN_alpha;
        data._LRN_beta = network._LRN_beta;
        data._maxout_k = network._maxout_k;
        data._bSparsenessPenalty = network._bSparsenessPenalty;
        data._sparsenessPenalty_p = network._sparsenessPenalty_p;
        data._sparsenessPenalty_beta = network._sparsenessPenalty_beta;
        data._bDenoising = network._bDenoising;
        data._denoising_p = network._denoising_p;

        if (network._denoising_p != 1.0f) {
            data._denoising_q = 1.0f / (1.0f - network._denoising_p);
        }
        else {
            data._denoising_q = std::numeric_limits<float>::infinity();
        }

        data._deltaBoost_one = network._deltaBoost_one;
        data._deltaBoost_zero = network._deltaBoost_zero;
        data._SMCE_oneTarget = network._SMCE_oneTarget;
        data._SMCE_zeroTarget = network._SMCE_zeroTarget;
        data._SMCE_oneScale = network._SMCE_oneScale;
        data._SMCE_zeroScale = network._SMCE_zeroScale;

        if (data._bShuffleIndices && network._mode == Mode::Training) {
            throw std::runtime_error("Copying constants failed during training mode.");
        }

        data._bShuffleIndices = network._bShuffleIndices && (network._mode == Mode::Training);

        data._pShuffleIndex = network._pShuffleIndex;

        CopyConstants();

        std::cout << "Finished setting Neural Network parameters in GpuContext";
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred during parameter setting: " << e.what();
    }
}

void GpuContext::SetRandomSeed(unsigned long seed)
{
    constexpr unsigned long factor = 76801ull;

    curandStatus_t crstatus = curandSetPseudoRandomGeneratorSeed(_RNG, seed + static_cast<unsigned long long>(static_cast<unsigned long>(_device)) * factor);

    if (crstatus != CURAND_STATUS_SUCCESS)
    {
        std::ostringstream errorMessage;
        errorMessage << "GpuContext::SetRandomSeed: Failed to set cuRand seed on GPU for process " << _device << ".\n";
        std::cerr << errorMessage.str();
        Shutdown();
        throw std::runtime_error("Failed to set cuRand seed on GPU.");
    }

    srand(seed);

    std::ostringstream logMessage;
    logMessage << "GpuContext::SetRandomSeed: Random seed set to " << seed << ".\n";
    std::cout << logMessage.str();
}

unsigned long GpuContext::ModifySeed(unsigned long seed) const {
    std::random_device rd;
    seed ^= rd();

    std::default_random_engine engine(seed);
    std::uniform_int_distribution<unsigned long> dist(0, 2);

    auto modify_seed = [&](unsigned long& s) {
        unsigned long rnd = dist(engine);
        if (rnd == 0) {
            s = (((s * PRIME_A + PRIME_B) | s) ^ PRIME_C) + PRIME_D;
        }
        else if (rnd == 1) {
            s ^= (s << 13);
            s += (s >> 11);
        }
        else {
            s ^= (s >> SHIFT_BITS);
            s += (s << 19);
        }
        s = Mix(s);
        s ^= rd();
        s = ((s << 7) | (s >> (sizeof(s) * 8 - 7))) + (s ^ 0x3FF00FF);
        s ^= ((s >> 21) & 0x12345FF);
        };

    std::ranges::for_each(std::views::iota(0, 30), [&](auto) { modify_seed(seed); });

    seed ^= XOR_MASK_A;

    std::ranges::for_each(std::views::iota(0, 25), [&](auto) {
        seed = Mix(seed);
        seed ^= rd();
        seed ^= ((seed >> 15) & 0x98765432) ^ 0x8D7C1235;
        });

    seed ^= XOR_MASK_B;

    seed = ((seed << 17) | (seed >> (sizeof(seed) * 8 - 17))) ^ XOR_MASK_C;

    std::cout << "Modified seed value: " << seed;

    return seed;
}

#ifdef USE_HIGHWAY

const HWY_FULL(float) df;
const uint32_t end_idx = std::min(bk + block_size, k);
for (uint32_t idx = bk; idx < end_idx; idx += Lanes(df)) {
    const auto aVector = Load(df, &localA[(i - local_start) * k + idx]);
    const auto bVector = Load(df, &transposedB[j * k + idx]);
    const auto mulResult = aVector * bVector;
    sum += mulResult;
}
localC[(i - local_start) * n + j] += GetLane(SumOfLanes(sum));

#else

template <typename T>
void verifySGEMMImpl(std::vector<T>& vA, std::vector<T>& vB, std::vector<T>& vC, uint32_t m, uint32_t k, uint32_t n, float tolerance, int numThreads = -1, bool printErrors = true) {
    if (vA.size() != m * k || vB.size() != k * n || vC.size() != m * n) {
        throw std::invalid_argument("Input matrix dimensions do not match vector sizes.");
    }

    const auto toleranceSquared = tolerance * tolerance;

    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        throw std::runtime_error("MPI should be configured with at least two processes.");
    }

    const uint32_t local_m = m / size;
    const uint32_t local_start = rank * local_m;
    const uint32_t local_end = local_start + local_m;
    const uint32_t block_size = 16;

    if (local_end > m) {
        throw std::runtime_error("Calculated submatrix exceeds original matrix dimensions.");
    }

    std::vector<T> localA(local_m * k);
    std::vector<T> localB(k * n);
    std::vector<T> localC(local_m * n);

    MPI_Datatype block_type;
    MPI_Type_vector(local_m, k, m, MPI_FLOAT, &block_type);
    MPI_Type_commit(&block_type);
    std::shared_ptr<void> block_type_guard(nullptr, [&block_type](void*) { MPI_Type_free(&block_type); });

    MPI_Scatterv(vA.data(), nullptr, nullptr, block_type, localA.data(), local_m * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vB.data(), k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    std::vector<T> transposedB(k * n);
    for (size_t idx = 0; idx < vB.size(); ++idx) {
        transposedB[(idx % n) * k + (idx / n)] = vB[idx];
    }

#pragma omp parallel for
    for (int bi = 0; bi < local_m; bi += block_size) {
        for (uint32_t bj = 0; bj < n; bj += block_size) {
            for (uint32_t bk = 0; bk < k; bk += block_size) {
                _mm_prefetch(reinterpret_cast<const char*>(&localA[(bi + block_size) * k + bk]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&transposedB[(bj + block_size) * k + bk]), _MM_HINT_T0);
                for (uint32_t i = bi; i < std::min(bi + block_size, local_m); ++i) {
                    for (uint32_t j = bj; j < std::min(bj + block_size, n); ++j) {
                        _mm_prefetch(reinterpret_cast<const char*>(&localA[(i - bi) * k + bk]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&transposedB[j * k + bk]), _MM_HINT_T0);
#ifdef __AVX512F__
                        __m512 sum = _mm512_setzero_ps();
                        const uint32_t end_idx = std::min(bk + block_size, k);
                        for (uint32_t idx = bk; idx < end_idx; idx += 16) {
                            __m512 aVector = _mm512_loadu_ps(&localA[(i - bi) * k + idx]);
                            __m512 bVector = _mm512_loadu_ps(&transposedB[j * k + idx]);
                            __m512 mulResult = _mm512_mul_ps(aVector, bVector);
                            sum = _mm512_add_ps(sum, mulResult);
                        }

                        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum, 0), _mm512_extractf32x8_ps(sum, 1));
                        sum256 = _mm256_hadd_ps(sum256, sum256);
                        sum256 = _mm256_hadd_ps(sum256, sum256);
                        float sum_elements[8];
                        _mm256_storeu_ps(sum_elements, sum256);

                        float temp_sum = 0.0f;
                        for (int k = 0; k < 8; ++k) {
                            temp_sum += sum_elements[k];
                        }

                        localC[(i - bi) * n + j] += temp_sum;
#else
                        localC[(i - bi) * n + j] = 0.0f;
                        for (uint32_t bk = 0; bk < k; ++bk) {
                            localC[(i - bi) * n + j] += localA[(i - bi) * k + bk] * transposedB[j * k + bk];
                        }
#endif
                    }
                }
            }
        }
    }

    MPI_Allgather(localC.data(), local_m* n, MPI_FLOAT, vC.data(), local_m* n, MPI_FLOAT, MPI_COMM_WORLD);

    if (printErrors && rank == 0) {
        float maxError = *std::max_element(vC.begin(), vC.end(),
            [&localC, &n](const T& a, size_t idx) {
                float diff = a - localC[static_cast<float>(idx) / n * n + idx % n];
                return diff * diff;
            });

        if (maxError > toleranceSquared) {
            std::cerr << "Error: Maximum squared error is above tolerance." << "\n";
        }
    }

    if (rank == 0) {
        std::cerr << "Matrix dimensions: " << m << " x " << k << " x " << n << "\n";
    }
}

void verifySGEMM(GpuBuffer<float>* pbA, GpuBuffer<float>* pbB, GpuBuffer<float>* pbC, uint32_t m, uint32_t k, uint32_t n, float tolerance = 0.000001f, int numThreads = -1, bool printErrors = true) {
    std::vector<float> vA(m * k);
    std::vector<float> vB(k * n);
    std::vector<float> vC(m * n);

    pbA->Download(vA.data());
    pbB->Download(vB.data());
    pbC->Download(vC.data());

    MPI_Init(NULL, NULL);

    verifySGEMMImpl(vA, vB, vC, m, k, n, tolerance, numThreads, printErrors);

    MPI_Finalize();
}

#endif