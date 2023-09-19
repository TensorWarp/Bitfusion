#include "GpuTypes.h"
#include "Types.h"
#include "Kernels.cuh"
#include <cstring>
#include <omp.h>
#include <cmath>
#include <span>
#include <iostream>
#include <vector>
#include <mpi.h>
#include <format>
#include <stdexcept>

// Define a constant float value with an acceptable error margin.
static const float cAcceptableError = 0.00001f;

// Declare a static instance of the GpuContext class named "gpu."
static GpuContext gpu;

// Define a function that returns a reference to the "gpu" instance.
GpuContext& getGpu() { return gpu; }

// Include the header file for integer types.
#include <stdint.h>

// Check for the compiler being Microsoft Visual C++.
#if defined(_MSC_VER)
// Include the header file for intrinsics in Microsoft Visual C++.
#include <intrin.h>
#include <random>

/// <summary>
/// Find the index of the most significant bit that is set to 1.
/// </summary>
/// <param name="x">The input integer.</param>
/// <returns>The index of the most significant bit (1-based) or 0 if x is 0.</returns>
static __forceinline int fls(int x)
{
    if (x == 0) return 0;
    unsigned long index;
    // Use the _BitScanReverse function to find the index of the most significant bit.
    _BitScanReverse(&index, static_cast<unsigned long>(x));
    return static_cast<int>(index) + 1;
}

// Check for the compiler being GCC (GNU Compiler Collection).
#elif defined(__GNUC__)

/// <summary>
/// Find the index of the most significant bit that is set to 1.
/// </summary>
/// <param name="x">The input integer.</param>
/// <returns>The index of the most significant bit (1-based) or 0 if x is 0.</returns>
static __inline int fls(int x)
{
    // Use the __builtin_clz function to find the index of the most significant bit.
    return x ? sizeof(x) * 8 - __builtin_clz(x) : 0;
}

// If the compiler is neither Microsoft Visual C++ nor GCC, display an error message.
#else
#error Unsupported compiler
#endif

/// <summary>
/// Constructor for the GpuContext class.
/// </summary>
GpuContext::GpuContext() :
    // Initialize various member variables.
    _bECCSupport(false),
    _bCanMapHostMemory(false),
    _bCPUValidate(false),
    _bUnifiedMemory(false),
    _acceptableError(cAcceptableError),
    _totalCPUMemory(0),
    _totalGPUMemory(0),
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

}

/// <summary>
/// Destructor for the GpuContext class.
/// </summary>
GpuContext::~GpuContext()
{

}

/// <summary>
/// Set whether CPU validation is enabled or disabled.
/// </summary>
/// <param name="bValidate">True to enable CPU validation, false to disable.</param>
void GpuContext::SetCPUValidate(bool bValidate)
{
    _bCPUValidate = bValidate;
}

/// <summary>
/// Initializes the GPU context, including MPI, GPU device selection, and GPU-related settings.
/// </summary>
/// <param name="argc">The number of command-line arguments.</param>
/// <param name="argv">An array of command-line arguments.</param>
void GpuContext::Startup(int argc, char** argv)
{
    int flag = 0;
    // Check if MPI (Message Passing Interface) is already initialized.
    MPI_Initialized(&flag);
    if (!flag) {
        // If MPI is not initialized, initialize it with the provided arguments.
        MPI_Init(&argc, &argv);
    }

    // Get the number of processes and the process ID within the MPI_COMM_WORLD communicator.
    MPI_Comm_size(MPI_COMM_WORLD, &_numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &_id);

    // Print a message indicating the initialization of the process.
    std::cout << std::format("GpuContext::Startup: Process %d out of %d initialized.\n", _id, _numprocs);

    char* cudaProfile = nullptr;
#ifdef _WIN32
    // On Windows, check for the CUDA_PROFILE environment variable using _dupenv_s.
    if (_dupenv_s(&cudaProfile, nullptr, "CUDA_PROFILE") == 0 && cudaProfile != nullptr) {
#else
    // On non-Windows systems, check for the CUDA_PROFILE environment variable using getenv.
    cudaProfile = getenv("CUDA_PROFILE");
    if (cudaProfile != nullptr) {
#endif
        char profile_log[512];
        char* cudaProfileLog = nullptr;
#ifdef _WIN32
        // On Windows, check for the CUDA_PROFILE_LOG environment variable using _dupenv_s.
        if (_dupenv_s(&cudaProfileLog, nullptr, "CUDA_PROFILE_LOG") == 0 && cudaProfileLog != nullptr) {
#else
        // On non-Windows systems, check for the CUDA_PROFILE_LOG environment variable using getenv.
        cudaProfileLog = getenv("CUDA_PROFILE_LOG");
        if (cudaProfileLog != nullptr) {
#endif
            snprintf(profile_log, sizeof(profile_log), "%s%d", cudaProfileLog, _id);
#ifdef _WIN32
            // Free the allocated memory for cudaProfileLog on Windows.
            free((void*)cudaProfileLog);
#else
            // Free the allocated memory for cudaProfileLog on non-Windows systems.
            free((void*)const_cast<char*>(cudaProfileLog));
#endif
        }
        else {
            // If CUDA_PROFILE_LOG is not set, create a default log filename.
            snprintf(profile_log, sizeof(profile_log), "cu%d.csv", _id);
        }

#ifdef _WIN32
        // Set the CUDA_PROFILE_LOG environment variable to the generated log filename on Windows.
        _putenv_s("CUDA_PROFILE_LOG", profile_log);
#else
        // Set the CUDA_PROFILE_LOG and CUDA_PROFILE_CSV environment variables on non-Windows systems.
        setenv("CUDA_PROFILE_LOG", profile_log, 1);
        setenv("CUDA_PROFILE_CSV", "1", 1);
#endif

#ifdef _WIN32
        // Free the allocated memory for cudaProfile on Windows.
        free(cudaProfile);
#else
        // Free the allocated memory for cudaProfile on non-Windows systems.
        free(cudaProfile);
#endif
    }

    // Initialize the device variable with -1 (indicates no CUDA device selected).
    int device = -1;

    // Initialize the GPU count to 0.
    int gpuCount = 0;

    // Declare a variable to store CUDA API call status.
    cudaError_t status;

    // Declare a structure to store CUDA device properties.
    cudaDeviceProp deviceProp;

    // Retrieve the number of CUDA-capable devices and store the status.
    status = cudaGetDeviceCount(&gpuCount);

    // If cudaGetDeviceCount fails, throw a runtime error with the error status.
    throw std::runtime_error("cudaGetDeviceCount failed with status: " + std::to_string(status));

    // Check if no CUDA-capable devices were found.
    if (gpuCount == 0) {
        // Output a message indicating no GPUs were found.
        std::cout << std::format("GpuContext::Startup: No CUDA-capable devices found, exiting.\n");

        // Reset the CUDA device and perform cleanup.
        cudaDeviceReset();
        Shutdown();
        exit(-1);
    }

    // Declare variables for MPI world size and rank.
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int length;

    // Declare an array to store the processor name.
    char myName[MPI_MAX_PROCESSOR_NAME + 1];

    // Create a vector to store processor names for all processes.
    std::vector<char> pName(world_size * (MPI_MAX_PROCESSOR_NAME + 1));

    // Create vectors for storing processor name count and displacement.
    std::vector<int> pNameCount(world_size);
    std::vector<int> pNameDisp(world_size);

    // Get the processor name for the current process and its length.
    MPI_Get_processor_name(myName, &length);

    // Copy the current processor name into the appropriate position in the vector.
    strcpy_s(&pName[static_cast<std::vector<char, std::allocator<char>>::size_type>(world_rank) * (MPI_MAX_PROCESSOR_NAME + 1)], MPI_MAX_PROCESSOR_NAME + 1, myName);

    // Initialize pNameCount and pNameDisp arrays.
    for (int i = 0; i < world_size; i++) {
        pNameCount[i] = MPI_MAX_PROCESSOR_NAME + 1;
        pNameDisp[i] = i * (MPI_MAX_PROCESSOR_NAME + 1);
    }

    // Gather processor names from all processes using MPI_Allgatherv.
    MPI_Allgatherv(myName, MPI_MAX_PROCESSOR_NAME + 1, MPI_CHAR, pName.data(), pNameCount.data(), pNameDisp.data(),
        MPI_CHAR, MPI_COMM_WORLD);

    // Initialize boolean flags for single-node and P2P communication.
    bool bSingleNode = true;
    bool bP2P = false;

    // Check if processor names of all processes match; if not, set bSingleNode to false.
    for (int i = 0; i < world_size; i++) {
        if (std::string(&pName[i * (MPI_MAX_PROCESSOR_NAME + 1)]) != myName)
            bSingleNode = false;
    }

    // Set CUDA device flags for mapping host memory.
    cudaSetDeviceFlags(cudaDeviceMapHost);

    // Initialize localCount and offset for GPU selection.
    int localCount = 0;
    int offset = 1;

    // Determine the localCount and offset for the current process.
    for (int i = 0; i < world_size; i++) {
        if (!strcmp(&pName[static_cast<std::vector<char, std::allocator<char>>::size_type>(i) * (MPI_MAX_PROCESSOR_NAME + 1)], myName)) {
            localCount++;
            if (i < world_rank)
                offset++;
        }
    }

    // If multiple processes share the same node, select a compatible GPU.
    if (localCount > 1) {
        int pos = 0;
        int device = -1;

        // Iterate through available GPUs to find a suitable one.
        while (offset > 0) {
#ifdef _WIN32
#else
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, pos);

            // Check if the GPU can map host memory and has a major version greater than or equal to 3.
            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3)) {
                device = pos;
                offset--;
            }
#endif
            pos++;
            if (pos == gpuCount)
                pos = 0;
        }

        char hostname[128];

#ifdef _WIN32
#else
        gethostname(hostname, sizeof(hostname) - 1);
#endif

        // Output information about the selected GPU.
        std::cout << std::format("GpuContext::Startup: Process %d running on device %d out of %d GPUs on %s\n", _id, device, gpuCount, hostname);
    }
    else {
        // Handle the case when only one process per node exists.
        // Initialize data structures for GPU selection based on criteria.
        std::vector<int> pGPUList(gpuCount);
        std::vector<unsigned int> pGPUScore(gpuCount);
        int gpus = 0;

        // Iterate through available GPUs to identify compatible ones.
        for (int i = 0; i < gpuCount; i++) {
            cudaGetDeviceProperties(&deviceProp, i);

            // Check if the GPU can map host memory and has a major version greater than or equal to 3.
            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3)) {
                pGPUList[gpus] = i;
                pGPUScore[gpus] = (static_cast<unsigned long long>(deviceProp.major) << 24) + (deviceProp.totalGlobalMem >> 20);
                gpus++;
            }
        }

        // Sort GPUs based on a scoring criterion.
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

        // Set the valid devices based on the selected GPU list.
        status = cudaSetValidDevices(pGPUList.data(), gpus);

        // Handle errors related to setting valid devices.
        if (status != cudaSuccess) {
            throw std::runtime_error("GpuContext::Startup: Error searching for compatible GPU");
        }

        // Allocate memory on the selected GPU to check compatibility.
        status = cudaFree(0);

        // Handle errors related to memory allocation on the GPU.
        if (status != cudaSuccess) {
            throw std::runtime_error("GpuContext::Startup: Error selecting compatible GPU");
        }

        // Get the current CUDA device.
        status = cudaGetDevice(&device);

        // Handle errors related to fetching the current GPU.
        if (status != cudaSuccess) {
            throw std::runtime_error("GpuContext::Startup: Error fetching current GPU");
        }

        // Handle the case when no compatible GPU was found.
        if (device == -1) {
            std::cout << std::format("GpuContext::Startup: No Kepler or later GPU located, exiting.\n");
            cudaDeviceReset();
            Shutdown();
            exit(-1);
        }

        // Set the CUDA device to the selected GPU.
        status = cudaSetDevice(device);

        // Handle errors related to setting the CUDA device.
        if (status != cudaSuccess) {
            throw std::runtime_error("GpuContext::Startup: Error setting CUDA device");
        }

        // Perform CUDA device synchronization.
        cudaDeviceSynchronize();

        // Initialize various GPU-related properties based on device properties.
        _pbAccumulator.reset(new GpuBuffer<unsigned long long int>((unsigned int)1, true));
        _data._pAccumulator = _pbAccumulator->_pDevData;

        cudaGetDeviceProperties(&deviceProp, _device);

        // Determine the SM version and set related properties.
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

        // Initialize maximum values for various data types.
        _data._maxUint32_t = 0xFFFFFFFF;
        _data._maxInt32_t = 0x7FFFFFFF;
        _data._maxUint64_t = 0xFFFFFFFFFFFFFFFF;
        _data._maxInt64_t = 0x7FFFFFFFFFFFFFFF;

        // Output information about the selected GPU.
        if (getGpu()._id == 0)
            std::cout << std::format("GpuContext::Startup: Enumerating GPUs in use.\n");

        // Iterate through processes and print GPU-related information.
        for (size_t i = 0; i < getGpu()._numprocs; i++) {
            if (static_cast<size_t>(getGpu()._id) == i)
                std::cout << std::format("Process: %lu, GPU: %s, running SM %d.%d\n", i, deviceProp.name, deviceProp.major, deviceProp.minor);
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Output information about single-node and P2P communication support.
        std::cout << std::format("GpuContext::Startup: Single node flag on GPU for process %d is %d\n", _device, bSingleNode);

        // Check if single-node mode is enabled and set P2P flag accordingly.
        if (bSingleNode) {
            bP2P = true;
            std::vector<int> pDevice(_numprocs);
            pDevice[_id] = device;

            // Gather device information from all processes.
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pDevice.data(), 1, MPI_INT, MPI_COMM_WORLD);

            std::vector<int> pUnifiedAddressing(_numprocs);
            cudaGetDeviceProperties(&deviceProp, device);
            pUnifiedAddressing[_id] = deviceProp.unifiedAddressing;

            // Gather unified addressing information from all processes.
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pUnifiedAddressing.data(), 1, MPI_INT, MPI_COMM_WORLD);

            // Check P2P compatibility between processes and update flags accordingly.
            for (int i = 0; i < _numprocs; i++) {
                if (pDevice[i] != device) {
                    int canAccessPeer;
                    std::cout << std::format("GpuContext::Startup: Testing P2P for processes {} and {}\n", device, pDevice[i]);
                    cudaError_t status = cudaDeviceCanAccessPeer(&canAccessPeer, device, pDevice[i]);

                    // Handle errors related to P2P access.
                    if (status != cudaSuccess) {
                        throw std::runtime_error("cudaDeviceCanAccessPeer");
                    }

                    if (canAccessPeer == 0) {
                        bP2P = false;
                    }
                    else {
                        status = cudaDeviceEnablePeerAccess(pDevice[i], 0);

                        // Handle errors related to enabling P2P access.
                        if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
                            throw std::runtime_error("cudaDeviceEnablePeerAccess");
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

        // Update single-node and P2P flags.
        _bSingleNode = bSingleNode;
        _bP2P = bP2P;

        // Print P2P support flags.
        printf("GpuContext::Startup: P2P support flags on GPU for process %d are %d %d\n", _device, _bP2P, _bSingleNode);

        // Perform collective logical AND operations to synchronize flags across all processes.
        MPI_Allreduce(MPI_IN_PLACE, &_bP2P, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

        // Check if P2P is not supported and print a message.
        if (!_bP2P) {
            if (_id == 0)
                std::cout << std::format("GpuContext::Startup: Not all GPUs can P2P between each other, falling back to MPI.\n");
        }

        // Perform collective logical AND operations to synchronize flags across all processes.
        MPI_Allreduce(MPI_IN_PLACE, &_bSingleNode, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

        // Check if single-node mode is not supported and print a message.
        if (!_bSingleNode) {
            if (_id == 0)
                std::cout << std::format("GpuContext::Startup: P2P support only works within a single node, falling back to MPI.\n");
        }

        // Get ECC support information from the device properties.
        cudaGetDeviceProperties(&deviceProp, device);
        _bECCSupport = deviceProp.ECCEnabled || deviceProp.tccDriver;

        // Convert the device name to lowercase for comparison.
        std::string deviceNameLower = deviceProp.name;
        std::transform(deviceNameLower.begin(), deviceNameLower.end(), deviceNameLower.begin(), ::tolower);

        // Check if the device name contains "tesla" and set ECC support accordingly.
        if (deviceNameLower.find("tesla") != std::string::npos) {
            _bECCSupport = true;
        }

        // Update variables related to memory mapping support.
        _bCanMapHostMemory = deviceProp.canMapHostMemory;
        _totalMemory = deviceProp.totalGlobalMem;

#ifdef GVERBOSE
        // Output detailed GPU information if GVERBOSE is defined.
        double memsize = (double)deviceProp.totalGlobalMem / (1024.0 * 1024.0);
        std::cout << std::format("GpuContext::Startup: Using GPU %d, %s, SM %d.%d, %.1f MBytes of memory\n", device, deviceProp.name, deviceProp.major, deviceProp.minor, memsize);
#endif

        // Initialize cuBLAS, cuDNN, and cuRAND libraries.
        cublasStatus_t cstatus = cublasCreate(&_cuBLASHandle);

        // Handle errors related to cuBLAS initialization.
        if (cstatus != CUBLAS_STATUS_SUCCESS) {
            std::cout << std::format("GpuContext::Startup: Failed to initialize cuBLAS on GPU for process %d, exiting.\n", _device);
            Shutdown();
            std::exit(-1);
        }

        cudnnStatus_t cdstatus = cudnnCreate(&_cuDNNHandle);

        // Handle errors related to cuDNN initialization.
        if (cdstatus != CUDNN_STATUS_SUCCESS) {
            std::cout << std::format("GpuContext::Startup: Failed to initialize cuDNN on GPU for process %d, exiting.\n", _device);
            Shutdown();
            std::exit(-1);
        }

        curandStatus_t crstatus = curandCreateGenerator(&_RNG, CURAND_RNG_PSEUDO_DEFAULT);

        // Handle errors related to cuRand initialization.
        if (crstatus != CURAND_STATUS_SUCCESS) {
            std::cout << std::format("GpuContext::Startup: Failed to initialize cuRand on GPU for process %d, exiting.\n", _device);
            Shutdown();
            std::exit(-1);
        }

        // Output a message indicating successful GPU initialization.
        std::cout << std::format("GpuContext::Startup: GPU for process %d initialized.\n", device);
    }
}

/// <summary>
/// Copies constants and data relevant to GPU kernels, loss functions, activations, and deltas.
/// </summary>
void GpuContext::CopyConstants()
{
    // Copy GPU data for kernels
    SetKernelsGpuData();

    // Copy GPU data for loss functions
    SetKLossGpuData();

    // Copy GPU data for activation functions
    SetKActivationGpuData();

    // Copy GPU data for delta calculations
    SetKDeltaGpuData();
}

/// <summary>
/// Sets the fast math mode for GPU operations using cuBLAS on devices with a compute capability of 7.0 or higher.
/// </summary>
/// <param name="flag">A boolean flag indicating whether to enable fast math mode.</param>
void GpuContext::SetFastMath(bool flag)
{
    // Helper function to print error messages
    auto printError = [&](const std::string& reason) {
        std::puts(std::format("GpuContext::SetFastMath: Failed to set math mode because {}\n", reason).c_str());
        };

    // Check if the GPU's compute capability is 7.0 or higher
    if (_sm_major >= 7)
    {
        // Set the cuBLAS math mode to either tensor op math or default math based on the flag
        cublasMath_t mathMode = flag ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;

        // Attempt to set the math mode
        if (cublasSetMathMode(_cuBLASHandle, mathMode) != CUBLAS_STATUS_SUCCESS)
        {
            printError("of an unknown issue");
        }
    }
    else
    {
        // Print an error if the GPU's compute capability is less than 7.0
        printError("GPU SM revision is <7.0");
    }
}

/// <summary>
/// Shuts down the GPU context by releasing resources and finalizing GPU-related libraries.
/// </summary>
void GpuContext::Shutdown()
{
    // Reset the accumulator buffer
    _pbAccumulator.reset();

    // Helper function to shut down a GPU library and print status
    auto shutdownLibrary = [&](const char* libraryName, auto destroyFunc, auto handle, auto successStatus) {
        // Prepare a message indicating library shutdown
        std::string message = std::format("GpuContext::Shutdown: Shutting down {} on GPU for process {}\n", libraryName, _device);
        std::puts(message.c_str());

        // Attempt to destroy the library handle
        auto status = destroyFunc(handle);

        // Check the status and print the corresponding message
        if (status != successStatus) {
            message = std::format("GpuContext::Shutdown: Failed to shut down {} on GPU for process {}\n", libraryName, _device);
        }
        else {
            message = std::format("GpuContext::Shutdown: {} shut down on GPU for process {}\n", libraryName, _device);
        }
        std::puts(message.c_str());
        };

    // Shutdown cuBLAS, cuDNN, and cuRand libraries
    shutdownLibrary("cuBLAS", cublasDestroy, _cuBLASHandle, CUBLAS_STATUS_SUCCESS);
    shutdownLibrary("cuDNN", cudnnDestroy, _cuDNNHandle, CUDNN_STATUS_SUCCESS);
    shutdownLibrary("cuRand", curandDestroyGenerator, _RNG, CURAND_STATUS_SUCCESS);

    // Reset the CUDA device
    cudaDeviceReset();

    // Finalize MPI communications
    MPI_Finalize();

    // Print a final message indicating process finalization
    std::string finalMessage = std::format("GpuContext::Shutdown: Process {} out of {} finalized.\n", _id, _numprocs);
    std::puts(finalMessage.c_str());
}

/// <summary>
/// Associates a neural network configuration with the GPU context and copies relevant network parameters to the GPU.
/// </summary>
/// <param name="pNetwork">Pointer to the neural network to associate with the GPU context.</param>
void GpuContext::SetNeuralNetwork(Network* pNetwork)
{
    // Associate the neural network with the GPU context
    _pNetwork = pNetwork;

    // Copy relevant network parameters to GPU data
    _data._LRN_k = pNetwork->_LRN_k;
    _data._LRN_n = pNetwork->_LRN_n;
    _data._LRN_alpha = pNetwork->_LRN_alpha;
    _data._LRN_beta = pNetwork->_LRN_beta;
    _data._maxout_k = pNetwork->_maxout_k;
    _data._bSparsenessPenalty = pNetwork->_bSparsenessPenalty;
    _data._sparsenessPenalty_p = pNetwork->_sparsenessPenalty_p;
    _data._sparsenessPenalty_beta = pNetwork->_sparsenessPenalty_beta;
    _data._bDenoising = pNetwork->_bDenoising;
    _data._denoising_p = pNetwork->_denoising_p;
    _data._denoising_q = 1.0f / (1.0f - pNetwork->_denoising_p);
    _data._deltaBoost_one = pNetwork->_deltaBoost_one;
    _data._deltaBoost_zero = pNetwork->_deltaBoost_zero;
    _data._SMCE_oneTarget = pNetwork->_SMCE_oneTarget;
    _data._SMCE_zeroTarget = pNetwork->_SMCE_zeroTarget;
    _data._SMCE_oneScale = pNetwork->_SMCE_oneScale;
    _data._SMCE_zeroScale = pNetwork->_SMCE_zeroScale;

    // Determine whether to shuffle indices on the GPU based on network mode
    _data._bShuffleIndices = pNetwork->_bShuffleIndices && (pNetwork->_mode == Mode::Training);

    // Copy shuffle index data to GPU
    _data._pShuffleIndex = pNetwork->_pShuffleIndex;

    // Copy the updated constants to the GPU
    CopyConstants();
}

/// <summary>
/// Sets the random seed for the GPU and CPU random number generators.
/// </summary>
/// <param name="seed">The random seed to be used for initialization.</param>
void GpuContext::SetRandomSeed(unsigned long seed)
{
    // Constants for computing the GPU seed
    constexpr unsigned long factor = 76801ull;

    // Set the random seed for the cuRand library on the GPU
    curandStatus_t crstatus = curandSetPseudoRandomGeneratorSeed(_RNG, seed + static_cast<unsigned long long>(static_cast<unsigned long>(_device)) * factor);

    // Check if setting the GPU seed was successful
    if (crstatus != CURAND_STATUS_SUCCESS)
    {
        // Handle the error and shut down if necessary
        if (getGpu()._id == 0)
        {
            std::cerr << std::format("GpuContext::SetRandomSeed: Failed to set cuRand seed on GPU for process {}.\n", _device);
        }
        Shutdown();
        throw std::runtime_error("Failed to set cuRand seed on GPU.");
    }

    // Set the random seed for the CPU-based random number generator
    srand(seed);

    // Print a message indicating the random seed set (for process 0)
    if (getGpu()._id == 0)
    {
        std::cout << std::format("GpuContext::SetRandomSeed: Random seed set to {}.\n", seed);
    }
}

/// <summary>
/// Modifies the given seed value.
/// </summary>
/// <param name="seed">The seed to be modified.</param>
/// <returns>The modified seed value.</returns>
unsigned long GpuContext::ModifySeed(unsigned long seed) const
{
    // Constants used for mixing the seed value
    constexpr unsigned long PRIME_A = 2654435761ul;
    constexpr unsigned long PRIME_B = 63689ul;
    constexpr unsigned long PRIME_C = 378551ul;
    constexpr unsigned long PRIME_D = 6367ul;
    constexpr unsigned long XOR_MASK_A = 0x5A17A17Aul;
    constexpr unsigned long XOR_MASK_B = 0xC3A5C3A5ul;
    constexpr unsigned long XOR_MASK_C = 0x81958195ul;
    constexpr unsigned long SHIFT_BITS = 7;

    // Lambda function for mixing the seed value
    auto mix = [](unsigned long x) -> unsigned long {
        x ^= (x >> 17);
        x += 0xABCD1234u;
        x ^= (x << 9);
        x ^= (x >> 27);
        return x;
        };

    // Generate a random number from a hardware source
    std::random_device rd;
    seed ^= rd();

    // Create a random number generator and distribution
    std::default_random_engine engine(seed);
    std::uniform_int_distribution<unsigned long> dist(0, 2);

    // Perform seed modification loop
    for (int i = 0; i < 30; ++i) {
        unsigned long rnd = dist(engine);

        // Apply different seed modification operations based on the random value
        if (rnd == 0) {
            seed = (((seed * PRIME_A + PRIME_B) | seed) ^ PRIME_C) + PRIME_D;
        }
        else if (rnd == 1) {
            seed ^= (seed << 13);
            seed += (seed >> 11);
        }
        else {
            seed ^= (seed >> SHIFT_BITS);
            seed += (seed << 19);
        }

        // Mix the seed value using the defined lambda function
        seed = mix(seed);
        seed ^= rd();
        seed = ((seed << 7) | (seed >> (sizeof(seed) * 8 - 7))) + (seed ^ 0x3FF00FF);
        seed ^= ((seed >> 21) & 0x12345FF);
    }

    // Apply XOR operation with XOR_MASK_A
    seed ^= XOR_MASK_A;

    // Perform another seed modification loop
    for (int i = 0; i < 25; ++i) {

        // Mix the seed value using the defined lambda function
        seed = mix(seed);
        seed ^= rd();
        seed ^= ((seed >> 15) & 0x98765432) ^ 0x8D7C1235;
    }

    // Apply XOR operation with XOR_MASK_B
    seed ^= XOR_MASK_B;

    // Perform a final seed modification step
    seed = ((seed << 17) | (seed >> (sizeof(seed) * 8 - 17))) ^ XOR_MASK_C;

    // Return the modified seed value
    return seed;
}

/// <summary>
/// Retrieves the GPU and CPU memory usage in kilobytes.
/// </summary>
/// <param name="gpuMemory">Pointer to store the GPU memory usage in kilobytes.</param>
/// <param name="cpuMemory">Pointer to store the CPU memory usage in kilobytes.</param>
void GpuContext::GetMemoryUsage(int* gpuMemory, int* cpuMemory) const
{
    *gpuMemory = (int)(_totalGPUMemory / 1024ll);
    *cpuMemory = (int)(_totalCPUMemory / 1024ll);
    return;
}

/// <summary>
/// Initializes the MPI (Message Passing Interface) library.
/// </summary>
template <typename T>
void initializeMPI() {
    MPI_Init(NULL, NULL);
}

/// <summary>
/// Finalizes the MPI (Message Passing Interface) library.
/// </summary>
template <typename T>
void finalizeMPI() {
    MPI_Finalize();
}

/// <summary>
/// Downloads data from a GPU buffer to a vector in CPU memory.
/// </summary>
/// <typeparam name="T">The data type of the elements in the buffer.</typeparam>
/// <param name="pb">Pointer to the GPU buffer containing the data.</param>
/// <param name="data">Reference to a vector where the downloaded data will be stored.</param>
template <typename T>
void downloadData(GpuBuffer<T>* pb, std::vector<T>& data) {
    pb->Download(data.data());
}

/// <summary>
/// Verifies the result of the SGEMM (Single-Precision General Matrix Multiply) operation with MPI and OpenMP parallelization.
/// This function is a template that can be used with various data types.
/// </summary>
/// <typeparam name="T">The data type of the matrix elements.</typeparam>
/// <param name="pbA">Pointer to the input matrix A stored in GPU memory.</param>
/// <param name="pbB">Pointer to the input matrix B stored in GPU memory.</param>
/// <param name="pbC">Pointer to the output matrix C stored in GPU memory.</param>
/// <param name="m">The number of rows in matrices A and C.</param>
/// <param name="k">The number of columns in matrices A and rows in matrix B.</param>
/// <param name="n">The number of columns in matrices B and C.</param>
template <typename T>
void verifySGEMMHelper(GpuBuffer<T>* pbA, GpuBuffer<T>* pbB, GpuBuffer<T>* pbC, uint32_t m, uint32_t k, uint32_t n) {
    // Declare variables to store MPI rank and size
    int mpi_rank, mpi_size;

    // Get the MPI rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Get the total number of MPI processes
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Distribute work among MPI processes
    int block_size = m / mpi_size;
    int start_row = mpi_rank * block_size;
    int end_row = (mpi_rank == mpi_size - 1) ? m : start_row + block_size;

    // Allocate memory for matrices A, B, and C
    std::vector<T> vA(m * k);
    std::vector<T> vB(k * n);
    std::vector<T> vC(m * n);

    // Generate random data for matrices A and B (for demonstration purposes)
    if (mpi_rank == 0) {
        std::generate(vA.begin(), vA.end(), []() { return static_cast<T>(std::rand()) / RAND_MAX; });
        std::generate(vB.begin(), vB.end(), []() { return static_cast<T>(std::rand()) / RAND_MAX; });
    }

    // Broadcast data from root MPI process to all other processes
    MPI_Bcast(vA.data(), m * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vB.data(), k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Initialize matrix C with zeros
    std::fill(vC.begin(), vC.end(), static_cast<T>(0));

    // Parallelize within each MPI process using OpenMP
#pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

        // Loop through matrix multiplication in blocks
        for (int i = start_row; i < end_row; i += 32) {
            for (uint32_t j = 0; j < n; j += 32) {
                // Define a block-level matrix to accumulate results
                std::array<std::array<T, 32>, 32> block_sum = {};

                // Loop through the submatrices
                for (uint32_t kk = 0; kk < k; kk += 32) {
                    for (int ii = 0; ii < 32; ii += 8) {
                        for (int jj = 0; jj < 32; jj += 8) {
                            // Using AVX-512 registers for 8x8 matrix multiplication
                            std::array<std::array<__m512, 8>, 8> blockA{}, blockB{};

                            // Load data into AVX-512 registers
                            for (int iii = 0; iii < 8; iii++) {
                                blockA[iii][thread_id] = _mm512_loadu_ps(vA.data() + (i + ii + iii) * k + kk);
                                blockB[iii][thread_id] = _mm512_loadu_ps(vB.data() + (kk + jj + iii) * n + j);
                            }

                            // Perform matrix multiplication using AVX-512 intrinsics
                            for (int iii = 0; iii < 8; iii++) {
                                for (int jjj = 0; jjj < 8; jjj++) {
                                    __m512 sum = _mm512_setzero_ps();
                                    for (int kkk = 0; kkk < 8; kkk++) {
                                        sum = _mm512_fmadd_ps(blockA[iii][kkk], blockB[jjj][kkk], sum);
                                        blockA[iii][kkk] = _mm512_castsi512_ps(_mm512_srli_epi32(_mm512_castps_si512(blockA[iii][kkk]), 1));
                                        blockB[jjj][kkk] = _mm512_castsi512_ps(_mm512_srli_epi32(_mm512_castps_si512(blockB[jjj][kkk]), 1));
                                    }
                                    block_sum[ii + iii][jj + jjj] += _mm512_reduce_add_ps(sum);
                                }
                            }
                        }
                    }
                }

                // Accumulate block-level results into the final result matrix
                for (int ii = 0; ii < 32; ii++) {
                    for (int jj = 0; jj < 32; jj++) {
                        vC[(i + ii) * n + j + jj] = std::reduce(block_sum[ii][jj].begin(), block_sum[ii][jj].end(), static_cast<T>(0));
                    }
                }
            }
        }
    }

    // Gather results from all MPI processes
    MPI_Allgather(MPI_IN_PLACE, m * n / mpi_size, MPI_FLOAT, vC.data(), m * n / mpi_size, MPI_FLOAT, MPI_COMM_WORLD);

    // Perform additional computation on the result matrix (e.g., normalization)
    for (int i = 0; i < m * n; i++) {
        vC[i] /= static_cast<T>(m * n);
    }

    // Measure and print execution time
    if (mpi_rank == 0) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        std::cout << "Execution time: " << elapsed_time.count() << " seconds" << '\n';
    }
}

/// <summary>
/// Verifies the result of the SGEMM (Single-Precision General Matrix Multiply) operation with MPI initialization, computation, and finalization.
/// This function is a template that can be used with various data types.
/// </summary>
/// <typeparam name="T">The data type of the matrix elements.</typeparam>
/// <param name="pbA">Pointer to the input matrix A stored in GPU memory.</param>
/// <param name="pbB">Pointer to the input matrix B stored in GPU memory.</param>
/// <param name="pbC">Pointer to the output matrix C stored in GPU memory.</param>
/// <param name="m">The number of rows in matrices A and C.</param>
/// <param name="k">The number of columns in matrices A and rows in matrix B.</param>
/// <param name="n">The number of columns in matrices B and C.</param>
template <typename T>
void verifySGEMM(GpuBuffer<T>* pbA, GpuBuffer<T>* pbB, GpuBuffer<T>* pbC, uint32_t m, uint32_t k, uint32_t n) {
    // Initialize MPI
    initializeMPI();

    // Call the SGEMM verification helper function
    verifySGEMMHelper(pbA, pbB, pbC, m, k, n);

    // Finalize MPI
    finalizeMPI();
}

/// <summary>
/// Verifies the result of the SGEMM operation for matrices A, B, and C with no transposition using MPI.
/// </summary>
/// <typeparam name="T">The data type of the matrix elements.</typeparam>
/// <param name="pbA">Pointer to the input matrix A stored in GPU memory.</param>
/// <param name="pbB">Pointer to the input matrix B stored in GPU memory.</param>
/// <param name="pbC">Pointer to the output matrix C stored in GPU memory.</param>
/// <param name="m">The number of rows in matrices A and C.</param>
/// <param name="k">The number of columns in matrices A and rows in matrix B.</param>
/// <param name="n">The number of columns in matrices B and C.</param>
template <typename T>
void verifySGEMMNT(GpuBuffer<T>* pbA, GpuBuffer<T>* pbB, GpuBuffer<T>* pbC, uint32_t m, uint32_t k, uint32_t n) {
    // Initialize MPI
    initializeMPI();

    // Call the SGEMM verification helper function
    verifySGEMMHelper(pbA, pbB, pbC, m, k, n);

    // Finalize MPI
    finalizeMPI();
}

/// <summary>
/// Verifies the result of the SGEMM operation for matrices A, B, and C using MPI.
/// </summary>
/// <param name="pbA">Pointer to the input matrix A stored in GPU memory.</param>
/// <param name="pbB">Pointer to the input matrix B stored in GPU memory.</param>
/// <param name="pbC">Pointer to the output matrix C stored in GPU memory.</param>
/// <param name="m">The number of rows in matrices A and C.</param>
/// <param name="k">The number of columns in matrices A and rows in matrix B.</param>
/// <param name="n">The number of columns in matrices B and C.</param>
template <typename T>
void verifySGEMMTN(GpuBuffer<T>* pbA, GpuBuffer<T>* pbB, GpuBuffer<T>* pbC, uint32_t m, uint32_t k, uint32_t n) {
    // Initialize MPI
    initializeMPI();

    // Call the SGEMM verification helper function
    verifySGEMMHelper(pbA, pbB, pbC, m, k, n);

    // Finalize MPI
    finalizeMPI();
}