# Contributing to the Project

If you are interested in contributing to the project, there are a number of ways you can get involved. You can:

- Suggest improvements for NLP & other domains.
- Contribute code. If you are a developer, you can contribute code to the project by submitting pull requests.
- Spread the word. If you think the project is useful, you can help spread the word by sharing it with your colleagues and friends.

We're still in the project's early developmental stages, and we highly value your input and ideas for improvement. Don't hesitate to open a pull request or submit an issue. Additionally, we're actively working on ROCm support to ensure compatibility with AMD GPUs.

For easy access to our list of pending tasks, you can navigate to the "Task List" view in Visual Studio, which includes all the TODO items.

## Install Visual Studio

We highly recommend Visual Studio as our preferred tool for C++ projects, It offers a comprehensive and user-friendly development environment with powerful features tailored specifically for C++ programming with plugins for both CUDA & AMD. To install Visual Studio for your C++ projects, follow these steps:

1. Visit the official Visual Studio website (https://visualstudio.microsoft.com) and navigate to the Downloads section.
2. Choose the edition of Visual Studio that suits your needs. There are different editions available, such as Community (free), Professional, and Enterprise. Click on the corresponding download button.
3. Once the installer is downloaded, run it and select the desired installation options. Make sure to include the necessary components for C++ development.
4. Review and accept the license terms, then click on the Install button to begin the installation process. It may take some time to download and install all the necessary files.
5. Once the installation is complete, launch Visual Studio. You will be prompted to sign in with a Microsoft account. You can choose to sign in or skip this step.
6. After signing in, you will be presented with the Visual Studio start page. From here you can open this project and link the external dependencies.

### Installl Cuda Toolkit 12.2
The NVIDIA CUDA Toolkit provides a development environment for creating high performance GPU-accelerated applications. Download and simple install it for Visual Studio 2022

https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11

### Install AMD HIP SDK
The AMD HIP SDK is a software development kit (SDK) that allows us to utilize both CUDA & HIP in one codebase overtime.

https://www.amd.com/en/developer/rocm-hub/hip-sdk.html

### Install cuDNN
The NVIDIA CUDA Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers. To get started sign up for the developer program and agree to the terms of Ethical AI next push the download button.

1. In the CUDA installation folder, In my case: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\
Open folder v12.2 side by side with the downloaded cuDNN folder.
2. Copy the respective files from cuDNN to CUDA installation folder. From bin to bin, include to include, lib/x64 to lib/x64.

### Install vcpkg
Vcpkg is an C/C++ dependency manager from Microsoft, Installing vcpkg is a two-step process: first, clone the repo, then run the bootstrapping script to produce the vcpkg binary. The repo can be cloned anywhere, and will include the vcpkg binary after bootstrapping as well as any libraries that are installed from the command line. It is recommended to clone vcpkg as a submodule for CMake projects, but to install it globally for MSBuild projects. If installing globally, we recommend a short install path like: C:\src\vcpkg or C:\dev\vcpkg, since otherwise you may run into path issues for some port build systems.

> Make sure you are in the directory you want the tool installed to before doing this such as the C drive folder.

Step 1: Clone the vcpkg repo (You need to have git installed on your local machine)

```bash
git clone https://github.com/Microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
```

### Link the librariers
To link the external libraries such as CUDA Toolkit 12.2, Microsoft MPI, NetCDF, and cuDNN to the project in Visual Studio 2022, you can follow these steps:

1. install & download the MSMPI redistributable package to your local computer from the vendor folder.

2. Open the project and the developer command line console in Visual Studio 2022.

```bash
vcpkg install
```

2. Right-click on the project in the Solution Explorer, and select "Properties" from the context menu.
3. In the project properties window, navigate to the "Configuration Properties" section and select "VC++ Directories".
4. In the "Include Directories" field, add the paths to the header files of the external libraries. These paths will vary depending on the installation location of each library (Separate multiple paths with semicolons)
5. Click "Apply" and then "OK" to save the changes to the project properties.

## Build with Visual Studio

Go the tab build and choose build solution or press F7 to build the project.
