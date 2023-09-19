# Build stage
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

# Set an environment variable to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libatlas-base-dev \
    pkg-config \
    software-properties-common \
    unzip \
    wget \
    cmake \
    libopenmpi-dev \
    libjsoncpp-dev \
    libhdf5-dev \
    zlib1g-dev \
    libnetcdf-dev \
    libnetcdf-c++4-dev \
    g++-13 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set g++-13 as the default C++ compiler
RUN update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 50 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-13 50

# Download and install the CUB library (used for CUDA primitives)
RUN cd /tmp && \
    wget https://github.com/NVIDIA/cub/archive/2.1.0.zip && \
    unzip 2.1.0.zip && \
    cp -rf cub-2.1.0/cub/ /usr/local/include/ && \
    rm -rf /tmp/*

# Set environment variables for OpenMPI and library paths
ENV PATH=/usr/local/openmpi/bin/:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/openmpi/lib/:${LD_LIBRARY_PATH}

# Copy the contents of the current directory into /opt and build/install the application
COPY . /opt

RUN cd /opt && \
    make install

# Runtime stage
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Copy necessary files and libraries from the builder stage
COPY --from=builder /usr/local/openmpi /usr/local/openmpi
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /opt /opt

# Add OpenMPI and application binaries to the PATH
ENV PATH=/usr/local/openmpi/bin/:/opt/bin/:${PATH}