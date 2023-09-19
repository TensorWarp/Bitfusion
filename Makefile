# Check if the default compiler supports C++20
CXX_SUPPORTS_CXX20 := $(shell echo "int main() {return 0;}" | $(CXX) -std=c++20 -x c++ -o /dev/null -)

# Check if the compiler supports C++20 and set the appropriate flags
ifeq ($(CXX_SUPPORTS_CXX20),)
    $(error The default compiler does not support C++20. Please use a compatible compiler or update your compiler settings.)
endif

# Specify the shell to use
SHELL := /bin/sh

# Set the default installation prefix to the current directory
PREFIX ?= $(CURDIR)

# Set the default build directory
BUILD_DIR ?= $(abspath $(CURDIR)/../../build)

# Define compilers
CC := mpiCC
NVCC := nvcc
CXX := mpiCC

# Set the DEBUG flag (0 for release, 1 for debug)
DEBUG ?= 0

# Define common compilation flags
COMMON_FLAGS := -std=c++20 -fPIC -DOMPI_SKIP_MPICXX -MMD -MP

# Conditionally add debug flags if DEBUG is set to 1
ifeq ($(DEBUG), 1)
    COMMON_FLAGS += -g -O0
else
    COMMON_FLAGS += -O3
endif

CFLAGS := $(COMMON_FLAGS)
CUDA_FLAGS := $(COMMON_FLAGS) --device-debug --generate-line-info --compiler-options=-fPIC --ptxas-options="-v" \
    -gencode arch=compute_70,code=sm_70 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_52,code=sm_52 \
    -DOMPI_SKIP_MPICXX

CU_INCLUDES := -I/usr/local/include -isystem /usr/local/cuda/include \
               -isystem /usr/lib/openmpi/include -isystem /usr/include/jsoncpp \
               -IB40C -IB40C/KernelCommon -I$(BUILD_DIR)/include

CU_LIBS := -L/usr/lib/atlas-base -L/usr/local/cuda/lib64 -L/usr/local/lib
CU_LOADLIBS := -lcudnn -lcurand -lcublas -lcudart -ljsoncpp -lnetcdf_c++4 -lnetcdf -lblas -ldl -lstdc++

SUBDIRS := src src/utils src/runtime tests

.PHONY: all source utils runtime install clean rebuild help

.DEFAULT_GOAL := help

COLOR_RESET = \033[0m
COLOR_YELLOW = \033[33m

define PRINT_SECTION
    @echo -e "$(COLOR_YELLOW)************  $(1) mode ************$(COLOR_RESET)"
endef

all: source utils runtime

# Rule to handle building of subdirectories
$(SUBDIRS):
	$(MAKE) -C $@

install: all
    mkdir -p $(PREFIX)
    find $(BUILD_DIR) -type f -exec cp -rfp {} $(PREFIX) \;

clean:
    $(foreach dir,$(SUBDIRS),$(MAKE) -C $(dir) clean;)

rebuild: clean all

help:
    @echo "Usage: make [target] [VARIABLE=value]"
    @echo ""
    @echo "Targets:"
    @echo "  all       Build all targets (default)"
    @echo "  source    Build the 'src' directory"
    @echo "  utils     Build the 'src/utils' directory"
    @echo "  runtime   Build the 'src/runtime' directory"
    @echo "  install   Install the built files to PREFIX (default: current directory)"
    @echo "  clean     Clean all subdirectories"
    @echo "  rebuild   Clean and rebuild all targets"
    @echo "  help      Display this help message"
    @echo ""
    @echo "Variables:"
    @echo "  PREFIX    Installation directory (default: current directory)"
    @echo "  DEBUG     Debug mode (0 for release, 1 for debug)"
    @echo ""
    @echo "Examples:"
    @echo "  make PREFIX=/usr/local install"
    @echo "  make DEBUG=1 source"
    @echo ""

ifndef VERBOSE
.SILENT:
endif

DEP_DIR := $(BUILD_DIR)/deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEP_DIR)/$*.Td

# Ensure dependency directory exists
$(DEP_DIR):
	mkdir -p $(DEP_DIR)

define CXX_COMPILE
    $(CC) $(DEPFLAGS) $(CFLAGS) $(CU_INCLUDES) -c -o $@ $<
    @mv -f $(DEP_DIR)/$*.Td $(DEP_DIR)/$*.d
endef

define CUDA_COMPILE
    $(NVCC) $(DEPFLAGS) $(CU_FLAGS) $(CU_INCLUDES) -c -o $@ $<
    @mv -f $(DEP_DIR)/$*.Td $(DEP_DIR)/$*.d
endef

-include $(DEP_DIR)/*.d

.PHONY: test

test: all
    cd tests && $(MAKE) test
