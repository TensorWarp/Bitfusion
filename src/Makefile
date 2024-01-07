SHELL := /bin/sh
VPATH := .
BUILD_DIR := build
HEADERS_DIR := .
SOURCES_DIR := .
HEADERS_BUILD_DIR := $(BUILD_DIR)/include/system
CPP_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/system/cpp
CUDA_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/system/cuda
LIB_BUILD_DIR := $(BUILD_DIR)/lib

CC := gcc
NVCC := nvcc
CFLAGS := -Wall -Werror -I$(HEADERS_DIR)
CUDA_FLAGS := -std=c++20 -arch=sm_80 -Xcompiler -Wall,-Werror,-fPIC,-fopenmp -Xptxas -v -I$(HEADERS_DIR)
CUDA_INCLUDES := -I../runtime/utils
LDFLAGS := -L$(LIB_BUILD_DIR)
CUDA_LIBS := -lcudart -lcublas -lcudnn -lcurand -lcusparse -lcufft
CUDA_LOADLIBS := -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs

HEADERS := $(wildcard $(HEADERS_DIR)/*.h $(HEADERS_DIR)/*.hpp)
CPP_SOURCES := $(wildcard $(SOURCES_DIR)/*.c $(SOURCES_DIR)/*.cc $(SOURCES_DIR)/*.cpp)
CUDA_SOURCES := $(wildcard $(SOURCES_DIR)/*.cu)
CPP_OBJECTS := $(patsubst $(SOURCES_DIR)/%.cpp, $(CPP_OBJS_BUILD_DIR)/%.o, $(CPP_SOURCES))
CUDA_OBJECTS := $(patsubst $(SOURCES_DIR)/%.cu, $(CUDA_OBJS_BUILD_DIR)/%.o, $(CUDA_SOURCES))

-include ../Makefile.inc

.PHONY: all clean distclean directories

all: $(LIB_BUILD_DIR)/libsource.a $(LIB_BUILD_DIR)/libsource_system.so

$(LIB_BUILD_DIR)/libsource.a: $(CPP_OBJECTS) $(CUDA_OBJECTS) | directories
	ar -cvr $@ $^
	cp -r $(HEADERS) $(HEADERS_BUILD_DIR)

$(LIB_BUILD_DIR)/libsource_system.so: $(CPP_OBJECTS) $(CUDA_OBJECTS) | directories
	$(CC) -shared $(LDFLAGS) $(CUDA_LIBS) $^ -o $@ $(CUDA_LOADLIBS)

directories:
	@mkdir -p $(CPP_OBJS_BUILD_DIR) $(CUDA_OBJS_BUILD_DIR) $(LIB_BUILD_DIR) $(HEADERS_BUILD_DIR)

clean:
	@echo "Cleaning up..."
	@$(RM) -f $(CPP_OBJS_BUILD_DIR)/*.o $(CPP_OBJS_BUILD_DIR)/*.d
	@$(RM) -f $(CUDA_OBJS_BUILD_DIR)/*.o $(CUDA_OBJS_BUILD_DIR)/*.d
	@$(RM) -rf $(LIB_BUILD_DIR)

distclean: clean
	@echo "Performing full clean..."
	@$(RM) -rf $(BUILD_DIR)

$(CPP_OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling $<"
	@$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(CUDA_OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cu
	@mkdir -p $(@D)
	@echo "Compiling CUDA $<"
	@$(NVCC) $(CUDA_FLAGS) --keep-dir $(CUDA_OBJS_BUILD_DIR) $(CUDA_INCLUDES) -MMD -MP -c $< -o $@

-include $(CPP_OBJECTS:.o=.d)
-include $(CUDA_OBJECTS:.o=.d)
