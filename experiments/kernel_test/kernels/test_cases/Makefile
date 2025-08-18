UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Paths (override from env if needed)
LIBTORCH ?= ../libtorch
ABI ?= 0

# Optional: Python include if you need it
PYTHON_INCLUDE ?=

# Compiler
ifeq ($(UNAME_S),Darwin)
	CXX ?= clang++
else
	CXX ?= g++
endif

# OpenMP flags per-OS
ifeq ($(UNAME_S),Darwin)
	# Homebrew libomp assumed; override LIBOMP if different
	LIBOMP ?= /opt/homebrew/opt/libomp
	OMP_CFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
	OMP_LDFLAGS := -L$(LIBOMP)/lib -lomp
	ACCEL := -framework Accelerate
else
	OMP_CFLAGS := -fopenmp
	OMP_LDFLAGS := -fopenmp
	ACCEL :=
endif

# SIMD flags per-arch
ifeq ($(UNAME_M),x86_64)
	SIMD_FLAGS ?= -mavx2 -mssse3 -mfma
else ifeq ($(UNAME_M),aarch64)
	SIMD_FLAGS ?= -march=armv8-a+simd
else
	SIMD_FLAGS ?=
endif

# Common flags
CXXFLAGS := -O3 -std=c++17 $(OMP_CFLAGS) \
	-I$(LIBTORCH)/include \
	-I$(LIBTORCH)/include/torch/csrc/api/include \
	$(if $(PYTHON_INCLUDE),-I$(PYTHON_INCLUDE),) \
	-D_GLIBCXX_USE_CXX11_ABI=$(ABI) \
	$(SIMD_FLAGS) \
	-flto \
	-ffast-math -fno-signed-zeros -fno-trapping-math -fassociative-math \
	-funroll-loops \
	-fomit-frame-pointer \
	-fstrict-aliasing \
	-DNDEBUG

LDFLAGS := -L$(LIBTORCH)/lib -ltorch -ltorch_cpu -lc10 \
	$(OMP_LDFLAGS) \
	-Wl,-rpath,$(LIBTORCH)/lib \
	$(ACCEL) \
	-flto

# Sources shared by all small test binaries built from a single .cpp
EXTRA_SRC := ../src/kernel_helper.cpp ../src/gemm.cpp binary_matmul.cpp

%: %.cpp $(EXTRA_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
