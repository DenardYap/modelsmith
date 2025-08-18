UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Paths (override from env if needed)
LIBTORCH ?= ../../../../deps/libtorch
ABI ?= 0

# Compiler
ifeq ($(UNAME_S),Darwin)
	CXX ?= clang++
else
	CXX ?= g++
endif

# OpenMP flags per-OS
ifeq ($(UNAME_S),Darwin)
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

CXXFLAGS := -O3 -std=c++17 $(OMP_CFLAGS) \
	-I$(LIBTORCH)/include \
	-I$(LIBTORCH)/include/torch/csrc/api/include \
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

SRC_DIR := ..

benchmark: benchmark.cpp $(SRC_DIR)/gemm.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: benchmark_complete
# Disable LTO for this target to avoid potential linker issues with c10 symbols
benchmark_complete: CXXFLAGS := $(filter-out -flto,$(CXXFLAGS))
benchmark_complete: LDFLAGS := $(filter-out -flto,$(LDFLAGS))
benchmark_complete: CXXFLAGS := $(subst -D_GLIBCXX_USE_CXX11_ABI=0,-D_GLIBCXX_USE_CXX11_ABI=1,$(CXXFLAGS))
benchmark_complete: LDFLAGS := $(subst -D_GLIBCXX_USE_CXX11_ABI=0,-D_GLIBCXX_USE_CXX11_ABI=1,$(LDFLAGS))
benchmark_complete: benchmark_complete.cpp $(SRC_DIR)/binary_matmul.cpp $(SRC_DIR)/gemm.cpp $(SRC_DIR)/kernel_helper.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: benchmark_binary_matmul
# Disable LTO for this target to avoid potential linker issues with c10 symbols
benchmark_binary_matmul: CXXFLAGS := $(filter-out -flto,$(CXXFLAGS))
benchmark_binary_matmul: LDFLAGS := $(filter-out -flto,$(LDFLAGS))
benchmark_binary_matmul: benchmark_binary_matmul.cpp $(SRC_DIR)/gemm.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: test_pack
test_pack: CXXFLAGS := $(filter-out -flto,$(CXXFLAGS))
test_pack: LDFLAGS := $(filter-out -flto,$(LDFLAGS))
test_pack: CXXFLAGS := $(subst -D_GLIBCXX_USE_CXX11_ABI=0,-D_GLIBCXX_USE_CXX11_ABI=1,$(CXXFLAGS))
test_pack: LDFLAGS := $(subst -D_GLIBCXX_USE_CXX11_ABI=0,-D_GLIBCXX_USE_CXX11_ABI=1,$(LDFLAGS))
test_pack: $(SRC_DIR)/test_pack_binary_matrix.cpp $(SRC_DIR)/kernel_helper.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f benchmark
	rm -f benchmark_complete


