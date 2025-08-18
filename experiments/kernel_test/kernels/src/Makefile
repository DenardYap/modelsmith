LIBTORCH = /Users/bernardyap/Desktop/ModelSmith/experiments/kernel_test/kernels/libtorch
PYTHON_INCLUDE = /Users/bernardyap/.pyenv/versions/3.7.9/include/python3.7m
LIBOMP = /opt/homebrew/opt/libomp

CXX = g++
CXXFLAGS = -O3 -Xpreprocessor -fopenmp -std=c++17 \
	-I$(LIBTORCH)/include \
	-I$(LIBTORCH)/include/torch/csrc/api/include \
	-I$(PYTHON_INCLUDE) \
	-I$(LIBOMP)/include \
	-D_GLIBCXX_USE_CXX11_ABI=0 \
	-march=native -mtune=native -mcpu=apple-m2 \
	-flto \
	-ffast-math -fno-signed-zeros -fno-trapping-math -fassociative-math \
	-funroll-loops -fvectorize \
	-fomit-frame-pointer \
	-fstrict-aliasing \
	-DNDEBUG \
	-mfma

LDFLAGS = -L$(LIBTORCH)/lib -ltorch -ltorch_cpu -lc10 \
	-L$(LIBOMP)/lib -lomp \
	-Wl,-rpath,$(LIBTORCH)/lib \
	-framework Accelerate \
	-flto

EXTRA_SRC := ../src/kernel_helper.cpp ../src/gemm.cpp binary_matmul.cpp 

%: %.cpp $(EXTRA_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
