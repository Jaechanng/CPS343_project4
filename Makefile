# Compiler
CXX = g++
CXXFLAGS = -Wall -O3

# Libraries
HDF5LIB = -lhdf5
BLASLIB = -lcblas

# Targets
TARGETS = proj4-cublas proj4-cuda cuda_stream

all: $(TARGETS)

#cuBLAS version (CUDA)
proj4-cublas: proj4-cublas.cu
	nvcc -O3 proj4-cublas.cu -o proj4-cublas -lhdf5 -lcublas

#CUDA version
proj4-cuda: proj4-cuda.cu
	nvcc -lineinfo -arch=compute_61 -code=sm_61,sm_75,sm_80 -O3 proj4-cuda.cu -o proj4-cuda -lhdf5 -lcublas

#CUDA streams version
cuda_stream: cuda_stream.cu
	nvcc -lineinfo -arch=compute_61 -code=sm_61,sm_75,sm_80 -O3 cuda_stream.cu -o cuda_stream -lhdf5 -lcublas

# clean
clean:
	rm -f $(TARGETS)
