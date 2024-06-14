# Makefile

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -Xcompiler -fPIC -w -g

# CUDA and TensorRT Include directories
INCLUDES = -I/usr/local/cuda/include -I/usr/local/trt/include -I../include

# CUDA and TensorRT Library directories
LIB_DIRS = -L/usr/local/trt/lib

# Libraries
LIBS = -lnvinfer -lnvparsers -lnvonnxparser -lnvinfer_plugin

# Source files
SRC_FILES = onnx2trt.cpp

# Output file
OUTPUT_LIB = onnx2trt

# Default target
all: $(OUTPUT_LIB)

# Rule for building the shared library
$(OUTPUT_LIB): $(SRC_FILES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC_FILES) -o $(OUTPUT_LIB) $(LIB_DIRS) $(LIBS)

# Clean target
clean:
	rm -f $(OUTPUT_LIB)