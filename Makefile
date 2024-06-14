# Makefile

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -Xcompiler -fPIC -w -g

# CUDA and TensorRT Include directories
INCLUDES = -I/usr/local/cuda/include -I/usr/local/trt/include -I./include

# CUDA and TensorRT Library directories
LIB_DIRS = -L/usr/local/cuda/lib64 -L/usr/local/trt/lib -L./src

# Libraries
LIBS = -lnvinfer -lcudart -lnvparsers -lnvonnxparser -lcublas -lcudnn -lnvrtc -lnvinfer_plugin -ldetector `pkg-config --cflags --libs opencv4`

# Source files
SRC_FILES = test_img.cpp

# Output file
OUTPUT_FILE = test_img

# Source files
SRC_FILES_2 = test_yuv.cpp

# Output file
OUTPUT_FILE_2 = test_yuv

# Default target
all: $(OUTPUT_FILE) $(OUTPUT_FILE_2)

# Rule for building the shared library
$(OUTPUT_FILE): $(SRC_FILES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC_FILES) -o $(OUTPUT_FILE) $(LIB_DIRS) $(LIBS)



$(OUTPUT_FILE_2): $(SRC_FILES_2)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC_FILES_2) -o $(OUTPUT_FILE_2) $(LIB_DIRS) $(LIBS)


# Clean target
clean:
	rm -f $(OUTPUT_FILE)
	rm -f $(OUTPUT_FILE_2)