all:
	nvcc -std=c++11 -g -G -gencode arch=compute_70,code=sm_70 -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_86,code=sm_86 kernel.cu CCL.cu group_box.cpp SubtitleDetector.cpp main.cpp -o main -lcudart -I/usr/local/trt/include -L/usr/local/trt/lib -lnvinfer -lcudart -lnvparsers -lnvonnxparser -lcublas -lcudnn -lnvrtc -lnvinfer_plugin
clean:
	rm kernel
