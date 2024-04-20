all:
	nvcc -std=c++11 -g -G kernel.cu CCL.cu group_box.cpp SubtitleDetector.cpp main.cpp -o main -lcudart -I/usr/local/trt/include -L/usr/local/trt/lib -lnvinfer -lcudart -lnvparsers -lnvonnxparser -lcublas -lcudnn -lnvrtc -lnvinfer_plugin
clean:
	rm kernel
