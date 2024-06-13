all:
	# nvcc -std=c++11 -g -G kernel.cu CCL.cu group_box.cpp SubtitleDetector.cpp main.cpp -o main -lcudart -I/usr/local/trt/include -L/usr/local/trt/lib -lnvinfer -lcudart -lnvparsers -lnvonnxparser -lcublas -lcudnn -lnvrtc -lnvinfer_plugin
	nvcc -std=c++11 -g -G kernel.cu CCL.cu group_box.cpp SubtitleDetector.cpp test_img.cpp -o test_img -lcudart -I/usr/local/trt/include -L/usr/local/trt/lib -lnvinfer -lcudart -lnvparsers -lnvonnxparser -lcublas -lcudnn -lnvrtc -lnvinfer_plugin `pkg-config --cflags --libs opencv4`
clean:
	rm kernel
