# nvcc -Xcompiler -fPIC -w -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_75,code=sm_75 -c kernel.cu -o kernel.o
# nvcc -Xcompiler -fPIC -w -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_75,code=sm_75 -c label.cu -o label.o
# nvcc -Xcompiler -fPIC -w -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_75,code=sm_75 -c group_box.cpp -o group_box.o
# nvcc -Xcompiler -fPIC -w -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_75,code=sm_75 -c SubtitleDetector.cpp -o SubtitleDetector.o -I/data/source/TensorRT-8.0.1.6/include


# g++ kernel.o label.o group_box.o SubtitleDetector.o -fPIC -shared -o detector.so -L/usr/local/cuda/lib64 -L/data/source/TensorRT-8.0.1.6/lib -lnvinfer -lcudart -lnvparsers -lnvonnxparser -lcublas -lcudnn -lnvrtc -lnvinfer_plugin

#nvcc -Xcompiler -fPIC -w -g -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_75,code=sm_75 kernel.cu CCL.cu group_box.cpp SubtitleDetector.cpp -shared -o libdetector.so -I/data/source/TensorRT-8.0.1.6/include -L/usr/local/cuda/lib64 -L/data/source/TensorRT-8.0.1.6/lib -lnvinfer -lcudart -lnvparsers -lnvonnxparser -lcublas -lcudnn -lnvrtc -lnvinfer_plugin

nvcc -Xcompiler -fPIC -w -g -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_86,code=sm_86 kernel.cu CCL.cu group_box.cpp SubtitleDetector.cpp -shared -o libdetector.so -I/usr/local/cuda/include -I/usr/local/trt/include -L/usr/local/cuda/lib64 -L/usr/local/trt/lib -lnvinfer -lcudart -lnvparsers -lnvonnxparser -lcublas -lcudnn -lnvrtc -lnvinfer_plugin
