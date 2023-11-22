## C++ implementation of CRAFT text detector with TensorRT 
CRAFT: Character-Region Awareness For Text detection | [Paper ](https://arxiv.org/abs/1904.01941) | [Official Pytorch code](https://github.com/clovaai/CRAFT-pytorch)

### Overview

This is a C++ implementation for the CRAFT text detector with TensorRT for accelerated inference. Compared to the official PyTorch implementation, it significantly improves text detection efficiency and facilitates deployment.

### Getting started

#### Requirements

- CUDA
- TensorRT


#### Generate trt engine

- Download the trained models

    - Official pretrained model: [craft_mlt_25k.pth](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)


- Pth to Onnx

    ```
    python torch2onnx.py --usefp16 --torch_path ./pretrained/craft_mlt_25k.pth
    ```

- Onnx to trt engine
    ```
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/trt/lib
    g++ -g onnx2trt.cpp -o onnx2trt -I/usr/local/trt/include -L/usr/local/trt/lib -I/usr/local/cuda/include -lnvinfer -lnvparsers -lnvonnxparser -lnvinfer_plugin
    ./onnx2trt ./pretrained/craft_mlt_25k_fp16.onnx ./pretrained/craft_mlt_25k_fp16_dynamic_shape.cache
    ```

#### Make and run demo
```
make
./main
```

#### Interface Specification

- Initialization and loading of the TRT engine
    ```
    void infer_init(int height, int width, const char* engine_path, float ratio)
    ```


    * `height`: Height of the video/image

    * `width`: Width of the video/image

    * `engine_path`: Path to the engine

    * `ratio`: Scaling ratio for the input image, ranging from (0, 1], typically taken as 0.5

- Inference
    ```
    vector<int> infer_pipe(uint8_t **in_yuv, int format, int* line_size);
    ```    

    * `in_yuv`: Memory address of NV12 Y and UV planes

    * `format`: When format = 0, it indicates input is in RGB format; when format = 1, it indicates input is in YUV format

    Returns a vector, sequentially storing x_min, x_max, y_min, y_max for each box.

- Destruction
    ```
    void destroyObj()
    ```
    Call this function after all inferences are completed.