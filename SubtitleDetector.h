/*
 * Detector.h
 * Author: yinghao
 */

#pragma once
#ifndef SUBTITLE_DETECTOR_H_
#define SUBTITLE_DETECTOR_H_


#include <iostream>
// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "NvOnnxParser.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <string>
#include "logging.h"
#include <assert.h>
#include <stdexcept>
#include <fstream>
#include <memory>
#include <cuda_fp16.h>
#include "kernel.h"
#include "group_box.h"
#include "component.h"
#include "CCL.cuh"
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
// extern "C"{
static Logger gLogger{Logger::Severity::kINFO};

// struct ConnectedComponent {
//     int label;
//     int startX, startY;
//     int endX, endY;
//     int area;
// };
extern "C"
{
void infer_init(int height, int width, const char* engine_path, float ratio);
vector<int> infer_pipe(uint8_t **in_yuv, int format, int* line_size);
void destroyObj();
vector<int> infer_pipe_rgb(uint8_t *rgb);
}
class Detector {
public:
    Detector(){}
    ~Detector();
    void buildEngine();
    void prepareContext();
    void load_engine_and_create_context();
    void onnx2trt();
    void initData(int height, int width, const char* onnx_path, float ratio);
    vector<int> inference(uint8_t **in_yuv, int* line_size);
    vector<int> inference_rgb_in(uint8_t *in_rgb);
    half *sdr_rgb;
private:
    nvinfer1::ILogger* logger_;
    float text_threshold_;
    float link_threshold_;
    float low_text_;
    bool poly_;
    float mag_ratio_;
    bool usefp16_;
    std::string backend_;
    bool use_easy_ocr_post_;
    string engine_path_, onnx_path_;
    
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    nvinfer1::IBuilder* builder_;
    uint8_t *input_yuv;
    int height, width;
    int p_count;
    int new_height, new_width;
    int scale_width, scale_height;
    // uint8_t *src_yuv_i_d;
    float *src_yuv_f_d;
    
    half *bilinear_out;
    half *pad_out;

    half *score_map;
    half *fea;
    // torch::nn::ModuleHolder<CRAFT> net_;

    float *text_score;
    float *link_score;
    uint8_t *text_score_comb;
    uint8_t *text_score_comb_dilate;
    cudaStream_t streams[3];
    cudaStream_t stream1;
    
    // label
    unsigned int *label_map;
    ConnectedComponent* components;
    ConnectedComponent* components_real;
    int *num_of_box;
    int *num_of_box_h;
    ConnectedComponent* components_real_h;

    // uint8_t *src_rgb_i_d;
    int dilate_radio;
};
// }
#endif
