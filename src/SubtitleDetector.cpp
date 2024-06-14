/*
@File    :   SubtitleDetector.cpp
@Time    :   2023/08/31 15:40:13
@Author  :   zhuyinghao
@Desc    :   定义Detector相关功能函数
*/
#include "SubtitleDetector.h"

#define USE_ENGINE_CACHE 1
#define DEBUG1 0


// void label(unsigned char *d_image, unsigned char *d_image_dilate, unsigned int  *d_labels, ConnectedComponent *components, ConnectedComponent *components_real, ConnectedComponent *components_real_h, int *count, int *count_h, unsigned int X, unsigned int Y, unsigned int XY, cudaStream_t stream1, cudaStream_t stream2, int dilate_radio);

// void Detector::initData(int height, int width, const char* onnx_path, const char* engine_path, float text_threshold, float low_text, float link_threshold, float mag_ratio, bool poly, bool usefp16){
void Detector::initData(int height, int width, const char* engine_path, float ratio){
    // this->input_yuv = in_yuv;


    text_threshold_ = 0.7;
    link_threshold_ = 0.4;
    low_text_ = 0.4;
    poly_ = false;
    mag_ratio_ = ratio;
    usefp16_ = true;
    dilate_radio = 3;

    this->height = height;
    this->width = width;
    this->p_count = height * width;
    if(mag_ratio_<0.1){
        scale_height = 540;
        scale_width = 960;
    }
    else{
        scale_height = height * mag_ratio_;
        scale_width = width * mag_ratio_;
    }
    new_width = scale_width;
    new_height = scale_height;
    // get padding size
    if (scale_width != ((scale_width>>5)<<5))
		this->new_width = (((scale_width>>5)+1)<<5);
	if (scale_height != ((scale_height>>5)<<5))
		this->new_height = (((scale_height>>5)+1)<<5);


    string engine_str(engine_path);
    engine_path_ = engine_str;
    // engine_path_ = modelDirStr + to_string(new_width) + "x" + to_string(new_height) + ".cache";

    // cudaMalloc(&this->src_yuv_i_d, p_count*sizeof(uint8_t)*3/2);
    // if input is rgb format
    // cudaMalloc(&this->src_rgb_i_d, p_count*sizeof(uint8_t)*3);
    cudaMalloc(&this->src_yuv_f_d, p_count*sizeof(float)*3);
    cudaMalloc(&this->sdr_rgb, p_count*sizeof(half)*3);
    cudaMalloc(&this->bilinear_out, scale_height*scale_width*sizeof(half)*3);
    cudaMalloc(&this->pad_out, new_height*new_width*sizeof(half)*3);
    // score_map shape: (batch_size, new_height/2, new_width/2, 2)
    cudaMalloc(&score_map, new_height*new_width*sizeof(half)/2);
    cudaMalloc(&fea, new_height*new_width*sizeof(half)*8);
    
    cudaMalloc(&text_score, new_height*new_width*sizeof(float)/4);
    cudaMalloc(&link_score, new_height*new_width*sizeof(float)/4);
    cudaMalloc(&text_score_comb, new_height*new_width*sizeof(uint8_t)/4);
    cudaMalloc(&text_score_comb_dilate, new_height*new_width*sizeof(uint8_t)/4);
    cudaMalloc(&label_map, new_height*new_width*sizeof(unsigned int)/4);
    cudaMalloc(&components, new_height*new_width*sizeof(ConnectedComponent)/4);
    cudaMalloc(&components_real, new_height*new_width*sizeof(ConnectedComponent)/4);
    cudaMalloc(&num_of_box, sizeof(int));

    components_real_h = (ConnectedComponent*)malloc(new_height*new_width*sizeof(ConnectedComponent)/4);
    num_of_box_h = (int*)malloc(sizeof(int));

    for (int c = 0; c < 3; c++) {
        cudaStreamCreate(&(streams[c]));
    }
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
}

Detector::~Detector(){
    cout << "~Detector..........." << endl;
    // cudaFree(src_yuv_i_d);
    // cudaFree(src_rgb_i_d);
    cudaFree(src_yuv_f_d);
    cudaFree(sdr_rgb);
    cudaFree(bilinear_out);
    cudaFree(pad_out);
    cudaFree(score_map);
    cudaFree(fea);
    cudaFree(text_score);
    cudaFree(link_score);
    cudaFree(text_score_comb);
    cudaFree(label_map);
    cudaFree(components);
    cudaFree(components_real);
    cudaFree(num_of_box);
    cudaFree(text_score_comb_dilate);
    for (int c = 0; c < 3; c++) {
        cudaStreamDestroy(streams[c]);
    }
    cudaStreamDestroy(stream1);
    free(num_of_box_h);
    free(components_real_h);

    if(this->context_){
        this->context_->destroy();
    }
    if(this->engine_)
        this->engine_->destroy();
    // if(this->builder_)
    //     this->builder_->destroy();

    // context->destroy();
    // engine->destroy();
    // runtime->destroy();

}

void Detector::onnx2trt(){
    this->builder_ = createInferBuilder(gLogger);
    uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    INetworkDefinition* network = this->builder_->createNetworkV2(flag);
    IParser* parser = createParser(*network, gLogger);
    // printf("onnx_path:%s\n", onnx_path_);
    parser->parseFromFile(this->onnx_path_.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
    
    auto config = this->builder_->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 33);
    // Dims4 inputDims{1, 3, this->new_height, this->new_width};
    // printf("%d", this->new_height);
    // network->getInput(0)->setDimensions(inputDims);

    IOptimizationProfile* profile = this->builder_->createOptimizationProfile();
    profile->setDimensions("input1", OptProfileSelector::kMIN, Dims4(1, 3, 272, 480));
    profile->setDimensions("input1", OptProfileSelector::kOPT, Dims4(1, 3, 1088, 1920));
    profile->setDimensions("input1", OptProfileSelector::kMAX, Dims4(1, 3, 1280, 2048));
    config->addOptimizationProfile(profile);
    config->setFlag(BuilderFlag::kFP16);
    
    // ICudaEngine* engine;
    this->engine_ = this->builder_->buildEngineWithConfig(*network, *config);
    assert(this->engine_);
    cout << "successful to build the engine" <<endl;

#if USE_ENGINE_CACHE   
    nvinfer1::IHostMemory* ModelStream = this->engine_->serialize(); 

    std::ofstream outfile(this->engine_path_.c_str(), std::ios::out | std::ios::binary);
    if (!outfile.is_open())
    {
        throw std::runtime_error{"Failed to open engineCache!"};
    }
    uint8_t* data = (uint8_t*)ModelStream->data();
    outfile.write((char*)data, ModelStream->size());
    outfile.close();
#endif
    cout << "save trt cache done." << endl;
    // this->engine_->destroy();
    parser->destroy(); 
    network->destroy();
    // builder->destroy();

    
    builder_->destroy();
    config->destroy();
}

void Detector::load_engine_and_create_context(){
    fstream fs;
    fs.open(this->engine_path_, std::ios::binary | std::ios::in);
    if (!fs) {
        throw std::runtime_error{"Failed to open engineCache!"};
    }
    cout << "trt file exist, build engine..."  << endl;
    fs.seekg(0, std::ios::end);
    int length = fs.tellg();
    fs.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data(new char[length]);
    fs.read(data.get(), length);
    fs.close();
    nvinfer1::IRuntime* trtRuntime = createInferRuntime(gLogger.getTRTLogger());
    this->engine_ = trtRuntime->deserializeCudaEngine(data.get(), length, nullptr);
    
    assert(this->engine_ != nullptr);
    std::cout << "deserialize cuda engine done." << endl;
    this->context_ = this->engine_->createExecutionContext();
    this->context_->setBindingDimensions(0, Dims4(1, 3, this->new_height, this->new_width));
}

void Detector::buildEngine()
{
#if USE_ENGINE_CACHE
     // get TRT context
    fstream fs;
    fs.open(this->engine_path_, std::ios::binary | std::ios::in);
    if (!fs) {
        cout << "convert onnx model to tensorRT model." << endl;
#if TEST_TIME
        double start;
        double end;
        start = getCurrentMSTime();
#endif
        onnx2trt();   
        // fs.open(this->engineCache, std::ios::binary | std::ios::in);
#if TEST_TIME
        end = getCurrentMSTime();
        double tcS = (double)(end - start) / 1000;
        printf("onnx to tensorRT model cost time: %f\n", tcS);
#endif
    }
    else{
        cout << "trt file exist, build engine..."  << endl;
        fs.seekg(0, std::ios::end);
        int length = fs.tellg();
        //std::cout << "length::" << length << std::endl;
        fs.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        fs.read(data.get(), length);
        fs.close();
        nvinfer1::IRuntime* trtRuntime = createInferRuntime(gLogger.getTRTLogger());
        this->engine_ = trtRuntime->deserializeCudaEngine(data.get(), length, nullptr);
        std::cout << "deserialize cuda engine done." << endl;
        assert(this->engine_ != nullptr);
        cout << "the tensorRT engine is not nullptr" << endl;

    }
    this->context_ = this->engine_->createExecutionContext();
    this->context_->setBindingDimensions(0, Dims4(1, 3, this->new_height, this->new_width));
    //this->context = engine->createExecutionContext();
#else
#if DEBUG
    cout << "call onnxToTRT" << endl;
#endif
    onnxToTRT();
#endif
}



vector<int> Detector::inference(uint8_t **src_yuv_i_d, int* line_size){
    
// #if DEBUG1
//     uint8* src_y;
//     cudaMalloc(&pad_out_f, new_height*new_width*sizeof(float)*3);
//     float *pad_out_f_h = (float*)malloc(new_height*new_width*sizeof(float)*3);

//     half2float(pad_out, pad_out_f, new_height, new_width);
//     cudaDeviceSynchronize();
//     cudaMemcpy(pad_out_f_h, pad_out_f, new_height*new_width*sizeof(float)*3, cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();
//     dumpbin("pad_2.rgb", pad_out_f_h, new_height*new_width*sizeof(float)*3);
// #endif
    // 将sdr yuv数据归一化成[0,1]的float, 同时420p转换成444p
#if DEBUG
    uint8_t* t =  (uint8_t*)malloc(height*width*sizeof(uint8_t));
    cudaMemcpy(t, src_yuv_i_d[0], height*width*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    dumpbin("y.yuv", t, height*width*sizeof(uint8_t));

#endif
    nv12_yConvertTo(src_yuv_f_d, src_yuv_i_d[0], 1/219.0f, 16.0f, width, height, line_size[0]);
    cudaDeviceSynchronize();
    nv12_uvConvertTo(src_yuv_f_d+p_count, src_yuv_f_d+2*p_count, src_yuv_i_d[1], 1/224.0f, 16.0f, width, height, width/2, height/2, line_size[1]);
    cudaDeviceSynchronize();

#if DEBUG1
    float* t =  (float*)malloc(height*width*sizeof(float)*3);
    cudaMemcpy(t, src_yuv_f_d, height*width*sizeof(float)*3, cudaMemcpyDeviceToHost);
    dumpbin("yuv444p.yuv", t, height*width*sizeof(float)*3);

#endif

    // yuv to rgb
    yuv444p2rgb_709(src_yuv_f_d, sdr_rgb, height, width);
    cudaDeviceSynchronize();

#if DEBUG
    float* pad_out_f;
    cudaMalloc(&pad_out_f, height*width*sizeof(float)*3);
    float *pad_out_f_h = (float*)malloc(height*width*sizeof(float)*3);

    half2float(pad_out, pad_out_f, height, width);
    cudaDeviceSynchronize();
    cudaMemcpy(pad_out_f_h, pad_out_f, height*width*sizeof(float)*3, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    dumpbin("rgb.rgb", pad_out_f_h, height*width*sizeof(float)*3);
#endif
    

    // resize
    doscale(streams[2], 1, sdr_rgb, bilinear_out, height, width, scale_height, scale_width);
    cudaDeviceSynchronize();
    // padding
    pad(bilinear_out, pad_out, scale_height, scale_width, new_height, new_width);
    cudaDeviceSynchronize();
#if DEBUG
    float* pad_out_f;
    cudaMalloc(&pad_out_f, new_height*new_width*sizeof(float)*3);
    float *pad_out_f_h = (float*)malloc(new_height*new_width*sizeof(float)*3);

    half2float(pad_out, pad_out_f, new_height, new_width);
    cudaDeviceSynchronize();
    cudaMemcpy(pad_out_f_h, pad_out_f, new_height*new_width*sizeof(float)*3, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    dumpbin("pad_2.rgb", pad_out_f_h, new_height*new_width*sizeof(float)*3);
#endif
    // inference
    void* buffers[3];
    buffers[0] = pad_out;
    buffers[1] = fea;
    buffers[2] = score_map;
    bool resFlag = this->context_->enqueueV2(buffers, stream1, nullptr);
    cudaDeviceSynchronize();
    if (!resFlag)
    {
        throw std::runtime_error{"Some errors occured when doing inference!"};
    }
#if DEBUG2
    float *score_map_f;
    cudaMalloc(&score_map_f, new_height*new_width*sizeof(float)/2);
    float *score_map_f_h = (float*)malloc(new_height*new_width*sizeof(float)/2);
    half2float(score_map, score_map_f, new_height/2, new_width/2);
    cudaDeviceSynchronize();
    cudaMemcpy(score_map_f_h, score_map_f, new_height*new_width*sizeof(float)/2, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("save\n");
    dumpbin("./score_2_2080.bin", score_map_f_h, new_height*new_width*sizeof(float)/2);

#endif
    // 二值化score map
    threshold_map(score_map, text_score, link_score, text_score_comb, text_threshold_, link_threshold_, low_text_, new_height/2, new_width/2);
    cudaDeviceSynchronize();
#if DEBUG
    uint8_t *text_score_comb_h = (uint8_t*)malloc(new_height*new_width*sizeof(uint8_t)/4);
    cudaMemcpy(text_score_comb_h, text_score_comb, new_height*new_width*sizeof(uint8_t)/4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    dumpbin("/data/QCVLib/QTD/cpp/score_int.bin", text_score_comb_h, new_height*new_width*sizeof(uint8_t)/4);
#endif
    // 对二值化的score map做膨胀操作
    dilate(text_score_comb, text_score_comb_dilate, dilate_radio, new_height/2, new_width/2);
    cudaDeviceSynchronize();
    // 标记连通域
    connectedComponentLabeling(label_map, text_score_comb, new_width/2, new_height/2);
    cudaDeviceSynchronize();
    // 获取每个联通域的坐标，保存在components_real中，同时统计连通域数量num_of_box，最后将num_of_box个box拷贝到cpu上, components_real_h是cpu上的指针
    get_box_from_label(text_score_comb, label_map, components, components_real, components_real_h, num_of_box, num_of_box_h, new_width/2, new_height/2);
    cudaDeviceSynchronize();
#if DEBUG
    unsigned int *label_h = (unsigned int*)malloc(new_height*new_width*sizeof(unsigned int)/4);
    cudaMemcpy(label_h, label_map, new_height*new_width*sizeof(unsigned int)/4, cudaMemcpyDeviceToHost);
    dumpbin("/data/QCVLib/QTD/cpp/label_h.bin", label_h, new_height*new_width*sizeof(unsigned int)/4);

#endif
    // cout<<"nums of boxes: "<<*num_of_box_h<<endl;
    // cout<<components_real_h[0].area<<endl;
    // cout<<components_real_h[0].startX<<endl;
    // cout<<components_real_h[0].endX<<endl;
    // cout<<components_real_h[0].startY<<endl;
    // cout<<components_real_h[0].endY<<endl;
    vector<Box> boxes;
    // 将components_real_h中的box转换成vector<Box>
    for(int i=0;i<*num_of_box_h;i++){
        Box box;
        box.x_min = components_real_h[i].startX;
        box.x_max = components_real_h[i].endX;
        box.y_min = components_real_h[i].startY;
        box.y_max = components_real_h[i].endY;
        box.y_center = (box.y_min + box.y_max) / 2;
        box.height = box.y_max - box.y_min;
        box.width = box.x_max - box.x_min;
        boxes.push_back(box);
    }

    // for (const Box& box : boxes) {
    //     std::cout << "x_min: " << box.x_min << ", x_max: " << box.x_max << ", y_min: " << box.y_min << ", y_max: " << box.y_max << "\n";
    // }
    // 合并中心点在同一水平线上的bo x
    vector<Box> merge_list = groupTextBox(boxes);

    // for (const Box& box : merge_list) {
    //     std::cout << "x_min: " << (int)box.x_min << ", x_max: " << (int)box.x_max << ", y_min: " << (int)box.y_min << ", y_max: " << (int)box.y_max << "\n";
    // }
    vector<int> res;
    for (const Box& box : merge_list){
        res.push_back((int)(box.x_min*2/mag_ratio_));
        res.push_back((int)(box.x_max*2/mag_ratio_));
        res.push_back((int)(box.y_min*2/mag_ratio_));
        res.push_back((int)(box.y_max*2/mag_ratio_));
    }
    return res;

}
vector<int> Detector::inference_rgb_in(uint8_t *src_rgb_i_d){
    // this->input_yuv = in_yuv;

    // cudaMemcpy(src_rgb_i_d, in_rgb, height*width*sizeof(uint8_t)*3, cudaMemcpyHostToDevice);

    rgb_norm(src_rgb_i_d, sdr_rgb, height, width);
    // cout<< new_height << new_width <<endl;
    // resize
    doscale(streams[2], 1, sdr_rgb, bilinear_out, height, width, scale_height, scale_width);
    cudaDeviceSynchronize();
    // padding
    pad(bilinear_out, pad_out, scale_height, scale_width, new_height, new_width);
    cudaDeviceSynchronize();
#if DEBUG1
    float* pad_out_f;
    cudaMalloc(&pad_out_f, new_height*new_width*sizeof(float)*3);
    float *pad_out_f_h = (float*)malloc(new_height*new_width*sizeof(float)*3);

    half2float(pad_out, pad_out_f, new_height, new_width);
    cudaDeviceSynchronize();
    cudaMemcpy(pad_out_f_h, pad_out_f, new_height*new_width*sizeof(float)*3, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    dumpbin("pad_2.rgb", pad_out_f_h, new_height*new_width*sizeof(float)*3);
#endif
    // inference
    void* buffers[3];
    buffers[0] = pad_out;
    buffers[1] = fea;
    buffers[2] = score_map;
    bool resFlag = this->context_->enqueueV2(buffers, stream1, nullptr);
    cudaDeviceSynchronize();
    if (!resFlag)
    {
        throw std::runtime_error{"Some errors occured when doing inference!"};
    }
#if DEBUG
    float *score_map_f;
    cudaMalloc(&score_map_f, new_height*new_width*sizeof(float)/2);
    float *score_map_f_h = (float*)malloc(new_height*new_width*sizeof(float)/2);
    half2float(score_map, score_map_f, new_height/2, new_width);
    cudaMemcpy(score_map_f_h, score_map_f, new_height*new_width*sizeof(float)/2, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    dumpbin("/data/QCVLib/QTD/cpp/score_2_2080.bin", score_map_f_h, new_height*new_width*sizeof(float)/2);

#endif
    // 二值化score map
    threshold_map(score_map, text_score, link_score, text_score_comb, text_threshold_, link_threshold_, low_text_, new_height/2, new_width/2);
    cudaDeviceSynchronize();
#if DEBUG3
    uint8_t *text_score_comb_h = (uint8_t*)malloc(new_height*new_width*sizeof(uint8_t)/4);


    cudaMemcpy(text_score_comb_h, text_score_comb, new_height*new_width*sizeof(uint8_t)/4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    dumpbin("score_int.bin", text_score_comb_h, new_height*new_width*sizeof(uint8_t)/4);
#endif
    // 标记连通域，获取每个联通域的坐标，保存在components_real中，同时统计连通域数量num_of_box，最后将num_of_box个box拷贝到cpu上
    // label(text_score_comb, text_score_comb_dilate, label_map, components, components_real, components_real_h, num_of_box, num_of_box_h, new_width/2, new_height/2, new_width*new_height/4, streams[0], streams[1], dilate_radio);
    dilate(text_score_comb, text_score_comb_dilate, dilate_radio, new_height/2, new_width/2);
    cudaDeviceSynchronize();
    connectedComponentLabeling(label_map, text_score_comb, new_width/2, new_height/2);
    cudaDeviceSynchronize();
    get_box_from_label(text_score_comb, label_map, components, components_real, components_real_h, num_of_box, num_of_box_h, new_width/2, new_height/2);
    cudaDeviceSynchronize();

#if DEBUG4
    unsigned int *label_h = (unsigned int*)malloc(new_height*new_width*sizeof(unsigned int)/4);
    cudaMemcpy(label_h, label_map, new_height*new_width*sizeof(unsigned int)/4, cudaMemcpyDeviceToHost);
    dumpbin("label_h.bin", label_h, new_height*new_width*sizeof(unsigned int)/4);

#endif
    
    vector<Box> boxes;
    for(int i=0;i<*num_of_box_h;i++){
        Box box;
        box.x_min = components_real_h[i].startX;
        box.x_max = components_real_h[i].endX;
        box.y_min = components_real_h[i].startY;
        box.y_max = components_real_h[i].endY;
        box.y_center = (box.y_min + box.y_max) / 2;
        box.height = box.y_max - box.y_min;
        box.width = box.x_max - box.x_min;
        boxes.push_back(box);
    }

    // for (const Box& box : boxes) {
    //     std::cout << "x_min: " << box.x_min << ", x_max: " << box.x_max << ", y_min: " << box.y_min << ", y_max: " << box.y_max << "\n";
    // }
    // 合并中心点在
    vector<Box> merge_list = groupTextBox(boxes);

    // for (const Box& box : merge_list) {
    //     std::cout << "x_min: " << (int)box.x_min << ", x_max: " << (int)box.x_max << ", y_min: " << (int)box.y_min << ", y_max: " << (int)box.y_max << "\n";
    // }
    vector<int> res;
    for (const Box& box : merge_list){
        res.push_back((int)(box.x_min*2/mag_ratio_));
        res.push_back((int)(box.x_max*2/mag_ratio_));
        res.push_back((int)(box.y_min*2/mag_ratio_));
        res.push_back((int)(box.y_max*2/mag_ratio_));
    }
    return res;

}
// extern "C"
// {

Detector* getObj()
{
	// return new Detector(engine_path, text_threshold, link_threshold, low_text, poly, usefp16, mag_ratio, onnx_path);
	return new Detector();
	// return reinterpret_cast<void*>(obj);
    // return detector;
    // printf("ff\n");
}
// }

extern "C"
{
Detector *detector;
void infer_init(int height, int width, const char* engine_path, float ratio){
    detector = getObj();
    detector->initData(height, width, engine_path, ratio);
    // detector->buildEngine();
    detector->load_engine_and_create_context();
}
}
extern "C"{
vector<int> infer_pipe(uint8_t **in_yuv, int* line_size){
        return detector->inference(in_yuv, line_size);
}
vector<int> infer_pipe_rgb(uint8_t *rgb){
    return detector->inference_rgb_in(rgb);
}
}
extern "C"{

// void python_api(uint8_t *in_yuv, int* size, int** data, int format) {
//     vector<int> vec = infer_pipe(in_yuv, format);
    
//     *size = vec.size();
//     *data = new int[*size];
    
//     for (int i = 0; i < *size; ++i) {
//         (*data)[i] = vec[i];
//     }
// }

}

extern "C"{

void destroyObj()
{
    // free(s2hdr10Obj->getSrc());
    delete detector;
}



}

