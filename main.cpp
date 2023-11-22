#include "SubtitleDetector.h"
int main(){
    
    string engine_path = "./pretrained/craft_mlt_25k_fp16_dynamic_shape.cache";
    
    int height = 2160;
    int width = 3840;
    
    int p_count = height * width;

    infer_init(height, width, engine_path.c_str(), 0.5);
    

    // 开辟内存和显存地址
    uint8_t *yuv;
    yuv = (uint8_t*) malloc(p_count*sizeof(uint8_t)*3/2);
    // 读取sdr yuv
    initbuf("./4k-B_fengqiluoyang_17min_27min_toufa_nv12.yuv", yuv, sizeof(uint8_t)*p_count*3/2);

    // cpu to gpu mem
    uint8_t *yuv_i_d;
    cudaMalloc(&yuv_i_d, p_count*sizeof(uint8_t)*3/2);
    cudaMemcpy(yuv_i_d, yuv, p_count*sizeof(uint8_t)*3/2, cudaMemcpyHostToDevice);
  

    int format = 1; // 0: RGB, 1: NV12

    uint8_t ** in_yuv_pointer = new uint8_t*[2];
    in_yuv_pointer[0] = yuv_i_d;
    in_yuv_pointer[1] = yuv_i_d + p_count;
    int* line_size = new int[2];
    line_size[0] = width;
    line_size[1] = width;

    printf("start inference");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);

    
    vector<int> res = infer_pipe(in_yuv_pointer, format, line_size);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time = %g ms.\n", elapsed_time);

    for (int& point : res) {
        // std::cout << "x_min: " << (int)box.x_min << ", x_max: " << (int)box.x_max << ", y_min: " << (int)box.y_min << ", y_max: " << (int)box.y_max << "\n";
        cout<<point<<endl;
    }

    destroyObj();
}
