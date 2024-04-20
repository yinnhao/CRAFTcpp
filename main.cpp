#include "SubtitleDetector.h"
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: ./main <engine_path> <height> <width> <yuv_file_path>" << std::endl;
        return 1;
    }

    std::string enginePath = argv[1];
    int height = std::stoi(argv[2]);
    int width = std::stoi(argv[3]);
    std::string yuvFilePath = argv[4];
    
    int p_count = height * width;
    // 分辨率缩放 ratio = 0.5
    infer_init(height, width, enginePath.c_str(), 0.5);
    

    // 开辟内存和显存地址
    uint8_t *yuv;
    yuv = (uint8_t*) malloc(p_count*sizeof(uint8_t)*3/2);
    // 读取sdr yuv
    initbuf(const_cast<char*>(yuvFilePath.c_str()), yuv, sizeof(uint8_t)*p_count*3/2);

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
    cudaEvent_t start, stop;

    printf("start inference\n");
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n = 10; // Number of iterations
    float total_elapsed_time = 0;
    vector<int> res;

    for (int i = 0; i < n; i++) {
        cudaEventRecord(start);
        cudaEventQuery(start);

        res = infer_pipe(in_yuv_pointer, format, line_size);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_elapsed_time += elapsed_time;
        printf("Iteration %d: Time = %g ms.\n", i+1, elapsed_time);
    }

    float average_elapsed_time = total_elapsed_time / n;
    printf("Average Time = %g ms.\n", average_elapsed_time);

    for (int i = 0; i < res.size(); i += 4) {
        int x_min = res[i];
        int x_max = res[i + 1];
        int y_min = res[i + 2];
        int y_max = res[i + 3];
        std::cout << "text " << (i / 4 + 1) << ": " << std::endl;
        std::cout << "x_min = " << x_min << std::endl;
        std::cout << "x_max = " << x_max << std::endl;
        std::cout << "y_min = " << y_min << std::endl;
        std::cout << "y_max = " << y_max << std::endl;
    }


    destroyObj();
}
