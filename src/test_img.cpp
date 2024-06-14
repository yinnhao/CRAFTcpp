#include "SubtitleDetector.h"
#include <opencv2/opencv.hpp>
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./main <engine_path> <input_path>" << std::endl;
        return 1;
    }

    std::string engine_path = argv[1];
    std::string img_path = argv[2];
    
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    
    if(image.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    int height = image.rows;
    int width = image.cols;

    int p_count = height * width * 3;
    
    // 分辨率缩放 ratio = 0.5
    infer_init(height, width, engine_path.c_str(), 0.5);
    
    
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // 检查是否为连续的内存
    if (!image.isContinuous()) {
        std::cout << "cv::Mat is not continuous" << std::endl;
        image = image.clone();  // 创建一个连续的副本
    }
    // cv::Mat的rgb存储顺序与字幕检测中rgb的存储顺序不一样，Mat是每个像素的r, g, b依次保存，字幕检测代码中是先保存所有像素的r，再保存所有像素的g，再保存所有像素的b
    uint8_t *rgb;
    rgb = (uint8_t*) malloc(p_count*sizeof(uint8_t));

    for (int c = 0; c < 3; ++c)
    {
       for (int w2 = 0; w2 < width; ++w2)
       {
           for (int h2 = 0; h2 < height; ++h2)
           {
                int dst_index = c * width * height + h2 * width + w2;
                int src_index = h2 * width * 3 + w2 * 3 + c;
                rgb[dst_index] = image.data[src_index];
           }
       }
    }



    // cpu to gpu mem
    uint8_t *d_image;
    cudaMalloc(&d_image, p_count*sizeof(uint8_t));
    cudaMemcpy(d_image, rgb, p_count*sizeof(uint8_t), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;

    printf("start inference\n");
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n = 1; // Number of iterations
    float total_elapsed_time = 0;
    vector<int> res;

    for (int i = 0; i < n; i++) {
        cudaEventRecord(start);
        cudaEventQuery(start);

        res = infer_pipe_rgb(d_image);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_elapsed_time += elapsed_time;
        printf("Iteration %d: Time = %g ms.\n", i+1, elapsed_time);
    }

    float average_elapsed_time = total_elapsed_time / n;
    printf("Average Time = %g ms.\n", average_elapsed_time);
    cv::Rect rect;
    for (int i = 0; i < res.size(); i += 4) {
        int x_min = res[i];
        int x_max = res[i + 1];
        int y_min = res[i + 2];
        int y_max = res[i + 3];
        // 画矩形框
        rect.x = x_min;
        rect.y = y_min;
        rect.height = y_max - y_min + 1;
        rect.width = x_max - x_min + 1;
        cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 2);
        
        std::cout << "text " << (i / 4 + 1) << ": " << std::endl;
        std::cout << "x_min = " << x_min << std::endl;
        std::cout << "x_max = " << x_max << std::endl;
        std::cout << "y_min = " << y_min << std::endl;
        std::cout << "y_max = " << y_max << std::endl;
    }
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite("./res/out.jpg", image);
    destroyObj();
}
