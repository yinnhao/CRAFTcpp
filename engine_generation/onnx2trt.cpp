/*
@File    :   onnx2trt.cpp
@Time    :   2023/08/31 15:40:51
@Author  :   zhuyinghao
@Desc    :   读取onnx文件，转成dynamic shape的engine文件，保存下来
*/
#include <iostream>
#include <NvInfer.h>
#include "NvOnnxParser.h"
#include "logging.h"
#include <string>
#include <fstream>
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;

static Logger gLogger{Logger::Severity::kINFO};
void onnx2trt(string onnx_path, string engine_path){

    nvinfer1::ICudaEngine* engine;
    nvinfer1::IBuilder* builder;

    // 创建builder
    builder = createInferBuilder(gLogger);
    uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    // 创建network
    INetworkDefinition* network = builder->createNetworkV2(flag);
    IParser* parser = createParser(*network, gLogger);
    // 从onnx导入网络
    parser->parseFromFile(onnx_path.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
    // 配置builder
    auto config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 33);
    // 设置动态尺寸的范围
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("input1", OptProfileSelector::kMIN, Dims4(1, 3, 186, 240));
    profile->setDimensions("input1", OptProfileSelector::kOPT, Dims4(1, 3, 1088, 1920));
    profile->setDimensions("input1", OptProfileSelector::kMAX, Dims4(1, 3, 1280, 2048));
    config->addOptimizationProfile(profile);
    config->setFlag(BuilderFlag::kFP16);
    
    // 通过builder创建engine
    engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);
    cout << "successful to build the engine" <<endl;

    // 序列化engine
    nvinfer1::IHostMemory* ModelStream = engine->serialize(); 
    // 保存到文件中
    std::ofstream outfile(engine_path.c_str(), std::ios::out | std::ios::binary);
    if (!outfile.is_open())
    {
        throw std::runtime_error{"Failed to open engineCache!"};
    }
    uint8_t* data = (uint8_t*)ModelStream->data();
    outfile.write((char*)data, ModelStream->size());
    outfile.close();

    cout << "save trt cache done." << endl;
    parser->destroy(); 
    network->destroy();
    config->destroy();
    builder->destroy();
    engine->destroy();

}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input onnx path> <output engine path>" << std::endl;
        return 1; // Return an error code
    }

    std::string string1 = argv[1];
    std::string string2 = argv[2];

    std::cout << "input onnx path: " << string1 << std::endl;
    std::cout << "output engine path: " << string2 << std::endl;
    onnx2trt(string1, string2);
    return 0; // Return success
}
