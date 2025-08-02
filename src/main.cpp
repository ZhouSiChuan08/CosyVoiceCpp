#include <iostream>
#include <chrono>

#include <torch/script.h>  // 导入 torch 库 必须先导入 torch/script.h
#include <torch/torch.h>
#include <tokenizers_cpp.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "debugTools.h"
#include "tokenizer.h"
#include "cosyvoice.h"
#include "tensorRT.h"

using tokenizers::Tokenizer;
Logger gLogger; // 全局日志器实例

// 全局变量
int global_argc;
char **global_argv;
int main(int argc, char *argv[])
{
    global_argc = argc;  // 输入参数个数
    global_argv = argv;  // 输入参数数组

    try
    {
        //输入数据
        std::unique_ptr<CosyVoice> voice;

        std::string cmdFilePath = "../command.json";
        std::string audioDirPath = "../audio";
        bool isUseTensorRT = LoadValueFromJsonFile(cmdFilePath, "useTRT").get<bool>();
        std::cout << "useTensorRT: " << isUseTensorRT << std::endl;
        bool isUseCuda = LoadValueFromJsonFile(cmdFilePath, "useCUDA").get<bool>();
        torch::DeviceType deviceType = isUseCuda ? torch::kCUDA : torch::kCPU;
        voice = std::make_unique<CosyVoice>("../model", deviceType, isUseTensorRT);
        
        while (true)
        {
            stdCoutInColor(1, StringColor::GREEN, "准备好指令后, 按回车继续... ");
            std::getline(std::cin, std::string());

            // 如果是否使用CUDA 和 是否使用TensorRT 发生变更, 重新创建模型
            bool isUseTensorRTNew = LoadValueFromJsonFile(cmdFilePath, "useTRT").get<bool>();
            bool isUseCudaNew = LoadValueFromJsonFile(cmdFilePath, "useCUDA").get<bool>();
            if (isUseTensorRTNew!= isUseTensorRT || isUseCudaNew!= isUseCuda)
            {
                isUseTensorRT = isUseTensorRTNew;
                isUseCuda = isUseCudaNew;
                deviceType = isUseCudaNew ? torch::kCUDA : torch::kCPU;
                voice.reset(new CosyVoice("../model", deviceType, isUseTensorRTNew));
            }

            // 设定种子值
            uint64_t userSeed = LoadValueFromJsonFile(cmdFilePath, "seed").get<uint64_t>();
            uint64_t seed = 0;
            if (userSeed != 0)
            {
                seed = userSeed;
            }
            else
            {
                seed = getRandSeed();
                stdCoutInColor(1, StringColor::BLUE, "随机种子: " + std::to_string(seed) + "\n");
            }
            torch::manual_seed(seed);
            
            std::string spkId = LoadValueFromJsonFile(cmdFilePath, "spkId").get<std::string>();
            std::string ttsText = LoadValueFromJsonFile(cmdFilePath, "ttsText").get<std::string>();

            std::string instructText = LoadValueFromJsonFile(cmdFilePath, "instructText").get<std::string>();
            
            CosyVoiceInput input = {spkId, ttsText, instructText, audioDirPath};

            TicToc time;
            voice->createModelInput(input);
            torch::Tensor audio;
            try
            {
                audio = voice->inference();
            }
            catch(const std::exception& e)
            {
                stdCerrInColor(1, "In main, 推理失败: %s\n", e.what());
            }
            std::cout << "audio is on " << (audio.is_cuda() ? "CUDA" : "CPU")  << std::endl;

            
            std::string audioName = getLocalTimestampForFilename() + ".wav";
            std::filesystem::path audioPath(input.audioDirPath);
            audioPath = audioPath / audioName;
            saveAudioWav(audioPath.string(), audio, voice->sampleRate);
            stdCoutInColor(1, StringColor::GREEN, "In main, 👌👌👌音频保存成功: %s\n", audioPath.string().c_str());

            double duration =  time.toc() / 1000.0;
            stdCoutInColor(1, StringColor::BLUE, "推理结束, 耗时: " + std::to_string(duration) + " s\n");
        
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}