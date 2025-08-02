#ifndef ONNX_H
#define ONNX_H

#include <filesystem>
#include <torch/torch.h>
#include <onnxruntime_cxx_api.h>
#include "debugTools.h"
enum class TensorType {
    INT32, INT64, FLOAT32, FLOAT64
};

/**
 * @brief 将Ort::Value转换为torch::Tensor
 * @brief 必须确保 Ort::Value 的生命周期覆盖 tensor 的使用期
 * @param ortValue Ort::Value 格式的输入数据
 * @param device torch::DeviceType 类型，指定推理设备
 * @return torch::Tensor
 */
torch::Tensor ortValueToTorchTensor(Ort::Value& ortValue, torch::DeviceType device = torch::kCPU);



class ONNXModel 
{
public:
    ONNXModel(const std::string& modelPath_, const std::string& envId_ ,torch::DeviceType device_ = torch::kCPU, OrtLoggingLevel logLevel = ORT_LOGGING_LEVEL_WARNING);
    // ORT_LOGGING_LEVEL_VERBOSE  // 最详细（调试用）
    // ORT_LOGGING_LEVEL_INFO     // 一般信息（推荐默认）
    // ORT_LOGGING_LEVEL_WARNING  // 仅警告
    // ORT_LOGGING_LEVEL_ERROR    // 仅错误
    // ORT_LOGGING_LEVEL_FATAL    // 致命错误

    /**
     * @brief 获取输入、输出节点名称
     */
    void getInputOutputNames();

    /**
     * @brief 执行推理
     * @param inputData 输入数据
     * @param isShowInfo 是否显示推理输入输出信息
     * @return std::vector<Ort::Value> 推理结果
     */
    std::vector<Ort::Value> inference(std::vector<Ort::Value>& inputData, bool isShowInfo = false);

    /**
     * @brief 将torch.tensor转换为ONNX格式的输入数据
     * @param type TensorType 输入数据类型
     * @param inputTensor torch.tensor格式的输入数据
     * @param shape 输入数据的shape
     * @return Ort::Value 转换后的ONNX格式的输入数据
     */
    Ort::Value torchTensorToOrtValue(TensorType type, torch::Tensor& inputTensor, const std::vector<int64_t>& shape);

    /**
     * @brief 将torch.tensor转换为ONNX格式的输入数据【自动推断转换类型】
     * @param inputTensor torch.tensor格式的输入数据
     * @param shape 输入数据的shape
     * @return Ort::Value 转换后的ONNX格式的输入数据
     */
    Ort::Value torchTensorToOrtValue(torch::Tensor& inputTensor, const std::vector<int64_t>& shape);


    std::vector<Ort::AllocatedStringPtr> inputNamesPtrs;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtrs;
    std::vector<const char*> outputNames;
    std::vector<const char*> inputNames;
    size_t outputCount;
    size_t inputCount;
private:
    std::string modelPath;                         // 模型路径
    std::string envId;                             // 环境ID
    std::unique_ptr<Ort::Env> env;                 // 环境
    Ort::SessionOptions sessionOptions;            // 会话选项
    std::unique_ptr<Ort::Session> session;         // 会话
    Ort::AllocatorWithDefaultOptions allocator;    // 管理输入输出节点名称指针的内存分配器
    std::unique_ptr<Ort::MemoryInfo>  memoryInfo;  // 管理创建的Ort::Value
    torch::DeviceType device;                      // 推理设备类型

};

#endif // ONNX_H