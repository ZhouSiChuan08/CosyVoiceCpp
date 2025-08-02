#ifndef TENSORRT_H
#define TENSORRT_H

#include <fstream>
#include <tuple>
#include <torch/torch.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "debugTools.h"

#define CUDA_CHECK(status)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // 只输出警告及以上级别的日志
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] ";
            switch (severity) {
                case Severity::kINTERNAL_ERROR: std::cout << "INTERNAL_ERROR: "; break;
                case Severity::kERROR: std::cout << "ERROR: "; break;
                case Severity::kWARNING: std::cout << "WARNING: "; break;
                case Severity::kINFO: std::cout << "INFO: "; break;
                case Severity::kVERBOSE: std::cout << "VERBOSE: "; break;
            }
            std::cout << msg << std::endl;
        }
    }
};

/**
 * @brief 获取libtorch的Tensor的指针, 用于TensorRT的推理
 * @brief 自动判断数据类型, INT64将被转换为INT32
 * @param dataPtr 数据指针
 * @param shape 形状容器
 */
std::tuple<void*, std::vector<int64_t>> getTorchTensorPtrShape(torch::Tensor& tensor);

/**
 * @brief 从数据指针读取数据到Tensor 依赖外部数据，务必clone()拷贝或保证数据指针有效期
 * @param dataPtr 数据指针
 * @param shape 形状容器
 * @return Tensor
 */
torch::Tensor dataPtrToTensor(void* dataPtr, const nvinfer1::Dims& dims, const nvinfer1::DataType& dataType, torch::DeviceType device = torch::kCPU);

/**
 * @brief TensorRT推理引擎
 */
class TensorRTEngine
{
public:
    TensorRTEngine(const std::string& enginePath_, std::vector<std::string> inputNames_, std::vector<std::string> outputNames_);
    ~TensorRTEngine();

    /**
     * @brief 动态输入时, 设置输入尺寸确定输出尺寸, 以便分配输出内存
     * @param inputDims 各输入尺寸
     * @return 各输出尺寸
     */
    std::vector<nvinfer1::Dims> getOutputDims(std::vector<nvinfer1::Dims> inputDims);

    /**
     * @brief 为输出分配内存
     * @param outputDims 各输出尺寸
     * @param types 各输出数据类型
     * @return 各输出内存指针
     */
    std::vector<void*> allocateOutputBuffers(const std::vector<nvinfer1::Dims>& outputDims, const std::vector<nvinfer1::DataType>& types);

    /**
     * @brief 集中销毁输出内存
     * @param outputBuffers 各输出内存指针
     */
    void freeOutputBuffers(std::vector<void*>& outputBuffers);

    /**
     * @brief 设置输入Tensor地址 输出Tensor地址
     * @param inputBuffers 各输入内存指针
     * @param outputBuffers 各输出内存指针
     */
    void setInputOutputAddress(const std::vector<void*>& inputBuffers, const std::vector<void*>& outputBuffers);

    /**
     * @brief 拷贝输出数据到容器
     * @param ouputVec 输出容器
     * @param dataType 输出数据类型
     * @param dims 输出尺寸
     * @param outputPtr 输出内存指针
     */
    template <typename T>
    void copyCudaDataToHost(std::vector<T>& ouputVec, const nvinfer1::DataType& dataType, const nvinfer1::Dims& dims, void* outputPtr)
    {
        int tmpSize = 1;
        for (size_t i = 0; i < dims.nbDims; i++)
        {
            tmpSize *= dims.d[i];
        }
        ouputVec.resize(tmpSize);
        switch (dataType)
        {
        case nvinfer1::DataType::kFLOAT:
            CUDA_CHECK(cudaMemcpyAsync(ouputVec.data(), outputPtr, tmpSize * sizeof(float), cudaMemcpyDeviceToHost, this->stream));
            break;
        case nvinfer1::DataType::kINT32:
            CUDA_CHECK(cudaMemcpyAsync(ouputVec.data(), outputPtr, tmpSize * sizeof(int32_t), cudaMemcpyDeviceToHost, this->stream));
            break;
        case nvinfer1::DataType::kINT64:
            CUDA_CHECK(cudaMemcpyAsync(ouputVec.data(), outputPtr, tmpSize * sizeof(int64_t), cudaMemcpyDeviceToHost, this->stream));
        default:
            stdCerrInColor(1, "In copyCudaDataToHost, 不支持的数据类型: {}", dataType);
            throw std::runtime_error("In copyCudaDataToHost, 不支持的数据类型");
            break;
        }
        CUDA_CHECK(cudaStreamSynchronize(this->stream));
    }
    /**
     * @brief 推理 只负责输入 需手动复制出输出数据到容器, 手动释放内存
     * @param std::vector<torch::Tensor>& inputTensors 输入数据
     * @param std::vector<nvinfer1::DataType>& outTypes 输出数据类型
     * @return 各输出内存指针
     */
    std::vector<void*> inference(std::vector<torch::Tensor>& inputTensors, const std::vector<nvinfer1::DataType>& outTypes);

    std::vector<nvinfer1::Dims> outputDims;  // 各输出尺寸
private:
    Logger gLogger;                         // 全局日志器实例
    cudaStream_t stream;                    // CUDA流
    nvinfer1::IRuntime* runtime;            // 运行时环境
    nvinfer1::ICudaEngine* engine;          // 引擎
    nvinfer1::IExecutionContext* context;   // 上下文

    std::string enginePath;                       // 引擎文件路径
    std::vector<std::string> inputNames;          // 输入名称
    std::vector<std::string> outputNames;         // 输出名称
};

#endif // TENSORRT_H