#include "onnx.h"

torch::Tensor ortValueToTorchTensor(Ort::Value& ortValue, torch::DeviceType device)
{
    // if (torch::cuda::is_available())
    // {
    //     // 默认是CPU, 若可用则改为CUDA
    //     device = torch::kCUDA;
    //     // device = torch::kCPU;
    // }
    
    // 1. 验证输入是张量类型
    if (!ortValue.IsTensor()) {
        throw std::runtime_error("In ortValueToTorchTensor, Ort::Value is not a tensor!");
    }
    // 2. 获取张量信息
    auto tensorInfo = ortValue.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensorInfo.GetShape();
    ONNXTensorElementDataType ortType = tensorInfo.GetElementType();

    // 3. 获取原始数据指针
    void* rawData = ortValue.GetTensorMutableData<void>();

    // 4. 转换为对应的LibTorch张量
    try
    {
        switch (ortType) 
        {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: 
            {  // float32
                float* floatData = static_cast<float*>(rawData);
                return torch::from_blob(floatData, shape, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)).to(device);
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: 
            {  // int32
                int32_t* intData = static_cast<int32_t*>(rawData);
                return torch::from_blob(intData, shape, torch::TensorOptions().dtype(torch::kInt32).requires_grad(false)).to(device);
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: 
            {  // int64
                int64_t* longData = static_cast<int64_t*>(rawData);
                return torch::from_blob(longData, shape, torch::TensorOptions().dtype(torch::kInt64).requires_grad(false)).to(device);
            }
            default:
                throw std::runtime_error("In ortValueToTorchTensor, Unsupported ONNX data type");
        }
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In ortValueToTorchTensor, 转换失败: %s\n", e.what());
        throw;
    }
}

ONNXModel::ONNXModel(const std::string& modelPath_, const std::string& envId_ ,torch::DeviceType device_, OrtLoggingLevel logLevel)
{
    try 
    {
        modelPath = modelPath_;
        envId = envId_;
        device = device_;
        env = std::make_unique<Ort::Env>(logLevel, envId.c_str());
        std::filesystem::path modelPathFile = modelPath;  // 自动转换编码
        if (device == torch::kCUDA)
        {
            sessionOptions.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});  // 启用 CUDA
            memoryInfo = std::make_unique<Ort::MemoryInfo>("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        }
        else
        {
            memoryInfo = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
        }
        
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(*env, modelPathFile.c_str(), sessionOptions);
        getInputOutputNames();
    } catch (const Ort::Exception& e) 
    {
        stdCerrInColor(1, "In ONNXModel::ONNXModel: 读取onnx模型失败: %s\n", e.what());
    }
}

void ONNXModel::getInputOutputNames()
{
    if (session != nullptr)
    {
        
        // 先获取输入节点名称
        inputCount = session->GetInputCount();
        outputCount = session->GetOutputCount();
        for (size_t i = 0; i < inputCount; i++)
        {
            Ort::AllocatedStringPtr inputNamePtr = session->GetInputNameAllocated(i, allocator);  
            inputNamesPtrs.push_back(session->GetInputNameAllocated(i, allocator));   // 存起来, 保持其生命周期
            inputNames.push_back(inputNamesPtrs[i].get());
        }
        for (size_t i = 0; i < outputCount; i++)
        {
            Ort::AllocatedStringPtr outputNamePtr = session->GetOutputNameAllocated(i, allocator);
            outputNamesPtrs.push_back(session->GetOutputNameAllocated(i, allocator));
            outputNames.push_back(outputNamesPtrs[i].get());
        }  
    }
}

Ort::Value ONNXModel::torchTensorToOrtValue(TensorType type, torch::Tensor& inputTensor, const std::vector<int64_t>& shape)
{
    // CreateTensor会拷贝shape, 但不拷贝Tensor, 需保证Tensoor生命周期
    inputTensor = inputTensor.contiguous().to(device);    // 确保内存连续, 数据在CPU上
    // 获取Tensor的内存信息
    void* tensorPtr = inputTensor.data_ptr();             // 空类型指针, 纯粹的地址容器, 没有类型信息, 解引用非法
    size_t tensorBytes = inputTensor.numel() * inputTensor.element_size();
    
    try
    {
        switch (type) {
            case TensorType::FLOAT32:
            {
                return Ort::Value::CreateTensor(*memoryInfo, tensorPtr, tensorBytes, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            }
            case TensorType::INT32:
            {
                return Ort::Value::CreateTensor(*memoryInfo, tensorPtr, tensorBytes, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
            }
            case TensorType::INT64:
            {

                return Ort::Value::CreateTensor(*memoryInfo, tensorPtr, tensorBytes, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
            }
            default:
                throw std::runtime_error("不支持的数据类型");
        }
    
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In torchTensorToOrtValue: 转换Tensor为Ort::Value失败: %s\n", e.what());
    }
}

Ort::Value ONNXModel::torchTensorToOrtValue(torch::Tensor& inputTensor, const std::vector<int64_t>& shape) 
{
    // 确保内存连续且在CPU上
    inputTensor = inputTensor.contiguous().to(device);
    
    // 获取Tensor的内存信息
    void* tensorPtr = inputTensor.data_ptr();
    size_t tensorBytes = inputTensor.numel() * inputTensor.element_size();

    // 自动推断ONNX数据类型
    ONNXTensorElementDataType onnxType;
    switch (inputTensor.scalar_type()) {
        case torch::kFloat32:
            onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            break;
        case torch::kInt32:
            onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            break;
        case torch::kInt64:
            onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            break;
        // 可根据需要添加更多类型支持
        default:
            throw std::runtime_error("Unsupported tensor type: " + 
                                   std::string(torch::toString(inputTensor.scalar_type())));
    }

    try {
        return Ort::Value::CreateTensor(
            *memoryInfo, 
            tensorPtr, 
            tensorBytes, 
            shape.data(), 
            shape.size(), 
            onnxType
        );
    }
    catch(const std::exception& e) {
        stdCerrInColor(1, "In torchTensorToOrtValue: 转换Tensor为Ort::Value失败: %s\n", e.what());
        throw;  // 重新抛出异常
    }
}


std::vector<Ort::Value> ONNXModel::inference(std::vector<Ort::Value>& inputData, bool isShowInfo)
{
    std::vector<Ort::Value> outputs;
    if (isShowInfo)
    {
        for (size_t i = 0; i < inputCount; i++)
        {
            std::vector<int64_t> shape = inputData[i].GetTensorTypeAndShapeInfo().GetShape();
            auto elementType = inputData[i].GetTensorTypeAndShapeInfo().GetElementType();
            stdCoutInColor(1, StringColor::YELLOW, "输入节点名称：%s 类型：%d\n", inputNames[i], elementType);
            std::cout << "形状：" << shape << std::endl;
        }
    }
    try
    {
        if (isShowInfo)
        {
            stdCoutInColor(1, StringColor::YELLOW, "开始推理\n");
        }
        outputs = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputData.data(), inputCount, outputNames.data(), outputCount);
        if (isShowInfo)
        {   
            stdCoutInColor(1, StringColor::YELLOW, "推理结束\n");
        }
        if (isShowInfo)
        {
            for (size_t i = 0; i < outputCount; i++)
            {
                std::vector<int64_t> OutPutshape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
                stdCoutInColor(1, StringColor::BLUE, "输出节点名称：%s 类型：%d\n", outputNames[i], outputs[i].GetTensorTypeAndShapeInfo().GetElementType());
                std::cout << "输出张量形状：" << OutPutshape << std::endl;
            }
        }
    } catch (const Ort::Exception& e) 
    {
        stdCerrInColor(1, "In ONNXModel::inference, ONNX Runtime 错误: %s\n", e.what());
        throw;
    } catch (const std::bad_alloc& e) 
    {
        stdCerrInColor(1, "In ONNXModel::inference, 内存分配失败: %s\n", e.what());
        throw;
    } catch (const std::exception& e) 
    {
        stdCerrInColor(1, "In ONNXModel::inference, 一般异常: %s\n", e.what());
        throw;
    }
    return outputs;
}