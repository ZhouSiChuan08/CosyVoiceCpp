#include "tensorRT.h"

std::tuple<void*, std::vector<int64_t>> getTorchTensorPtrShape(torch::Tensor& tensor)
{
    void* dataPtr = nullptr;
    std::vector<int64_t> shape = tensor.sizes().vec();
    if (!tensor.is_cuda())
    {
        tensor = tensor.to(torch::kCUDA);  //将tensor转移到CUDA设备
    }
    tensor = tensor.contiguous();      //将tensor变成连续的内存布局
    auto dataType = tensor.dtype().toScalarType();
    switch (dataType)
    {
        case torch::kInt32:
            dataPtr = tensor.data_ptr<int32_t>();
            break;
        case torch::kInt64:
            // {
            //  tensor = tensor.to(torch::kInt32);
                dataPtr = tensor.data_ptr<int64_t>();
            // }
            break;
        case torch::kFloat16:
            dataPtr = tensor.data_ptr<c10::Half>();
            break;
        case torch::kFloat32:
            dataPtr = tensor.data_ptr<float>();
            break;
        default:
            throw std::runtime_error("In getTorchTensorPtrShape, 不支持的数据类型");
            break;
    }
    return std::make_tuple(dataPtr, shape);
}

torch::Tensor dataPtrToTensor(void* dataPtr, const nvinfer1::Dims& dims, const nvinfer1::DataType& dataType, torch::DeviceType device)
{
    torch::Tensor tensor;
    if (dims.nbDims > 0 && dims.d == nullptr) 
    {
        throw std::runtime_error("Invalid dims: non-zero nbDims but null dims.d");
    }
    std::vector<int64_t> shapeVec(dims.d, dims.d + dims.nbDims);
    switch (dataType)
    {
        case nvinfer1::DataType::kFLOAT:
            tensor = torch::from_blob(static_cast<float*>(dataPtr), shapeVec, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)).to(device);
            break;
        case nvinfer1::DataType::kINT32:
            tensor = torch::from_blob(static_cast<int32_t*>(dataPtr), shapeVec, torch::TensorOptions().dtype(torch::kInt32).requires_grad(false)).to(device);
            break;
        case nvinfer1::DataType::kINT64:
            tensor = torch::from_blob(static_cast<int64_t*>(dataPtr), shapeVec, torch::TensorOptions().dtype(torch::kInt64).requires_grad(false)).to(device);
            break;
        default:
            throw std::runtime_error("In dataPtrToTensor, 不支持的数据类型");
            break;
    }
    return tensor;
}

TensorRTEngine::TensorRTEngine(const std::string& enginePath_, std::vector<std::string> inputNames_, std::vector<std::string> outputNames_) :
    enginePath(enginePath_),
    inputNames(inputNames_),
    outputNames(outputNames_),
    runtime(nullptr),
    engine(nullptr),
    context(nullptr)
{
    // 加载模型
    runtime = nvinfer1::createInferRuntime(gLogger);
    std::ifstream fin(enginePath, std::ios::binary);
    std::string modelData = "";
    // 简而言之就是超快的文件读取
    while (fin.peek() != EOF) { // 使用fin.peek()防止文件读取时无限循环

        std::stringstream buffer;
        buffer << fin.rdbuf();
        modelData.append(buffer.str());
    }
    fin.close();
    engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
    context = engine->createExecutionContext();
    int64_t memorySize =  engine->getDeviceMemorySizeV2();
    stdCoutInColor(1, StringColor::BLUE,"In TensorRTEngine, 加载模型 %s 成功, 占用显存大小为 %.3lf MB %.3lf GB\n", enginePath.c_str() ,memorySize / (1024 * 1024.0), memorySize / (1024 * 1024 * 1024.0));
}

TensorRTEngine::~TensorRTEngine()
{
    CUDA_CHECK(cudaStreamDestroy(stream));
}

std::vector<nvinfer1::Dims> TensorRTEngine::getOutputDims(std::vector<nvinfer1::Dims> inputDims)
{
    std::vector<nvinfer1::Dims> outputDims_;
    // 先设置输入尺寸
    for (size_t i = 0; i < inputNames.size(); i++)
    {
        context->setInputShape(inputNames[i].c_str(), inputDims[i]);
    }
    // 再获取输出尺寸
    for (size_t i = 0; i < outputNames.size(); i++)
    {
        nvinfer1::Dims dims = context->getTensorShape(outputNames[i].c_str());
        outputDims_.push_back(dims);
    }
    return outputDims_;
}

std::vector<void*> TensorRTEngine::allocateOutputBuffers(const std::vector<nvinfer1::Dims>& outputDims, const std::vector<nvinfer1::DataType>& types)
{
    std::vector<void*> outputBuffers;
    if (outputDims.size() != outputNames.size() || outputDims.size() != types.size())
    {
        stdCerrInColor(1, "In allocateOutputBuffers, 输出尺寸数量、数据类型数量与名称数量不匹配");
    }
    else
    {
        // 遍历每个输出维度, 分配内存
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        //stdCoutInColor(1, StringColor::BLUE,"In allocateOutputBuffers, 分配输出内存, 剩余显存大小为 %.3lf MB %.3lf GB / %.3lf GB\n", free / (1024 * 1024.0), free / (1024 * 1024 * 1024.0), total / (1024 * 1024 * 1024.0));
        for (size_t i = 0; i < outputDims.size(); i++)
        {
            auto dims = outputDims[i];
            int dimsCount = dims.nbDims;
            int tmpOutputSize = 1;
            for (size_t j = 0; j < dimsCount; j++)
            {
                if (dims.d[j] == -1)
                {
                    stdCerrInColor(1, "In allocateOutputBuffers, 未确定输出 %s 的 第 %d 动态维度值(0索引起点), 无法分配内存", outputNames[j].c_str(), j);
                    break;
                }
                tmpOutputSize *= dims.d[j];
            }
            switch (types[i])
            {
                case nvinfer1::DataType::kFLOAT:
                    {
                        void* outputBuffer;
                        CUDA_CHECK(cudaMalloc(&outputBuffer, tmpOutputSize * sizeof(float)));
                        outputBuffers.push_back(outputBuffer);
                    }
                    break;
                case nvinfer1::DataType::kINT32:
                    {
                        void* outputBuffer;
                        CUDA_CHECK(cudaMalloc(&outputBuffer, tmpOutputSize * sizeof(int32_t)));
                        outputBuffers.push_back(outputBuffer);
                    }
                    break;
                case nvinfer1::DataType::kINT64:
                    {
                        void* outputBuffer;
                        CUDA_CHECK(cudaMalloc(&outputBuffer, tmpOutputSize * sizeof(int64_t)));
                        outputBuffers.push_back(outputBuffer);
                    }
                    break;
                default:
                    stdCerrInColor(1, "In allocateOutputBuffers, 不支持的数据类型 %d", types[i]);
                    throw std::runtime_error("In allocateOutputBuffers, 不支持的数据类型");
                    break;
            }
        }
    }
    return outputBuffers;
}

void TensorRTEngine::freeOutputBuffers(std::vector<void*>& outputBuffers)
{
    for (auto& outputBuffer : outputBuffers)
    {
        CUDA_CHECK(cudaFree(outputBuffer));
    }
}

void TensorRTEngine::setInputOutputAddress(const std::vector<void*>& inputBuffers, const std::vector<void*>& outputBuffers)
{
    // 设置输入地址
    for (size_t i = 0; i < inputNames.size(); i++)
    {
        context->setTensorAddress(inputNames[i].c_str(), inputBuffers[i]);
    }
    // 设置输出地址
    for (size_t i = 0; i < outputNames.size(); i++)
    {
        context->setTensorAddress(outputNames[i].c_str(), outputBuffers[i]);
    }
}

std::vector<void*> TensorRTEngine::inference(std::vector<torch::Tensor>& inputTensors, const std::vector<nvinfer1::DataType>& outTypes)
{
    // 先获取输入数据指针和形状
    std::vector<void*> inputBuffers(inputTensors.size());
    std::vector<nvinfer1::Dims> inputDims(inputTensors.size());
    for (size_t i = 0; i < inputTensors.size(); i++)
    {
        auto [dataPtr, shapes] = getTorchTensorPtrShape(inputTensors[i]);
        nvinfer1::Dims dims;
        dims.nbDims = shapes.size();
        for (size_t j = 0; j < shapes.size(); j++)
        {
            dims.d[j] = shapes[j];
        }
        inputBuffers[i] = dataPtr;
        inputDims[i] = dims;
    }
    // 确定输入形状后, 再获取输出形状
    outputDims = getOutputDims(inputDims);
    

    // 分配输出内存 得到输出数据指针
    std::vector<void*> outputBuffers = allocateOutputBuffers(outputDims, outTypes);

    // 分配 cuda stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 开始绑定数据
    setInputOutputAddress(inputBuffers, outputBuffers);

    // 执行推理
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);     // 等待计算完成
    return outputBuffers;
}