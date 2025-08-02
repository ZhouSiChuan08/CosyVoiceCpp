#include "cosyVoice.h"

void saveAudioWav(const std::string& filename, const torch::Tensor& audio, int sample_rate) {
    // 检查输入张量
    AT_ASSERT(audio.dim() == 1 || audio.dim() == 2, "Audio must be 1D (mono) or 2D (multi-channel)");
    AT_ASSERT(audio.dtype() == torch::kFloat32, "Audio tensor must be float32");

    // 转换为 CPU 和连续内存
    auto audio_cpu = audio.contiguous().cpu();
    float* audio_data = audio_cpu.data_ptr<float>();

    // 设置 WAV 文件参数
    SF_INFO sfinfo;
    sfinfo.samplerate = sample_rate;
    sfinfo.channels = audio.dim() == 1 ? 1 : audio.size(0);  // 单声道或多声道
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;  // 16-bit PCM WAV

    // 写入文件
    SNDFILE* file = sf_open(filename.c_str(), SFM_WRITE, &sfinfo);
    if (!file) {
        throw std::runtime_error("In saveAudioWav(), Failed to open WAV file for writing");
    }

    // 如果是多声道，需要转置数据（Libsndfile 期望交错存储）
    if (audio.dim() == 2) {
        std::vector<float> interleaved(audio_cpu.numel());
        int64_t num_samples = audio_cpu.size(1);
        int64_t num_channels = audio_cpu.size(0);
        for (int64_t i = 0; i < num_samples; ++i) {
            for (int64_t c = 0; c < num_channels; ++c) {
                interleaved[i * num_channels + c] = audio_cpu[c][i].item<float>();
            }
        }
        sf_write_float(file, interleaved.data(), interleaved.size());
    } else {
        sf_write_float(file, audio_data, audio_cpu.numel());
    }

    sf_close(file);
}

CosyVoiceLlm::CosyVoiceLlm(
    const std::vector<std::string>& modelPaths, 
    const std::vector<std::string>& envIds,
    torch::DeviceType device_,
    OrtLoggingLevel logLevel
    ) : device(device_)
{
    // 初始化模型
    speechTokenSize = 6561;
    sampling = 25;
    maxTokenTextRatio = 20;
    minTokenTextRatio = 2;

    llmEncoder_ONNX  = std::make_unique<ONNXModel>(modelPaths[0], envIds[0], device, logLevel);
    llmDecoder1_ONNX = std::make_unique<ONNXModel>(modelPaths[1], envIds[1], device, logLevel);
    llmDecoder2_ONNX = std::make_unique<ONNXModel>(modelPaths[2], envIds[2], device, logLevel);
}

torch::Tensor CosyVoiceLlm::randomSampling(const torch::Tensor& weightedScores)
{
    torch::Tensor topIds = weightedScores.softmax(0).multinomial(1, true);
    return topIds;
}

torch::Tensor CosyVoiceLlm::nucleusSampling(const torch::Tensor& weightedScores, const float top_p, const int top_k)
{
    std::vector<torch::Tensor> prob_vec;
    std::vector<torch::Tensor> indices_vec;
    float cumProb = 0.0;
    auto [sorted_value, sorted_idx] = weightedScores.softmax(0).sort(-1, true);
    for (int i = 0; i < sorted_idx.size(0); ++i) 
    {
        if (cumProb < top_p && i < top_k) 
        {
            cumProb += sorted_value[i].item<double>();
            prob_vec.push_back(sorted_value[i]);
            indices_vec.push_back(sorted_idx[i]);
        } else {
            break;
        }
    }
    // 将tensor容器转为tensor
    torch::Tensor prob_tensor = torch::stack(prob_vec).to(weightedScores.device());
    torch::Tensor indices_tensor = torch::stack(indices_vec).to(weightedScores.device());
    auto sampled_idx = prob_tensor.multinomial(1, /*replacement=*/true);
    torch::Tensor top_ids = indices_tensor.index({sampled_idx});
    
    return top_ids;
}

torch::Tensor CosyVoiceLlm::rasSampling(const torch::Tensor& weightedScores, const std::vector<int64_t>& decoded_tokens, 
    const float top_p, const int top_k, const int win_size, const float tau_r)
{
    torch::Tensor top_ids = nucleusSampling(weightedScores, top_p, top_k);
    if (decoded_tokens.size() >= win_size) {
        auto recent_tokens = torch::tensor(
            std::vector<int64_t>(decoded_tokens.end() - win_size, decoded_tokens.end()),
            torch::TensorOptions().dtype(torch::kLong).device(weightedScores.device())
        );
        
        int64_t rep_num = (recent_tokens == top_ids).sum().item<int64_t>();
        
        // If repetition exceeds threshold, fall back to random sampling
        if (rep_num >= win_size * tau_r) {
            top_ids = randomSampling(weightedScores);
        }
    }
    return top_ids;
}

torch::Tensor CosyVoiceLlm::samplingIds(const torch::Tensor& weightedScores, const std::vector<int64_t>& decoded_tokens, const int sampling, bool ignore_eos)
{
    int num_trials = 0, max_trials = 100;
    torch::Tensor top_ids;
    while (true)
    {
        top_ids = rasSampling(weightedScores, decoded_tokens);
        bool isContain = (speechTokenSize == top_ids).any().item<bool>();
        if (!ignore_eos || !isContain)
        {
            break;
        }
        num_trials++;
        if (num_trials >= max_trials)
        {
            stdCerrInColor(1, "In CosyVoiceLlm::samplingIds(), 采样次数过多, 请检查模型是否正确\n");
        }
    }
    return top_ids;
}

std::vector<Ort::Value> CosyVoiceLlm::llmEncoderInference(std::vector<torch::Tensor>& input)
{
    std::vector<Ort::Value> llmEncoderOutputs;
    try
    {
        std::vector<Ort::Value> llmEncoderInputs;
        for (auto& tensor : input)
        {
            Ort::Value value = llmEncoder_ONNX->torchTensorToOrtValue(tensor, tensor.sizes().vec());
            llmEncoderInputs.emplace_back(std::move(value));
        }
        llmEncoderOutputs = llmEncoder_ONNX->inference(llmEncoderInputs, false);
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In CosyVoiceLlm::llmEncoderInference(), 推理失败, 错误信息: %s\n", e.what());
    }
    return llmEncoderOutputs;
}

std::vector<torch::Tensor> CosyVoiceLlm::llmEncoderInferenceTRT(std::vector<torch::Tensor>& inputTensors)
{
    std::vector<torch::Tensor> outputTensors;
    if (llmEncoder_TRT)
    {
        std::vector<void*> outputBuffers = llmEncoder_TRT->inference(inputTensors, std::vector<nvinfer1::DataType>{nvinfer1::DataType::kFLOAT});
        // 手动从显卡拷贝数据到 CPU
        std::vector<float> outputData;
        llmEncoder_TRT->copyCudaDataToHost(outputData, nvinfer1::DataType::kFLOAT, llmEncoder_TRT->outputDims[0], outputBuffers[0]);
        // 转换为张量
        outputTensors.emplace_back(dataPtrToTensor(outputData.data(), llmEncoder_TRT->outputDims[0], nvinfer1::DataType::kFLOAT, inputTensors[0].device().type()).clone());
        // 手动释放显存资源
        llmEncoder_TRT->freeOutputBuffers(outputBuffers);
    }
    else
    {
        stdCerrInColor(1, "In CosyVoiceLlm::llmEncoderInferenceTRT(), 推理失败, 错误信息: 错误启用TensorRT推理, 模型指针为空\n");
    }
    return outputTensors;
}

std::vector<int64_t> CosyVoiceLlm::llmDecoder1_Decoder2_Inference(Ort::Value& lmInput, const torch::Tensor& ttsTextTokenLen)
{
    std::vector<int64_t> outTokens;
    try
    {
        size_t minLen = ttsTextTokenLen.item<int>() * minTokenTextRatio;
        size_t maxLen = ttsTextTokenLen.item<int>() * maxTokenTextRatio;
        std::vector<Ort::Value> decoder1_Inputs;
        for(size_t i = 0; i < maxLen; i++)
        {
            std::vector<Ort::Value> decoder1_Outputs;
            // 首次循环准备空张量作为第一个输入
            if (i == 0)
            {
                decoder1_Inputs.emplace_back(std::move(lmInput));
                for (size_t j = 0; j < 48; j++)
                {
                    torch::Tensor cacheInputTensor = torch::zeros({1, 2, 0, 64}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
                    Ort::Value cacheInput = llmDecoder1_ONNX->torchTensorToOrtValue(TensorType::FLOAT32, cacheInputTensor, {1, 2, 0, 64});
                    decoder1_Inputs.emplace_back(std::move(cacheInput));
                }
            }
            // 执行 llm_decoder_1 推理
            decoder1_Outputs = llmDecoder1_ONNX->inference(decoder1_Inputs, false);
            torch::Tensor logp = ortValueToTorchTensor(decoder1_Outputs[0], device);
            logp = logp.squeeze(0);
            bool ignoreEos = true;
            if (i < minLen)
            {
                ignoreEos = true;
            }
            else
            {
                ignoreEos = false;
            }
            torch::Tensor topIds = samplingIds(logp, outTokens, sampling, ignoreEos);
            int topIdsInt = topIds.item<int>();
            if (topIdsInt == speechTokenSize)
            {
                break;
            }
            else if (topIdsInt > speechTokenSize)
            {
                continue;
            }
            outTokens.push_back(topIdsInt);
            // 执行 llm_decoder_2 推理
            topIds = topIds.to(torch::kInt64);
            std::vector<Ort::Value> decoder2_Inputs;
            Ort::Value topIdValue = llmDecoder2_ONNX->torchTensorToOrtValue(TensorType::INT64, topIds, {1});
            decoder2_Inputs.emplace_back(std::move(topIdValue));
            std::vector<Ort::Value> decoder2_Outputs = llmDecoder2_ONNX->inference(decoder2_Inputs, false);
            // 重新装填 decoder1_Inputs
            decoder1_Inputs.clear();
            decoder1_Inputs.emplace_back(std::move(decoder2_Outputs[0]));
            for (size_t j = 0; j < 48; j++)
            {
                decoder1_Inputs.emplace_back(std::move(decoder1_Outputs[j+1]));
            }
            std::vector<int64_t>  cacheInputShape = decoder1_Inputs[1].GetTensorTypeAndShapeInfo().GetShape();
            
        }
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In CosyVoiceLlm::llmDecoder1_Decoder2_Inference(), 推理失败, 错误信息: %s\n", e.what());
    }
    return outTokens;
}

std::vector<int64_t> CosyVoiceLlm::llmInference(std::map<ModelInputKey, torch::Tensor>& modelInput)
{
    // llm 推理
    std::vector<int64_t> outTokens;

    // 1 先执行 llm_encoder 推理
    // 准备 llm_encoder 输入
    std::vector<torch::Tensor> llmEncoderInputsTensors = {
        modelInput[ModelInputKey::text], 
        modelInput[ModelInputKey::prompt_text],
        modelInput[ModelInputKey::llm_prompt_speech_token]
    };
    std::vector<Ort::Value> llmEncoder_OutputsValue;
    std::vector<torch::Tensor> llmEncoder_OutputsTensor;
    if (llmEncoder_ONNX){
        // ONNX推理
        try
        {
            llmEncoder_OutputsValue = llmEncoderInference(llmEncoderInputsTensors);
        }
        catch(const std::exception& e)
        {
            stdCerrInColor(1, "In CosyVoiceLlm::llmInference(), llm_encoder ONNX推理失败, 错误信息: %s\n", e.what());
        }
    }
    else
    {
        stdCerrInColor(1, "In CosyVoiceLlm::llmInference(), 推理失败, 错误信息: 错误启用 llm_encoder 推理, 模型指针为空\n");
    }
    // 2 执行 decoder1 和 decoder2 推理
    if(!llmEncoder_OutputsValue.empty() && llmDecoder1_ONNX && llmDecoder2_ONNX)
    {
        // ONNX推理
        outTokens = llmDecoder1_Decoder2_Inference(std::move(llmEncoder_OutputsValue[0]), modelInput[ModelInputKey::text_len]);
    }
    else
    {
        stdCerrInColor(1, "In CosyVoiceLlm::llmInference(), 推理失败, 错误信息: 错误启用 llm 推理, 模型指针为空/encoder输出为空\n");
    }
    return outTokens;
}

CosyVoiceFlow::CosyVoiceFlow(
    const std::vector<std::string>& modelPaths, 
    const std::vector<std::string>& envIds,
    bool useTRT_,
    torch::DeviceType device_,
    OrtLoggingLevel logLevel
    ) : useTRT(useTRT_), device(device_)
{
    // 初始化模型
    if (useTRT)
    {
        std::vector<std::string> flow1_InputNames = {"token", "mask"};
        std::vector<std::string> flow1_OutputNames = {"h"};
        flow1_TRT = std::make_unique<TensorRTEngine>(modelPaths[0], flow1_InputNames, flow1_OutputNames);
        
        std::vector<std::string> flow2_1_InputNames = {"embedding192"};
        std::vector<std::string> flow2_1_OutputNames = {"embedding80"};
        flow2_1_TRT = std::make_unique<TensorRTEngine>(modelPaths[1], flow2_1_InputNames, flow2_1_OutputNames);

        std::vector<std::string> flow2_2_InputNames = {"x", "mask", "mu", "t", "spks", "cond", "cache_step_0", "cache_step_1", "cache_step_2", "cache_step_3", "cache_step_4", "cache_step_5", "cache_step_6"};
        std::vector<std::string> flow2_2_OutputNames = {
            "x_out", "cache_out_0", "cache_out_1", "cache_out_2", "cache_out_3", "cache_out_4", "cache_out_5", "cache_out_6"
        };
        flow2_2_TRT = std::make_unique<TensorRTEngine>(modelPaths[2], flow2_2_InputNames, flow2_2_OutputNames);
    }
    else
    {
        flow1_ONNX  = std::make_unique<ONNXModel>(modelPaths[0], envIds[0], device, logLevel);
        flow2_1_ONNX = std::make_unique<ONNXModel>(modelPaths[1], envIds[1], device, logLevel);
        flow2_2_ONNX = std::make_unique<ONNXModel>(modelPaths[2], envIds[2], device, logLevel);
    }
}

torch::Tensor CosyVoiceFlow::makePadMask(const torch::Tensor& lengths, int maxLen)
{
    torch::Tensor mask;
    try
    {
        int batchSize = lengths.size(0);
        if (maxLen > 0)
        {}
        else
        {
            maxLen = lengths.max().item<int>();
        }
        torch::Tensor seqRange = torch::arange(0, maxLen, torch::TensorOptions().dtype(torch::kInt64).device(lengths.device()));
        torch::Tensor seqRangeExpand = seqRange.unsqueeze(0).expand({batchSize, maxLen});
        torch::Tensor seqLengthExpand = lengths.unsqueeze(-1);
        mask = seqRangeExpand >= seqLengthExpand;
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In CosyVoiceFlow::makePadMask(), 生成mask失败, 错误信息: %s\n", e.what());
    }
    return mask;
}

std::tuple<torch::Tensor, torch::Tensor> CosyVoiceFlow::flow1Preprocess(const torch::Tensor& token, const torch::Tensor& tokenLen, 
        const torch::Tensor& promptToken, const torch::Tensor& promptTokenLen)
{
    torch::Tensor token_, mask;
    try
    {
        token_ = torch::concat({promptToken, token}, 1);
        torch::Tensor tokenLen_ = promptTokenLen + tokenLen;
        mask = (~makePadMask(tokenLen_)).unsqueeze(-1).to(torch::TensorOptions().dtype(torch::kFloat32).device(token.device()));
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In CosyVoiceFlow::flow1Preprocess(), 预处理失败, 错误信息: %s\n", e.what());
    }
    return std::make_tuple(token_, mask);
}

std::tuple<
        torch::Tensor, 
        torch::Tensor, 
        torch::Tensor, 
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        std::map<std::string, torch::Tensor>
    > CosyVoiceFlow::inferenceFlow2_2_Input(const torch::Tensor& h, const torch::Tensor& promptFeat, const torch::Tensor& embedding80)
{
    torch::Tensor x, mask, mu, t, spks, cond, tSpan, dt;
    std::map<std::string, torch::Tensor> cache;
    int flowDecoderRequiredCacheSize = 50;
    int offset = 0;
    int outputSize = 80;
    cache["down_blocks_conv_cache"] = torch::zeros({10, 1, 2, 832, 2}).to(h.device());
    cache["down_blocks_kv_cache"] = torch::zeros({10, 1, 4, 2, flowDecoderRequiredCacheSize, 512, 2}).to(h.device());
    cache["mid_blocks_conv_cache"] = torch::zeros({10, 12, 2, 512, 2}).to(h.device());
    cache["mid_blocks_kv_cache"] = torch::zeros({10, 12, 4, 2, flowDecoderRequiredCacheSize, 512, 2}).to(h.device());
    cache["up_blocks_conv_cache"] = torch::zeros({10, 1, 2, 1024, 2}).to(h.device());
    cache["up_blocks_kv_cache"] = torch::zeros({10, 1, 4, 2, flowDecoderRequiredCacheSize, 512, 2}).to(h.device());
    cache["final_blocks_conv_cache"] = torch::zeros({10, 2, 256, 2}).to(h.device());

    int melLen1 = promptFeat.size(1);
    int melLen2 = h.size(1) - promptFeat.size(1);

    torch::Tensor conds = torch::zeros({1, melLen1 + melLen2, outputSize}).to(h.device()).to(h.dtype());
    // conds[:, :mel_len1] = prompt_feat
    conds.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, melLen1)},
        promptFeat
    );
    conds = conds.transpose(1, 2);
    int melLenSum = melLen1 + melLen2;
    torch::Tensor melLenSumTensor = torch::tensor({melLenSum}).to(h.device());
    mask = (~makePadMask(melLenSumTensor)).to(h);

    mu = h.transpose(1, 2).contiguous();
    mask = mask.unsqueeze(1);
    spks = embedding80;
    cond = conds;

    int nTimesteps = 10, temperature = 1.0;
    torch::Tensor randNoise = torch::randn({1, 80, 50 * 300});
    torch::Tensor z = randNoise.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(0, mu.size(2) + offset)
    }).to(mu.device()).to(mu.dtype()) * temperature;
    // z = z[:, :, offset:]
    z = z.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(offset, torch::indexing::None)
    });

    tSpan = torch::linspace(0, 1, nTimesteps + 1, torch::TensorOptions().device(mu.device()).dtype(mu.dtype()));
    tSpan = 1 - torch::cos(tSpan * 0.5 * torch::tensor(M_PI));

    t = tSpan[0];
    dt = tSpan[1] - tSpan[0];
    t = t.unsqueeze(0);
    x = z;
    return std::make_tuple(x, mask, mu, t, spks, cond, tSpan, dt, cache);
}

torch::Tensor CosyVoiceFlow::flowInference(torch::Tensor& token, torch::Tensor& tokenLen, torch::Tensor& promptToken, torch::Tensor& promptTokenLen, 
        torch::Tensor& promptFeat, torch::Tensor& embedding192)
{
    torch::Tensor feat;
    int melLen1 = promptFeat.size(1);
    // 预处理
    auto [token_, mask] = flow1Preprocess(token, tokenLen, promptToken, promptTokenLen);

    // 执行 flow_1 推理
    torch::Tensor hTensor;
    if (useTRT && flow1_TRT)
    {
        // TRT推理
        try
        {
            std::vector<torch::Tensor> inputTensors = {token_, mask};
            std::vector<void*> outputBuffers = flow1_TRT->inference(inputTensors, std::vector<nvinfer1::DataType>{nvinfer1::DataType::kFLOAT});
            // 拷贝数据
            std::vector<float> outputData;
            flow1_TRT->copyCudaDataToHost(outputData, nvinfer1::DataType::kFLOAT, flow1_TRT->outputDims[0], outputBuffers[0]);
            // 转换为张量
            hTensor = torch::from_blob(outputData.data(), {flow1_TRT->outputDims[0].d[0], flow1_TRT->outputDims[0].d[1], flow1_TRT->outputDims[0].d[2]}, torch::kFloat32).clone();
            // 手动释放显存资源
            flow1_TRT->freeOutputBuffers(outputBuffers);
        }
        catch(const std::exception& e)
        {
            stdCerrInColor(1, "In CosyVoiceFlow::flowInference(), flow_1 TRT推理失败, 错误信息: %s\n", e.what());
        }
    }
    else
    {
        // ONNX推理
        Ort::Value tokenValue = flow1_ONNX->torchTensorToOrtValue(TensorType::INT64, token_, token_.sizes().vec());
        Ort::Value maskValue = flow1_ONNX->torchTensorToOrtValue(TensorType::FLOAT32, mask, mask.sizes().vec());
        std::vector<Ort::Value> flow1_Inputs;
        flow1_Inputs.emplace_back(std::move(tokenValue));
        flow1_Inputs.emplace_back(std::move(maskValue));
        try
        {
            std::vector<Ort::Value> flow1_Outputs = flow1_ONNX->inference(flow1_Inputs, false);
            Ort::Value h = std::move(flow1_Outputs[0]);
            hTensor = ortValueToTorchTensor(h, device).clone();
        }
        catch(const std::exception& e)
        {
            stdCerrInColor(1, "In CosyVoiceFlow::flowInference(), flow_1 ONNX推理失败, 错误信息: %s\n", e.what());
        }
    }

    // 准备 flow_2_1 输入
    torch::Tensor embedding80Tensor;
    if (useTRT && flow2_1_TRT)
    {
        // 执行 flow_2_1 TRT推理
        try
        {
            std::vector<torch::Tensor> inputTensors = {embedding192};
            std::vector<void*> outputBuffers = flow2_1_TRT->inference(inputTensors, std::vector<nvinfer1::DataType>{nvinfer1::DataType::kFLOAT});
            // 拷贝数据
            std::vector<float> outputData;
            flow2_1_TRT->copyCudaDataToHost(outputData, nvinfer1::DataType::kFLOAT, flow2_1_TRT->outputDims[0], outputBuffers[0]);
            // 转换为张量
            embedding80Tensor = torch::from_blob(outputData.data(), {flow2_1_TRT->outputDims[0].d[0], flow2_1_TRT->outputDims[0].d[1]}, torch::kFloat32).clone();
            // 手动释放显存资源
            flow2_1_TRT->freeOutputBuffers(outputBuffers);
        }
        catch(const std::exception& e)
        {
            stdCerrInColor(1, "In CosyVoiceFlow::flowInference(), flow_2_1 TRT推理失败, 错误信息: %s\n", e.what());
        }
        
    }
    else
    {
        Ort::Value embedding192Value = flow2_1_ONNX->torchTensorToOrtValue(TensorType::FLOAT32, embedding192, embedding192.sizes().vec());
        // 执行 flow_2_1 ONNX 推理
        try
        {
            std::vector<Ort::Value> flow2_1_Inputs;
            flow2_1_Inputs.emplace_back(std::move(embedding192Value));
            std::vector<Ort::Value> flow2_1_Outputs = flow2_1_ONNX->inference(flow2_1_Inputs, false);
            Ort::Value embedding80Value = std::move(flow2_1_Outputs[0]);
            embedding80Tensor = ortValueToTorchTensor(embedding80Value, device).clone();
        }
        catch(const std::exception& e)
        {
            stdCerrInColor(1, "In CosyVoiceFlow::flowInference(), flow_2_1 推理失败, 错误信息: %s\n", e.what());
        }
    }
    // 准备flow2_2 输入
    auto [x, mask_, mu, t, spks, cond, tSpan, dt, cache] = inferenceFlow2_2_Input(hTensor, promptFeat, embedding80Tensor);
    torch::Tensor xIn = torch::zeros({2, 80, x.size(2)}, torch::TensorOptions().device(x.device()).dtype(x.dtype()));
    torch::Tensor maskIn = torch::zeros({2, 1, x.size(2)}, torch::TensorOptions().device(x.device()).dtype(x.dtype()));
    torch::Tensor muIn = torch::zeros({2, 80, x.size(2)}, torch::TensorOptions().device(x.device()).dtype(x.dtype()));
    torch::Tensor tIn = torch::zeros({2}, torch::TensorOptions().device(x.device()).dtype(x.dtype()));
    torch::Tensor spksIn = torch::zeros({2, 80}, torch::TensorOptions().device(x.device()).dtype(x.dtype()));
    torch::Tensor condIn = torch::zeros({2, 80, x.size(2)}, torch::TensorOptions().device(x.device()).dtype(x.dtype()));
    int flowCacheSize = cache["down_blocks_kv_cache"].size(4);
    std::vector<torch::Tensor> sol;
    // 执行 flow2_2 推理
    try
    {
        for (size_t step = 1; step < tSpan.size(0); step++)
        {
            xIn.copy_(x.expand_as(xIn));               // x_in[:] = x
            maskIn.copy_(mask_.expand_as(maskIn));     // mask_in[:] = mask_
            muIn[0] = mu.squeeze(0);
            tIn.copy_(t);                              // t_in[:] = t.unsqueeze(0)
            spksIn[0] = spks.squeeze(0);
            condIn[0] = cond.squeeze(0);
            std::map<std::string, torch::Tensor> cacheStep;
            for (const auto& item : cache)
            {
                std::string key = item.first;
                torch::Tensor value = item.second;
                cacheStep[key] = value[step-1];
            }
            std::vector<std::pair<std::string, torch::Tensor>> flow_2_2_Dict = {
                {"x", xIn},
                {"mask", maskIn},
                {"mu", muIn},
                {"t", tIn},
                {"spks", spksIn},
                {"cond", condIn},
                {"cache_step_0", cacheStep["down_blocks_conv_cache"]},
                {"cache_step_1", cacheStep["down_blocks_kv_cache"]},
                {"cache_step_2", cacheStep["mid_blocks_conv_cache"]},
                {"cache_step_3", cacheStep["mid_blocks_kv_cache"]},
                {"cache_step_4", cacheStep["up_blocks_conv_cache"]},
                {"cache_step_5", cacheStep["up_blocks_kv_cache"]},
                {"cache_step_6", cacheStep["final_blocks_conv_cache"]}
            };
            // 执行 flow2_2 推理
            torch::Tensor xOut;
            std::vector<torch::Tensor> cacheStepOut;
            torch::Tensor dphiDt;
            if (useTRT && flow2_2_TRT)
            {
                // TRT推理
                try
                {
                    // 准备 flow2_2 输入
                    std::vector<torch::Tensor> flow2_2_Inputs;
                    for (auto& item : flow_2_2_Dict)
                    {
                        flow2_2_Inputs.emplace_back(item.second);
                    }
                    std::vector<void*> outputBuffers = flow2_2_TRT->inference(flow2_2_Inputs, std::vector<nvinfer1::DataType>{
                        nvinfer1::DataType::kFLOAT,  // x_out
                        nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT, // cache_out_0 ~ cache_out_6
                        nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT
                    });
                    // 手动拷贝数据
                    std::vector<std::vector<float>> outputDatas(8);  // flow2_2 有8个输出
                    for (size_t i = 0; i < 8; i++)
                    {
                        flow2_2_TRT->copyCudaDataToHost(outputDatas[i], nvinfer1::DataType::kFLOAT, flow2_2_TRT->outputDims[i], outputBuffers[i]);
                        // 转换为张量
                        if (i == 0)
                        {
                            xOut = dataPtrToTensor(outputDatas[i].data(), flow2_2_TRT->outputDims[i], nvinfer1::DataType::kFLOAT, x.device().type()).clone();
                            dphiDt = xOut;
                        }
                        else
                        {
                            cacheStepOut.emplace_back(dataPtrToTensor(outputDatas[i].data(), flow2_2_TRT->outputDims[i], nvinfer1::DataType::kFLOAT, x.device().type()).clone());
                        }
                    }
                    // 释放显存
                    flow2_2_TRT->freeOutputBuffers(outputBuffers);
                }
                catch(const std::exception& e)
                {
                    stdCerrInColor(1, "In CosyVoiceFlow::flowInference(), flow_2_2 TRT推理失败, 错误信息: %s\n", e.what());
                }
                
            }
            else
            {
                // ONNX推理
                try
                {
                    // 准备 flow2_2 输入
                    std::vector<Ort::Value> flow2_2_Inputs;
                    for (auto& item : flow_2_2_Dict)
                    {
                        Ort::Value value = flow2_2_ONNX->torchTensorToOrtValue(item.second, item.second.sizes().vec());
                        flow2_2_Inputs.emplace_back(std::move(value));
                    }
                    
                    std::vector<Ort::Value> flow2_2_Outputs = flow2_2_ONNX->inference(flow2_2_Inputs, false);
                    xOut = ortValueToTorchTensor(flow2_2_Outputs[0], device).clone();
                    dphiDt = xOut;
                    
                    for (size_t i = 1; i < 8; i++)
                    {
                        cacheStepOut.emplace_back(ortValueToTorchTensor(flow2_2_Outputs[i], device).clone());
                    }
                }
                catch(const std::exception& e)
                {
                    stdCerrInColor(1, "In CosyVoiceFlow::flowInference(), flow_2_2 ONNX推理失败, 错误信息: %s\n", e.what());
                }
            }
            if (flowCacheSize != 0 && xIn.size(2) >= flowCacheSize)
            {
                cache["down_blocks_conv_cache"][step - 1] = cacheStepOut[0];
                cache["down_blocks_kv_cache"][step - 1] = cacheStepOut[1].index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(-flowCacheSize, torch::indexing::None)
                });
                cache["mid_blocks_conv_cache"][step - 1] = cacheStepOut[2];
                cache["mid_blocks_kv_cache"][step - 1] = cacheStepOut[3].index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(-flowCacheSize, torch::indexing::None)
                });
                cache["up_blocks_conv_cache"][step - 1] = cacheStepOut[4];
                cache["up_blocks_kv_cache"][step - 1] = cacheStepOut[5].index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(-flowCacheSize, torch::indexing::None)
                });
                cache["final_blocks_conv_cache"][step - 1] = cacheStepOut[6];
            }
            torch::Tensor cfgDphiDt;
            std::vector<torch::Tensor> splitVec =  torch::split(dphiDt, {x.size(0), x.size(0)}, 0);
            dphiDt = splitVec[0];
            cfgDphiDt = splitVec[1];
            x = x + dt * dphiDt;
            t = t + dt;
            sol.emplace_back(x);
            if (step < tSpan.size(0) - 1)
            {
                dt = tSpan[step + 1] - t;
            }
        }
        feat = sol.back().to(torch::kFloat32);
        feat = feat.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(melLen1, torch::indexing::None)
        });
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In CosyVoiceFlow::flowInference(), flow_2_2 推理失败, 错误信息: %s\n", e.what());
    }
    return feat;
}

CosyVoiceHift::CosyVoiceHift(
    const std::vector<std::string>& modelPaths, 
    const std::vector<std::string>& envIds,
    torch::DeviceType device_,
    OrtLoggingLevel logLevel
    ) : 
    device(device_)
{
    // 初始化模型
    hift1 = std::make_unique<ONNXModel>(modelPaths[0], envIds[0], device, logLevel);
    hift2 = std::make_unique<ONNXModel>(modelPaths[1], envIds[1], device, logLevel);
}

std::tuple<torch::Tensor, torch::Tensor> CosyVoiceHift::hiftStft(const torch::Tensor& x)
{
    // STFT参数设置
    int64_t nFft = 16;
    int64_t hopLength = 4;
    int64_t winLength = 16;

    // 创建汉宁窗 /*periodic=*/true 对应fftbins=True
    torch::Tensor window = torch::hann_window(winLength, true, torch::TensorOptions().dtype(torch::kFloat32)).to(x.device());

    // 计算STFT
    torch::Tensor spec = torch::stft(
        x,                       // 输入信号
        nFft,                    // FFT点数
        hopLength,               // 帧移
        winLength,               // 窗口长度
        window,                  // 窗口函数
        /*center=*/true,         // 默认值
        /*pad_mode=*/"reflect",  // 默认值
        /*normalized=*/false,    // 默认值
        /*onesided=*/true,       // 默认值
        /*return_complex=*/true  // 返回复数张量 
    );
    torch::Tensor realSpec = torch::view_as_real(spec);   // [B, F, TT, 2]
    // 分离实部和虚部
    torch::Tensor real = realSpec.index({"...", 0}); // [B, F, TT] "..."表示自动填充维度, 指定0表示选择最后以一个维度的第一个元素
    torch::Tensor imag = realSpec.index({"...", 1}); // [B, F, TT]
    
    return std::make_tuple(real, imag);
}

torch::Tensor CosyVoiceHift::hiftIstft(torch::Tensor& magnitude, torch::Tensor& phase)
{
    torch::Tensor inverseTransform;

    int64_t nFft = 16;
    int64_t hopLength = 4;
    int64_t winLength = 16;

    // 创建汉宁窗
    torch::Tensor window = torch::hann_window(winLength, true, torch::TensorOptions().dtype(torch::kFloat32)).to(magnitude.device());

    magnitude = torch::clip(magnitude, -INFINITY, 1e2);  // -INFINITY不限制最小值
    torch::Tensor real = magnitude * torch::cos(phase);
    torch::Tensor imag = magnitude * torch::sin(phase);
    inverseTransform = torch::istft(
        torch::complex(real, imag),                      // 输入信号
        nFft,                    // FFT点数
        hopLength,               // 帧移
        winLength,               // 窗口长度
        window                   // 窗口函数
    );

    return inverseTransform;
}

torch::Tensor CosyVoiceHift::getAudio(torch::Tensor& magnitude, torch::Tensor& phase)
{
    float audioLimit = 0.99;
    torch::Tensor x = hiftIstft(magnitude, phase);
    x = torch::clamp(x, -audioLimit, audioLimit);
    return x.detach().cpu();
}

torch::Tensor CosyVoiceHift::hiftInference(torch::Tensor& ttsMel)
{
    torch::Tensor audio;
    // 准备 hift1 输入
    Ort::Value ttsMelValue = hift1->torchTensorToOrtValue(ttsMel, ttsMel.sizes().vec());
    std::vector<Ort::Value> hift1Inputs;
    hift1Inputs.emplace_back(std::move(ttsMelValue));
    // 执行 hift1 推理
    std::vector<Ort::Value> hift1_Outputs;
    try
    {
        hift1_Outputs = hift1->inference(hift1Inputs, false);
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In CosyVoiceHift::hiftInference(), hift_1 推理失败, 错误信息: %s\n", e.what());
    }
    torch::Tensor s = ortValueToTorchTensor(hift1_Outputs[0], device);
    auto[sReal, sImag] = hiftStft(s.squeeze(1));

    // 准备 hift2 输入
    std::vector<Ort::Value> hift2_Inputs;
    Ort::Value ttsMelValue2 = hift2->torchTensorToOrtValue(ttsMel, ttsMel.sizes().vec());
    Ort::Value sRealValue = hift2->torchTensorToOrtValue(sReal, sReal.sizes().vec());
    Ort::Value sImagValue = hift2->torchTensorToOrtValue(sImag, sImag.sizes().vec());
    hift2_Inputs.emplace_back(std::move(ttsMelValue2));
    hift2_Inputs.emplace_back(std::move(sRealValue));
    hift2_Inputs.emplace_back(std::move(sImagValue));
    // 执行 hift2 推理
    std::vector<Ort::Value> hift2_Outputs;
    try
    {
        hift2_Outputs = hift2->inference(hift2_Inputs, false);
        torch::Tensor magnitude = ortValueToTorchTensor(hift2_Outputs[0], device);
        torch::Tensor phase = ortValueToTorchTensor(hift2_Outputs[1], device);
        audio = getAudio(magnitude, phase);
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In CosyVoiceHift::hiftInference(), hift_2 推理失败, 错误信息: %s\n", e.what());
    }
    return audio;
}

CosyVoice::CosyVoice(const std::string& modelDirPath, torch::DeviceType device_type, bool useTRT_) : 
    modelDir(modelDirPath), tokenizer(modelDir), device(device_type), useTRT(useTRT_)
{
    // 初始化设备
    initDevice(device_type);

    // 加载音色模型、台词
    loadSpkInfo();

    // 初始化模型
    initModule();

    // 保存音频采样率
    sampleRate = 24000;
}

void CosyVoice::initDevice(torch::DeviceType device_type)
{
    // 默认选择CPU 但如果指定GPU, 则选择GPU
    // 指定GPU, 但GPU不可用, 则使用CPU
    if (device_type == torch::kCUDA && !torch::cuda::is_available()) {
        stdCoutInColor(1, StringColor::YELLOW, "In CosyVoice::initDevice(), GPU 不可用, 使用CPU\n");
        device = torch::kCPU;
    }
    stdCoutInColor(1, StringColor::YELLOW, "In CosyVoice::initDevice(), 设备类型: %s\n", device == torch::kCPU? "CPU" : "GPU");
}

void CosyVoice::initModule()
{
    // 配置模型路径 推理环境
    std::vector<std::string> llmModelNames = { "llm_encoder_export.onnx", "llm_decoder_1_export.onnx", "llm_decoder_2_export.onnx" };
    std::vector<std::string> flowModelNames = { "flow_1_export.onnx", "flow_2_1_export.onnx", "flow_2_2_export.onnx" };
    std::vector<std::string> hiftModelNames = { "hift_1_export.onnx", "hift_2_export.onnx" };
    if (useTRT)
    {
        flowModelNames[0] = "flow_1.trt";
        flowModelNames[1] = "flow_2_1.trt";
        flowModelNames[2] = "flow_2_2.trt";
    }
    std::vector<std::string> llmEnvIds = { "llm_encoder", "llm_decoder_1", "llm_decoder_2" };
    std::vector<std::string> flowEnvIds = { "flow_1", "flow_2_1", "flow_2_2" };
    std::vector<std::string> hiftEnvIds = { "hift_1", "hift_2" };

    std::vector<std::string> llmModelPaths = {(modelDir / llmModelNames[0]).string(), (modelDir / llmModelNames[1]).string(), (modelDir / llmModelNames[2]).string()};
    std::vector<std::string> flowModelPaths = {(modelDir / flowModelNames[0]).string(), (modelDir / flowModelNames[1]).string(), (modelDir / flowModelNames[2]).string()};
    std::vector<std::string> hiftModelPaths = {(modelDir / hiftModelNames[0]).string(), (modelDir / hiftModelNames[1]).string()};

    // 初始化模型 加载模型到内存
    llm = std::make_unique<CosyVoiceLlm>(llmModelPaths, llmEnvIds, device);
    flow = std::make_unique<CosyVoiceFlow>(flowModelPaths, flowEnvIds, useTRT, device);
    hift = std::make_unique<CosyVoiceHift>(hiftModelPaths, hiftEnvIds, device);
}

torch::Tensor CosyVoice::textToTensor(const std::string& text, bool isShowResult)
{
    // 调用tokenizer进行分词
    std::vector<int> tokens = tokenizer.tokenize(text);
    // 创建张量并指定数据类型
    torch::Tensor tensor;
    try
    {
        tensor = torch::tensor(tokens, torch::dtype(torch::kInt32)).unsqueeze(0);  // 增加一维
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In CosyVoice::textToTensor(), 转换tensor失败, 错误信息: %s\n", e.what());
    }
    // 将张量移动到指定设备
    tensor = tensor.to(device);
    if (isShowResult)
    {
        stdCoutInColor(1, StringColor::YELLOW, "In CosyVoice::textToTensor(), text: %s\n", text.c_str());
        stdCoutInColor(1, StringColor::YELLOW, "In CosyVoice::textToTensor(), tensor: \n");
        std::cout << tensor << std::endl;
    }
    return tensor;
}

torch::Tensor CosyVoice::tensorSizeToTensor(const torch::Tensor& tensor_, bool isShowResult)
{
    torch::Tensor tensor;
    try
    {
        tensor = torch::tensor({tensor_.size(1)}, torch::dtype(torch::kInt32)).to(device);
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In tensorSizeToTensor(), 转换tensor失败, 错误信息: %s\n", e.what());
    }
    if (isShowResult)
    {
        stdCoutInColor(1, StringColor::YELLOW, "In tensorSizeToTensor(), tensor: \n");
    }
    return tensor;
}

void CosyVoice::loadSpkInfo()
{
    stdCoutInColor(1, StringColor::YELLOW, "In CosyVoice::loadSpkInfo(), 加载音色模型\n");
    // 拼接出音色文件的路径
    std::filesystem::path spkInfoPath = modelDir / "spkInfo.jit";  // 无视平台拼接
    std::filesystem::path spkLinesPath = modelDir / "spkLines.json"; 
    std::string spkInfoPathStr = spkInfoPath.string();
    std::string spkLinesPathStr = spkLinesPath.string();
    stdCoutInColor(1, StringColor::BLUE, "In CosyVoice::loadSpkInfo(), 音色模型路径: %s\n", spkInfoPathStr.c_str());
    stdCoutInColor(1, StringColor::BLUE, "In CosyVoice::loadSpkInfo(), 音色台词路径: %s\n", spkLinesPathStr.c_str());
    // 加载音色模型
    try
    {
        spkInfo = torch::jit::load(spkInfoPathStr, device);
        // 查看可用音色
        int i = 0;  // 每三个属性提取一个音色id
        for (const auto& attr : spkInfo.named_attributes())
        {
            std::string spkId = attr.name.substr(0, attr.name.find("_embedding"));
            if (i % 3 == 0)
            {
                if (spkId == "training")  // 读取到倒数第二个属性(training)时结束循环
                {
                    break;
                }
                spkList.push_back(spkId);
                spkLines[spkId] = LoadValueFromJsonFile(spkLinesPathStr, spkId).get<std::string>();
                stdCoutInColor(1, StringColor::BLUE, "In CosyVoice::loadSpkInfo(), 可用音色id: %s\n", spkId.c_str());
                stdCoutInColor(1, StringColor::BLUE, "In CosyVoice::loadSpkInfo(), 台词: %s\n", spkLines[spkId].c_str());
            }
            i++;
        }
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In CosyVoice::loadSpkInfo(), 加载读取音色模型失败, 错误信息: %s\n", e.what());
    }
}

torch::Tensor CosyVoice::getSpkTensor(const std::string& spkId, const SpkTensorKey tensorName)
{
    torch::Tensor tensor;
    // 先确定音色列表是否含有输入的音色id
    if (std::find(spkList.begin(), spkList.end(), spkId) == spkList.end())
    {
        stdCerrInColor(1, "In CosyVoice::getSpkTensor(), 音色id %s 不存在 返回空tensor\n", spkId.c_str());
    }
    else
    {
        try
        {
            std::string totalKey = spkId + "_" + TensorKeyMeta::name(tensorName);
            tensor = spkInfo.attr(totalKey).toTensor();
        }
        catch(const std::exception& e)
        {
            stdCerrInColor(1, "In CosyVoice::getSpkTensor(), 获取音色 %s 张量 %s 失败, 错误信息: %s\n", spkId.c_str(), TensorKeyMeta::name(tensorName).c_str(), e.what());
        }
        
    }
    return tensor;
}

void CosyVoice::createModelInput(const CosyVoiceInput& input)
{
    std::string spkId = input.spkId;                // 音色id
    std::string ttsText = input.ttsText;            // 语音转换文本
    std::string instructText = input.instructText;  // 情景控制文本
    // tts_text
    torch::Tensor ttsTextToken    = textToTensor(ttsText);
    torch::Tensor ttsTextTokenLen = tensorSizeToTensor(ttsTextToken);
    // prompt_text
    std::string promptText           = instructText + "<|endofprompt|>" + spkLines[spkId];
    torch::Tensor promptTextToken    = textToTensor(promptText);
    torch::Tensor promptTextTokenLen = tensorSizeToTensor(promptTextToken);
    // speech
    torch::Tensor speechToken    = getSpkTensor(spkId, SpkTensorKey::speech_token);
    torch::Tensor speechTokenLen = tensorSizeToTensor(speechToken);

    torch::Tensor speechFeat     = getSpkTensor(spkId, SpkTensorKey::speech_feat);
    torch::Tensor speechFeatLen  = tensorSizeToTensor(speechFeat);

    torch::Tensor embedding      = getSpkTensor(spkId, SpkTensorKey::embedding);

    try
    {
        modelInput = std::map<ModelInputKey, torch::Tensor>
        {
            { ModelInputKey::text, ttsTextToken }, { ModelInputKey::text_len, ttsTextTokenLen },
            { ModelInputKey::prompt_text, promptTextToken }, { ModelInputKey::prompt_text_len, promptTextTokenLen },
            { ModelInputKey::llm_prompt_speech_token, speechToken }, { ModelInputKey::llm_prompt_speech_token_len, speechTokenLen },
            { ModelInputKey::flow_prompt_speech_token, speechToken }, { ModelInputKey::flow_prompt_speech_token_len, speechTokenLen },
            { ModelInputKey::prompt_speech_feat, speechFeat }, { ModelInputKey::prompt_speech_feat_len, speechFeatLen },
            { ModelInputKey::llm_embedding, embedding }, { ModelInputKey::flow_embedding, embedding }
        };
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In CosyVoice::createModelInput(), 创建模型输入失败, 错误信息: %s\n", e.what());
    }
}

torch::Tensor CosyVoice::inference()
{
    torch::Tensor audio;
    if (!modelInput.empty() && llm && flow && hift)
    {
        stdCoutInColor(1, StringColor::YELLOW, "In CosyVoice::inference(), 开始推理\n");
        std::vector<int64_t> outTokens;
        try
        {
            outTokens = llm->llmInference(modelInput);
        }
        catch(const std::exception& e)
        {
            stdCerrInColor(1, "In CosyVoice::inference(), llm 推理失败, 错误信息: %s\n", e.what());
        }
        stdCoutInColor(1, StringColor::GREEN, "In CosyVoice::inference(), llm 推理完成\n");

        torch::Tensor ttsMel;
        try
        {
            // 2 准备 flow 输入
            torch::Tensor outTokensTensor = torch::from_blob(
                outTokens.data(),                            // 数据指针
                {static_cast<int64_t>(outTokens.size())},    // 形状
                torch::TensorOptions().dtype(torch::kInt64)  // 类型和设备, 数据源自CPU, 无法在内部指定设备为GPU
            ).unsqueeze(0).to(device);     // 只能先在CPU上创建张量, 然后转移到GPU上
            // 把整型容器转化为Tensor
            torch::Tensor outTokensTensorLen = torch::tensor({outTokensTensor.size(1)}, torch::dtype(torch::kInt32)).to(outTokensTensor.device());
            // flow 推理
            ttsMel =  flow->flowInference(
                outTokensTensor, outTokensTensorLen,
                modelInput[ModelInputKey::llm_prompt_speech_token], modelInput[ModelInputKey::llm_prompt_speech_token_len],
                modelInput[ModelInputKey::prompt_speech_feat],
                modelInput[ModelInputKey::llm_embedding] 
            );
        }
        catch(const std::exception& e)
        {
            stdCerrInColor(1, "In CosyVoice::inference(), flow 推理失败, 错误信息: %s\n", e.what());
        }
        stdCoutInColor(1, StringColor::GREEN, "In CosyVoice::inference(), flow 推理完成\n");

        try
        {
            // 3 准备 hift 输入 推理
            audio =  hift->hiftInference(ttsMel);
        }
        catch(const std::exception& e)
        {
            stdCerrInColor(1, "In CosyVoice::inference(), hift 推理失败, 错误信息: %s\n", e.what());
        }
        stdCoutInColor(1, StringColor::GREEN, "In CosyVoice::inference(), hift 推理完成\n"); 
    }
    else if(modelInput.empty())
    {
        stdCerrInColor(1, "In CosyVoice::inference(), 模型输入为空, 请先调用createModelInput()初始化模型输入\n");
    }
    else if(!llm)
    {
        stdCerrInColor(1, "In CosyVoice::inference(), llm模型为空, 请先初始化llm模型\n");
    }
    else if(!flow)
    {
        stdCerrInColor(1, "In CosyVoice::inference(), flow流模型为空, 请先初始化flow模型\n");
    }
    else if(!hift)
    {
        stdCerrInColor(1, "In CosyVoice::inference(), hift模型为空, 请先初始化hift模型\n");
    }
    return audio;
}