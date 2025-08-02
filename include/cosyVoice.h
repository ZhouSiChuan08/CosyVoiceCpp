#ifndef COSYVOICE_H
#define COSYVOICE_H

#include <map>
#include <filesystem>
#include <torch/script.h>
#include <torch/torch.h>
#include <sndfile.h>
#include "tensorRT.h"
#include "onnx.h"
#include "tokenizer.h"
#include "debugTools.h"
#include "toolFun.h"


// tensor键值对名强类型定义
// 音色tensor键值对名
enum class SpkTensorKey {
    embedding, speech_feat, speech_token
};
enum class ModelInputKey {
    text,        text_len, 
    prompt_text, prompt_text_len, 
    llm_prompt_speech_token,  llm_prompt_speech_token_len,
    flow_prompt_speech_token, flow_prompt_speech_token_len, 
    prompt_speech_feat, prompt_speech_feat_len,
    llm_embedding,      flow_embedding
};
struct TensorKeyMeta {
    // 获取键名
    static std::string name(SpkTensorKey key) {
        switch (key) {
            case SpkTensorKey::embedding:    return "embedding";
            case SpkTensorKey::speech_feat:  return "speech_feat";
            case SpkTensorKey::speech_token: return "speech_token";
            default: throw std::invalid_argument("Unknown SpkTensorKey");
        }
    }
    static std::string name(ModelInputKey key) {
        switch (key) {
            case ModelInputKey::text:                            return "text";
            case ModelInputKey::text_len:                        return "text_len";
            case ModelInputKey::prompt_text:                     return "prompt_text";
            case ModelInputKey::prompt_text_len:                 return "prompt_text_len";
            case ModelInputKey::llm_prompt_speech_token:         return "llm_prompt_speech_token";
            case ModelInputKey::llm_prompt_speech_token_len:     return "llm_prompt_speech_token_len";
            case ModelInputKey::flow_prompt_speech_token:        return "flow_prompt_speech_token";
            case ModelInputKey::flow_prompt_speech_token_len:    return "flow_prompt_speech_token_len";
            case ModelInputKey::prompt_speech_feat:              return "prompt_speech_feat";
            case ModelInputKey::prompt_speech_feat_len:          return "prompt_speech_feat_len";
            case ModelInputKey::llm_embedding:                   return "llm_embedding";
            case ModelInputKey::flow_embedding:                  return "flow_embedding";
            default: throw std::invalid_argument("Unknown ModelInputKey");
        }
    }
};

void saveAudioWav(const std::string& filename, const torch::Tensor& audio, int sample_rate);

/**
 * @brief CosyVoiceLlm llm模块
 */
class CosyVoiceLlm {
public:
    CosyVoiceLlm(const std::vector<std::string>& modelPaths, const std::vector<std::string>& envIds, torch::DeviceType device_ = torch::kCPU, OrtLoggingLevel logLevel = ORT_LOGGING_LEVEL_WARNING);

    /**
     * @brief 复刻采样函数random_sampling
     * @param weighted_scores 权重分数
     * @return torch::Tensor 采样结果
     */
    torch::Tensor randomSampling(const torch::Tensor& weightedScores);

    /**
     * @brief 复刻采样函数nucleus_sampling
     * @param weighted_scores 权重分数
     * @param top_p 置信度阈值
     * @param top_k 采样数量
     * @return torch::Tensor 采样结果
     */
    torch::Tensor nucleusSampling(const torch::Tensor& weightedScores, const float top_p = 0.8, const int top_k = 25);

    /**
     * @brief 复刻采样函数name:cosyvoice.utils.common.ras_sampling
     */
    torch::Tensor rasSampling(const torch::Tensor& weightedScores, const std::vector<int64_t>& decoded_tokens, 
        const float top_p=0.8, const int top_k=25, const int win_size=10, const float tau_r=0.1);
    
    /**
     * @brief 复刻采样函数sampling_ids
     * @param weighted_scores 权重分数
     * @param decoded_tokens 解码结果
     * @param sampling 采样策略
     * @param ignore_eos 是否忽略eos
     * @return torch::Tensor 采样结果
     */
    torch::Tensor samplingIds(const torch::Tensor& weightedScores, const std::vector<int64_t>& decoded_tokens, const int sampling, bool ignore_eos=true);

    /**
     * @brief llmEncoder ONNX推理
     * @param std::vector<torch::Tensor> 输入张量 元素数量为3 text, prompt_text, llm_prompt_speech_token
     * @return std::vector<Ort::Value> output 输出
     */
    std::vector<Ort::Value> llmEncoderInference(std::vector<torch::Tensor>& input);

    /**
     * @brief llmEncoder TRT推理
     * @param std::vector<torch::Tensor> 输入张量 元素数量为3 text, prompt_text, llm_prompt_speech_token
     * @return std::vector<Ort::Value> output 输出
     */
    std::vector<torch::Tensor> llmEncoderInferenceTRT(std::vector<torch::Tensor>& inputTensors);

    /**
     * @brief llmDecoder1 llmDecoder2推理
     * @param Ort::Value lmInput 输入
     * @param torch::Tensor ttsTextTokenLen 输入tts文本token长度
     * @return std::vector<int64_t> 输出tokens
     */
    std::vector<int64_t> llmDecoder1_Decoder2_Inference(Ort::Value& lmInput, const torch::Tensor& ttsTextTokenLen);

    /**
     * @brief llm 推理
     * @param std::map<ModelInputKey, torch::Tensor> 模型的所有可输入张量 
     */
    std::vector<int64_t> llmInference(std::map<ModelInputKey, torch::Tensor>& modelInput);

private:
    // 成员变量
    int speechTokenSize;      // 最大token索引值
    int sampling;
    size_t maxTokenTextRatio;
    size_t minTokenTextRatio;
    bool useTRT;
    torch::DeviceType device;

    // onnx模型
    std::unique_ptr<ONNXModel> llmEncoder_ONNX;
    std::unique_ptr<ONNXModel> llmDecoder1_ONNX;
    std::unique_ptr<ONNXModel> llmDecoder2_ONNX;

    // TensorRT模型
    std::unique_ptr<TensorRTEngine> llmEncoder_TRT;
};

/**
 * @brief CosyVoiceFlow flow模块
 */
class CosyVoiceFlow {
public:
    CosyVoiceFlow(const std::vector<std::string>& modelPaths, const std::vector<std::string>& envIds ,bool useTRT_ = false, torch::DeviceType device_ = torch::kCPU, OrtLoggingLevel logLevel = ORT_LOGGING_LEVEL_WARNING);

    /**
     * @brief 复刻make_pad_mask函数 
     * @param lengths 输入长度
     * @param maxLen 最大长度
     * @return torch::Tensor 填充掩码
     */
    torch::Tensor makePadMask(const torch::Tensor& lengths, int maxLen=0);

    /**
     * @brief flow1输入预处理
     * @param token 输入token
     * @param tokenLen 输入token长度
     * @param promptToken 输入提示token
     * @param promptTokenLen 输入提示token长度
     * @return std::tuple<torch::Tensor, torch::Tensor> 预处理结果
     */
    std::tuple<torch::Tensor, torch::Tensor> flow1Preprocess(const torch::Tensor& token, const torch::Tensor& tokenLen, 
        const torch::Tensor& promptToken, const torch::Tensor& promptTokenLen);

    /**
     * @brief 复刻 inference_flow_2_2_input_函数
     * @param h 输入隐藏状态
     * @param promptFeat 输入提示特征
     * @param embedding80 输入embedding
     * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::map<std::string, torch::Tensor>> 预处理结果
     */
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
    > CosyVoiceFlow::inferenceFlow2_2_Input(const torch::Tensor& h, const torch::Tensor& promptFeat, const torch::Tensor& embedding80);

    /**
     * @brief flow 推理
     * @param token 输入token
     * @param tokenLen 输入token长度
     * @param promptToken 输入提示token
     * @param promptTokenLen 输入提示token长度
     * @param promptFeat 输入提示特征
     * @param embedding192 输入embedding
     * @return torch::Tensor 输出mel谱张量
     */
    torch::Tensor flowInference(torch::Tensor& token, torch::Tensor& tokenLen, torch::Tensor& promptToken, torch::Tensor& promptTokenLen, 
        torch::Tensor& promptFeat, torch::Tensor& embedding192);

private:
    // onnx模型
    std::unique_ptr<ONNXModel> flow1_ONNX;
    std::unique_ptr<ONNXModel> flow2_1_ONNX;
    std::unique_ptr<ONNXModel> flow2_2_ONNX;
    // TensorRT模型
    std::unique_ptr<TensorRTEngine> flow1_TRT;
    std::unique_ptr<TensorRTEngine> flow2_1_TRT;
    std::unique_ptr<TensorRTEngine> flow2_2_TRT;

    bool useTRT;
    torch::DeviceType device;
};

/**
 * @brief CosyVoiceHift hift模块
 */
class CosyVoiceHift {
public:
    CosyVoiceHift(const std::vector<std::string>& modelPaths, const std::vector<std::string>& envIds , torch::DeviceType device_ = torch::kCPU, OrtLoggingLevel logLevel = ORT_LOGGING_LEVEL_WARNING);

    /**
     * @brief 复刻 hift_stft 函数 快速傅里叶变换
     * @param x 输入信号
     * @return std::tuple<torch::Tensor, torch::Tensor> 实部和虚部
     */
    std::tuple<torch::Tensor, torch::Tensor> hiftStft(const torch::Tensor& x);

    /**
     * @brief 复刻 hift_istft 函数 逆快速傅里叶变换
     * @param magnitude 幅度
     * @param phase 相位
     * @return torch::Tensor 逆变换结果
     */
    torch::Tensor hiftIstft(torch::Tensor& magnitude, torch::Tensor& phase);

    /**
     * @brief 复刻 get_audio 函数 获取音频
     * @param magnitude 幅度
     * @param phase 相位
     * @return torch::Tensor 音频 剔除梯度, 在CPU上
     */
    torch::Tensor getAudio(torch::Tensor& magnitude, torch::Tensor& phase);

    /**
     * @brief hift 推理
     * @param ttsMel 输入mel
     * @return torch::Tensor 输出音频张量
     */
    torch::Tensor hiftInference(torch::Tensor& ttsMel);

private:
    torch::DeviceType device;
    std::unique_ptr<ONNXModel> hift1;  // onnx模型
    std::unique_ptr<ONNXModel> hift2;
};

// 用户输入部分
struct CosyVoiceInput {
    std::string spkId;                            // 音色id
    std::string ttsText;                          // 待转换语音的文本
    std::string instructText;                     // 情景控制文本
    std::string audioDirPath;                     // 音频保存目录
};

/**
 * @brief CosyVoice 声音合成类
 */
class CosyVoice {
public:
    /**
     * @brief CosyVoice 构造函数
     * @param modelDirPath_ 模型文件目录 自动加载模型文件和分词器文件
     * @param device_type 设备类型，默认使用CPU
     * @param useTRT_ 是否使用TensorRT
     */
    CosyVoice(const std::string& modelDirPath_, torch::DeviceType device_type = torch::kCPU, bool useTRT_ = false);

    /**
     * @brief 初始化设备信息
     * @param device_type 设备类型，默认使用CPU
     */
    void initDevice(torch::DeviceType device_type = torch::kCPU);

    /**
     * @brief 模块初始化
     */
    void initModule();


    /**
     *@brief 文本转张量
     * @param text 输入文本
     * @return 转换后的张量
     */
    torch::Tensor textToTensor(const std::string& text, bool isShowResult = false);

    /**
     * @brief 张量尺寸转张量 只转换第二个维度
     * @param tensor 输入张量
     * @return 转换后的张量
     */
    torch::Tensor tensorSizeToTensor(const torch::Tensor& tensor_, bool isShowResult = false);

    /**
     * @brief 加载音色模型
     */
    void loadSpkInfo();

    /**
     * @brief 根据音色id和tensor键名获取对应tensor
     * @param spkId 音色id
     * @param tensorName 张量键名
     * @return torch::Tensor 对应张量
     */
    torch::Tensor getSpkTensor(const std::string& spkId, const SpkTensorKey tensorName);

    /**
     * @brief 构造模型输入
     * @param CosyVoiceInput 用户输入
     */
    void createModelInput(const CosyVoiceInput& input);

    /**
     * @brief 执行推理
     */
    torch::Tensor inference();

    // 保存音频参数
    int sampleRate;                               // 采样率
private:
    // 成员变量
    torch::DeviceType device;                         // 存储设备信息
    const std::filesystem::path modelDir;         // 模型文件目录

    const std::string tokenizerPath;              // 用于分词的tokenizer配置文件路径
    HuggingFaceTokenizer tokenizer;               // 用于分词的tokenizer

    torch::jit::Module spkInfo;                   // 音色模型
    std::vector<std::string> spkList;             // 音色列表
    std::map<std::string, std::string> spkLines;  // 音色id和音色台词的映射表

    // 子模块
    std::unique_ptr<CosyVoiceLlm> llm;            // llm模块
    std::unique_ptr<CosyVoiceFlow> flow;          // flow模块
    std::unique_ptr<CosyVoiceHift> hift;          // hift模块

    bool useTRT;                                  // 是否使用TensorRT

    // 模型输入
public:
    std::map<ModelInputKey, torch::Tensor> modelInput;  // 模型输入张量字典

};



#endif // COSYVOICE_H