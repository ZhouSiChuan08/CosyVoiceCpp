#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <tokenizers_cpp.h>
#include <torch/script.h>

#include "debugTools.h"
using tokenizers::Tokenizer;

/**
 * @brief 读取token配置文件
 * @param path token配置文件路径
 * @return token配置文件内容
 */
std::string loadBytesFromFile(const std::string& path);

/**
 * @brief 打印编码结果
 * @param ids 编码结果容器
 */
void printEncodeResult(const std::vector<int>& ids);

/**
 * @brief 测试分词器是否正确工作
 * @param tok 分词器对象
 * @param print_vocab 是否打印词表
 * @param check_id_back 是否检查id映射示例
 */
void testTokenizer(std::shared_ptr<Tokenizer> tok,const std::string& prompt = "Hello world!", bool check_id_back = true);

/**
 * @brief 基于Hugging Face的分词器
 */
class HuggingFaceTokenizer 
{
public:
    /**
     * @brief 分词器构造函数
     * @param tokenizerDir_ 分词器配置目录路径
     */
    HuggingFaceTokenizer(const std::filesystem::path& tokenizerDir_);
    
    /**
     *@brief 根据分词器配置文件tokenizer.json创建分词器
     * @note 该函数会读取配置文件，并根据配置文件创建分词器对象。
     */
    void createTokenizer();

    /**
     * @brief 获取分词器对象
     * @return 分词器对象
     */
    std::shared_ptr<Tokenizer> getTokenizer();

    /**
     * @brief 对自然语言进行编码
     */
    std::vector<int> tokenize(const std::string& text);

private:
    const std::filesystem::path tokenizerDir;   // 分词器配置文件路径
    std::shared_ptr<Tokenizer> tokenizer;  // 分词器对象
};

#endif // TOKENIZER_H