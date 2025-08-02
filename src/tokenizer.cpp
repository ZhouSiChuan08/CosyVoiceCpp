#include "tokenizer.h"

std::string loadBytesFromFile(const std::string& path) 
{
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        stdCerrInColor(1, "In loadBytesFromFile, 无法打开文件: " + path + " 已退出程序.\n");
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());  // 获取当前指针位置, 即获取文件大小
    fs.seekg(0, std::ios::beg);                     // 指针回到文件开头
    data.resize(size);
    fs.read(data.data(), size);
    stdCoutInColor(1, StringColor::BLUE, "In loadBytesFromFile, 从 " + path + " 读取 " + std::to_string(size) + " bytes数据.\n");
    fs.close();
    return data;
}

void printEncodeResult(const std::vector<int>& ids) 
{
    std::cout << "tokens=[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "]" << std::endl;
}

void testTokenizer(std::shared_ptr<Tokenizer> tok,const std::string& prompt, bool check_id_back)
{
    stdCoutInColor(1, StringColor::YELLOW, "In testTokenizer, 开始测试Tokenizer功能...\n");
    // Check #1. Encode and Decode
    std::vector<int> ids = tok->Encode(prompt);
    std::string decoded_prompt = tok->Decode(ids);
    printEncodeResult(ids);
    std::cout << "decode=\"" << decoded_prompt << "\"" << std::endl;
    if (prompt != decoded_prompt) {
        stdCerrInColor(1, "In testTokenizer, 编码和解码结果不一致! 已退出程序.\n");
        exit(1);
    }
    // Check #2. IdToToken and TokenToId
    stdCoutInColor(1, StringColor::BLUE, "In TestTokenizer, 开始测试IdToToken和TokenToId功能...\n");
    std::vector<int32_t> ids_to_test = {0, 1, 2, 3, 32, 33, 34, 130, 131, 1000};
    for (auto id : ids_to_test) {
        auto token = tok->IdToToken(id);
        auto id_new = tok->TokenToId(token);
        std::cout << "id=" << id << ", token=\"" << token << "\", id_new=" << id_new << std::endl;
        if (check_id_back) {
            if (id != id_new)
            {
                stdCerrInColor(1, "In testTokenizer, Id和Token的转换不一致! 已退出程序.\n");
                exit(1);
            }
        }
    }
    // Check #3. GetVocabSize
    stdCoutInColor(1, StringColor::BLUE, "In testTokenizer, 开始测试GetVocabSize功能...\n");
    auto vocab_size = tok->GetVocabSize();
    std::cout << "vocab_size=" << vocab_size << std::endl;
}

HuggingFaceTokenizer::HuggingFaceTokenizer(const std::filesystem::path& tokenizerDir_) : tokenizerDir(tokenizerDir_)
{
    createTokenizer();
}

void HuggingFaceTokenizer::createTokenizer()
{
    if (tokenizerDir.empty())
    {
        stdCerrInColor(1, "In HuggingFaceTokenizer::createTokenizer, tokenizerDir为空! 已退出程序.\n");
        exit(1);
    }
    else
    {
        stdCoutInColor(1, StringColor::YELLOW, "In HuggingFaceTokenizer::createTokenizer, 开始创建分词器, 类型: Huggingface\n");
        auto start = std::chrono::high_resolution_clock::now();
        // Read blob from file.
        std::filesystem::path tokenizerJsonPath = tokenizerDir / "tokenizer.json";
        auto tokenizer_blob  = loadBytesFromFile(tokenizerJsonPath.string());
        // Note: all the current factory APIs takes in-memory blob as input.
        // This gives some flexibility on how these blobs can be read.
        std::unique_ptr<Tokenizer> tok = Tokenizer::FromBlobJSON(tokenizer_blob);
        tokenizer = std::move(tok);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        stdCoutInColor(1, StringColor::GREEN, "In HuggingFaceTokenizer::createTokenizer, 创建分词器成功, 耗时: " + std::to_string(duration) + " ms\n");
        testTokenizer(tokenizer);
    }
}

std::shared_ptr<Tokenizer> HuggingFaceTokenizer::getTokenizer()
{
    return tokenizer;
}

std::vector<int> HuggingFaceTokenizer::tokenize(const std::string& text) {
    std::vector<int> ids;
    ids = tokenizer->Encode(text);
    return ids;
}