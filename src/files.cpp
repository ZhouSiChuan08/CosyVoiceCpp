//getline 按行读取文件流 ifstream， 字符串流 ifstream  直到无更多内容可读
#include <fstream> // 包含ifstream的头文件
#include <sstream> // 包含stringstream的头文件
#include <vector>  // 包含vector的头文件
#include <string>  // 包含string的头文件
#include <iostream> // 包含iostream的头文件，用于输入输出
#include "global.h" // 包含全局变量的头文件


std::vector<std::string> splitStr(const std::string& str, const char delimiter)  //
{
    std::vector<std::string> result; // 定义一个空的vector
    std::istringstream iss(str); // 定义一个字符串流对象
    std::string item; // 定义一个空字符串变量
    // 使用while循环和std::getline函数来读取iss中的内容
    while (std::getline(iss, item, delimiter)) 
    {
        result.push_back(item); // 如果读取成功，将得到的子字符串添加到result向量中
    }
    return result; // 返回包含所有子字符串的向量
}

std::string trim(const std::string& str) 
{
    std::string result = str;
    // 去除开头的空白字符
    result.erase(0, result.find_first_not_of(" \t\n\r\f\v"));
    // 去除结尾的空白字符
    result.erase(result.find_last_not_of(" \t\n\r\f\v") + 1);
    return result;
}

std::vector<std::vector<std::string>> readAny(const std::string& filePath, const std::vector<int>& columnIndex, const char delimiter, const std::string flag, int skipRows)
{
    std::ifstream file(filePath, std::ios::in);    // 定义一个输入文件流对象
    if(!file.is_open())
    {
        std::cout  << "终止程序, 无法打开文件: " << filePath << std::endl;
        exit(1);
    }
    int skipLine = 0;                             // 记录跳过的行数
    std::string line;                             // 保存每行数据
    std::vector<std::string> splitedLine;         // 保存每行数据分割后的结果
    std::vector<std::vector<std::string>> result; // 保存所有行数据
    while (std::getline(file, line))
    {
        skipLine++;
        if (skipLine <= skipRows)         // 跳过指定行数
        {
            continue;
        }
        splitedLine = splitStr(line, delimiter); //分割字符串
        // 先删除空白元素
        for (auto it = splitedLine.begin(); it !=  splitedLine.end();)
        {
            if (trim(*it).empty())
            {
                it = splitedLine.erase(it);
            }
            else
            {
                it++;
            } 
        }
        // 每行数据不为空
        if (splitedLine.size() != 0)
        {
            if (flag != "flag")  // 启用标志位过滤
            {
                if (splitedLine[0] != flag)
                {
                    //不符合标志位, 什么也不干
                }
                else
                {
                    //符合标志位, 继续处理数据 根据输入索引提取数据
                    std::vector<std::string> temp;
                    for (size_t i = 0; i < columnIndex.size(); i++)
                    {
                        temp.push_back(splitedLine[columnIndex[i]]);
                    }
                    result.push_back(temp);
                }
                
            }
            else
            {
                // 不启用标志位过滤, 直接处理数据 根据输入索引提取数据
                std::vector<std::string> temp;
                for (size_t i = 0; i < columnIndex.size(); i++)
                {
                    temp.push_back(splitedLine[columnIndex[i]]);
                }
                result.push_back(temp);
            }
        } 
    }
    file.close();
    return result;
}


