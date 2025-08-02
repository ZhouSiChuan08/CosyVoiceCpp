#ifndef TOOLFUN_H
#define TOOLFUN_H


#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdint>
#include <random>
#include <nlohmann/json.hpp>
#include "debugTools.h"


using json = nlohmann::json;

/**
 * @brief 从json文件中读取值
 * @param filePath json文件路径
 * @param key 要读取的key值
 * @return 返回读取到的json对象 需手动解析
 */
json LoadValueFromJsonFile(const std::string& filePath, const std::string& key);

/**
 * @brief 获取本地时间, 用以文件命名
 * @return 返回格式化的本地时间字符串
 */
std::string getLocalTimestampForFilename();

class TicToc
{
  public:
    TicToc();
    /**
     * @brief 开始计时, 刷新当前时间字符串
     */
    void tic();

    /**
     * @brief 结束计时并返回运行时间
     * @return 返回运行时间, 单位为毫秒
     */
    double toc();

    /**
     * @brief 获取本地时间, 用以文件命名
     */
    void getLocalt();

    /**
     * @brief 将三字母月份转为对应数字字符串
     * @param month 三字母月份字符串
     * @return 返回对应数字字符串
     */
    std::string monthToNumber(const std::string& month);

    time_t start_t;                  //可打印的时间格式  UNIX时间戳
    char *start_t_loacal;            //可打印的时间格式  当地日期时间戳
    struct tm *satrt_t_utc;          //UTC时间戳
    std::string starTimeStamp;
 private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

/**
 * @brief 生成0~UINT64_MAX之间的随机数
 * @return 返回随机数
 */
uint64_t getRandSeed();

#endif // TOOLFUN_H