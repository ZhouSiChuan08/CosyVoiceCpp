#ifndef TIC_TOC_H
#define TIC_TOC_H
#pragma once  //保证该h文件只被include一次 但只有部分编译器指令支持该预处理指令
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <map>
#include <string>
class TicToc
{
  public:
    TicToc()  //构造器
    {
        tic();
    }
    void tic()  //开始计时
    {
        start = std::chrono::system_clock::now();  //开始时刻
        //获取不同类型的时间
        start_t = std::chrono::system_clock::to_time_t(start);  //time_t类型 UNIX时间戳
        start_t_loacal = ctime(&start_t);  //本地时间
        satrt_t_utc = gmtime(&start_t);  //UTC标准时间
        //获取当地时间字符串
        get_localt();
    }
    double toc()
    {
        end = std::chrono::system_clock::now();  //结束时刻
        std::chrono::duration<double> elapsed_seconds = end - start;  //持续时长
        return elapsed_seconds.count() * 1000;  //毫秒
    }

    //获取本地日期字符串
    void get_localt()
    {
        std::string start_t_loacal_string(start_t_loacal);
        start_t_loacal_ymdhms = start_t_loacal_string.substr(20, 4) + monthToNumber(start_t_loacal_string.substr(4, 3)) 
                                + start_t_loacal_string.substr(8, 2) + start_t_loacal_string.substr(11, 2) 
                                + start_t_loacal_string.substr(14, 2) + start_t_loacal_string.substr(17, 2);
    }
    //将三字母月份转为对应数字字符串
    std::string monthToNumber(const std::string& month) 
    {
        // 创建一个月份缩写到数字的映射
        std::map<std::string, std::string> monthMap = {
            {"Jan", "01"}, {"Feb", "02"}, {"Mar", "03"},
            {"Apr", "04"}, {"May", "05"}, {"Jun", "06"},
            {"Jul", "07"}, {"Aug", "08"}, {"Sep", "09"},
            {"Oct", "10"}, {"Nov", "11"}, {"Dec", "12"}
        };

        // 查找月份缩写并返回对应的数字
        auto it = monthMap.find(month);
        if (it != monthMap.end()) {
            return it->second;
        } else {
            return "Invalid month";
        }
    }
    time_t start_t;           //可打印的时间格式  UNIX时间戳
    char *start_t_loacal;     //可打印的时间格式  当地日期时间戳
    struct tm *satrt_t_utc;   //UTC时间戳
    std::string start_t_loacal_ymdhms;
 private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};
#endif //TIC_TOC_H