#ifndef ZTRACET_H
#define ZTRACET_H

#include "global.h"
#include <iostream>
#include <fstream>  //写入文件
#include <string>
//打印至文件 ztracet()
/**
 * @brief 打印信息至文件
 * @param 信息级别 信息格式  对应变量列表 无变量使用nullptr
*/
bool ztracet(const int level, const std::string& format)  //无参数列表的情况
{
    if (global_argc >= 2)
    {
        if (level == atoi(global_argv[global_argc-1]))
        {
            std::string filePath = LogFile;
            if (!foutC.is_open())
            {
                foutC.open(filePath, std::ios::app);  //打开logPath,std::ios::ate标志位表示文件打开后定位到文件末尾
                if (!foutC.is_open()) 
                {
                    std::cerr << "无法打开文件：" << filePath << std::endl;
                    return false;
                }   
            }
            //开始写入
            char buffer[TRACESIZE];
            snprintf(buffer, sizeof(buffer), format.c_str(), nullptr);  //格式化写入
            foutC << buffer;
        }
    }
    return true;
}

#endif  //ZTRACET_H