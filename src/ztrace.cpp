#ifndef ZTRACE_H
#define ZTRACE_H
#include "global.h"
#include <iostream>
#include <string>
//打印至终端 ztrace()
/**
 * @brief 打印信息至终端
 * @param 信息级别 信息格式  对应变量列表 无变量使用nullptr
*/
bool ztrace(const int level, const std::string& format)  //无参数列表的情况
{
    if (global_argc >= 2)
    {
        if (level == atoi(global_argv[global_argc-1]))
        {
            //开始打印
            char buffer[TRACESIZE];
            snprintf(buffer, sizeof(buffer), format.c_str(), nullptr);  //格式化写入
            std::cout << buffer;  
        }
    }
    return true;
}

#endif  //ZTRACE_H