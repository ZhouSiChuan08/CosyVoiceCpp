#ifndef DEBUGTOOLS_H
#define DEBUGTOOLS_H
#include <iostream>
#include <string>

#define DEBUGSTRSIZE 1024
extern int  global_argc;
extern char **global_argv;
enum class StringColor {
    RED,
    GREEN,
    YELLOW,
    BLUE
};
/**
 * @brief 让标准输出流的字符串带上颜色
 * @param color 枚举类型颜色
 * @return 颜色字符串
 */
std::string getColorfullString(StringColor color, const std::string& str);

/**
 * @brief 让标准输出流的字符串带上颜色
 */
template<typename... Args>
void stdCoutInColor(const int level,  StringColor color, const std::string& format, Args... args)
{
    if (global_argc >= 2)
    {
        if (level == atoi(global_argv[global_argc-1]))  // 最后一个参数决定打印等级
        {
            //开始打印
            char buffer[DEBUGSTRSIZE];
            snprintf(buffer, sizeof(buffer), format.c_str(), args...);  // 格式化写入
            std::string colorStr = getColorfullString(color, buffer);   // 获取颜色字符串
            std::cout << colorStr << std::flush;                        // 打印到标准输出流

        }
    }
}

/**
 * @brief 让标准错误流的字符串带上红色
 */
template<typename... Args>
void stdCerrInColor(const int level,  const std::string& format, Args... args)
{
    if (global_argc >= 2)
    {
        if (level == atoi(global_argv[global_argc-1]))  // 最后一个参数决定打印等级
        {
            //开始打印
            char buffer[DEBUGSTRSIZE];
            snprintf(buffer, sizeof(buffer), format.c_str(), args...);             // 格式化写入
            std::string colorStr = getColorfullString(StringColor::RED, buffer);   // 获取颜色字符串
            std::cerr << colorStr;                                                 // 打印到标准错误流
        }
    }
}

#endif // DEBUGTOOLS_H