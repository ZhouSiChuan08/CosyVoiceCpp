#include "global.h"
#include <cstdlib>  //引入std::exit
#include <thread>   //引入std::this_thread::sleep_for

/**
 * @brief 从终端获取指令【quit】以停止程序 同时关闭文件指针以保存文件
 * @return None
 */
void zquit()
{
    while (true) {
        std::cout << "\033[0;34m \nEnter a command ( 'zquit' to quit):\n \033[0m";
        std::string command;
        std::cin >> command;
        if (command == "zquit") 
        {
            //关闭ztracet文件
            if (foutC.is_open())
            {
                foutC.close();
            }
            //关闭其他log文件
            for (std::ofstream* fout : fouts) 
            {
                if (fout->is_open()) 
                {
                    fout->close();
                }
            }
            std::exit(0);
            break;
        } 
        else
        {
            std::cout << "\033[0;35m wrong! I need [zquit].\033[0m";
        }
    }
}
/**
 * @brief 阻塞函数，用于阻止main返回
 * @return None
 */
void zblock()
{
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}
