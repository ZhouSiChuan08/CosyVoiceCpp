#include <libconfig.h++>
#include <string>
#include <string.h>
#include "global.h"

//定义
int speed_GNSS;
int speed_SSR;
std::string file_GNSS;
std::string file_SSR;
std::string COM_GNSS;
std::string COM_SSR;


/**
 * @brief 读取配置文件
 * @param null
 * @return 是否成功读取 bool
*/
bool zoptions()
{
    libconfig::Config cfg;  //创建配置对象
    // 读取配置文件
    if (global_argc >= 2)
    {
        //判断cfg文件存在与格式
        try {
        cfg.readFile(global_argv[1]);
        } catch (const libconfig::FileIOException &fioex) {  //处理文件输入/输出相关的错误
            std::cerr << "I/O error while reading file." << std::endl;
            return(EXIT_FAILURE);
        } catch (const libconfig::ParseException &pex) {  //处理配置文件格式不正确的错误
            std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                << " - " << pex.getError() << std::endl;
            return(EXIT_FAILURE);
        }

        //开始读取配置
        //获取根设置
        const libconfig::Setting &root = cfg.getRoot();  //&root是引用变量 是获取到引用的别名
        //基本读取框架  root就是cfg中最外层变量名
        try
        {
            //输入文件及输出文件
            const libconfig::Setting &sendfileVar = root["sendfile"];
            sendfileVar.lookupValue("speed_GNSS", speed_GNSS);
            sendfileVar.lookupValue("speed_SSR", speed_SSR);
            sendfileVar.lookupValue("file_GNSS", file_GNSS);
            sendfileVar.lookupValue("file_SSR", file_SSR);
            sendfileVar.lookupValue("COM_GNSS", COM_GNSS);
            sendfileVar.lookupValue("COM_SSR", COM_SSR);
                 
        }
        catch(const libconfig::SettingNotFoundException &nfex) 
        {
            std::cerr << "Setting not found: " << nfex.what() << std::endl;
            return false;
        }
        return true;
    }
    else
    {
        return false;
    }
}