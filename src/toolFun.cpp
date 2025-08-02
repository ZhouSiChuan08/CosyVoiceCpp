#include "toolFun.h"

json LoadValueFromJsonFile(const std::string& filePath, const std::string& key) 
{
    std::ifstream file(filePath);
    json config;
    try
    {
        file >> config;
        return config[key];
    }
    catch(const std::exception& e)
    {
        stdCerrInColor(1, "In LoadValueFromJsonFile, 解析配置文件变量 %s 出错, 主动退出程序: ",  key.c_str(), e.what());
        exit(1);
    }
}

std::string getLocalTimestampForFilename() {
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    
    // 线程安全地转换为本地时间结构
    std::tm local_tm;
    #ifdef _WIN32
        localtime_s(&local_tm, &now_time);  // Windows线程安全版本
    #else
        localtime_r(&now_time, &local_tm);  // POSIX线程安全版本
    #endif
    
    // 格式化为字符串 (YYYYMMDD_HHMMSS)
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
    
    // 添加毫秒（保证唯一性）
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ) % 1000;
    oss << "_" << std::setfill('0') << std::setw(3) << milliseconds.count();
    
    return oss.str();
}

TicToc::TicToc()
{
    // 创建就计时
    tic();  
}

void TicToc::tic()
{
    start = std::chrono::system_clock::now();  //开始时刻
    //获取不同类型的时间
    start_t = std::chrono::system_clock::to_time_t(start);  //time_t类型 UNIX时间戳
    start_t_loacal = ctime(&start_t);  //本地时间
    satrt_t_utc = gmtime(&start_t);  //UTC标准时间
    //获取当地时间字符串
    getLocalt();
}

double TicToc::toc()
{
    end = std::chrono::system_clock::now();  //结束时刻
    std::chrono::duration<double> elapsed_seconds = end - start;  //持续时长
    return elapsed_seconds.count() * 1000;  //毫秒
}

void TicToc::getLocalt()
{
    std::string start_t_loacal_string(start_t_loacal);
    starTimeStamp = start_t_loacal_string.substr(20, 4) + monthToNumber(start_t_loacal_string.substr(4, 3)) 
                            + start_t_loacal_string.substr(8, 2) + start_t_loacal_string.substr(11, 2) 
                            + start_t_loacal_string.substr(14, 2) + start_t_loacal_string.substr(17, 2);
}

std::string TicToc::monthToNumber(const std::string& month)
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

uint64_t getRandSeed()
{
    uint64_t seed = 0;  // 默认为0

    // 初始化随机数源
    std::random_device rd;
    
    // 初始化随机数生成器
    std::mt19937_64 gen(rd()); // 64位版本
    
    // 定义均匀分布范围 [0, UINT64_MAX]
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    seed = dist(gen);  // 产生随机数种子
    return seed;
}