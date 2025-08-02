#include "debugTools.h"
std::string getColorfullString(StringColor color, const std::string& str)
{
    switch (color)
    {
        case StringColor::RED:
            return "\033[31m" + str + "\033[0m";
        case StringColor::GREEN:
            return "\033[32m" + str + "\033[0m";
        case StringColor::YELLOW:
            return "\033[33m" + str + "\033[0m";
        case StringColor::BLUE:
            return "\033[34m" + str + "\033[0m";
        default:
            return str;
    }
}