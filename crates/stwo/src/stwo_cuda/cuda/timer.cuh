#pragma once
#include <chrono>
class timer
{
private:
    std::chrono::_V2::system_clock::time_point start_time;
    std::chrono::_V2::system_clock::time_point end_time;

public:
    void start(const char *message);
    void end(const char *message);
    timer();
    ~timer();
};