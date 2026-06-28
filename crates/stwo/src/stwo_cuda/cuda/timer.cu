#include "timer.cuh"
#include "stdio.h"
void timer::start(const char *message)
{
    // this->start_time = std::chrono::high_resolution_clock::now();
    // printf("[TIME]: start %s\n", message);
    // fflush(stdout);
}
void timer::end(const char *message)
{
    // this->end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed = this->end_time - this->start_time;
    // printf("[TIME]: end   %s elapse: %lf ms\n", message, elapsed.count());
    // fflush(stdout);
}
timer::timer()
{
}

timer::~timer()
{
}