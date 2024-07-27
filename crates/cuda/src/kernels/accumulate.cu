#include "field.h"

extern "C" __global__ void accumulate_kernel(unsigned int *dst, unsigned int *src, unsigned int n)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= n)
        return;
    unsigned int cur = add31(dst[idx], src[idx]);
    dst[idx] = cur;
}