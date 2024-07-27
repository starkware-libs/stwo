#include "field.h"

extern "C" __global__ void accumulate_kernel(M31 *dst, M31 *src, unsigned int n)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= n)
        return;
    M31 cur = dst[idx].add(src[idx]);
    dst[idx] = cur;
}