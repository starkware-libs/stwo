#include "field.h"

extern "C" __global__ void upsweep_kernel(unsigned int *src, unsigned int *dst, int n)
{
    int idx0 = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx0 >= n)
        return;
    int idx1 = idx0 + n;
    // a0' = a0 * a1 .
    src[idx1] = dst[idx0];
    dst[idx0] = mul31(dst[idx0], dst[idx1]);
}

extern "C" __global__ void downsweep_kernel(unsigned int *src, unsigned int *dst, int n)
{
    int idx0 = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx0 >= n)
        return;
    int idx1 = idx0 + n;
    // a0'' = inv(a0') * a1 .
    // a1'' = inv(a0') * a0 .
    unsigned int inva0a1 = dst[idx0];
    dst[idx0] = mul31(inva0a1, dst[idx1]);
    dst[idx1] = mul31(inva0a1, src[idx1]);
}