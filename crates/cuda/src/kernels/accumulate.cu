const int P = (1 << 31) - 1;
extern "C" __global__ void accumulate_kernel(int *dst, int *src)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int cur = dst[idx] + src[idx];
    cur = (cur > P) ? cur - P : cur;
    dst[idx] = cur;
}