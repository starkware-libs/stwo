#include "field.h"

template <typename T>
__device__ void upsweep(T *top_layer, T *bot_layer, int n)
{
    int idx0 = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx0 >= n)
        return;
    int idx1 = idx0 + n;
    bot_layer[idx0] = top_layer[idx0].mul(top_layer[idx1]);
}

template <typename T>
__device__ void downsweep(T *top_layer, T *bot_layer, T *dst_layer, int n)
{
    int idx0 = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx0 >= n)
        return;

    int idx1 = idx0 + n;
    auto inva0a1 = bot_layer[idx0];
    T temp = top_layer[idx0];
    dst_layer[idx0] = inva0a1.mul(top_layer[idx1]);
    dst_layer[idx1] = inva0a1.mul(temp);
}

// M31.
extern "C" __global__ void upsweep_m31_kernel(M31 *top_layer, M31 *bot_layer, int n)
{
    upsweep<M31>(top_layer, bot_layer, n);
}
extern "C" __global__ void downsweep_m31_kernel(M31 *top_layer, M31 *bot_layer, M31 *dst_layer, int n)
{
    downsweep<M31>(top_layer, bot_layer, dst_layer, n);
}

// QM31.
extern "C" __global__ void upsweep_qm31_kernel(QM31 *top_layer, QM31 *bot_layer, int n)
{
    upsweep<QM31>(top_layer, bot_layer, n);
}
extern "C" __global__ void downsweep_qm31_kernel(QM31 *top_layer, QM31 *bot_layer, QM31 *dst_layer, int n)
{
    downsweep<QM31>(top_layer, bot_layer, dst_layer, n);
}