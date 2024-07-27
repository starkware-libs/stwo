#include "field.h"

template <typename T>
__device__ void upsweep(T *src, T *dst, int n)
{
    int idx0 = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx0 >= n)
        return;
    int idx1 = idx0 + n;
    // a0' = a0 * a1 .
    src[idx1] = dst[idx0];
    dst[idx0] = dst[idx0].mul(dst[idx1]);
}

template <typename T>
__device__ void downsweep(T *src, T *dst, int n)
{
    int idx0 = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx0 >= n)
        return;
    int idx1 = idx0 + n;
    // a0'' = inv(a0') * a1 .
    // a1'' = inv(a0') * a0 .
    auto inva0a1 = dst[idx0];
    dst[idx0] = inva0a1.mul(dst[idx1]);
    dst[idx1] = inva0a1.mul(src[idx1]);
}

// M31.
extern "C" __global__ void upsweep_m31_kernel(M31 *src, M31 *dst, int n)
{
    upsweep<M31>(src, dst, n);
}
extern "C" __global__ void downsweep_m31_kernel(M31 *src, M31 *dst, int n)
{
    downsweep<M31>(src, dst, n);
}

// QM31.
extern "C" __global__ void upsweep_qm31_kernel(QM31 *src, QM31 *dst, int n)
{
    upsweep<QM31>(src, dst, n);
}
extern "C" __global__ void downsweep_qm31_kernel(QM31 *src, QM31 *dst, int n)
{
    downsweep<QM31>(src, dst, n);
}