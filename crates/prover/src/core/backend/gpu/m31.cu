#include "m31.h"

extern "C" __global__ void mul(unsigned int *a, unsigned int *b, unsigned int *out, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        out[tid] = mul_m31(a[tid], b[tid]);
    }
}

extern "C" __global__ void add(unsigned int *a, unsigned int *b, unsigned int *out, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = add_m31(a[tid], b[tid]);
    }
}

extern "C" __global__ void reduce(unsigned int *out, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        reduce_m31(&out[tid]);
    }
}

extern "C" __global__ void sub(unsigned int *a, unsigned int *b, unsigned int *out, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        out[tid] = sub_m31(a[tid], b[tid]);
    }
}

extern "C" __global__ void neg(unsigned int *a, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        neg_m31(&a[tid]);
    }
}