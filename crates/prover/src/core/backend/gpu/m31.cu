#include "m31.h"

__device__ __constant__ int MODULUS = (1 << 31) - 1; 

// TODO: Check if using Shared memory per block over device for optimizations
extern "C" __global__  void mul_m31(unsigned int lhs, unsigned int rhs, unsigned int *out) {
    unsigned long long int a_e;
    unsigned long long int b_e;
    unsigned long long int prod_e;
    unsigned int prod_lows;
    unsigned int prod_highs;

    a_e = static_cast<unsigned long long int>(lhs);
    b_e = static_cast<unsigned long long int>(rhs);

    prod_e = a_e * b_e;
    
    // TODO:: look at optimizing through union (check performance)
    prod_lows = static_cast<unsigned int>(prod_e & 0x7FFFFFFF);

    prod_highs = static_cast<unsigned int>(prod_e >> 31);

    // add 
    *out = prod_lows + prod_highs; 
    *out = min(*out, *out - MODULUS);
}

extern "C" __global__  void add_m31(unsigned int lhs,  unsigned int rhs, unsigned int *out) {
    *out = lhs + rhs; 
    
    *out = min(*out, *out - MODULUS);
}

extern "C" __global__ void reduce_m31(unsigned int *f) {
    *f = min(*f, *f - MODULUS);
}

extern "C" __global__  void sub_m31(unsigned int lhs, unsigned int rhs, unsigned int *out) {
    *out = lhs - rhs; 
    *out = min(*out, *out + MODULUS);
}

extern "C" __global__  void neg_m31(unsigned int *f) {
    *f = MODULUS - *f;
}

extern "C" __global__ void mul(unsigned int *a, unsigned int *b, unsigned int *out, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        mul_m31(a[tid], b[tid], &out[tid]);
    }
}

extern "C" __global__ void add(unsigned int *a, unsigned int *b, unsigned int *out, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        add_m31(a[tid], b[tid], &out[tid]);
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
        sub_m31(a[tid], b[tid], &out[tid]);
    }
}

extern "C" __global__ void neg(unsigned int *a, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        neg_m31(&a[tid]);
    }
}