// m31.cuh
#ifndef M31_H
#define M31_H

__device__ unsigned int mul_m31(unsigned int lhs, unsigned int rhs);
__device__ void reduce_m31(unsigned int *f);
__device__ unsigned int sub_m31(unsigned int lhs, unsigned int rhs);
__device__ void neg_m31(unsigned int *f);
__device__ unsigned int add_m31(unsigned int lhs, unsigned int rhs);
extern "C" __global__ void is_zero(unsigned int *arr, bool *res, int size);

#endif 

__device__ __constant__ unsigned int MODULUS = (1 << 31) - 1; 

// TODO: Check if using Shared memory per block over device for optimizations
__device__  unsigned int mul_m31(unsigned int lhs, unsigned int rhs) {
    unsigned long long int a_e;
    unsigned long long int b_e;
    unsigned long long int prod_e;
    unsigned int prod_lows;
    unsigned int prod_highs;

    a_e = static_cast<unsigned long long int>(lhs);
    b_e = static_cast<unsigned long long int>(rhs);

    prod_e = a_e * b_e;
    
    prod_lows = static_cast<unsigned int>(prod_e & 0x7FFFFFFF);

    prod_highs = static_cast<unsigned int>(prod_e >> 31);

    // add 
    unsigned int out = prod_lows + prod_highs; 
    return min(out, out - MODULUS);
}

__device__ unsigned int add_m31(unsigned int lhs, unsigned int rhs) {
    unsigned int out = lhs + rhs; 
    return min(out, out - MODULUS);
}

__device__ void reduce_m31(unsigned int *f) {
    *f = min(*f, *f - MODULUS);
}

__device__  unsigned int sub_m31(unsigned int lhs, unsigned int rhs) {
    unsigned int out = lhs - rhs; 
    return min(out, out + MODULUS);
}

__device__  void neg_m31(unsigned int *f) {
    *f = MODULUS - *f;
}

extern "C" __global__ void is_zero(unsigned int *arr, bool *res, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size && arr[tid]) 
        *res = false; 
}