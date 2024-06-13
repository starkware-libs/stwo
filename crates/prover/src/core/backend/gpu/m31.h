// crypto_ops.h
#ifndef M31_H
#define M31_H

extern __device__ __constant__ int MODULUS;

extern "C" {
    __global__ void mul_m31(unsigned int *a, unsigned int *b, unsigned int *out);
    __global__ void add_m31(unsigned int *lhs, unsigned int *rhs, unsigned int *out);
    __global__ void reduce_m31(unsigned int *f);
    __global__ void sub_m31(unsigned int *lhs, unsigned int *rhs, unsigned int *out);
    __global__ void neg_m31(unsigned int *f);
}

#endif 