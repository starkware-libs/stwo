// m31.h
#ifndef M31_H
#define M31_H

    __device__ unsigned int mul_m31(unsigned int lhs, unsigned int rhs);
    __device__ void reduce_m31(unsigned int *f);
    __device__ unsigned int sub_m31(unsigned int lhs, unsigned int rhs);
    __device__ void neg_m31(unsigned int *f);
    __device__ unsigned int add_m31(unsigned int lhs, unsigned int rhs);

#endif 