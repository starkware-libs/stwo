#include "qm31.cuh"

extern "C" __global__ void mul(unsigned int *lhs, unsigned int *rhs, unsigned int *out, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        unsigned int idx = tid * 4; 
        mul_qm31(lhs + idx, rhs + idx, out + idx);
    }
}

extern "C" __global__ void is_zero(unsigned int *arr, bool *res, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size && arr[tid]) 
        *res = false; 
}