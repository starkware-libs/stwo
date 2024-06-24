#include "qm31.cuh"

extern "C" __global__ void mul(unsigned int *lhs, unsigned int *rhs, unsigned int *out, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        unsigned int idx = tid * 4; 
        mul_qm31(lhs + idx, rhs + idx, out + idx);
    }
}

