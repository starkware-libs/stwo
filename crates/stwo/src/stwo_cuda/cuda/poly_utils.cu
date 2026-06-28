#include "poly_utils.cuh"



__global__ void rescale(m31 *values, int size, m31 factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        values[idx] = mul(values[idx], factor);
    }
}
