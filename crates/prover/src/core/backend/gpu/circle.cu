typedef unsigned int uint32_t;

extern "C"
__global__ void sort_values(uint32_t *from, uint32_t *dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if(idx < (size >> 1)) {
            dst[idx] = from[idx << 1];
        } else {
            int tmp = idx - (size >> 1);
            dst[idx] = from[size - (tmp << 1) - 1];
        }
    }
}