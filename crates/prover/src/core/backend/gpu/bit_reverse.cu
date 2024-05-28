__device__ unsigned int bit_reverse(unsigned int n, int bits) {
    unsigned int reversed_n = __brev(n);
    return reversed_n >> (32 - bits);
}

extern "C"
__global__ void kernel(int *d_array, int size, int bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

        unsigned int rev_idx = bit_reverse(idx, bits);

        if (rev_idx > idx && idx < size) {
            int temp = d_array[idx];
            d_array[idx] = d_array[rev_idx];
            d_array[rev_idx] = temp;
        }
}