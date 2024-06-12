// These functions assume `bits` is at most 32

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

typedef struct {
    uint32_t a;
    uint32_t b;
} cm31;

typedef struct {
    cm31 a;
    cm31 b;
} qm31;

__device__ unsigned int bit_reverse(unsigned int n, int bits) {
    unsigned int reversed_n = __brev(n);
    return reversed_n >> (32 - bits);
}

template<typename T>
__global__ void bit_reverse_generic(T *array, int size, int bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int rev_idx = bit_reverse(idx, bits);

    if (rev_idx > idx && idx < size) {
        T temp = array[idx];
        array[idx] = array[rev_idx];
        array[rev_idx] = temp;
    }
}

extern "C"
__global__ void bit_reverse_basefield(uint32_t *array, int size, int bits) {
    return bit_reverse_generic(array, size, bits);
}

extern "C"
__global__ void bit_reverse_secure_field(qm31 *array, int size, int bits) {
    return bit_reverse_generic(array, size, bits);
}
