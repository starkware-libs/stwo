typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

const int LOG_MAX_NUM_CONCURRENT_THREADS = 14;
const uint32_t P = 2147483647;

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

typedef struct {
    uint32_t x;
    uint32_t y;
} point;

__device__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    // TODO: use mul from m31.cu
    return ((uint64_t) a * (uint64_t) b) % P;
}

__device__ uint32_t m31_add(uint32_t a, uint32_t b) {
    // TODO: use add from m31.cu
    return ((uint64_t) a + (uint64_t) b) % P;
}

__device__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    // TODO: use sub from m31.cu
    return ((uint64_t) a + (uint64_t) (P - b)) % P;
}

__device__ point point_mul(point &p1, point &p2) {
    return {
        m31_sub(m31_mul(p1.x, p2.x), m31_mul(p1.y, p2.y)),
        m31_add(m31_mul(p1.x, p2.y), m31_mul(p1.y, p2.x)),
    };
}

__device__ point point_square(point &p1) {
    return point_mul(p1, p1);
}

__device__ point one() {
    return {1, 0};
}

__device__ point pow_to_power_of_two(point p, int log_exponent) {
    int i = 0;
    while (i < log_exponent) {
        p = point_square(p);
        i++;
    }
    return p;
}

__device__ point point_pow(point p, int exponent) {
    point result = one();
    while (exponent > 0) {
        if (exponent & 1) {
            result = point_mul(p, result);
        }
        p = point_square(p);
        exponent >>= 1;
    }
    return result;
}

const point m31_circle_gen = {2, 1268011823};

__device__ unsigned int bit_reverse(unsigned int n, int bits) {
    unsigned int reversed_n = __brev(n);
    return reversed_n >> (32 - bits);
}

extern "C"
__global__ void precompute_twiddles(uint32_t *dst, point initial, point step, int offset, int size, int log_size) {
    // Computes one level of twiddles for a particular Coset.
    //      dst: twiddles array.
    //  initial: coset factor.
    //     step: generator of the group.
    //   offset: store values in dst[offset]
    //     size: coset size
    // log_size: log(size)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    size >>= 1;
    if (idx < size) {
        point pow = point_pow(step, bit_reverse(idx, log_size));
        dst[offset + idx] = point_mul(initial, pow).x;
    }
}
