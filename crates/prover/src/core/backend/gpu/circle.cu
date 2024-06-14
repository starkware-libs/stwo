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
    uint64_t v = ((uint64_t) a * (uint64_t) b);
    uint64_t w = v + (v >> 31);
    uint64_t u = v + (w >> 31);
    return u & P;}

__device__ uint32_t m31_add(uint32_t a, uint32_t b) {
    // TODO: use add from m31.cu
    return ((uint64_t) a + (uint64_t) b) % P;
}

__device__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    // TODO: use sub from m31.cu
    return ((uint64_t) a + (uint64_t) (P - b)) % P;
}

__device__ uint32_t m31_neg(uint32_t a) {
    // TODO: use sub from m31.cu
    return P - a;
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
__global__ void put_one(uint32_t *dst, int offset) {
    dst[offset] = 1;
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

    // TODO: when size is larger than the max number of concurrent threads,
    //       consecutive numbers can me computed with a multiplication within the same thread,
    //       instead of using another pow.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    size >>= 1;
    if (idx < size) {
        point pow = point_pow(step, bit_reverse(idx, log_size - 1));
        dst[offset + idx] = point_mul(initial, pow).x;
    }
}

__device__ int get_twiddle(uint32_t *twiddles, int index) {
    int k = index >> 2;
    if (index % 4 == 0) {
        return twiddles[2 * k + 1];
    } else if (index % 4 == 1) {
        return m31_neg(twiddles[2 * k + 1]);
    } else if (index % 4 == 2) {
        return m31_neg(twiddles[2 * k]);
    } else {
        return twiddles[2 * k];
    }
}

extern "C"
__global__ void ifft_circle_part(uint32_t *values, uint32_t *inverse_twiddles_tree, int values_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    if (idx < (values_size >> 1)) {
        uint32_t val0 = values[2 * idx];
        uint32_t val1 = values[2 * idx + 1];
        uint32_t twiddle = get_twiddle(inverse_twiddles_tree, idx);
        
        values[2 * idx] = m31_add(val0, val1);
        values[2 * idx + 1] = m31_mul(m31_sub(val0, val1), twiddle);
    }
}


extern "C"
__global__ void ifft_line_part(uint32_t *values, uint32_t *inverse_twiddles_tree, int values_size, int inverse_twiddles_size, int layer_domain_offset, int layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        int number_polynomials = 1 << layer;
        int h = idx / number_polynomials;
        int l = idx % number_polynomials;
        int idx0 = (h << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        uint32_t val0 = values[idx0];
        uint32_t val1 = values[idx1];
        uint32_t twiddle = inverse_twiddles_tree[layer_domain_offset + h];
        
        values[idx0] = m31_add(val0, val1);
        values[idx1] = m31_mul(m31_sub(val0, val1), twiddle);
    }
}

extern "C"
__global__ void rfft_circle_part(uint32_t *values, uint32_t *inverse_twiddles_tree, int values_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    if (idx < (values_size >> 1)) {
        uint32_t val0 = values[2 * idx];
        uint32_t val1 = values[2 * idx + 1];
        uint32_t twiddle = get_twiddle(inverse_twiddles_tree, idx);
        
        uint32_t temp = m31_mul(val1, twiddle);
        
        values[2 * idx] = m31_add(val0, temp);
        values[2 * idx + 1] = m31_sub(val0, temp);
    }
}


extern "C"
__global__ void rfft_line_part(uint32_t *values, uint32_t *inverse_twiddles_tree, int values_size, int inverse_twiddles_size, int layer_domain_offset, int layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        int number_polynomials = 1 << layer;
        int h = idx / number_polynomials;
        int l = idx % number_polynomials;
        int idx0 = (h << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        uint32_t val0 = values[idx0];
        uint32_t val1 = values[idx1];
        uint32_t twiddle = inverse_twiddles_tree[layer_domain_offset + h];
        
        uint32_t temp = m31_mul(val1, twiddle);
        
        values[idx0] = m31_add(val0, temp);
        values[idx1] = m31_sub(val0, temp);
    }
}

extern "C"
__global__ void rescale(uint32_t *values, int size, uint32_t factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size) {
        values[idx] = m31_mul(values[idx], factor);
    }
}