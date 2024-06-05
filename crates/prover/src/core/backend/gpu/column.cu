// These functions assume `bits` is at most 32

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

const uint32_t P = 2147483647;

__device__ uint32_t mul_m31(uint32_t a, uint32_t b) {
    // TODO: replace with a faster algorithm
    return ((uint64_t) a * (uint64_t) b) % P;
}

__device__ uint64_t pow_to_power_of_two(int n, uint32_t t) {
    int i = 0;
    while(i < n) {
        t = mul_m31(t, t);
        i++;
    }
    return t;
}

__device__ uint32_t inv_m31(uint32_t t) {
    uint64_t t0 = mul_m31(pow_to_power_of_two(2, t), t);
    uint64_t t1 = mul_m31(pow_to_power_of_two(1, t0), t0);
    uint64_t t2 = mul_m31(pow_to_power_of_two(3, t1), t0);
    uint64_t t3 = mul_m31(pow_to_power_of_two(1, t2), t0);
    uint64_t t4 = mul_m31(pow_to_power_of_two(8, t3), t3);
    uint64_t t5 = mul_m31(pow_to_power_of_two(8, t4), t3);
    return mul_m31(pow_to_power_of_two(7, t5), t2);
}

__device__ void new_forward_layer(uint32_t *from, uint32_t *dst, int index) {
    dst[index] = mul_m31(from[index << 1], from[(index << 1) + 1]);
}

__device__ void new_backward_layer(uint32_t *from, uint32_t *dst, int index) {
    int temp = dst[index << 1];
    dst[index << 1] = mul_m31(from[index], dst[(index << 1) + 1]);
    dst[(index << 1) + 1] = mul_m31(from[index], temp);
}

extern "C"
__global__ void batch_inverse(uint32_t *from, uint32_t *dst, uint32_t *inner_tree, int size, int log_size) {
    // Montgomery's trick.
    int index = threadIdx.x;

    if(size >= 2048) {
        size = 2048;
        log_size = 11;
    }

    from = &from[2 * blockIdx.x * blockDim.x];
    dst = &dst[2 * blockIdx.x * blockDim.x];
    inner_tree = &inner_tree[2 * blockIdx.x * blockDim.x];

    size = size >> 1;

    // Forward Pass
    if(index < size) {
        new_forward_layer(from, inner_tree, index);
    }
    int from_offset = 0;
    int dst_offset = size;
    size >>= 1;

    int step = 1;
    while(step < log_size) {
        __syncthreads();
        if(index < size) {
            new_forward_layer(&inner_tree[from_offset], &inner_tree[dst_offset], index);
        }
        from_offset = dst_offset;
        dst_offset = dst_offset + size;
        size >>= 1;
        step++;
    }

    // Compute inverse of cumulative product
    __syncthreads();
    if(index == 0){
        inner_tree[dst_offset - 1] = inv_m31(inner_tree[dst_offset - 1]);
    }
    

    // Backward Pass
    step = 0;
    size = 1;
    from_offset = dst_offset - 1;
    dst_offset = from_offset - (size << 1);
    while(step < log_size - 1) {
        __syncthreads();
        if(index < size) {
            new_backward_layer(&inner_tree[from_offset], &inner_tree[dst_offset], index);
        }
        size <<= 1;
        from_offset = dst_offset;
        dst_offset = from_offset - (size << 1);
        step++;
    }
    
    __syncthreads();
    if(index < size) {
        dst[index << 1] = mul_m31(inner_tree[index], from[(index << 1) + 1]);
        dst[(index << 1) + 1] = mul_m31(inner_tree[index], from[index << 1]);
    }
}
