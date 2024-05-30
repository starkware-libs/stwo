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

extern "C"
__global__ void batch_inverse(uint32_t *A, uint32_t *B, uint32_t *C, int size, int log_size) {
    // Montgomery's trick.
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size) {
        // Compute 2 cumulative products.
        int step = 0;
        while(step < log_size) {
            // Check if this thread has to do something in this step.
            if((index >> step) % 2 == 1) {
                A[index] = mul_m31(A[index], A[((index >> step) << step) - 1]);
            } else {
                B[index] = mul_m31(B[index], B[((index >> step) + 1) << step]); // TODO: Be aware of warp diversions
            }
            step++;
        }

        // Multiply cumulative.
        if(index == size - 1) {
            C[index] = A[index - 1];
        }
        if(0 < index && index < size - 1) {
            C[index] = mul_m31(A[index - 1], B[index + 1]);
        }
        if(index == 0) {
            C[index] = B[1];
        }
        
        __shared__ uint32_t inv;
        // Invert the cumulative product of all.
        if(index == 0) {
           inv = inv_m31(A[size - 1]);
        }
        __syncthreads();

        C[index] = mul_m31(C[index], inv);
    }
}
