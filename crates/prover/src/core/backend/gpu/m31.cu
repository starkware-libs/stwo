
__device__ __constant__ int MODULUS = (1 << 31) - 1; 

// TODO: Use Shared memory per block over device
extern "C" __global__  void mul_m31(unsigned int *a, unsigned int *b, unsigned int *out) {
    // unsigned int = u32
    // unsigned long long int = u64
    unsigned long long int a_e;
    unsigned long long int b_e;
    unsigned long long int prod_e;
    unsigned int prod_lows;
    unsigned int prod_highs;
    // replace 
    // Double b value
    //b_dbl <<= 1; 

    // Set up a word s.t. the lower half of each 64-bit word has the even 32-bit words of
    // the first operand.
    a_e = static_cast<unsigned long long int>(*a);
    b_e = static_cast<unsigned long long int>(*b);

    prod_e = a_e * b_e;
    
    // TODO:: look at optimizing through union (check performance)
   // prod_lows = static_cast<unsigned int>(prod_e_dbl & 0xFFFFFFFF);
    prod_lows = static_cast<unsigned int>(prod_e & 0x7FFFFFFF);

    //prod_lows >>= 1; 
    //prod_highs = static_cast<unsigned int>(prod_e_dbl >> 32);
    prod_highs = static_cast<unsigned int>(prod_e >> 31);

    // add 
    *out = prod_lows + prod_highs; 
    *out = min(*out, *out - MODULUS);
}

extern "C" __global__  void add_m31(unsigned int *lhs,  unsigned int *rhs, unsigned int *out) {
    *out = *lhs + *rhs; 
    
    *out = min(*out, *out - MODULUS);
}

extern "C" __global__ void reduce_m31(unsigned int *f) {
    *f = min(*f, *f - MODULUS);
}

extern "C" __global__  void sub_m31(unsigned int *lhs, unsigned int *rhs, unsigned int *out) {
    *out = *lhs - *rhs; 
    *out = min(*out, *out + MODULUS);
}

extern "C" __global__  void neg_m31(unsigned int *f) {
    *f = MODULUS - *f;
}

// Make sure size is equal to thread count
extern "C" __global__ void mul(unsigned int *a, unsigned int *b, unsigned int *out, int size) {
    unsigned int tid = threadIdx.x;

    if (tid < size) {
        mul_m31(&a[tid], &b[tid], &out[tid]);
    }
}

extern "C" __global__ void add(unsigned int *a, unsigned int *b, unsigned int *out, int size) {
    unsigned int tid = threadIdx.x;

    if (tid < size) {
        add_m31(&a[tid], &b[tid], &out[tid]);
    }
}

extern "C" __global__ void reduce(unsigned int *out, int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        reduce_m31(&out[tid]);
    }
}

extern "C" __global__ void sub(unsigned int *a, unsigned int *b, unsigned int *out, int size) {
    unsigned int tid = threadIdx.x;

    if (tid < size) {
        sub_m31(&a[tid], &b[tid], &out[tid]);
    }
}

extern "C" __global__ void neg(unsigned int *a, int size) {
    unsigned int tid = threadIdx.x;

    if (tid < size) {
        neg_m31(&a[tid]);
    }
}