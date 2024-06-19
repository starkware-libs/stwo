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

const uint32_t P = 2147483647;
const cm31 R = {2, 1};

/*##### M31 ##### */

__device__ uint32_t mul(uint32_t a, uint32_t b) {
    // TODO: use mul from m31.cu
    uint64_t v = ((uint64_t) a * (uint64_t) b);
    uint64_t w = v + (v >> 31);
    uint64_t u = v + (w >> 31);
    return u & P;
}

__device__ uint32_t add(uint32_t a, uint32_t b) {
    // TODO: use add from m31.cu
    return ((uint64_t) a + (uint64_t) b) % P;
}

__device__ uint32_t sub(uint32_t a, uint32_t b) {
    // TODO: use sub from m31.cu
    return ((uint64_t) a + (uint64_t) (P - b)) % P;
}

__device__ uint32_t neg(uint32_t a) {
    // TODO: use neg from m31.cu
    return P - a;
}

__device__ uint64_t pow_to_power_of_two(int n, uint32_t t) {
    int i = 0;
    while(i < n) {
        t = mul(t, t);
        i++;
    }
    return t;
}

__device__ uint32_t inv(uint32_t t) {
    uint64_t t0 = mul(pow_to_power_of_two(2, t), t);
    uint64_t t1 = mul(pow_to_power_of_two(1, t0), t0);
    uint64_t t2 = mul(pow_to_power_of_two(3, t1), t0);
    uint64_t t3 = mul(pow_to_power_of_two(1, t2), t0);
    uint64_t t4 = mul(pow_to_power_of_two(8, t3), t3);
    uint64_t t5 = mul(pow_to_power_of_two(8, t4), t3);
    return mul(pow_to_power_of_two(7, t5), t2);
}

/*##### CM1 ##### */

__device__ cm31 mul(cm31 x, cm31 y) {
    return {sub(mul(x.a, y.a), mul(x.b, y.b)), add(mul(x.a, y.b), mul(x.b, y.a))};
}

__device__ cm31 add(cm31 x, cm31 y) {
    return {add(x.a, y.a), add(x.b, y.b)};
}

__device__ cm31 sub(cm31 x, cm31 y) {
    return {sub(x.a, y.a), sub(x.b, y.b)};
}

__device__ cm31 neg(cm31 x) {
    return {neg(x.a), neg(x.b)};
}

__device__ cm31 inv(cm31 t) {
    uint32_t factor = inv(add(mul(t.a, t.a), mul(t.b, t.b)));
    return {mul(t.a, factor), mul(neg(t.b) , factor)};
}

/*##### Q31 ##### */

__device__ qm31 mul(qm31 x, qm31 y) {
    return {add(mul(x.a, y.a), mul(R, mul(x.b, y.b))), add(mul(x.a, y.b), mul(x.b, y.a))};
}

__device__ qm31 inv(qm31 t) {
    cm31 b2 = mul(t.b, t.b);
    cm31 ib2 = {neg(b2.b), b2.a};
    cm31 denom = sub(mul(t.a, t.a), add(add(b2, b2),ib2));
    cm31 denom_inverse = inv(denom);
    return {mul(t.a, denom_inverse), neg(mul(t.b, denom_inverse))};
}

/*##### batch inverse ##### */

template<typename T>
__device__ void new_forward_level(T *from, T *dst, int index) {
    // Computes the value of the parent from the multiplication of two children.
    // dst  : Pointer to the beginning of the parent's level.
    // from : Pointer to the beginning of the children level.
    // index: Index of the computed parent.
    dst[index] = mul(from[index << 1], from[(index << 1) + 1]);
}

template<typename T>
__device__ void new_backward_level(T *from, T *dst, int index) {
    // Computes the inverse of the two children from the inverse of the parent.
    // dst  : Pointer to the beginning of the children's level.
    // from : Pointer to the beginning of the parent's level.
    // index: Index of the computed children.
    T temp = dst[index << 1];
    dst[index << 1] = mul(from[index], dst[(index << 1) + 1]);
    dst[(index << 1) + 1] = mul(from[index], temp);
}

template<typename T>
__global__ void batch_inverse(T *from, T *dst, int size, int log_size, T *s_from, T *s_inner_tree) {
    // Input:
    // - from      : array of uint32_t representing field elements in M31.
    // - inner_tree: array of uint32_t used as an auxiliary variable.
    // - size      : size of "from" and "inner_tree".
    // - log_size  : log(size).
    // Output:
    // - dst       : array of uint32_t with the inverses of "from".
    //
    // Variation of Montgomery's trick to leverage GPU parallelization.
    // Construct a binary tree:
    //    - from      : stores the leaves
    //    - inner_tree: stores the inner nodes and the root.
    // 
    // The algorithm has three parts:
    //    - Cumulative product: each parent is the product of its children.
    //    - Compute inverse of root node
    //    - Backward pass: compute inverses of children using the fact that
    //          inv_left_child  = inv_parent * right_child
    //          inv_right_child = inv_parent * left_child
    int index = threadIdx.x;

    s_from[index] = from[2 * blockIdx.x * blockDim.x + index];
    s_from[index + blockDim.x] = from[2 * blockIdx.x * blockDim.x + index + blockDim.x];
    __syncthreads();

    dst = &dst[2 * blockIdx.x * blockDim.x];

    // Size tracks the number of threads working.

    size = size >> 1;

    // The first level is a special case because inner_tree and leaves
    // are stored in separate variables.
    if(index < size) {
        new_forward_level(s_from, s_inner_tree, index);
        // from      : | a_0       | a_1       | ... | a_(n/2 - 1)       |      ...    | a_(n-1)
        // inner_tree: | a_0 * a_1 | a_2 * a_3 | ... | a_(n-2) * a_(n-1) | empty | ... | empty   
    }

    int from_offset = 0;   // Offset at inner_tree to get the children.
    int dst_offset = size; // Offset at inner_tree to store the parents.
    size >>= 1;            // Next level is half the size.

    // Each step will compute one level of the inner_tree.
    // If size = 4 inner tree stores:
    // |       Level 1         |        Root           |
    // | a_0 * a_1 | a_2 * a_3 | a_0 * a_1 * a_2 * a_3 |
    int step = 1;
    while(step < log_size) {
        __syncthreads();

        if(index < size) {
            // Each thread computes one parent as the product of left and right children
            new_forward_level(&s_inner_tree[from_offset], &s_inner_tree[dst_offset], index);
        }

        from_offset = dst_offset;       // Output of this level is input of next one.
        dst_offset = dst_offset + size; // Skip the number of nodes computed.

        size >>= 1; // Next level is half the size.
        step++;
    }

    // Compute inverse of the root.
    __syncthreads();
    if(index == 0){
        s_inner_tree[dst_offset - 1] = inv(s_inner_tree[dst_offset - 1]);
    }
    
    // Backward Pass: compute the inverses of the children using the parents.
    step = 0;
    size = 1;
    from_offset = dst_offset - 1;
    dst_offset = from_offset - (size << 1);
    while(step < log_size - 1) {
        __syncthreads();
        if(index < size) {
            // Compute children inverses from parent inverses.
            new_backward_level(&s_inner_tree[from_offset], &s_inner_tree[dst_offset], index);
        }

        size <<= 1; // Each level doubles up its size.

        from_offset = dst_offset;               // Output of this level is input of next one.
        dst_offset = from_offset - (size << 1); // Size threads work but 2*size children are computed.

        step++;
    }
    
    __syncthreads();
    // The inner_tree has all its inverses computed, now
    // we have to compute the inverses of the leaves:
    
    if(index < size) {
        dst[index << 1] = mul(s_inner_tree[index], s_from[(index << 1) + 1]);
        dst[(index << 1) + 1] = mul(s_inner_tree[index], s_from[index << 1]);
    }
}

extern "C"
__global__ void batch_inverse_basefield(uint32_t *from, uint32_t *dst, int size, int log_size) {
    // Thread syncing happens within a block. 
    // Split the problem to feed them to multiple blocks.
    if(size >= 512) {
        size = 512;
        log_size = 9;
    }

    extern __shared__ uint32_t shared_basefield[];
    uint32_t *s_from_basefield = shared_basefield;
    uint32_t *s_inner_trees_basefield = &shared_basefield[size];

    batch_inverse(from, dst, size, log_size, s_from_basefield, s_inner_trees_basefield);
}

extern "C"
__global__ void batch_inverse_secure_field(qm31 *from, qm31 *dst, int size, int log_size) {
    // Thread syncing happens within a block. 
    // Split the problem to feed them to multiple blocks.
    if(size >= 1024) {
        size = 1024;
        log_size = 10;
    }

    extern __shared__ qm31 shared_qm31[];
    qm31 *s_from_qm31 = shared_qm31;
    qm31 *s_inner_trees_qm31 = &shared_qm31[size];
    batch_inverse(from, dst, size, log_size, s_from_qm31, s_inner_trees_qm31);
}