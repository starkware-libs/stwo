// These functions assume `bits` is at most 32

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

const uint32_t P = 2147483647;

__device__ uint32_t mul_m31(uint32_t a, uint32_t b) {
    // TODO: use mul_m31 from m31.cu
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

__device__ void new_forward_level(uint32_t *from, uint32_t *dst, int index) {
    // Computes the value of the parent from the multiplication of two children.
    // dst  : Pointer to the beginning of the parent's level.
    // from : Pointer to the beginning of the children level.
    // index: Index of the computed parent.
    dst[index] = mul_m31(from[index << 1], from[(index << 1) + 1]);
}

__device__ void new_backward_level(uint32_t *from, uint32_t *dst, int index) {
    // Computes the inverse of the two children from the inverse of the parent.
    // dst  : Pointer to the beginning of the children's level.
    // from : Pointer to the beginning of the parent's level.
    // index: Index of the computed children.
    int temp = dst[index << 1];
    dst[index << 1] = mul_m31(from[index], dst[(index << 1) + 1]);
    dst[(index << 1) + 1] = mul_m31(from[index], temp);
}

extern "C"
__global__ void batch_inverse(uint32_t *from, uint32_t *dst, uint32_t *inner_tree, int size, int log_size) {
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

    // Thread syncing happens within a block. 
    // Split the problem to feed them to multiple blocks.
    if(size >= 2048) {
        size = 2048;
        log_size = 11;
    }

    from = &from[2 * blockIdx.x * blockDim.x];
    dst = &dst[2 * blockIdx.x * blockDim.x];
    inner_tree = &inner_tree[2 * blockIdx.x * blockDim.x];

    // Size tracks the number of threads working.
    size = size >> 1;

    // The first level is a special case because inner_tree and leaves
    // are stored in separate variables.
    if(index < size) {
        new_forward_level(from, inner_tree, index);
        // from      : | a_0       | a_1       | ... | a_(n/2 - 1)       | ...   | a_(n-1)
        // inner_tree: | a_0 * a_1 | a_2 * a_3 | ... | a_(n-2) * a_(n-1) | empty | empty   
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
            new_forward_level(&inner_tree[from_offset], &inner_tree[dst_offset], index);
        }

        from_offset = dst_offset;       // Output of this level is input of next one.
        dst_offset = dst_offset + size; // Skip the number of nodes computed.

        size >>= 1; // Next level is half the size.
        step++;
    }

    // Compute inverse of the root.
    __syncthreads();
    if(index == 0){
        inner_tree[dst_offset - 1] = inv_m31(inner_tree[dst_offset - 1]);
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
            new_backward_level(&inner_tree[from_offset], &inner_tree[dst_offset], index);
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
        dst[index << 1] = mul_m31(inner_tree[index], from[(index << 1) + 1]);
        dst[(index << 1) + 1] = mul_m31(inner_tree[index], from[index << 1]);
    }
}

