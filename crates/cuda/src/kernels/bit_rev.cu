const int V_BITS = 5;

// Kernel is assumed to run with (32, 32) block size.
extern "C" __global__ void bit_rev_kernel(int *data, int m_bits)
{
    __shared__ int temp[1 << (2 * V_BITS)];
    int v_l = threadIdx.x;
    int v_lr = __brev(v_l) >> (32 - V_BITS);
    int v_h = threadIdx.y;
    int v_hr = __brev(v_h) >> (32 - V_BITS);
    int m = blockIdx.x;
    int mr = __brev(m) >> (32 - m_bits);

    // The current block bit reverses * m *, and * bitrev(m) *, and swaps them.
    if (m > mr)
    {
        // Exit the entire block.
        return;
    }

    // Coaslesced read into shared memory.
    temp[(v_lr << V_BITS) | v_hr] = data[(v_h << (V_BITS + m_bits)) | (m << V_BITS) | v_l];
    __syncthreads();

    int temp2 = data[(v_h << (V_BITS + m_bits)) | (mr << V_BITS) | v_l];
    data[(v_h << (V_BITS + m_bits)) | (mr << V_BITS) | v_l] = temp[(v_h << V_BITS) | v_l];
    temp[(v_h << V_BITS) | v_l] = temp2;

    if (m != mr)
    {
        __syncthreads();
        data[(v_h << (V_BITS + m_bits)) | (m << V_BITS) | v_l] = temp[(v_lr << V_BITS) | v_hr];
    }
}