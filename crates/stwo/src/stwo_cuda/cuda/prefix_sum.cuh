#ifndef PREFIX_SUM
#define PREFIX_SUM

__global__ void circle_domain_order_to_coset_order_kernel(const m31* in, m31* out, int n);

__global__ void coset_order_to_circle_domain_order_kernel(const m31* d_in, m31* d_out, int n);

extern "C"
void inclusive_prefix_sum(m31 *device_bit_rev_circle_domain_evals, unsigned len);

#endif