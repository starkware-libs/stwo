#ifndef POLY_IFFT_H
#define POLY_IFFT_H

#include "fields.cuh"
#include "utils.cuh"
#include "poly_utils.cuh"

__device__ __forceinline__ void exchg_dif(m31 &a, m31 &b, const m31 &twiddle) {
    const auto a_tmp = a;
    a = a_tmp + b;
    b = a_tmp - b;
    b = b * twiddle;
}

#define LOG_THREADS_PER_WARP 5


static const size_t LAUNCH_B2N_CONFIG_13_18[6][2] = {
    {7, 6}, {8, 6}, {7, 8}, {8, 8}, {9, 8}, {10, 8}
};

static const size_t LAUNCH_B2N_CONFIG_19_24[6][3] = {
    {7, 6, 6}, // 19
    {8, 6, 6}, // 20
    {7, 6, 8}, // 21
    {8, 6, 8}, // 22
    {7, 8, 8}, // 23
    {8, 8, 8}, // 24
};

static const size_t LAUNCH_B2N_CONFIG_25_29[5][4] = {
    {7, 6, 6, 6}, // 25
    {8, 6, 6, 6}, // 26
    {7, 8, 6, 6}, //27
    {8, 8, 6, 6}, //28
    {7, 8, 8, 6}, //29
};

extern "C"
void interpolate(int eval_domain_size, m31 *values, m31 *inverse_twiddles_tree, int inverse_twiddles_size, int values_size);

extern "C"
void interpolate_columns(int eval_domain_size, m31 **values, m31 *inverse_twiddles_tree, int inverse_twiddles_size, int values_size, int number_of_rows);

#endif // POLY_IFFT_H