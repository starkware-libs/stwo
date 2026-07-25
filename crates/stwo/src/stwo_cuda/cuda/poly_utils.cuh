#ifndef POLY_UTILS_H
#define POLY_UTILS_H

#include "fields.cuh"
#include "utils.cuh"

DEVICE_FORCEINLINE m31 get_circle_twiddle(m31 *twiddles, int index) {
    int k = index >> 2;
    if (index % 4 == 0) {
        return twiddles[2 * k + 1];
    } else if (index % 4 == 1) {
        return neg(twiddles[2 * k + 1]);
    } else if (index % 4 == 2) {
        return neg(twiddles[2 * k]);
    } else {
        return twiddles[2 * k];
    }
}

__global__ void rescale(m31 *values, int size, m31 factor);

#endif // POLY_UTILS_H
