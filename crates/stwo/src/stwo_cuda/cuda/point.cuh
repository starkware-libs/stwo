#ifndef POINT_H
#define POINT_H

#include "fields.cuh"
#include "utils.cuh"

typedef struct {
    m31 x;
    m31 y;
} point;

typedef struct {
    qm31 x;
    qm31 y;
} secure_field_point;

__device__ const point m31_circle_gen = {2, 1268011823};

/*##### Point ##### */

HOST_DEVICE_FORCEINLINE point one() {
    return {1, 0};
}

HOST_DEVICE_FORCEINLINE point point_mul(point &p1, point &p2) {
    return {
        sub(mul(p1.x, p2.x), mul(p1.y, p2.y)),
        add(mul(p1.x, p2.y), mul(p1.y, p2.x)),
    };
}

HOST_DEVICE_FORCEINLINE point point_square(point &p1) {
    return point_mul(p1, p1);
}

HOST_DEVICE_FORCEINLINE point point_pow(point p, int exponent) {
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


HOST_DEVICE_FORCEINLINE point point_inv(point &p) {
    return {p.x, neg(p.y)};
}

HOST_DEVICE_FORCEINLINE point point_neg(point &p) {
    return {p.x, neg(p.y)};
}

__host__ __forceinline__ point point_of_order(int n) {
    point result = m31_circle_gen;
    int log_exponent = 31 - log_2(n);
    while (log_exponent > 0) {
        result = point_square(result);
        log_exponent--;
    }
    return result;
}

#endif // POINT_H