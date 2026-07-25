#ifndef FIELDS_H
#define FIELDS_H

#include <cstdint>
#include <iostream>
typedef unsigned int uint32_t;
typedef uint32_t m31;
// typedef unsigned long long uint64_t;

typedef struct {
    m31 a;
    m31 b;
} cm31;

typedef struct {
    cm31 a;
    cm31 b;
} qm31;

__device__ const m31 P = 2147483647;
__device__ const cm31 R = {2, 1};

struct M31 {
    m31 value;

    __host__ __device__ M31() : value(0) { }
    __host__ __device__ M31(m31 v) : value(v) { }

    __host__ __device__ M31 operator+(const M31& other) const {
        uint64_t sum = (uint64_t)value + (uint64_t)other.value;
        return M31(sum < P ? (m31)sum : (m31)(sum - P));
    }
};

#define SECURE_EXTENSION_DEGREE 4
/*##### M31 ##### */

__host__ __device__ m31 mul(m31 a, m31 b);

__host__ __device__ m31 add(m31 a, m31 b);

__host__ __device__ m31 sub(m31 a, m31 b);

__host__ __device__ m31 neg(m31 a);

__host__ __device__ m31 eq_m31(m31 a, m31 b);

__host__ __device__ uint64_t pow_to_power_of_two(int n, m31 t);

__host__ __device__ m31 inv(m31 t);

__host__ __device__ m31 square(m31 x);

__host__ __device__ m31 div(m31 x, m31 y);

/*##### CM31 ##### */

__host__ __device__ cm31 mul(cm31 x, cm31 y);

__host__ __device__ cm31 add(cm31 x, cm31 y);

__host__ __device__ cm31 add(m31 x, cm31 y);

__host__ __device__ cm31 mul(m31 x, cm31 y);

__host__ __device__ cm31 sub(cm31 x, cm31 y);

__host__ __device__ cm31 sub(m31 x, cm31 y);

__host__ __device__ cm31 neg(cm31 x);

__host__ __device__ cm31 inv(cm31 t);

__host__ __device__ cm31 mul_by_scalar(cm31 x, m31 scalar);

__host__ __device__ cm31 div(cm31 x, m31 y);

__host__ __device__ m31 low_as_m31(uint32_t x);

__host__ __device__ m31 high_as_m31(uint32_t x);

/*##### QM31 ##### */

__host__ __device__ qm31 mul(qm31 x, qm31 y);

__host__ __device__ qm31 mul(qm31 x, cm31 y);

__host__ __device__ qm31 mul(m31 x, qm31 y);

__host__ __device__ qm31 add(qm31 x, qm31 y);

__host__ __device__ qm31 add(m31 x, qm31 y);

__host__ __device__ qm31 sub(qm31 x, qm31 y);

__host__ __device__ qm31 sub(m31 x, qm31 y);

__host__ __device__ qm31 inv(qm31 t);

__host__ __device__ qm31 mul_by_scalar(qm31 x, m31 scalar);

__host__ __device__ qm31 pow(qm31 x, uint64_t exp);

__host__ __device__ qm31 square(qm31 x);

__host__ __device__ qm31 div(qm31 x, m31 y);

__host__ __device__ void print_qm31(const qm31 data, const char *description);
__device__ void print_qm31_device(const qm31 data, const char *description);

__device__ m31 atomic_add(m31* address, m31 val);

__host__ __device__ void dump_m31_array_generic(const m31 *array, size_t size, const char *description);

__global__ void print_qm31_array(qm31 *array, int size);

__global__ void print_m31_array(m31 *array, int size);

#endif // FIELDS_H
