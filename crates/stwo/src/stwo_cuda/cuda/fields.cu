#include "fields.cuh"

__host__ __device__ m31 mul(m31 a, m31 b) {
    uint64_t v = ((uint64_t) a * (uint64_t) b);
    uint64_t w = v + (v >> 31);
    uint64_t u = v + (w >> 31);
    return u & P;
}

__host__ __device__ m31 add(m31 a, m31 b) {
    uint64_t sum = ((uint64_t) a + (uint64_t) b);
    return min(sum, sum - P);
}

__host__ __device__ m31 sub(m31 a, m31 b) {
    return add(a, P - b);
}

__host__ __device__ m31 neg(m31 a) {
    // P - 0 = P, but P should be normalized to 0 in M31
    // Without this check, neg(0) returns P which causes incorrect
    // results when used in mul() since mul() doesn't expect P as input
    m31 result = P - a;
    return result == P ? 0 : result;
}

__host__ __device__ m31 eq_m31(m31 a, m31 b) {
    return (a == b) ? 1 : 0;
}

__host__ __device__ uint64_t pow_to_power_of_two(int n, m31 t) {
    int i = 0;
    while (i < n) {
        t = mul(t, t);
        i++;
    }
    return t;
}

__host__ __device__ m31 inv(m31 t) {
    uint64_t t0 = mul(pow_to_power_of_two(2, t), t);
    uint64_t t1 = mul(pow_to_power_of_two(1, t0), t0);
    uint64_t t2 = mul(pow_to_power_of_two(3, t1), t0);
    uint64_t t3 = mul(pow_to_power_of_two(1, t2), t0);
    uint64_t t4 = mul(pow_to_power_of_two(8, t3), t3);
    uint64_t t5 = mul(pow_to_power_of_two(8, t4), t3);
    return mul(pow_to_power_of_two(7, t5), t2);
}

__host__ __device__ m31 square(m31 x) {
    return mul(x, x);
}

__host__ __device__ m31 div(m31 x, m31 y) {
    return mul(x, inv(y));
}

/*##### CM31 ##### */

__host__ __device__ cm31 mul(cm31 x, cm31 y) {
    return {sub(mul(x.a, y.a), mul(x.b, y.b)), add(mul(x.a, y.b), mul(x.b, y.a))};
}

__host__ __device__ cm31 add(cm31 x, cm31 y) {
    return {add(x.a, y.a), add(x.b, y.b)};
}

__host__ __device__ cm31 add(m31 x, cm31 y) {
    return {
            add(x, y.a),
            y.b,
    };}

__host__ __device__ cm31 mul(m31 x, cm31 y) {
    return {
            mul(x, y.a),
            mul(x, y.b),
    };
}

__host__ __device__ cm31 sub(cm31 x, cm31 y) {
    return {sub(x.a, y.a), sub(x.b, y.b)};
}

__host__ __device__ cm31 sub(m31 x, cm31 y) {
    return {sub(x, y.a), neg(y.b)};
}

__host__ __device__ cm31 neg(cm31 x) {
    return {neg(x.a), neg(x.b)};
}

__host__ __device__ cm31 inv(cm31 t) {
    m31 factor = inv(add(mul(t.a, t.a), mul(t.b, t.b)));
    return {mul(t.a, factor), mul(neg(t.b), factor)};
}

__host__ __device__ cm31 mul_by_scalar(cm31 x, m31 scalar) {
    return cm31 { mul(x.a, scalar), mul(x.b, scalar) };
}

__host__ __device__ cm31 div(cm31 x, m31 y) {
    return cm31{div(x.a, y), div(x.b, y)};
}

/*##### QM31 ##### */

__host__ __device__ qm31 mul(qm31 x, qm31 y) {
    // Karatsuba multiplication
    cm31 v0 = mul(x.a, y.a);
    cm31 v1 = mul(x.b, y.b);
    cm31 v2 = mul(add(x.a, x.b), add(y.a, y.b));
    return {
            add(v0, mul(R, v1)),
            sub(v2, add(v0, v1))
    };
}

__host__ __device__ qm31 mul(m31 x, qm31 y) {
    return {
            mul(x, y.a),
            mul(x, y.b),
    };
}

__host__ __device__ qm31 mul(qm31 x, cm31 y) {
    return { mul(x.a, y), mul(x.b, y) };
}


__host__ __device__ qm31 add(qm31 x, qm31 y) {
    return {add(x.a, y.a), add(x.b, y.b)};
}

__host__ __device__ qm31 add(m31 x, qm31 y) {
    return {
            add(x, y.a),
            y.b,
    };
}

__host__ __device__ qm31 sub(qm31 x, qm31 y) {
    return {sub(x.a, y.a), sub(x.b, y.b)};
}

__host__ __device__ qm31 sub(m31 x, qm31 y) {
    return {sub(x, y.a), neg(y.b)};
}

__host__ __device__ qm31 mul_by_scalar(qm31 x, m31 scalar) {
    return qm31 { mul_by_scalar(x.a, scalar), mul_by_scalar(x.b, scalar) };
}

__host__ __device__ qm31 inv(qm31 t) {
    cm31 b2 = mul(t.b, t.b);
    cm31 ib2 = {neg(b2.b), b2.a};
    cm31 denom = sub(mul(t.a, t.a), add(add(b2, b2), ib2));
    cm31 denom_inverse = inv(denom);
    return {mul(t.a, denom_inverse), neg(mul(t.b, denom_inverse))};
}

__host__ __device__ qm31 square(qm31 x) {
    return mul(x, x);
}

__host__ __device__ qm31 pow(qm31 x, uint64_t exp) {
    qm31 res = qm31{cm31{m31{1}, m31{0}}, cm31{m31{0}, m31{0}}};
    while (exp > 0) {
        if (exp & 1 == 1) {
            res = mul(res, x);
        }
        x = square(x);
        exp >>= 1;
    }
    return res;
}

__host__ __device__ qm31 div(qm31 x, m31 y) {
    return qm31 {
        div(x.a, y),
        div(x.b, y)
    };
}


__device__ m31 atomic_add(m31* address, m31 val) {
    unsigned int* addr_as_ui = reinterpret_cast<unsigned int*>(address);
    unsigned int old = *addr_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned int newVal = add(assumed, val);
        old = atomicCAS(addr_as_ui, assumed, newVal);
    } while (assumed != old);
        return old;
    }

__device__ void print_qm31_device(const qm31 data, const char *description) {
    printf("threadId: %d, %s: (%d + %di) + (%d + %di)u\n", blockIdx.x * blockDim.x + threadIdx.x, description, data.a.a, data.a.b, data.b.a, data.b.b);
}


__host__ __device__ void print_qm31(const qm31 data, const char *description) {
    printf("%s: (%d + %di) + (%d + %di)u\n", description, data.a.a, data.a.b, data.b.a, data.b.b);
}

__host__ __device__ void dump_m31_array_generic(const m31 *array, size_t size, const char *description) {
    if (array == NULL || description == NULL) {
        printf("invalid input。\n");
        return;
    }

    printf("%s: [", description);

    for (size_t i = 0; i < size; i++) {
        printf("M31(%d)", array[i]);
        if (i != size - 1) {
            printf(", ");
        }
    }

    printf("]\n");
}

__global__ void print_qm31_array(qm31 *array, int size) {
    for (int i = 0; i < size; i++) {
        print_qm31(array[i], "");
    }
}

__global__ void print_m31_array(m31 *array, int size) {
    dump_m31_array_generic(array, size, "");
}

__host__ __device__ m31 low_as_m31(uint32_t x) {
    return (m31) {0xffff & x};
}

__host__ __device__ m31 high_as_m31(uint32_t x) {
    return (m31) {x >> 16};
}
