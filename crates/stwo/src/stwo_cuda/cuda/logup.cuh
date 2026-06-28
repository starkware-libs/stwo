#ifndef LOGUP_H
#define LOGUP_H

#include "fields.cuh"
#include "utils.cuh"

template <int N>
struct LookupElementsBasic {
    qm31 z;
    qm31 alpha;
    qm31 alpha_powers[N];

    HOST_DEVICE_FORCEINLINE LookupElementsBasic() {
        z = qm31{{0, 0}, {0, 0}};
        alpha = qm31{{0, 0}, {0, 0}};
        for (int i = 0; i < N; ++i) {
            alpha_powers[i] = qm31{{0, 0}, {0, 0}};
        }
    }

    HOST_DEVICE_FORCEINLINE static LookupElementsBasic<N> dummy() {
        LookupElementsBasic<N> elements;
        // Match Rust: z = SecureField::from_u32_unchecked(1, 2, 3, 4)
        elements.z = qm31{{1, 2}, {3, 4}};
        elements.alpha = qm31{{1, 0}, {0, 0}};
        for (int i = 0; i < N; ++i) {
            elements.alpha_powers[i] = qm31{{1, 0}, {0, 0}};
        }
        return elements;
    }

    HOST_DEVICE_FORCEINLINE qm31 combine(const m31* values, int num_values) const {
        qm31 result = qm31{cm31{0, 0}, cm31{0, 0}};

        for (int i = 0; i < num_values; i++) {
            qm31 term = mul(alpha_powers[i], qm31{cm31{values[i], 0}, cm31{0, 0}});
            result = add(result, term);
        }

        result = sub(result, z);

        return result;
    }
};

HOST_DEVICE_FORCEINLINE void logup_col_write_frac(
    unsigned vec_index,
    qm31 numerator,
    qm31 denom,
    qm31 *denominator,
    m31 *numerator_0,
    m31 *numerator_1,
    m31 *numerator_2,
    m31 *numerator_3
) {
    numerator_0[vec_index] = numerator.a.a;
    numerator_1[vec_index] = numerator.a.b;
    numerator_2[vec_index] = numerator.b.a;
    numerator_3[vec_index] = numerator.b.b;
    denominator[vec_index] = denom;
}

#endif // LOGUP_H
