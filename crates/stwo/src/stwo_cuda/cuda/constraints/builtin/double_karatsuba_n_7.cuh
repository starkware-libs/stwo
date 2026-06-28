#ifndef DOUBLE_KARATSUBA_N_7_H
#define DOUBLE_KARATSUBA_N_7_H

#include "fields.cuh"
#include "utils.cuh"
#include "single_karatsuba_n_7.cuh"

// CUDA version DoubleKaratsubaN7::evaluate
// translated from cairo-air/src/components/subroutines/double_karatsuba_n_7_limb_max_bound_511.rs
// Uses 3 instances of SingleKaratsubaN7 to implement large number multiplication

DEVICE_FORCEINLINE void double_karatsuba_n_7_evaluate(
    m31 *input,   // input[0-55]: x[0-27], y[28-55]
    m31 *output   // output[0-54]: 55 limbs
) {
    // sk_0 = SingleKaratsubaN7(x_low[0-13], y_low[28-41])
    m31 sk_0_input[28];
    for (int i = 0; i < 14; i++) {
        sk_0_input[i] = input[i];           // x_low
        sk_0_input[14 + i] = input[28 + i]; // y_low
    }
    m31 sk_0[27];
    single_karatsuba_n_7_evaluate(sk_0_input, sk_0);

    // sk_1 = SingleKaratsubaN7(x_high[14-27], y_high[42-55])
    m31 sk_1_input[28];
    for (int i = 0; i < 14; i++) {
        sk_1_input[i] = input[14 + i];      // x_high
        sk_1_input[14 + i] = input[42 + i]; // y_high
    }
    m31 sk_1[27];
    single_karatsuba_n_7_evaluate(sk_1_input, sk_1);

    // x_sum = x_low + x_high (14 limbs)
    m31 x_sum[14];
    for (int i = 0; i < 14; i++) {
        x_sum[i] = add(input[i], input[14 + i]);
    }

    // y_sum = y_low + y_high (14 limbs)
    m31 y_sum[14];
    for (int i = 0; i < 14; i++) {
        y_sum[i] = add(input[28 + i], input[42 + i]);
    }

    // sk_2 = SingleKaratsubaN7(x_sum, y_sum)
    m31 sk_2_input[28];
    for (int i = 0; i < 14; i++) {
        sk_2_input[i] = x_sum[i];
        sk_2_input[14 + i] = y_sum[i];
    }
    m31 sk_2[27];
    single_karatsuba_n_7_evaluate(sk_2_input, sk_2);

    // Karatsuba combination for output
    // output[0-13] = sk_0[0-13]
    for (int i = 0; i < 14; i++) {
        output[i] = sk_0[i];
    }

    // output[14-26] = sk_0[14-26] + (sk_2 - sk_0 - sk_1)[0-12]
    for (int i = 0; i < 13; i++) {
        output[14 + i] = add(sk_0[14 + i], sub(sub(sk_2[i], sk_0[i]), sk_1[i]));
    }

    // output[27] = (sk_2 - sk_0 - sk_1)[13]
    output[27] = sub(sub(sk_2[13], sk_0[13]), sk_1[13]);

    // output[28-40] = sk_1[0-12] + (sk_2 - sk_0 - sk_1)[14-26]
    for (int i = 0; i < 13; i++) {
        output[28 + i] = add(sk_1[i], sub(sub(sk_2[14 + i], sk_0[14 + i]), sk_1[14 + i]));
    }

    // output[41-54] = sk_1[13-26]
    for (int i = 0; i < 14; i++) {
        output[41 + i] = sk_1[13 + i];
    }
}

#endif // DOUBLE_KARATSUBA_N_7_H
