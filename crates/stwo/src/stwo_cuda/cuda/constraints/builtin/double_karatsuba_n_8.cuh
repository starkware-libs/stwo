#ifndef DOUBLE_KARATSUBA_N_8_H
#define DOUBLE_KARATSUBA_N_8_H

#include "fields.cuh"
#include "utils.cuh"
#include "single_karatsuba_n_8.cuh"

// CUDA version DoubleKaratsubaN8LimbMaxBound4095::evaluate
// translated from cairo-air/src/components/subroutines/double_karatsuba_n_8_limb_max_bound_4095.rs
// Pure polynomial multiplication, calls 3 instances of SingleKaratsubaN8

DEVICE_FORCEINLINE void double_karatsuba_n_8_evaluate(
    // 64 input limbs: x[0-31], y[32-63]
    m31 *input,   // input[0-63]
    // 63 output limbs
    m31 *output   // output[0-62]
) {
    // sk_4 = SingleKaratsubaN8(input[0-15], input[32-47])
    // x_low[0-15], y_low[32-47]
    m31 sk_4_input[32];
    for (int i = 0; i < 16; i++) {
        sk_4_input[i] = input[i];           // x_low
        sk_4_input[16 + i] = input[32 + i]; // y_low
    }
    m31 sk_4[31];
    single_karatsuba_n_8_evaluate(sk_4_input, sk_4);

    // sk_9 = SingleKaratsubaN8(input[16-31], input[48-63])
    // x_high[16-31], y_high[48-63]
    m31 sk_9_input[32];
    for (int i = 0; i < 16; i++) {
        sk_9_input[i] = input[16 + i];      // x_high
        sk_9_input[16 + i] = input[48 + i]; // y_high
    }
    m31 sk_9[31];
    single_karatsuba_n_8_evaluate(sk_9_input, sk_9);

    // x_sum = input[0-15] + input[16-31] (16 limbs)
    m31 x_sum[16];
    for (int i = 0; i < 16; i++) {
        x_sum[i] = add(input[i], input[16 + i]);
    }

    // y_sum = input[32-47] + input[48-63] (16 limbs)
    m31 y_sum[16];
    for (int i = 0; i < 16; i++) {
        y_sum[i] = add(input[32 + i], input[48 + i]);
    }

    // sk_16 = SingleKaratsubaN8(x_sum, y_sum)
    m31 sk_16_input[32];
    for (int i = 0; i < 16; i++) {
        sk_16_input[i] = x_sum[i];
        sk_16_input[16 + i] = y_sum[i];
    }
    m31 sk_16[31];
    single_karatsuba_n_8_evaluate(sk_16_input, sk_16);

    // Output calculation using Karatsuba combination
    // output[0-15] = sk_4[0-15]
    for (int i = 0; i < 16; i++) {
        output[i] = sk_4[i];
    }

    // output[16-30] = sk_4[16-30] + (sk_16 - sk_4 - sk_9)[0-14]
    for (int i = 0; i < 15; i++) {
        output[16 + i] = add(sk_4[16 + i], sub(sub(sk_16[i], sk_4[i]), sk_9[i]));
    }

    // output[31] = (sk_16 - sk_4 - sk_9)[15]
    output[31] = sub(sub(sk_16[15], sk_4[15]), sk_9[15]);

    // output[32-46] = sk_9[0-14] + (sk_16 - sk_4 - sk_9)[16-30]
    for (int i = 0; i < 15; i++) {
        output[32 + i] = add(sk_9[i], sub(sub(sk_16[16 + i], sk_4[16 + i]), sk_9[16 + i]));
    }

    // output[47-62] = sk_9[15-30]
    for (int i = 0; i < 16; i++) {
        output[47 + i] = sk_9[15 + i];
    }
}

#endif // DOUBLE_KARATSUBA_N_8_H
