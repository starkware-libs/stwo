#ifndef SINGLE_KARATSUBA_N_7_H
#define SINGLE_KARATSUBA_N_7_H

#include "fields.cuh"
#include "utils.cuh"

// CUDA version SingleKaratsubaN7::evaluate
// translated from cairo-air/src/components/subroutines/single_karatsuba_n_7.rs
// Pure polynomial multiplication, no relation lookups

DEVICE_FORCEINLINE void single_karatsuba_n_7_evaluate(
    // 28 input limbs: x[0-13], y[14-27]
    m31 *input,  // input[0-27]
    // 27 output limbs
    m31 *output  // output[0-26]
) {
    // z0 = x_low[0-6] * y_low[14-20], 13 limbs
    m31 z0[13];
    z0[0] = mul(input[0], input[14]);
    z0[1] = add(mul(input[0], input[15]), mul(input[1], input[14]));
    z0[2] = add(add(mul(input[0], input[16]), mul(input[1], input[15])), mul(input[2], input[14]));
    z0[3] = add(add(add(mul(input[0], input[17]), mul(input[1], input[16])), mul(input[2], input[15])), mul(input[3], input[14]));
    z0[4] = add(add(add(add(mul(input[0], input[18]), mul(input[1], input[17])), mul(input[2], input[16])), mul(input[3], input[15])), mul(input[4], input[14]));
    z0[5] = add(add(add(add(add(mul(input[0], input[19]), mul(input[1], input[18])), mul(input[2], input[17])), mul(input[3], input[16])), mul(input[4], input[15])), mul(input[5], input[14]));
    z0[6] = add(add(add(add(add(add(mul(input[0], input[20]), mul(input[1], input[19])), mul(input[2], input[18])), mul(input[3], input[17])), mul(input[4], input[16])), mul(input[5], input[15])), mul(input[6], input[14]));
    z0[7] = add(add(add(add(add(mul(input[1], input[20]), mul(input[2], input[19])), mul(input[3], input[18])), mul(input[4], input[17])), mul(input[5], input[16])), mul(input[6], input[15]));
    z0[8] = add(add(add(add(mul(input[2], input[20]), mul(input[3], input[19])), mul(input[4], input[18])), mul(input[5], input[17])), mul(input[6], input[16]));
    z0[9] = add(add(add(mul(input[3], input[20]), mul(input[4], input[19])), mul(input[5], input[18])), mul(input[6], input[17]));
    z0[10] = add(add(mul(input[4], input[20]), mul(input[5], input[19])), mul(input[6], input[18]));
    z0[11] = add(mul(input[5], input[20]), mul(input[6], input[19]));
    z0[12] = mul(input[6], input[20]);

    // z2 = x_high[7-13] * y_high[21-27], 13 limbs
    m31 z2[13];
    z2[0] = mul(input[7], input[21]);
    z2[1] = add(mul(input[7], input[22]), mul(input[8], input[21]));
    z2[2] = add(add(mul(input[7], input[23]), mul(input[8], input[22])), mul(input[9], input[21]));
    z2[3] = add(add(add(mul(input[7], input[24]), mul(input[8], input[23])), mul(input[9], input[22])), mul(input[10], input[21]));
    z2[4] = add(add(add(add(mul(input[7], input[25]), mul(input[8], input[24])), mul(input[9], input[23])), mul(input[10], input[22])), mul(input[11], input[21]));
    z2[5] = add(add(add(add(add(mul(input[7], input[26]), mul(input[8], input[25])), mul(input[9], input[24])), mul(input[10], input[23])), mul(input[11], input[22])), mul(input[12], input[21]));
    z2[6] = add(add(add(add(add(add(mul(input[7], input[27]), mul(input[8], input[26])), mul(input[9], input[25])), mul(input[10], input[24])), mul(input[11], input[23])), mul(input[12], input[22])), mul(input[13], input[21]));
    z2[7] = add(add(add(add(add(mul(input[8], input[27]), mul(input[9], input[26])), mul(input[10], input[25])), mul(input[11], input[24])), mul(input[12], input[23])), mul(input[13], input[22]));
    z2[8] = add(add(add(add(mul(input[9], input[27]), mul(input[10], input[26])), mul(input[11], input[25])), mul(input[12], input[24])), mul(input[13], input[23]));
    z2[9] = add(add(add(mul(input[10], input[27]), mul(input[11], input[26])), mul(input[12], input[25])), mul(input[13], input[24]));
    z2[10] = add(add(mul(input[11], input[27]), mul(input[12], input[26])), mul(input[13], input[25]));
    z2[11] = add(mul(input[12], input[27]), mul(input[13], input[26]));
    z2[12] = mul(input[13], input[27]);

    // x_sum = x_low + x_high, 7 limbs
    m31 x_sum[7];
    for (int i = 0; i < 7; i++) {
        x_sum[i] = add(input[i], input[i + 7]);
    }

    // y_sum = y_low + y_high, 7 limbs
    m31 y_sum[7];
    for (int i = 0; i < 7; i++) {
        y_sum[i] = add(input[14 + i], input[21 + i]);
    }

    // z1_product = x_sum * y_sum, compute inline and combine with z0 and z2
    // output[0-6] = z0[0-6]
    for (int i = 0; i < 7; i++) {
        output[i] = z0[i];
    }

    // output[7-12] = z0[7-12] + (x_sum * y_sum - z0 - z2)[0-5]
    output[7] = add(z0[7], sub(sub(mul(x_sum[0], y_sum[0]), z0[0]), z2[0]));

    output[8] = add(z0[8], sub(sub(add(mul(x_sum[0], y_sum[1]), mul(x_sum[1], y_sum[0])), z0[1]), z2[1]));

    output[9] = add(z0[9], sub(sub(add(add(mul(x_sum[0], y_sum[2]), mul(x_sum[1], y_sum[1])), mul(x_sum[2], y_sum[0])), z0[2]), z2[2]));

    output[10] = add(z0[10], sub(sub(add(add(add(mul(x_sum[0], y_sum[3]), mul(x_sum[1], y_sum[2])), mul(x_sum[2], y_sum[1])), mul(x_sum[3], y_sum[0])), z0[3]), z2[3]));

    output[11] = add(z0[11], sub(sub(add(add(add(add(mul(x_sum[0], y_sum[4]), mul(x_sum[1], y_sum[3])), mul(x_sum[2], y_sum[2])), mul(x_sum[3], y_sum[1])), mul(x_sum[4], y_sum[0])), z0[4]), z2[4]));

    output[12] = add(z0[12], sub(sub(add(add(add(add(add(mul(x_sum[0], y_sum[5]), mul(x_sum[1], y_sum[4])), mul(x_sum[2], y_sum[3])), mul(x_sum[3], y_sum[2])), mul(x_sum[4], y_sum[1])), mul(x_sum[5], y_sum[0])), z0[5]), z2[5]));

    // output[13] = (x_sum * y_sum)[6] - z0[6] - z2[6]
    output[13] = sub(sub(add(add(add(add(add(add(mul(x_sum[0], y_sum[6]), mul(x_sum[1], y_sum[5])), mul(x_sum[2], y_sum[4])), mul(x_sum[3], y_sum[3])), mul(x_sum[4], y_sum[2])), mul(x_sum[5], y_sum[1])), mul(x_sum[6], y_sum[0])), z0[6]), z2[6]);

    // output[14-19] = z2[0-5] + (x_sum * y_sum)[7-12] - z0[7-12] - z2[7-12]
    output[14] = add(z2[0], sub(sub(add(add(add(add(add(mul(x_sum[1], y_sum[6]), mul(x_sum[2], y_sum[5])), mul(x_sum[3], y_sum[4])), mul(x_sum[4], y_sum[3])), mul(x_sum[5], y_sum[2])), mul(x_sum[6], y_sum[1])), z0[7]), z2[7]));

    output[15] = add(z2[1], sub(sub(add(add(add(add(mul(x_sum[2], y_sum[6]), mul(x_sum[3], y_sum[5])), mul(x_sum[4], y_sum[4])), mul(x_sum[5], y_sum[3])), mul(x_sum[6], y_sum[2])), z0[8]), z2[8]));

    output[16] = add(z2[2], sub(sub(add(add(add(mul(x_sum[3], y_sum[6]), mul(x_sum[4], y_sum[5])), mul(x_sum[5], y_sum[4])), mul(x_sum[6], y_sum[3])), z0[9]), z2[9]));

    output[17] = add(z2[3], sub(sub(add(add(mul(x_sum[4], y_sum[6]), mul(x_sum[5], y_sum[5])), mul(x_sum[6], y_sum[4])), z0[10]), z2[10]));

    output[18] = add(z2[4], sub(sub(add(mul(x_sum[5], y_sum[6]), mul(x_sum[6], y_sum[5])), z0[11]), z2[11]));

    output[19] = add(z2[5], sub(sub(mul(x_sum[6], y_sum[6]), z0[12]), z2[12]));

    // output[20-26] = z2[6-12]
    for (int i = 0; i < 7; i++) {
        output[20 + i] = z2[6 + i];
    }
}

#endif // SINGLE_KARATSUBA_N_7_H
