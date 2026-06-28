#ifndef SINGLE_KARATSUBA_N_8_H
#define SINGLE_KARATSUBA_N_8_H

#include "fields.cuh"
#include "utils.cuh"

// CUDA version SingleKaratsubaN8::evaluate
// translated from cairo-air/src/components/subroutines/single_karatsuba_n_8.rs
// Pure polynomial multiplication, no relation lookups

DEVICE_FORCEINLINE void single_karatsuba_n_8_evaluate(
    // 32 input limbs: x[0-15], y[16-31]
    m31 *input,  // input[0-31]
    // 31 output limbs
    m31 *output  // output[0-30]
) {
    // z0 = x_low[0-7] * y_low[16-23], 15 limbs
    m31 z0[15];
    z0[0] = mul(input[0], input[16]);
    z0[1] = add(mul(input[0], input[17]), mul(input[1], input[16]));
    z0[2] = add(add(mul(input[0], input[18]), mul(input[1], input[17])), mul(input[2], input[16]));
    z0[3] = add(add(add(mul(input[0], input[19]), mul(input[1], input[18])), mul(input[2], input[17])), mul(input[3], input[16]));
    z0[4] = add(add(add(add(mul(input[0], input[20]), mul(input[1], input[19])), mul(input[2], input[18])), mul(input[3], input[17])), mul(input[4], input[16]));
    z0[5] = add(add(add(add(add(mul(input[0], input[21]), mul(input[1], input[20])), mul(input[2], input[19])), mul(input[3], input[18])), mul(input[4], input[17])), mul(input[5], input[16]));
    z0[6] = add(add(add(add(add(add(mul(input[0], input[22]), mul(input[1], input[21])), mul(input[2], input[20])), mul(input[3], input[19])), mul(input[4], input[18])), mul(input[5], input[17])), mul(input[6], input[16]));
    z0[7] = add(add(add(add(add(add(add(mul(input[0], input[23]), mul(input[1], input[22])), mul(input[2], input[21])), mul(input[3], input[20])), mul(input[4], input[19])), mul(input[5], input[18])), mul(input[6], input[17])), mul(input[7], input[16]));
    z0[8] = add(add(add(add(add(add(mul(input[1], input[23]), mul(input[2], input[22])), mul(input[3], input[21])), mul(input[4], input[20])), mul(input[5], input[19])), mul(input[6], input[18])), mul(input[7], input[17]));
    z0[9] = add(add(add(add(add(mul(input[2], input[23]), mul(input[3], input[22])), mul(input[4], input[21])), mul(input[5], input[20])), mul(input[6], input[19])), mul(input[7], input[18]));
    z0[10] = add(add(add(add(mul(input[3], input[23]), mul(input[4], input[22])), mul(input[5], input[21])), mul(input[6], input[20])), mul(input[7], input[19]));
    z0[11] = add(add(add(mul(input[4], input[23]), mul(input[5], input[22])), mul(input[6], input[21])), mul(input[7], input[20]));
    z0[12] = add(add(mul(input[5], input[23]), mul(input[6], input[22])), mul(input[7], input[21]));
    z0[13] = add(mul(input[6], input[23]), mul(input[7], input[22]));
    z0[14] = mul(input[7], input[23]);

    // z2 = x_high[8-15] * y_high[24-31], 15 limbs
    m31 z2[15];
    z2[0] = mul(input[8], input[24]);
    z2[1] = add(mul(input[8], input[25]), mul(input[9], input[24]));
    z2[2] = add(add(mul(input[8], input[26]), mul(input[9], input[25])), mul(input[10], input[24]));
    z2[3] = add(add(add(mul(input[8], input[27]), mul(input[9], input[26])), mul(input[10], input[25])), mul(input[11], input[24]));
    z2[4] = add(add(add(add(mul(input[8], input[28]), mul(input[9], input[27])), mul(input[10], input[26])), mul(input[11], input[25])), mul(input[12], input[24]));
    z2[5] = add(add(add(add(add(mul(input[8], input[29]), mul(input[9], input[28])), mul(input[10], input[27])), mul(input[11], input[26])), mul(input[12], input[25])), mul(input[13], input[24]));
    z2[6] = add(add(add(add(add(add(mul(input[8], input[30]), mul(input[9], input[29])), mul(input[10], input[28])), mul(input[11], input[27])), mul(input[12], input[26])), mul(input[13], input[25])), mul(input[14], input[24]));
    z2[7] = add(add(add(add(add(add(add(mul(input[8], input[31]), mul(input[9], input[30])), mul(input[10], input[29])), mul(input[11], input[28])), mul(input[12], input[27])), mul(input[13], input[26])), mul(input[14], input[25])), mul(input[15], input[24]));
    z2[8] = add(add(add(add(add(add(mul(input[9], input[31]), mul(input[10], input[30])), mul(input[11], input[29])), mul(input[12], input[28])), mul(input[13], input[27])), mul(input[14], input[26])), mul(input[15], input[25]));
    z2[9] = add(add(add(add(add(mul(input[10], input[31]), mul(input[11], input[30])), mul(input[12], input[29])), mul(input[13], input[28])), mul(input[14], input[27])), mul(input[15], input[26]));
    z2[10] = add(add(add(add(mul(input[11], input[31]), mul(input[12], input[30])), mul(input[13], input[29])), mul(input[14], input[28])), mul(input[15], input[27]));
    z2[11] = add(add(add(mul(input[12], input[31]), mul(input[13], input[30])), mul(input[14], input[29])), mul(input[15], input[28]));
    z2[12] = add(add(mul(input[13], input[31]), mul(input[14], input[30])), mul(input[15], input[29]));
    z2[13] = add(mul(input[14], input[31]), mul(input[15], input[30]));
    z2[14] = mul(input[15], input[31]);

    // x_sum = x_low + x_high, 8 limbs
    m31 x_sum[8];
    for (int i = 0; i < 8; i++) {
        x_sum[i] = add(input[i], input[i + 8]);
    }

    // y_sum = y_low + y_high, 8 limbs
    m31 y_sum[8];
    for (int i = 0; i < 8; i++) {
        y_sum[i] = add(input[16 + i], input[24 + i]);
    }

    // z1_product = x_sum * y_sum, compute inline and combine with z0 and z2
    // output[0-7] = z0[0-7]
    for (int i = 0; i < 8; i++) {
        output[i] = z0[i];
    }

    // output[8-14] = z0[8-14] + (x_sum * y_sum - z0 - z2)[0-6]
    output[8] = add(z0[8], sub(sub(mul(x_sum[0], y_sum[0]), z0[0]), z2[0]));

    output[9] = add(z0[9], sub(sub(add(mul(x_sum[0], y_sum[1]), mul(x_sum[1], y_sum[0])), z0[1]), z2[1]));

    output[10] = add(z0[10], sub(sub(add(add(mul(x_sum[0], y_sum[2]), mul(x_sum[1], y_sum[1])), mul(x_sum[2], y_sum[0])), z0[2]), z2[2]));

    output[11] = add(z0[11], sub(sub(add(add(add(mul(x_sum[0], y_sum[3]), mul(x_sum[1], y_sum[2])), mul(x_sum[2], y_sum[1])), mul(x_sum[3], y_sum[0])), z0[3]), z2[3]));

    output[12] = add(z0[12], sub(sub(add(add(add(add(mul(x_sum[0], y_sum[4]), mul(x_sum[1], y_sum[3])), mul(x_sum[2], y_sum[2])), mul(x_sum[3], y_sum[1])), mul(x_sum[4], y_sum[0])), z0[4]), z2[4]));

    output[13] = add(z0[13], sub(sub(add(add(add(add(add(mul(x_sum[0], y_sum[5]), mul(x_sum[1], y_sum[4])), mul(x_sum[2], y_sum[3])), mul(x_sum[3], y_sum[2])), mul(x_sum[4], y_sum[1])), mul(x_sum[5], y_sum[0])), z0[5]), z2[5]));

    output[14] = add(z0[14], sub(sub(add(add(add(add(add(add(mul(x_sum[0], y_sum[6]), mul(x_sum[1], y_sum[5])), mul(x_sum[2], y_sum[4])), mul(x_sum[3], y_sum[3])), mul(x_sum[4], y_sum[2])), mul(x_sum[5], y_sum[1])), mul(x_sum[6], y_sum[0])), z0[6]), z2[6]));

    // output[15] = (x_sum * y_sum)[7] - z0[7] - z2[7]
    output[15] = sub(sub(add(add(add(add(add(add(add(mul(x_sum[0], y_sum[7]), mul(x_sum[1], y_sum[6])), mul(x_sum[2], y_sum[5])), mul(x_sum[3], y_sum[4])), mul(x_sum[4], y_sum[3])), mul(x_sum[5], y_sum[2])), mul(x_sum[6], y_sum[1])), mul(x_sum[7], y_sum[0])), z0[7]), z2[7]);

    // output[16-22] = z2[0-6] + (x_sum * y_sum)[8-14] - z0[8-14] - z2[8-14]
    output[16] = add(z2[0], sub(sub(add(add(add(add(add(add(mul(x_sum[1], y_sum[7]), mul(x_sum[2], y_sum[6])), mul(x_sum[3], y_sum[5])), mul(x_sum[4], y_sum[4])), mul(x_sum[5], y_sum[3])), mul(x_sum[6], y_sum[2])), mul(x_sum[7], y_sum[1])), z0[8]), z2[8]));

    output[17] = add(z2[1], sub(sub(add(add(add(add(add(mul(x_sum[2], y_sum[7]), mul(x_sum[3], y_sum[6])), mul(x_sum[4], y_sum[5])), mul(x_sum[5], y_sum[4])), mul(x_sum[6], y_sum[3])), mul(x_sum[7], y_sum[2])), z0[9]), z2[9]));

    output[18] = add(z2[2], sub(sub(add(add(add(add(mul(x_sum[3], y_sum[7]), mul(x_sum[4], y_sum[6])), mul(x_sum[5], y_sum[5])), mul(x_sum[6], y_sum[4])), mul(x_sum[7], y_sum[3])), z0[10]), z2[10]));

    output[19] = add(z2[3], sub(sub(add(add(add(mul(x_sum[4], y_sum[7]), mul(x_sum[5], y_sum[6])), mul(x_sum[6], y_sum[5])), mul(x_sum[7], y_sum[4])), z0[11]), z2[11]));

    output[20] = add(z2[4], sub(sub(add(add(mul(x_sum[5], y_sum[7]), mul(x_sum[6], y_sum[6])), mul(x_sum[7], y_sum[5])), z0[12]), z2[12]));

    output[21] = add(z2[5], sub(sub(add(mul(x_sum[6], y_sum[7]), mul(x_sum[7], y_sum[6])), z0[13]), z2[13]));

    output[22] = add(z2[6], sub(sub(mul(x_sum[7], y_sum[7]), z0[14]), z2[14]));

    // output[23-30] = z2[7-14]
    for (int i = 0; i < 8; i++) {
        output[23 + i] = z2[7 + i];
    }
}

#endif // SINGLE_KARATSUBA_N_8_H
