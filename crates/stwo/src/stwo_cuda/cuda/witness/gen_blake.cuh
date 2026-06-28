
#ifndef GEN_BLAKE_H
#define GEN_BLAKE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define NUM_INPUT_WORDS_G 6
#define NUM_OUTPUT_WORDS_G 4

HOST_DEVICE_FORCEINLINE uint32_t rotate_right(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// input: [a, b, c, d, m0, m1]
HOST_DEVICE_FORCEINLINE void blake_deduce_output(
    const uint32_t input[NUM_INPUT_WORDS_G],
    uint32_t output[NUM_OUTPUT_WORDS_G]
) {
    uint32_t a = input[0];
    uint32_t b = input[1];
    uint32_t c = input[2];
    uint32_t d = input[3];
    uint32_t m0 = input[4];
    uint32_t m1 = input[5];

    a = a + b + m0;
    d ^= a;
    d = rotate_right(d, 16);

    c += d;
    b ^= c;
    b = rotate_right(b, 12);

    a = a + b + m1;
    d ^= a;
    d = rotate_right(d, 8);

    c += d;
    b ^= c;
    b = rotate_right(b, 7);

    output[0] = a;
    output[1] = b;
    output[2] = c;
    output[3] = d;
}

#endif // GEN_BLAKE_H