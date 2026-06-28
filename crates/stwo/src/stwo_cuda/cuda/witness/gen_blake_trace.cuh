
#ifndef GEN_BLAKE_TRACE_H
#define GEN_BLAKE_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"
#include "gen_blake_round_sigma_trace.cuh"

extern __constant__  m31 BLAKE_SIGMA_DEV[N_BLAKE_ROUNDS][N_BLAKE_SIGMA_COLS];;

DEVICE_FORCEINLINE void blake_round_sigma_deduce_output(
    m31 round,
    m31 *out_0, m31 *out_1, m31 *out_2,  m31 *out_3,  m31 *out_4,  m31 *out_5,  m31 *out_6,  m31 *out_7,
    m31 *out_8, m31 *out_9, m31 *out_10, m31 *out_11, m31 *out_12, m31 *out_13, m31 *out_14, m31 *out_15
) {
    *out_0  = BLAKE_SIGMA_DEV[round][0];
    *out_1  = BLAKE_SIGMA_DEV[round][1];
    *out_2  = BLAKE_SIGMA_DEV[round][2];
    *out_3  = BLAKE_SIGMA_DEV[round][3];
    *out_4  = BLAKE_SIGMA_DEV[round][4];
    *out_5  = BLAKE_SIGMA_DEV[round][5];
    *out_6  = BLAKE_SIGMA_DEV[round][6];
    *out_7  = BLAKE_SIGMA_DEV[round][7];
    *out_8  = BLAKE_SIGMA_DEV[round][8];
    *out_9  = BLAKE_SIGMA_DEV[round][9];
    *out_10 = BLAKE_SIGMA_DEV[round][10];
    *out_11 = BLAKE_SIGMA_DEV[round][11];
    *out_12 = BLAKE_SIGMA_DEV[round][12];
    *out_13 = BLAKE_SIGMA_DEV[round][13];
    *out_14 = BLAKE_SIGMA_DEV[round][14];
    *out_15 = BLAKE_SIGMA_DEV[round][15];
}



#endif // GEN_BLAKE_TRACE_H