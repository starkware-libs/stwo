
#ifndef GEN_BLAKE_ROUND_SIGMA_TRACE_H
#define GEN_BLAKE_ROUND_SIGMA_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define N_BLAKE_ROUNDS 10
#define N_BLAKE_SIGMA_COLS 16

const m31 BLAKE_SIGMA[N_BLAKE_ROUNDS][N_BLAKE_SIGMA_COLS] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 }
};

const m31 SIGMA_DEDUCE_SOURCE[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0};
const m31 SIGMA_SEQ[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

extern "C"
void blake_round_sigma_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 **mults,
    unsigned mults_col_size,
    unsigned mults_row_log_size
);

extern "C"
void generate_blake_round_sigma_interaction_traces(
    void *blake_round_sigma,

    m31 **lookup_blake_round_sigma,

    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);

void init_blake_round_sigma_constants_only_once();
#endif // GEN_BLAKE_ROUND_SIGMA_TRACE_H