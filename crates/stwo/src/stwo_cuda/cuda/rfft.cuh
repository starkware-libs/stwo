#ifndef POLY_RFFT_H
#define POLY_RFFT_H

#include "fields.cuh"

#define LOG_WARP 5

static const size_t LAUNCH_N2B_CONFIG_13_19[7][2] = {
    {6, 7}, // 13
    {6, 8}, // 14
    {8, 7},  //15
    {8, 8},  //16
    {6, 11}, //17
    {8, 10},// 18
    {8, 11} //19
};

static const size_t LAUNCH_N2B_CONFIG_20_27[8][3] = {
    {6, 6, 8},
    {6, 8, 7},
    {6, 8, 8},
    {8, 8, 7},
    {8, 8, 8},
    {6, 8, 11},
    {8, 8, 10},
    {8, 8, 11}
};

static const size_t LAUNCH_N2B_CONFIG_28_30[3][4] = {
    {6, 6, 6, 10},
    {6, 6, 6, 11},
    {6, 6, 8, 10}
};

extern "C"
void evaluate(int eval_domain_size, m31 *values, m31 *twiddles_tree, int twiddles_size, int values_size);

extern "C"
void evaluate_columns(const int *eval_domain_sizes, m31 **values, m31 *twiddles_tree, int twiddles_size, int number_of_columns, const int *column_sizes);

#endif // POLY_RFFT_H