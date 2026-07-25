#ifndef ACCUMULATE_H
#define ACCUMULATE_H

#include "fields.cuh"

extern "C"
void accumulate(int size, m31 **left_columns, m31 **right_columns);

extern "C"
void lift_and_accumulate(
    int col_size,
    m31 *col_0, m31 *col_1, m31 *col_2, m31 *col_3,
    m31 *curr_0, m31 *curr_1, m31 *curr_2, m31 *curr_3,
    int log_ratio
);

#endif // ACCUMULATE_H