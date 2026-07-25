#ifndef BATCH_INVERSE_H
#define BATCH_INVERSE_H

#include "fields.cuh"

extern "C"
void batch_inverse_base_field(m31 *from, m31 *dst, int size);

extern "C"
void batch_inverse_secure_field(qm31 *from, qm31 *dst, int size);

#endif // BATCH_INVERSE_H
