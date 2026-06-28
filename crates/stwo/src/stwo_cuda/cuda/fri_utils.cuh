#ifndef FRI_UTILS_H
#define FRI_UTILS_H

#include "fields.cuh"
#include "utils.cuh"
DEVICE_FORCEINLINE const qm31 getEvaluation(m31 **eval_values, const uint32_t index) {
    return {{eval_values[0][index],
                    eval_values[1][index]},
            {eval_values[2][index],
                    eval_values[3][index]}};
}

#endif // FRI_UTILS_H