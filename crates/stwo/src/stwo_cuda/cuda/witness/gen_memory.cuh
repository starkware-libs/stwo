
#ifndef GEN_MEMORY_H
#define GEN_MEMORY_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define LARGE_MEMORY_VALUE_ID_BASE 0x40000000U
#define DEFAULT_ID (LARGE_MEMORY_VALUE_ID_BASE - 1)

typedef struct {
    uint64_t lo;
    uint64_t hi;
} u128;

typedef enum {
    MEMORY_VALUE_ID_SMALL = 0,
    MEMORY_VALUE_ID_F252 = 1,
    MEMORY_VALUE_ID_EMPTY = 2
} MemoryValueIdTag;

typedef struct {
    MemoryValueIdTag tag;
    uint32_t value; // only valid if tag != EMPTY
} MemoryValueId;

typedef struct {
    uint32_t encoded;
} EncodedMemoryValueId;

HOST_DEVICE_FORCEINLINE EncodedMemoryValueId encode_memory_value_id(const MemoryValueId* mv) {
    EncodedMemoryValueId out;
    switch (mv->tag) {
        case MEMORY_VALUE_ID_SMALL:
            out.encoded = mv->value;
            break;
        case MEMORY_VALUE_ID_F252:
            out.encoded = mv->value | LARGE_MEMORY_VALUE_ID_BASE;
            break;
        case MEMORY_VALUE_ID_EMPTY:
            out.encoded = DEFAULT_ID;
            break;
        default:
            out.encoded = DEFAULT_ID;
            break;
    }
    return out;
}

HOST_DEVICE_FORCEINLINE MemoryValueId decode_memory_value_id(const EncodedMemoryValueId* emv) {
    MemoryValueId out;
    if (emv->encoded == DEFAULT_ID) {
        out.tag = MEMORY_VALUE_ID_EMPTY;
        out.value = 0;
        return out;
    }
    uint32_t tag = emv->encoded >> 30;
    uint32_t val = emv->encoded & 0x3FFFFFFFU;
    if (tag == 0) {
        out.tag = MEMORY_VALUE_ID_SMALL;
        out.value = val;
    } else if (tag == 1) {
        out.tag = MEMORY_VALUE_ID_F252;
        out.value = val;
    } else {
        out.tag = MEMORY_VALUE_ID_EMPTY;
        out.value = 0;
    }
    return out;
}

HOST_DEVICE_FORCEINLINE EncodedMemoryValueId encoded_memory_value_id_default() {
    EncodedMemoryValueId out;
    out.encoded = DEFAULT_ID;
    return out;
}

typedef enum {
    MEMORY_VALUE_SMALL = 0,
    MEMORY_VALUE_F252 = 1
} MemoryValueTag;

typedef struct {
    MemoryValueTag tag;
    union {
        u128 small;
        uint32_t f252[8];
    } value;
} MemoryValue;

HOST_DEVICE_FORCEINLINE void u128_to_4_limbs(u128 x, uint32_t out[4]) {
    out[0] = (uint32_t)(x.lo & 0xFFFFFFFFULL);
    out[1] = (uint32_t)((x.lo >> 32) & 0xFFFFFFFFULL);
    out[2] = (uint32_t)(x.hi & 0xFFFFFFFFULL);
    out[3] = (uint32_t)((x.hi >> 32) & 0xFFFFFFFFULL);
}

HOST_DEVICE_FORCEINLINE u128 memory_value_as_small(const MemoryValue* mv) {
    if (mv->tag != MEMORY_VALUE_SMALL) {
        fprintf(stderr, "Cannot convert F252 to u128\n");
        abort();
    }
    return mv->value.small;
}

HOST_DEVICE_FORCEINLINE void memory_value_as_u256(const MemoryValue* mv, uint32_t out[8]) {
    if (mv->tag == MEMORY_VALUE_SMALL) {
        uint32_t limbs[4];
        u128_to_4_limbs(mv->value.small, limbs);
        out[0] = limbs[0];
        out[1] = limbs[1];
        out[2] = limbs[2];
        out[3] = limbs[3];
        out[4] = out[5] = out[6] = out[7] = 0;
    } else if (mv->tag == MEMORY_VALUE_F252) {
        memcpy(out, mv->value.f252, sizeof(uint32_t) * 8);
    } else {
        fprintf(stderr, "Invalid MemoryValue tag\n");
        abort();
    }
}

#endif // GEN_MEMORY_H
