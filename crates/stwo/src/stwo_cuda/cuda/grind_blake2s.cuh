#ifndef GRIND_BLAKE2S_H
#define GRIND_BLAKE2S_H

#include <cstdint>

extern "C"
uint64_t grind_blake2s(const uint32_t* prefixed_digest, uint32_t pow_bits);

#endif // GRIND_BLAKE2S_H
