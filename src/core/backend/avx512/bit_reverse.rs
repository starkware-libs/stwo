use std::arch::x86_64::{__m512i, _mm512_permutex2var_epi32};

use crate::core::fields::m31::BaseField;

const VEC_BITS: u32 = 4;
const W_BITS: u32 = 3;
const MIN_LOG_SIZE: u32 = 2 * W_BITS + VEC_BITS;

pub fn bit_reverse_m31(data: &mut [[BaseField; 16]]) {
    assert!(data.len().is_power_of_two());
    assert!(data.len().ilog2() >= MIN_LOG_SIZE);

    // V W1 A W0 [V]

    let data_bits = data.len().ilog2();
    let a_bits = data_bits - 2 * W_BITS - VEC_BITS;
    // TODO: if threading, over a.
    // TODO: Go over a in an L2/L3 cache friendly way.

    // Total needed cache size: 2*2^(W_BITS+VEC_BITS) = 2^15 B = 32KB.
    for a in 0u32..(1 << a_bits) {
        for w0 in 0u32..(1 << W_BITS) {
            for w1 in 0u32..(1 << W_BITS) {
                let idx = (((w1 << a_bits) | a) << W_BITS) | w0;
                let idxr = idx.reverse_bits() >> (32 - (data_bits - VEC_BITS));
                if idx > idxr {
                    continue;
                }

                let values0 = std::array::from_fn(|i| {
                    data[(idx + ((i as u32) << (2 * W_BITS + a_bits))) as usize]
                });
                let values0 = bit_reverse16(values0);

                if idx == idxr {
                    // Palindrome.
                    for i in 0..16 {
                        data[(idx + ((i as u32) << (2 * W_BITS + a_bits))) as usize] =
                            values0[i as usize];
                    }
                    continue;
                }
                let values1 = std::array::from_fn(|i| {
                    data[(idxr + ((i as u32) << (2 * W_BITS + a_bits))) as usize]
                });
                let values1 = bit_reverse16(values1);

                for i in 0..16 {
                    data[(idx + ((i as u32) << (2 * W_BITS + a_bits))) as usize] =
                        values1[i as usize];
                    data[(idxr + ((i as u32) << (2 * W_BITS + a_bits))) as usize] =
                        values0[i as usize];
                }
            }
        }
    }
}

#[allow(dead_code)]
fn bit_reverse16(data: [[BaseField; 16]; 16]) -> [[BaseField; 16]; 16] {
    let mut data: [__m512i; 16] = unsafe { std::mem::transmute(data) };
    // abcd0123 => 0abc123d
    const L: __m512i = unsafe {
        core::mem::transmute([
            0b00000, 0b10000, 0b00001, 0b10001, 0b00010, 0b10010, 0b00011, 0b10011, 0b00100,
            0b10100, 0b00101, 0b10101, 0b00110, 0b10110, 0b00111, 0b10111,
        ])
    };
    const H: __m512i = unsafe {
        core::mem::transmute([
            0b01000, 0b11000, 0b01001, 0b11001, 0b01010, 0b11010, 0b01011, 0b11011, 0b01100,
            0b11100, 0b01101, 0b11101, 0b01110, 0b11110, 0b01111, 0b11111,
        ])
    };
    for _ in 0..4 {
        unsafe {
            data = [
                _mm512_permutex2var_epi32(data[0], L, data[1]),
                _mm512_permutex2var_epi32(data[2], L, data[3]),
                _mm512_permutex2var_epi32(data[4], L, data[5]),
                _mm512_permutex2var_epi32(data[6], L, data[7]),
                _mm512_permutex2var_epi32(data[8], L, data[9]),
                _mm512_permutex2var_epi32(data[10], L, data[11]),
                _mm512_permutex2var_epi32(data[12], L, data[13]),
                _mm512_permutex2var_epi32(data[14], L, data[15]),
                _mm512_permutex2var_epi32(data[0], H, data[1]),
                _mm512_permutex2var_epi32(data[2], H, data[3]),
                _mm512_permutex2var_epi32(data[4], H, data[5]),
                _mm512_permutex2var_epi32(data[6], H, data[7]),
                _mm512_permutex2var_epi32(data[8], H, data[9]),
                _mm512_permutex2var_epi32(data[10], H, data[11]),
                _mm512_permutex2var_epi32(data[12], H, data[13]),
                _mm512_permutex2var_epi32(data[14], H, data[15]),
            ];
        }
    }
    unsafe { std::mem::transmute(data) }
}

#[cfg(test)]
mod tests {
    use super::bit_reverse16;
    use crate::core::backend::avx512::bit_reverse::bit_reverse_m31;
    use crate::core::fields::m31::BaseField;
    use crate::core::utils::bit_reverse;

    #[test]
    fn test_bit_reverse16() {
        let data: [u32; 256] = std::array::from_fn(|i| i as u32);
        let expected: [u32; 256] = std::array::from_fn(|i| (i as u32).reverse_bits() >> 24);
        unsafe {
            let data = bit_reverse16(std::mem::transmute(data));
            assert_eq!(std::mem::transmute::<_, [u32; 256]>(data), expected);
        }
    }

    #[test]
    fn test_bit_reverse() {
        const SIZE: usize = 1 << 15;
        let data: Vec<_> = (0..SIZE as u32)
            .map(BaseField::from_u32_unchecked)
            .collect();
        let expected = bit_reverse(data.clone());
        let mut data: Vec<_> = data.into_iter().array_chunks::<16>().collect();
        let expected: Vec<_> = expected.into_iter().array_chunks::<16>().collect();

        bit_reverse_m31(&mut data);
        assert_eq!(data, expected);
    }
}
