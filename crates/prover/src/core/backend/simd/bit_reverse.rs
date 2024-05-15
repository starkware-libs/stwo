use std::simd::u32x16;

use super::m31::PackedBaseField;
use crate::core::utils::bit_reverse_index;

const VEC_BITS: u32 = 4;
const W_BITS: u32 = 3;
pub const MIN_LOG_SIZE: u32 = 2 * W_BITS + VEC_BITS;

/// Bit reverses packed M31 values.
/// Given an array `A[0..2^n)`, computes `B[i] = A[bit_reverse(i)]`.
pub fn bit_reverse_m31(data: &mut [PackedBaseField]) {
    assert!(data.len().is_power_of_two());
    assert!(data.len().ilog2() >= MIN_LOG_SIZE);

    // Indices in the array are of the form v_h w_h a w_l v_l, with
    // |v_h| = |v_l| = VEC_BITS, |w_h| = |w_l| = W_BITS, |a| = n - 2*W_BITS - VEC_BITS.
    // The loops go over a, w_l, w_h, and then swaps the 16 by 16 values at:
    //   * w_h a w_l *   <->   * rev(w_h a w_l) *.
    // These are 1 or 2 chunks of 2^W_BITS contiguous AVX512 vectors.

    let log_size = data.len().ilog2();
    let a_bits = log_size - 2 * W_BITS - VEC_BITS;

    // TODO(spapini): when doing multithreading, do it over a.
    for a in 0u32..(1 << a_bits) {
        for w_l in 0u32..(1 << W_BITS) {
            let w_l_rev = w_l.reverse_bits() >> (32 - W_BITS);
            for w_h in 0u32..(w_l_rev + 1) {
                let idx = ((((w_h << a_bits) | a) << W_BITS) | w_l) as usize;
                let idx_rev = bit_reverse_index(idx, log_size - VEC_BITS);

                // In order to not swap twice, only swap if idx <= idx_rev.
                if idx > idx_rev {
                    continue;
                }

                // Read first chunk.
                // TODO(spapini): Think about optimizing a_bits.
                let chunk0 = std::array::from_fn(|i| unsafe {
                    *data.get_unchecked(idx + (i << (2 * W_BITS + a_bits)))
                });
                let values0 = bit_reverse16(chunk0);

                if idx == idx_rev {
                    // Palindrome index. Write into the same chunk.
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..16 {
                        unsafe {
                            *data.get_unchecked_mut(idx + (i << (2 * W_BITS + a_bits))) =
                                values0[i];
                        }
                    }
                    continue;
                }

                // Read bit reversed chunk.
                let chunk1 = std::array::from_fn(|i| unsafe {
                    *data.get_unchecked(idx_rev + (i << (2 * W_BITS + a_bits)))
                });
                let values1 = bit_reverse16(chunk1);

                for i in 0..16 {
                    unsafe {
                        *data.get_unchecked_mut(idx + (i << (2 * W_BITS + a_bits))) = values1[i];
                        *data.get_unchecked_mut(idx_rev + (i << (2 * W_BITS + a_bits))) =
                            values0[i];
                    }
                }
            }
        }
    }
}

/// Bit reverses 256 M31 values, packed in 16 words of 16 elements each.
fn bit_reverse16(data: [PackedBaseField; 16]) -> [PackedBaseField; 16] {
    let mut data: [u32x16; 16] = unsafe { std::mem::transmute(data) };
    // L is an input to _mm512_permutex2var_epi32, and it is used to
    // interleave the first half of a with the first half of b.
    const _L: u32x16 = unsafe {
        core::mem::transmute([
            0b00000, 0b10000, 0b00001, 0b10001, 0b00010, 0b10010, 0b00011, 0b10011, 0b00100,
            0b10100, 0b00101, 0b10101, 0b00110, 0b10110, 0b00111, 0b10111,
        ])
    };
    // H is an input to _mm512_permutex2var_epi32, and it is used to interleave the second half
    // interleave the second half of a with the second half of b.
    const _H: u32x16 = unsafe {
        core::mem::transmute([
            0b01000, 0b11000, 0b01001, 0b11001, 0b01010, 0b11010, 0b01011, 0b11011, 0b01100,
            0b11100, 0b01101, 0b11101, 0b01110, 0b11110, 0b01111, 0b11111,
        ])
    };

    // Denote the index of each element in the 16 packed M31 words as abcd:0123,
    // where abcd is the index of the packed word and 0123 is the index of the element in the word.
    // Bit reversal is achieved by applying the following permutation to the index for 4 times:
    //   abcd:0123 => 0abc:123d
    // This is how it looks like at each iteration.
    //   abcd:0123
    //   0abc:123d
    //   10ab:23dc
    //   210a:3dcb
    //   3210:dcba
    for _ in 0..4 {
        // Apply the abcd:0123 => 0abc:123d permutation.
        // _mm512_permutex2var_epi32() with L allows us to interleave the first half of 2 words.
        // For example, the second call interleaves 0010:0xyz (low half of register 2) with
        // 0011:0xyz (low half of register 3), and stores the result in register 1 (0001).
        // This results in
        //    0001:xyz0 (even indices of register 1) <= 0010:0xyz (low half of register2), and
        //    0001:xyz1 (odd indices of register 1)  <= 0011:0xyz (low half of register 3)
        // or 0001:xyzw <= 001w:0xyz.
        let (d0, d8) = data[0].interleave(data[1]);
        let (d1, d9) = data[2].interleave(data[3]);
        let (d2, d10) = data[4].interleave(data[5]);
        let (d3, d11) = data[6].interleave(data[7]);
        let (d4, d12) = data[8].interleave(data[9]);
        let (d5, d13) = data[10].interleave(data[11]);
        let (d6, d14) = data[12].interleave(data[13]);
        let (d7, d15) = data[14].interleave(data[15]);
        data = [
            d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15,
        ];
    }
    unsafe { std::mem::transmute(data) }
}

#[cfg(test)]
mod tests {
    use super::{bit_reverse16, bit_reverse_m31};
    use crate::core::backend::simd::column::BaseFieldVec;
    use crate::core::backend::Column;
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
        let mut expected = data.clone();
        bit_reverse(&mut expected);
        let mut data: BaseFieldVec = data.into_iter().collect();

        bit_reverse_m31(&mut data.data[..]);
        assert_eq!(data.to_cpu(), expected);
    }
}
