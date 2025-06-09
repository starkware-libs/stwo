use std::array;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::column::{BaseColumn, SecureColumn};
use super::m31::PackedBaseField;
use super::SimdBackend;
use crate::core::backend::cpu::bit_reverse as cpu_bit_reverse;
use crate::core::backend::simd::utils::UnsafeMut;
use crate::core::backend::ColumnOps;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::utils::bit_reverse_index;
use crate::parallel_iter;

const VEC_BITS: u32 = 4;

const W_BITS: u32 = 3;

pub const MIN_LOG_SIZE: u32 = 2 * W_BITS + VEC_BITS;

impl ColumnOps<BaseField> for SimdBackend {
    type Column = BaseColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        // Fallback to cpu bit_reverse.
        if column.data.len().ilog2() < MIN_LOG_SIZE {
            cpu_bit_reverse(column.as_mut_slice());
            return;
        }

        bit_reverse_m31(&mut column.data);
    }
}

impl ColumnOps<SecureField> for SimdBackend {
    type Column = SecureColumn;

    fn bit_reverse_column(_column: &mut SecureColumn) {
        todo!()
    }
}

/// Bit reverses M31 values.
///
/// Given an array `A[0..2^n)`, computes `B[i] = A[bit_reverse(i)]`.
pub fn bit_reverse_m31(data: &mut [PackedBaseField]) {
    assert!(data.len().is_power_of_two());
    assert!(data.len().ilog2() >= MIN_LOG_SIZE);

    // Indices in the array are of the form v_h w_h a w_l v_l, with
    // |v_h| = |v_l| = VEC_BITS, |w_h| = |w_l| = W_BITS, |a| = n - 2*W_BITS - VEC_BITS.
    // The loops go over a, w_l, w_h, and then swaps the 16 by 16 values at:
    //   * w_h a w_l *   <->   * rev(w_h a w_l) *.
    // These are 1 or 2 chunks of 2^W_BITS contiguous `u32x16` vectors.

    let log_size = data.len().ilog2();
    let a_bits = log_size - 2 * W_BITS - VEC_BITS;
    let data = UnsafeMut(data);

    parallel_iter!(0u32..(1 << a_bits)).for_each(|a| {
        let data = unsafe { data.get() };
        for w_l in 0u32..1 << W_BITS {
            let w_l_rev = w_l.reverse_bits() >> (u32::BITS - W_BITS);
            for w_h in 0..w_l_rev + 1 {
                let idx = ((((w_h << a_bits) | a) << W_BITS) | w_l) as usize;
                let idx_rev = bit_reverse_index(idx, log_size - VEC_BITS);

                // In order to not swap twice, only swap if idx <= idx_rev.
                if idx > idx_rev {
                    continue;
                }

                // Read first chunk.
                // TODO(andrew): Think about optimizing a_bits. What does this mean?
                let chunk0 = array::from_fn(|i| unsafe {
                    *data.get_unchecked_mut(idx + (i << (2 * W_BITS + a_bits)))
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
                let chunk1 = array::from_fn(|i| unsafe {
                    *data.get_unchecked_mut(idx_rev + (i << (2 * W_BITS + a_bits)))
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
    })
}

/// Bit reverses 256 M31 values, packed in 16 words of 16 elements each.
fn bit_reverse16(mut data: [PackedBaseField; 16]) -> [PackedBaseField; 16] {
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
        // `interleave` allows us to interleave the first half of 2 words.
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

    data
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::{bit_reverse16, bit_reverse_m31, MIN_LOG_SIZE};
    use crate::core::backend::cpu::bit_reverse as cpu_bit_reverse;
    use crate::core::backend::simd::column::BaseColumn;
    use crate::core::backend::simd::m31::{PackedM31, N_LANES};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, ColumnOps};
    use crate::core::fields::m31::BaseField;

    #[test]
    fn test_bit_reverse16() {
        let values: BaseColumn = (0..N_LANES * 16).map(BaseField::from).collect();
        let mut expected = values.to_cpu();
        cpu_bit_reverse(&mut expected);

        let res = bit_reverse16(values.data.try_into().unwrap());

        assert_eq!(res.map(PackedM31::to_array).as_flattened(), expected);
    }

    #[test]
    fn bit_reverse_m31_works() {
        const SIZE: usize = 1 << 15;
        let data: Vec<_> = (0..SIZE).map(BaseField::from).collect();
        let mut expected = data.clone();
        cpu_bit_reverse(&mut expected);

        let mut res: BaseColumn = data.into_iter().collect();
        bit_reverse_m31(&mut res.data[..]);

        assert_eq!(res.to_cpu(), expected);
    }

    #[test]
    fn bit_reverse_small_column_works() {
        const LOG_SIZE: u32 = MIN_LOG_SIZE - 1;
        let column = (0..1 << LOG_SIZE).map(BaseField::from).collect_vec();
        let mut expected = column.clone();
        cpu_bit_reverse(&mut expected);

        let mut res = column.iter().copied().collect::<BaseColumn>();
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut res);

        assert_eq!(res.to_cpu(), expected);
    }

    #[test]
    fn bit_reverse_large_column_works() {
        const LOG_SIZE: u32 = MIN_LOG_SIZE;
        let column = (0..1 << LOG_SIZE).map(BaseField::from).collect_vec();
        let mut expected = column.clone();
        cpu_bit_reverse(&mut expected);

        let mut res = column.iter().copied().collect::<BaseColumn>();
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut res);

        assert_eq!(res.to_cpu(), expected);
    }
}
