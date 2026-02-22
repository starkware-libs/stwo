use std::simd::{simd_swizzle, u32x16};

use crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
use crate::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
use crate::prover::backend::simd::m31::PackedBaseField;
// TODO(andrew): Examine usage of unsafe in SIMD FFT.
pub struct UnsafeMut<T: ?Sized>(pub *mut T);
impl<T: ?Sized> UnsafeMut<T> {
    /// # Safety
    ///
    /// Returns a raw mutable pointer.
    pub const unsafe fn get(&self) -> *mut T {
        self.0
    }
}

unsafe impl<T: ?Sized> Send for UnsafeMut<T> {}
unsafe impl<T: ?Sized> Sync for UnsafeMut<T> {}

pub struct UnsafeConst<T>(pub *const T);
impl<T> UnsafeConst<T> {
    /// # Safety
    ///
    /// Returns a raw constant pointer.
    pub const unsafe fn get(&self) -> *const T {
        self.0
    }
}

unsafe impl<T> Send for UnsafeConst<T> {}
unsafe impl<T> Sync for UnsafeConst<T> {}

/// A helper function to compute the lift of a column of PackedM31 values.
///
/// # Intro
///
/// Given a column C of log_size n, containing u32x16 values, the goal is to compute
/// its "lifting" to log_size m (m >= n). Here, "lifting" means the following:
///
/// 1. Interpret column C as the vector of evaluations of a circle polynomial `p`, of degree < n, on
///    the canonical coset of log_size n, in bit reversed order.
///
/// 2. The lift of C to log_size m is, by definition, the vector of evaluations of the polynomial `p
///    ∘ πᵐ⁻ⁿ` on the canonical coset of log_size m, in bit reversed order. Here `π` is the doubling
///    map.
///
/// # Arguments
///
/// - `x`: the evaluation of the un-lifted polynomial that we wish to lift.
/// - `log_ratio`: the log ratio between the lifted domain and the base domain (in the above
///   example, it's m - n).
/// - `idx`: the index in the vector of lifted evaluations that we wish to compute.
///
/// # Returns
///
/// - A PackedM31 corresponding to the values of the lifted polynomial on the `idx`-th, ..., `idx +
///   15`-th points of the lifted domain, where the order is the bit reversed order.
pub fn to_lifted_simd(x: u32x16, log_ratio: u32, idx: usize) -> u32x16 {
    let idx_mod_ratio = idx % (1 << log_ratio);
    match log_ratio {
        0 => x,
        1 => match idx_mod_ratio % 2 {
            0 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_1[0]),
            1 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_1[1]),
            _ => unreachable!(),
        },
        2 => match idx_mod_ratio % 4 {
            0 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_2[0]),
            1 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_2[1]),
            2 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_2[2]),
            3 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_2[3]),
            _ => unreachable!(),
        },
        _ => match idx_mod_ratio >> (log_ratio - 3) {
            0 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[0]),
            1 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[1]),
            2 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[2]),
            3 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[3]),
            4 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[4]),
            5 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[5]),
            6 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[6]),
            7 => simd_swizzle!(x, LIFTING_SWIZZLES_LOG_RATIO_GREATER_2[7]),
            _ => unreachable!(),
        },
    }
}

#[rustfmt::skip]
const LIFTING_SWIZZLES_LOG_RATIO_1: [[usize; 16]; 2] = [
    [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7],
    [8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15, 14, 15],
];
#[rustfmt::skip]
const LIFTING_SWIZZLES_LOG_RATIO_2: [[usize; 16]; 4] = [
    [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3],
    [4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7],
    [8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11],
    [12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15],
];
#[rustfmt::skip]
const LIFTING_SWIZZLES_LOG_RATIO_GREATER_2: [[usize; 16]; 8] = [
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
    [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5],
    [6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
    [8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9],
    [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11],
    [12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13],
    [14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15],
];

#[inline(always)]
pub fn transpose_packed_leaf(
    packed_values: [[PackedBaseField; SECURE_EXTENSION_DEGREE]; PACKED_LEAF_SIZE],
) -> [[PackedBaseField; SECURE_EXTENSION_DEGREE]; PACKED_LEAF_SIZE] {
    let coord_arrays = packed_values.map(|coords| coords.map(PackedBaseField::to_array));

    core::array::from_fn(|offset| {
        core::array::from_fn(|coord| {
            PackedBaseField::from_array(core::array::from_fn(|lane| {
                let src_packed = lane / 4;
                let src_lane = (lane % 4) * 4 + offset;
                coord_arrays[src_packed][coord][src_lane]
            }))
        })
    })
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
pub mod swizzle {
    use std::simd::Swizzle;

    /// Used with [`Swizzle::concat_swizzle`] to interleave the even values of two vectors.
    pub struct InterleaveEvens;
    impl<const N: usize> Swizzle<N> for InterleaveEvens {
        const INDEX: [usize; N] = parity_interleave(false);
    }

    /// Used with [`Swizzle::concat_swizzle`] to interleave the odd values of two vectors.
    pub struct InterleaveOdds;

    impl<const N: usize> Swizzle<N> for InterleaveOdds {
        const INDEX: [usize; N] = parity_interleave(true);
    }

    const fn parity_interleave<const N: usize>(odd: bool) -> [usize; N] {
        let mut res = [0; N];
        let mut i = 0;
        while i < N {
            res[i] = (i % 2) * N + (i / 2) * 2 + if odd { 1 } else { 0 };
            i += 1;
        }
        res
    }

    #[cfg(test)]
    mod tests {
        use std::simd::{u32x4, Swizzle};

        use super::{InterleaveEvens, InterleaveOdds};

        #[test]
        fn interleave_evens() {
            let lo = u32x4::from_array([0, 1, 2, 3]);
            let hi = u32x4::from_array([4, 5, 6, 7]);

            let res = InterleaveEvens::concat_swizzle(lo, hi);

            assert_eq!(res, u32x4::from_array([0, 4, 2, 6]));
        }

        #[test]
        fn interleave_odds() {
            let lo = u32x4::from_array([0, 1, 2, 3]);
            let hi = u32x4::from_array([4, 5, 6, 7]);

            let res = InterleaveOdds::concat_swizzle(lo, hi);

            assert_eq!(res, u32x4::from_array([1, 5, 3, 7]));
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::transpose_packed_leaf;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
    use crate::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
    use crate::prover::backend::simd::m31::PackedBaseField;
    use crate::qm31;

    #[test]
    fn test_transpose_leaf() {
        // Create input and expected output (before packing them into SIMD). The input is a column
        // of 64 QM31s. The expected output is 4 columns of 16 QM31s each.
        let mut input_col = vec![];
        (0..16 * 4).for_each(|row| {
            input_col.push(qm31!(10 * row, 10 * row + 1, 10 * row + 2, 10 * row + 3))
        });
        let mut expected_output: [Vec<[M31; 4]>; 4] = [const { vec![] }; 4];
        for chunk in &input_col.iter().chunks(4) {
            chunk
                .into_iter()
                .enumerate()
                .for_each(|(i, val)| expected_output[i].push(val.to_m31_array()));
        }

        // Pack the input column and expected output columns into SIMD elements.
        let input_col: Vec<_> = input_col.iter().map(|x| x.to_m31_array()).collect();
        let packed_input: [[PackedBaseField; SECURE_EXTENSION_DEGREE]; PACKED_LEAF_SIZE] =
            std::array::from_fn(|packed_row| {
                std::array::from_fn(|coord| {
                    PackedBaseField::from_array(std::array::from_fn(|lane| {
                        input_col[packed_row * 16 + lane][coord]
                    }))
                })
            });
        let packed_expected_output: [[PackedBaseField; SECURE_EXTENSION_DEGREE]; PACKED_LEAF_SIZE] =
            std::array::from_fn(|leaf| {
                std::array::from_fn(|coord| {
                    PackedBaseField::from_array(std::array::from_fn(|lane| {
                        expected_output[leaf][lane][coord]
                    }))
                })
            });

        let actual = transpose_packed_leaf(packed_input);
        // TODO(Leo): implement PartialEq for PackedM31 so that it doesn't conflict with stwo-cairo.
        for offset in 0..PACKED_LEAF_SIZE {
            for coord in 0..SECURE_EXTENSION_DEGREE {
                assert_eq!(
                    actual[offset][coord].to_array(),
                    packed_expected_output[offset][coord].to_array()
                );
            }
        }
    }
}
