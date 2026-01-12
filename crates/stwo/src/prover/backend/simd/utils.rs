use std::simd::{simd_swizzle, u32x16};
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
