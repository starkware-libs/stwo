use std::simd::Swizzle;

use num_traits::One;

use crate::core::backend::simd::m31::{PackedM31, N_LANES};
use crate::core::fields::m31::M31;

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

// TODO(Gali): Remove #[allow(dead_code)].
#[allow(dead_code)]
pub fn generate_secure_powers(felt: M31) -> PackedM31 {
    let arr: [M31; N_LANES] = (0..N_LANES)
        .scan(M31::one(), |acc, _| {
            let res = *acc;
            *acc *= felt;
            Some(res)
        })
        .collect::<Vec<M31>>()
        .try_into()
        .expect("Failed generating secure powers.");

    PackedM31::from_array(arr)
}

#[cfg(test)]
mod tests {
    use std::simd::{u32x4, Swizzle};

    use num_traits::One;

    use super::{InterleaveEvens, InterleaveOdds};
    use crate::core::fields::m31::M31;
    use crate::core::fields::FieldExpOps;

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

    #[test]
    fn generate_secure_powers_works() {
        let felt = M31(2);

        let powers = super::generate_secure_powers(felt);
        let powers = powers.to_array();

        assert_eq!(powers[0], M31::one());
        assert_eq!(powers[1], felt);
        assert_eq!(powers[7], felt.pow(7));
    }
}
