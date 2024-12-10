use std::simd::Swizzle;

use itertools::Itertools;

use crate::core::backend::simd::m31::N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::fields::qm31::SecureField;
use crate::core::utils::generate_secure_powers;

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
pub fn generate_secure_powers_simd(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
    let base_arr = generate_secure_powers(felt, N_LANES).try_into().unwrap();
    let base = PackedSecureField::from_array(base_arr);
    let step = PackedSecureField::broadcast(base_arr[N_LANES - 1] * felt);
    let size = n_powers.div_ceil(N_LANES);

    (0..size)
        .scan(base, |acc, _| {
            let res = *acc;
            *acc *= step;
            Some(res)
        })
        .flat_map(|x| x.to_array())
        .take(n_powers)
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use std::simd::{u32x4, Swizzle};

    use super::{generate_secure_powers_simd, InterleaveEvens, InterleaveOdds};
    use crate::core::utils::generate_secure_powers;
    use crate::qm31;

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
    fn test_generate_secure_powers_simd() {
        let felt = qm31!(1, 2, 3, 4);
        let n_powers = 100;

        let cpu_powers = generate_secure_powers(felt, n_powers);
        let simd_powers = generate_secure_powers_simd(felt, n_powers);

        assert_eq!(simd_powers, cpu_powers);
    }
}
