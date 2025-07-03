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
