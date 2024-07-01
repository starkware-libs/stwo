use std::simd::Swizzle;

use rayon::prelude::*;

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

pub trait MaybeParIter: IntoParallelIterator + IntoIterator {
    type Iter;
    fn maybe_par_iter(self) -> <Self as MaybeParIter>::Iter;
}
impl<I: IntoParallelIterator + IntoIterator> MaybeParIter for I {
    #[cfg(feature = "parallel")]
    type Iter = <Self as IntoParallelIterator>::Iter;
    #[cfg(not(feature = "parallel"))]
    type Iter = <Self as IntoIterator>::IntoIter;

    fn maybe_par_iter(self) -> <Self as MaybeParIter>::Iter {
        #[cfg(feature = "parallel")]
        return self.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        self.into_iter()
    }
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
