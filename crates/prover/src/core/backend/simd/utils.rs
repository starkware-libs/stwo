use std::simd::Swizzle;

/// Used with [`Swizzle::concat_swizzle`] to interleave the low halves of vectors `lo` and `hi`.
pub struct LoLoInterleaveHiLo;

impl<const N: usize> Swizzle<N> for LoLoInterleaveHiLo {
    const INDEX: [usize; N] = segment_interleave(false);
}

/// Used with [`Swizzle::concat_swizzle`] to interleave the high halves of vectors `lo` and `hi`.
pub struct LoHiInterleaveHiHi;

impl<const N: usize> Swizzle<N> for LoHiInterleaveHiHi {
    const INDEX: [usize; N] = segment_interleave(true);
}

/// Used with [`Swizzle::concat_swizzle`] to concat the even values of vectors `lo` and `hi`.
pub struct LoEvensConcatHiEvens;

impl<const N: usize> Swizzle<N> for LoEvensConcatHiEvens {
    const INDEX: [usize; N] = parity_concat(false);
}

/// Used with [`Swizzle::concat_swizzle`] to concat the odd values of vectors `lo` and `hi`.
pub struct LoOddsConcatHiOdds;

impl<const N: usize> Swizzle<N> for LoOddsConcatHiOdds {
    const INDEX: [usize; N] = parity_concat(true);
}

/// Used with [`Swizzle::concat_swizzle`] to interleave the even values of vectors `lo` and `hi`.
pub struct LoEvensInterleaveHiEvens;

impl<const N: usize> Swizzle<N> for LoEvensInterleaveHiEvens {
    const INDEX: [usize; N] = parity_interleave(false);
}

/// Used with [`Swizzle::concat_swizzle`] to interleave the odd values of vectors `lo` and `hi`.
pub struct LoOddsInterleaveHiOdds;

impl<const N: usize> Swizzle<N> for LoOddsInterleaveHiOdds {
    const INDEX: [usize; N] = parity_interleave(true);
}

const fn segment_interleave<const N: usize>(hi: bool) -> [usize; N] {
    let mut res = [0; N];
    let mut i = 0;
    while i < N {
        res[i] = (i % 2) * N + i / 2 + if hi { N / 2 } else { 0 };
        i += 1;
    }
    res
}

const fn parity_concat<const N: usize>(odd: bool) -> [usize; N] {
    let mut res = [0; N];
    let mut i = 0;
    while i < N {
        res[i] = i * 2 + if odd { 1 } else { 0 };
        i += 1;
    }
    res
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

    use super::LoLoInterleaveHiLo;
    use crate::core::backend::simd::utils::{
        LoEvensConcatHiEvens, LoEvensInterleaveHiEvens, LoHiInterleaveHiHi, LoOddsConcatHiOdds,
        LoOddsInterleaveHiOdds,
    };

    #[test]
    fn lo_lo_interleave_hi_lo() {
        let lo = u32x4::from_array([0, 1, 2, 3]);
        let hi = u32x4::from_array([4, 5, 6, 7]);

        let res = LoLoInterleaveHiLo::concat_swizzle(lo, hi);

        assert_eq!(res, u32x4::from_array([0, 4, 1, 5]));
    }

    #[test]
    fn lo_hi_interleave_hi_hi() {
        let lo = u32x4::from_array([0, 1, 2, 3]);
        let hi = u32x4::from_array([4, 5, 6, 7]);

        let res = LoHiInterleaveHiHi::concat_swizzle(lo, hi);

        assert_eq!(res, u32x4::from_array([2, 6, 3, 7]));
    }

    #[test]
    fn lo_evens_concat_hi_evens() {
        let lo = u32x4::from_array([0, 1, 2, 3]);
        let hi = u32x4::from_array([4, 5, 6, 7]);

        let res = LoEvensConcatHiEvens::concat_swizzle(lo, hi);

        assert_eq!(res, u32x4::from_array([0, 2, 4, 6]));
    }

    #[test]
    fn lo_odds_concat_hi_odds() {
        let lo = u32x4::from_array([0, 1, 2, 3]);
        let hi = u32x4::from_array([4, 5, 6, 7]);

        let res = LoOddsConcatHiOdds::concat_swizzle(lo, hi);

        assert_eq!(res, u32x4::from_array([1, 3, 5, 7]));
    }

    #[test]
    fn lo_evens_interleave_hi_evens() {
        let lo = u32x4::from_array([0, 1, 2, 3]);
        let hi = u32x4::from_array([4, 5, 6, 7]);

        let res = LoEvensInterleaveHiEvens::concat_swizzle(lo, hi);

        assert_eq!(res, u32x4::from_array([0, 4, 2, 6]));
    }

    #[test]
    fn lo_odds_interleave_hi_odds() {
        let lo = u32x4::from_array([0, 1, 2, 3]);
        let hi = u32x4::from_array([4, 5, 6, 7]);

        let res = LoOddsInterleaveHiOdds::concat_swizzle(lo, hi);

        assert_eq!(res, u32x4::from_array([1, 5, 3, 7]));
    }
}
