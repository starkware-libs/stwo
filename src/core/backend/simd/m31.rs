use std::mem::transmute;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::simd::cmp::SimdOrd;
use std::simd::{Simd, Swizzle};

use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};

use crate::core::fields::m31::{M31, P};
use crate::core::fields::FieldExpOps;

pub const LOG_N_LANES: usize = 2;
pub const N_LANES: usize = 1 << LOG_N_LANES;

pub const MODULUS: Simd<u32, N_LANES> = Simd::from_array([P; N_LANES]);

/// SIMD implementation of [`M31`].
#[derive(Copy, Clone, Debug)]
pub struct PackedBaseField(Simd<u32, N_LANES>);

impl PackedBaseField {
    pub fn broadcast(M31(value): M31) -> Self {
        Self(Simd::splat(value))
    }

    pub fn from_array(values: [M31; N_LANES]) -> PackedBaseField {
        Self(Simd::from_array(values.map(|M31(v)| v)))
    }

    pub fn to_array(self) -> [M31; N_LANES] {
        self.reduce().0.to_array().map(M31)
    }

    /// Reduces each word in the 512-bit register to the range `[0, P)`, excluding P.
    fn reduce(self) -> PackedBaseField {
        Self(Simd::simd_min(self.0, self.0 - MODULUS))
    }

    /// Interleaves self with other.
    /// Returns the result as two packed M31 elements.
    pub fn interleave(self, other: Self) -> (Self, Self) {
        let (a, b) = self.0.interleave(other.0);
        (Self(a), Self(b))
    }

    /// Deinterleaves self with other.
    /// Done by concatenating the even words of self with the even words of other, and the odd words
    /// The inverse of [Self::interleave].
    /// Returns the result as two packed M31 elements.
    pub fn deinterleave(self, other: Self) -> (Self, Self) {
        let (a, b) = self.0.deinterleave(other.0);
        (Self(a), Self(b))
    }

    /// Sums all the elements in the packed M31 element.
    pub fn pointwise_sum(self) -> M31 {
        self.to_array().into_iter().sum()
    }

    // TODO: Docs.
    pub fn double(self) -> Self {
        // TODO: Make more optimal.
        self + self
    }
}

impl Add for PackedBaseField {
    type Output = Self;

    /// Adds two packed M31 elements, and reduces the result to the range `[0,P]`.
    /// Each value is assumed to be in unreduced form, [0, P] including P.
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        // Add word by word. Each word is in the range [0, 2P].
        let c = self.0 + rhs.0;
        // Apply min(c, c-P) to each word.
        // When c in [P,2P], then c-P in [0,P] which is always less than [P,2P].
        // When c in [0,P-1], then c-P in [2^32-P,2^32-1] which is always greater than [0,P-1].
        Self(Simd::simd_min(c, c - MODULUS))
    }
}

impl AddAssign for PackedBaseField {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for PackedBaseField {
    type Output = Self;

    /// Computes the product of two packed M31 elements
    /// Each value is assumed to be in unreduced form, [0, P] including P.
    /// Returned values are in unreduced form, [0, P] including P.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        const fn interleave(odd: bool) -> [usize; N_LANES] {
            let mut res = [0; N_LANES];
            let mut i = 0;
            while i < res.len() {
                res[i] = (i % 2) * N_LANES + (i / 2) * 2 + if odd { 1 } else { 0 };
                i += 1;
            }
            res
        }

        struct InterleaveEvens;
        struct InterleaveOdds;

        impl Swizzle<N_LANES> for InterleaveEvens {
            const INDEX: [usize; N_LANES] = interleave(false);
        }

        impl Swizzle<N_LANES> for InterleaveOdds {
            const INDEX: [usize; N_LANES] = interleave(true);
        }

        const MASK_U32: Simd<u64, { N_LANES / 2 }> =
            Simd::from_array([0xFFFFFFFF; { N_LANES / 2 }]);

        unsafe {
            // Set up a word s.t. the lower half of each 64-bit word has the even 32-bit words of
            // the first operand.
            // let val0_e = transmute::<_, Simd<u64, { N_LANES / 2 }>>(self.0) & MASK_U32;
            let val0_e = transmute::<_, Simd<u64, { N_LANES / 2 }>>(self.0) & MASK_U32;
            // Set up a word s.t. the lower half of each 64-bit word has the odd 32-bit words of
            // the first operand.
            let val0_o = transmute::<_, Simd<u64, { N_LANES / 2 }>>(self.0) >> 32;

            // Double the second operand.
            let val1 = rhs.0 << 1;
            let val1_e = transmute::<_, Simd<u64, { N_LANES / 2 }>>(val1) & MASK_U32;
            let val1_o = transmute::<_, Simd<u64, { N_LANES / 2 }>>(val1) >> 32;

            // To compute prod = val0 * val1 start by multiplying
            // val0_e/o by val1_e/o.
            let prod_e_dbl = val0_e * val1_e;
            let prod_o_dbl = val0_o * val1_o;

            // The result of a multiplication holds val1*twiddle_dbl in as 64-bits.
            // Each 64b-bit word looks like this:
            //               1    31       31    1
            // prod_e_dbl - |0|prod_e_h|prod_e_l|0|
            // prod_o_dbl - |0|prod_o_h|prod_o_l|0|

            // Interleave the even words of prod_e_dbl with the even words of prod_o_dbl:
            // let prod_lows = _mm512_permutex2var_epi32(prod_e_dbl, EVENS_INTERLEAVE_EVENS,
            // prod_o_dbl);
            // prod_ls -    |prod_o_l|0|prod_e_l|0|
            // Divide by 2:
            // prod_ls -    |0|prod_o_l|0|prod_e_l|
            let prod_lows = Self(
                InterleaveEvens::concat_swizzle(
                    transmute::<_, Simd<u32, N_LANES>>(prod_e_dbl),
                    transmute::<_, Simd<u32, N_LANES>>(prod_o_dbl),
                ) >> 1,
            );

            // Interleave the odd words of prod_e_dbl with the odd words of prod_o_dbl:
            let prod_highs = Self(InterleaveOdds::concat_swizzle(
                transmute::<_, Simd<u32, N_LANES>>(prod_e_dbl),
                transmute::<_, Simd<u32, N_LANES>>(prod_o_dbl),
            ));
            // prod_hs -    |0|prod_o_h|0|prod_e_h|

            prod_lows + prod_highs
        }
    }
}

impl MulAssign for PackedBaseField {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for PackedBaseField {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self(MODULUS - self.0)
    }
}

/// Subtracts two packed M31 elements, and reduces the result to the range `[0,P]`.
/// Each value is assumed to be in unreduced form, [0, P] including P.
impl Sub for PackedBaseField {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        // Subtract word by word. Each word is in the range [-P, P].
        let c = self.0 - rhs.0;
        // Apply min(c, c+P) to each word.
        // When c in [0,P], then c+P in [P,2P] which is always greater than [0,P].
        // When c in [2^32-P,2^32-1], then c+P in [0,P-1] which is always less than
        // [2^32-P,2^32-1].
        Self(Simd::simd_min(c + MODULUS, c))
    }
}

impl SubAssign for PackedBaseField {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Zero for PackedBaseField {
    fn zero() -> Self {
        Self(Simd::from_array([0; N_LANES]))
    }

    fn is_zero(&self) -> bool {
        self.to_array().iter().all(M31::is_zero)
    }
}

impl One for PackedBaseField {
    fn one() -> Self {
        Self(Simd::<u32, N_LANES>::from_array([1; N_LANES]))
    }
}

impl FieldExpOps for PackedBaseField {
    fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "0 has no inverse");
        self.pow((P - 2) as u128)
    }
}

unsafe impl Pod for PackedBaseField {}

unsafe impl Zeroable for PackedBaseField {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use super::{PackedBaseField, N_LANES};
    use crate::core::fields::m31::{BaseField, M31, P};

    const LHS_VALUES: [BaseField; N_LANES] = [
        M31(0),
        M31(1),
        M31(2),
        M31((P - 1) / 2),
        // M31(10),
        // M31((P + 1) / 2),
        // M31(P - 2),
        // M31(P - 1),
        // M31(0),
        // M31(1),
        // M31(2),
        // M31(10),
        // M31((P - 1) / 2),
        // M31((P + 1) / 2),
        // M31(P - 2),
        // M31(P - 1),
        // // ==
        // M31(0),
        // M31(1),
        // M31(2),
        // M31(10),
        // M31((P - 1) / 2),
        // M31((P + 1) / 2),
        // M31(P - 2),
        // M31(P - 1),
        // M31(0),
        // M31(1),
        // M31(2),
        // M31(10),
        // M31((P - 1) / 2),
        // M31((P + 1) / 2),
        // M31(P - 2),
        // M31(P - 1),
    ];

    const RHS_VALUES: [BaseField; N_LANES] = [
        M31(0),
        M31(1),
        M31(2),
        M31((P - 1) / 2),
        // M31(10),
        // M31((P + 1) / 2),
        // M31(P - 2),
        // M31(P - 1),
        // M31(P - 1),
        // M31(P - 2),
        // M31((P + 1) / 2),
        // M31((P - 1) / 2),
        // M31(10),
        // M31(2),
        // M31(1),
        // M31(0),
        // // ==
        // M31(0),
        // M31(1),
        // M31(2),
        // M31(10),
        // M31((P - 1) / 2),
        // M31((P + 1) / 2),
        // M31(P - 2),
        // M31(P - 1),
        // M31(P - 1),
        // M31(P - 2),
        // M31((P + 1) / 2),
        // M31((P - 1) / 2),
        // M31(10),
        // M31(2),
        // M31(1),
        // M31(0),
    ];

    #[test]
    fn addition_works() {
        let lhs = PackedBaseField::from_array(LHS_VALUES);
        let rhs = PackedBaseField::from_array(RHS_VALUES);

        let res = (lhs + rhs).to_array();

        assert_eq!(res, array::from_fn(|i| LHS_VALUES[i] + RHS_VALUES[i]));
    }

    #[test]
    fn subtraction_works() {
        let lhs = PackedBaseField::from_array(array::from_fn(|i| LHS_VALUES[i]));
        let rhs = PackedBaseField::from_array(array::from_fn(|i| RHS_VALUES[i]));

        let res = (lhs - rhs).to_array();

        assert_eq!(res, array::from_fn(|i| LHS_VALUES[i] - RHS_VALUES[i]));
    }

    #[test]
    fn multiplication_works() {
        let lhs = PackedBaseField::from_array(LHS_VALUES);
        let rhs = PackedBaseField::from_array(RHS_VALUES);

        let res = (lhs * rhs).to_array();

        assert_eq!(res, array::from_fn(|i| LHS_VALUES[i] * RHS_VALUES[i]));
    }

    #[test]
    fn negation_works() {
        const VALUES: [BaseField; N_LANES] = LHS_VALUES;
        let values = PackedBaseField::from_array(VALUES);

        let res = (-values).to_array();

        assert_eq!(res, VALUES.map(|v| -v));
    }
}
