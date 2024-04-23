use std::mem::transmute;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::ptr;
use std::simd::cmp::SimdOrd;
use std::simd::{u32x16, Simd, Swizzle};

use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};

use crate::core::backend::simd::utils::{LoEvensInterleaveHiEvens, LoOddsInterleaveHiOdds};
use crate::core::fields::m31::{BaseField, M31, P};
use crate::core::fields::FieldExpOps;

// // #[cfg(any(target_feature = "neon", target_feature = "simd128"))]
// // #[cfg(any(target_feature = "neon", target_feature = "simd128"))]
// #[cfg(not(target_feature = "avx512"))]
// pub const LOG_N_LANES: u32 = 2;
// #[cfg(target_feature = "avx512")]
pub const LOG_N_LANES: u32 = 4;

pub const N_LANES: usize = 1 << LOG_N_LANES;

pub const MODULUS: Simd<u32, N_LANES> = Simd::from_array([P; N_LANES]);

/// Holds a vector of unreduced [`M31`] elements in the range `[0, P]`.
///
/// Implemented with [`std::simd`] to support multiple targets (avx512, neon, wasm etc.).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
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

    /// Reduces each element of the vector to the range `[0, P)`.
    fn reduce(self) -> PackedBaseField {
        Self(Simd::simd_min(self.0, self.0 - MODULUS))
    }

    /// Interleaves two vectors.
    pub fn interleave(self, other: Self) -> (Self, Self) {
        let (a, b) = self.0.interleave(other.0);
        (Self(a), Self(b))
    }

    /// Deinterleaves two vectors.
    pub fn deinterleave(self, other: Self) -> (Self, Self) {
        let (a, b) = self.0.deinterleave(other.0);
        (Self(a), Self(b))
    }

    /// Sums all the elements in the vector.
    pub fn pointwise_sum(self) -> M31 {
        self.to_array().into_iter().sum()
    }

    /// Doubles each element.
    pub fn double(self) -> Self {
        // TODO: Make more optimal.
        self + self
    }

    pub fn into_simd(self) -> Simd<u32, N_LANES> {
        self.0
    }

    /// # Safety
    ///
    /// Vector elements must be in the range `[0, P]`.
    pub unsafe fn from_simd(v: Simd<u32, N_LANES>) -> Self {
        Self(v)
    }

    /// # Safety
    ///
    /// Behavior is undefined if the pointer does not have the same alignment as
    /// [`PackedBaseField`]. The loaded `u32` values must be in the range `[0, P]`.
    pub unsafe fn load(mem_addr: *const u32) -> Self {
        Self(ptr::read(mem_addr as *const u32x16))
    }

    /// # Safety
    ///
    /// Behavior is undefined if the pointer does not have the same alignment as
    /// [`PackedBaseField`].
    pub unsafe fn store(self, dst: *mut u32) {
        ptr::write(dst as *mut u32x16, self.0)
    }
}

impl Add for PackedBaseField {
    type Output = Self;

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

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        const MASK_U32: Simd<u64, { N_LANES / 2 }> =
            Simd::from_array([0xFFFFFFFF; { N_LANES / 2 }]);

        unsafe {
            // TODO: This multiplication should be platform specific. avx512 should use
            // `_mm512_mul_epu32` and avoid the bit masks. wasm and neon can do pairwise
            // multiplication instruction.

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
                LoEvensInterleaveHiEvens::concat_swizzle(
                    transmute::<_, Simd<u32, N_LANES>>(prod_e_dbl),
                    transmute::<_, Simd<u32, N_LANES>>(prod_o_dbl),
                ) >> 1,
            );

            // Interleave the odd words of prod_e_dbl with the odd words of prod_o_dbl:
            let prod_highs = Self(LoOddsInterleaveHiOdds::concat_swizzle(
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

impl From<[BaseField; N_LANES]> for PackedBaseField {
    fn from(v: [BaseField; N_LANES]) -> Self {
        Self::from_array(v)
    }
}

#[cfg(test)]
mod tests {
    use std::array;
    use std::iter::zip;

    use aligned::{Aligned, A64};

    use super::PackedBaseField;
    use crate::core::fields::m31::{BaseField, M31, P};

    const LHS_VALUES: [BaseField; 16] = [
        M31(0),
        M31(1),
        M31(2),
        M31((P - 1) / 2),
        M31(10),
        M31((P + 1) / 2),
        M31(P - 2),
        M31(P - 1),
        M31(0),
        M31(1),
        M31(2),
        M31(10),
        M31((P - 1) / 2),
        M31((P + 1) / 2),
        M31(P - 2),
        M31(P - 1),
    ];

    const RHS_VALUES: [BaseField; 16] = [
        M31(0),
        M31(1),
        M31(2),
        M31((P - 1) / 2),
        M31(10),
        M31((P + 1) / 2),
        M31(P - 2),
        M31(P - 1),
        M31(P - 1),
        M31(P - 2),
        M31((P + 1) / 2),
        M31((P - 1) / 2),
        M31(10),
        M31(2),
        M31(1),
        M31(0),
    ];

    #[test]
    fn addition_works() {
        for (lhs, rhs) in zip(LHS_VALUES.array_chunks(), RHS_VALUES.array_chunks()) {
            let packed_lhs = PackedBaseField::from_array(*lhs);
            let packed_rhs = PackedBaseField::from_array(*rhs);

            let res = packed_lhs + packed_rhs;

            assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] + rhs[i]));
        }
    }

    #[test]
    fn subtraction_works() {
        for (lhs, rhs) in zip(LHS_VALUES.array_chunks(), RHS_VALUES.array_chunks()) {
            let packed_lhs = PackedBaseField::from_array(*lhs);
            let packed_rhs = PackedBaseField::from_array(*rhs);

            let res = packed_lhs - packed_rhs;

            assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] - rhs[i]));
        }
    }

    #[test]
    fn multiplication_works() {
        for (lhs, rhs) in zip(LHS_VALUES.array_chunks(), RHS_VALUES.array_chunks()) {
            let packed_lhs = PackedBaseField::from_array(*lhs);
            let packed_rhs = PackedBaseField::from_array(*rhs);

            let res = packed_lhs - packed_rhs;

            assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] - rhs[i]));
        }
    }

    #[test]
    fn negation_works() {
        for value in LHS_VALUES.array_chunks() {
            let packed_value = PackedBaseField::from_array(*value);

            let res = -packed_value;

            assert_eq!(res.to_array(), array::from_fn(|i| -value[i]));
        }
    }

    #[test]
    fn load_works() {
        let v: Aligned<A64, [u32; 16]> = Aligned(array::from_fn(|i| i as u32));

        let res = unsafe { PackedBaseField::load(v.as_ptr()) };

        assert_eq!(res.to_array().map(|v| v.0), *v);
    }

    #[test]
    fn store_works() {
        let v = PackedBaseField::from_array(array::from_fn(BaseField::from));

        let mut res: Aligned<A64, [u32; 16]> = Aligned([0; 16]);
        unsafe { v.store(res.as_mut_ptr()) };

        assert_eq!(*res, v.to_array().map(|v| v.0));
    }
}
