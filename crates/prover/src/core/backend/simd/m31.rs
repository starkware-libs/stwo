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
    /// Constructs a new instance with all vector elements set to `value`.
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
    fn mul(self, rhs: Self) -> Self {
        // TODO: Come up with a better approach than `cfg`ing on target_feature.
        cfg_if::cfg_if! {
            if #[cfg(all(target_feature = "neon", target_arch = "aarch64"))] {
                _mul_neon(self, rhs)
            } else if #[cfg(all(target_feature = "simd128", target_arch = "wasm32"))] {
                _mul_wasm(self, rhs)
            } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
                _mul_avx512(self, rhs)
            } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2f"))] {
                _mul_avx2(self, rhs)
            } else {
                _mul_simd(self, rhs)
            }
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

#[cfg(target_arch = "aarch64")]
fn _mul_neon(a: PackedBaseField, b: PackedBaseField) -> PackedBaseField {
    use core::arch::aarch64::{int32x2_t, vqdmull_s32};
    use std::simd::u32x4;

    use crate::core::backend::simd::utils::{LoEvensConcatHiEvens, LoOddsConcatHiOdds};

    let [a0, a1, a2, a3, a4, a5, a6, a7]: [int32x2_t; 8] = unsafe { transmute(a) };
    let [b0, b1, b2, b3, b4, b5, b6, b7]: [int32x2_t; 8] = unsafe { transmute(b) };

    // Each c_i contains |0|prod_lo|prod_hi|0|0|prod_lo|prod_hi|0|
    let c0: u32x4 = unsafe { transmute(vqdmull_s32(a0, b0)) };
    let c1: u32x4 = unsafe { transmute(vqdmull_s32(a1, b1)) };
    let c2: u32x4 = unsafe { transmute(vqdmull_s32(a2, b2)) };
    let c3: u32x4 = unsafe { transmute(vqdmull_s32(a3, b3)) };
    let c4: u32x4 = unsafe { transmute(vqdmull_s32(a4, b4)) };
    let c5: u32x4 = unsafe { transmute(vqdmull_s32(a5, b5)) };
    let c6: u32x4 = unsafe { transmute(vqdmull_s32(a6, b6)) };
    let c7: u32x4 = unsafe { transmute(vqdmull_s32(a7, b7)) };

    // *_lo contain `|prod_lo|0|prod_lo|0|prod_lo0|0|prod_lo|0|`.
    let mut c0_c1_lo: u32x4 = LoEvensConcatHiEvens::concat_swizzle(c0, c1);
    let mut c2_c3_lo: u32x4 = LoEvensConcatHiEvens::concat_swizzle(c2, c3);
    let mut c4_c5_lo: u32x4 = LoEvensConcatHiEvens::concat_swizzle(c4, c5);
    let mut c6_c7_lo: u32x4 = LoEvensConcatHiEvens::concat_swizzle(c6, c7);

    // *_hi contain `|0|prod_hi|0|prod_hi|0|prod_hi|0|prod_hi|`.
    let c0_c1_hi: u32x4 = LoOddsConcatHiOdds::concat_swizzle(c0, c1);
    let c2_c3_hi: u32x4 = LoOddsConcatHiOdds::concat_swizzle(c2, c3);
    let c4_c5_hi: u32x4 = LoOddsConcatHiOdds::concat_swizzle(c4, c5);
    let c6_c7_hi: u32x4 = LoOddsConcatHiOdds::concat_swizzle(c6, c7);

    // *_lo contain `|0|prod_lo|0|prod_lo|0|prod_lo|0|prod_lo|`.
    c0_c1_lo >>= 1;
    c2_c3_lo >>= 1;
    c4_c5_lo >>= 1;
    c6_c7_lo >>= 1;

    let lo: PackedBaseField = unsafe { transmute([c0_c1_lo, c2_c3_lo, c4_c5_lo, c6_c7_lo]) };
    let hi: PackedBaseField = unsafe { transmute([c0_c1_hi, c2_c3_hi, c4_c5_hi, c6_c7_hi]) };

    lo + hi
}

#[cfg(target_arch = "wasm32")]
fn _mul_wasm(a: PackedBaseField, b: PackedBaseField) -> PackedBaseField {
    use core::arch::wasm32::{i64x2_extmul_high_u32x4, i64x2_extmul_low_u32x4, v128};
    use std::simd::u32x4;

    use crate::core::backend::simd::utils::{LoEvensConcatHiEvens, LoOddsConcatHiOdds};

    let [a0, a1, a2, a3]: [v128; 4] = unsafe { transmute(a) };

    let b_double = b.0 << 1;
    let [b_double0, b_double1, b_double2, b_double3]: [v128; 4] = unsafe { transmute(b_double) };

    let c0_lo: u32x4 = unsafe { transmute(i64x2_extmul_low_u32x4(a0, b_double0)) };
    let c0_hi: u32x4 = unsafe { transmute(i64x2_extmul_high_u32x4(a0, b_double0)) };
    let c1_lo: u32x4 = unsafe { transmute(i64x2_extmul_low_u32x4(a1, b_double1)) };
    let c1_hi: u32x4 = unsafe { transmute(i64x2_extmul_high_u32x4(a1, b_double1)) };
    let c2_lo: u32x4 = unsafe { transmute(i64x2_extmul_low_u32x4(a2, b_double2)) };
    let c2_hi: u32x4 = unsafe { transmute(i64x2_extmul_high_u32x4(a2, b_double2)) };
    let c3_lo: u32x4 = unsafe { transmute(i64x2_extmul_low_u32x4(a3, b_double3)) };
    let c3_hi: u32x4 = unsafe { transmute(i64x2_extmul_high_u32x4(a3, b_double3)) };

    let mut c0_even: u32x4 = LoEvensConcatHiEvens::concat_swizzle(c0_lo, c0_hi);
    let mut c1_even: u32x4 = LoEvensConcatHiEvens::concat_swizzle(c1_lo, c1_hi);
    let mut c2_even: u32x4 = LoEvensConcatHiEvens::concat_swizzle(c2_lo, c2_hi);
    let mut c3_even: u32x4 = LoEvensConcatHiEvens::concat_swizzle(c3_lo, c3_hi);

    c0_even >>= 1;
    c1_even >>= 1;
    c2_even >>= 1;
    c3_even >>= 1;

    let c0_odd: u32x4 = LoOddsConcatHiOdds::concat_swizzle(c0_lo, c0_hi);
    let c1_odd: u32x4 = LoOddsConcatHiOdds::concat_swizzle(c1_lo, c1_hi);
    let c2_odd: u32x4 = LoOddsConcatHiOdds::concat_swizzle(c2_lo, c2_hi);
    let c3_odd: u32x4 = LoOddsConcatHiOdds::concat_swizzle(c3_lo, c3_hi);

    let even: PackedBaseField = unsafe { transmute([c0_even, c1_even, c2_even, c3_even]) };
    let odd: PackedBaseField = unsafe { transmute([c0_odd, c1_odd, c2_odd, c3_odd]) };

    even + odd
}

#[cfg(target_arch = "x86_64")]
fn _mul_avx512(a: PackedBaseField, b: PackedBaseField) -> PackedBaseField {
    use std::arch::x86_64::{__m512i, _mm512_add_epi32, _mm512_mul_epu32, _mm512_srli_epi64};

    let a: __m512i = unsafe { transmute(a) };
    let b: __m512i = unsafe { transmute(b) };

    // Set up a word s.t. the lower half of each 64-bit word has the even 32-bit words of
    // the first operand.
    let a_e = a;
    // Set up a word s.t. the lower half of each 64-bit word has the odd 32-bit words of
    // the first operand.
    let a_o = unsafe { _mm512_srli_epi64(a, 32) };

    // Double the second operand.
    let b_dbl = unsafe { _mm512_add_epi32(b, b) };
    let b_dbl_e = b_dbl;
    let b_dbl_o = unsafe { _mm512_srli_epi64(b_dbl, 32) };

    // To compute prod = a * b start by multiplying a_e/odd by b_dbl_e/odd.
    let prod_dbl_e: u32x16 = unsafe { transmute(_mm512_mul_epu32(a_e, b_dbl_e)) };
    let prod_dbl_o: u32x16 = unsafe { transmute(_mm512_mul_epu32(a_o, b_dbl_o)) };

    // The result of a multiplication holds a*b in as 64-bits.
    // Each 64b-bit word looks like this:
    //               1    31       31    1
    // prod_dbl_e - |0|prod_e_h|prod_e_l|0|
    // prod_dbl_o - |0|prod_o_h|prod_o_l|0|

    // Interleave the even words of prod_dbl_e with the even words of prod_dbl_o:
    let mut prod_lo = LoEvensInterleaveHiEvens::concat_swizzle(prod_dbl_e, prod_dbl_o);
    // prod_lo -    |prod_dbl_o_l|0|prod_dbl_e_l|0|
    // Divide by 2:
    prod_lo >>= 1;
    // prod_lo -    |0|prod_o_l|0|prod_e_l|

    // Interleave the odd words of prod_dbl_e with the odd words of prod_dbl_o:
    let prod_hi = LoOddsInterleaveHiOdds::concat_swizzle(prod_dbl_e, prod_dbl_o);
    // prod_hi -    |0|prod_o_h|0|prod_e_h|

    unsafe { PackedBaseField::from_simd(prod_lo) + PackedBaseField::from_simd(prod_hi) }
}

#[cfg(target_arch = "x86_64")]
fn _mul_avx2(a: PackedBaseField, b: PackedBaseField) -> PackedBaseField {
    use std::arch::x86_64::{__m256i, _mm256_add_epi32, _mm256_mul_epu32, _mm256_srli_epi64};

    let [a0, a1]: [__m256i; 2] = unsafe { transmute(a) };
    let [b0, b1]: [__m256i; 2] = unsafe { transmute(b) };

    // Set up a word s.t. the lower half of each 64-bit word has the even 32-bit words of
    // the first operand.
    let a0_e = a0;
    let a1_e = a1;
    // Set up a word s.t. the lower half of each 64-bit word has the odd 32-bit words of
    // the first operand.
    let a0_o = unsafe { _mm256_srli_epi64(a0, 32) };
    let a1_o = unsafe { _mm256_srli_epi64(a1, 32) };

    // Double the second operand.
    let b0_dbl = unsafe { _mm256_add_epi32(b0, b0) };
    let b1_dbl = unsafe { _mm256_add_epi32(b1, b1) };
    let b0_dbl_e = b0_dbl;
    let b1_dbl_e = b1_dbl;
    let b0_dbl_o = unsafe { _mm256_srli_epi64(b0_dbl, 32) };
    let b1_dbl_o = unsafe { _mm256_srli_epi64(b1_dbl, 32) };

    // To compute prod = a * b start by multiplying a0/1_e/odd by b0/1_e/odd.
    let prod0_dbl_e = unsafe { _mm256_mul_epu32(a0_e, b0_dbl_e) };
    let prod0_dbl_o = unsafe { _mm256_mul_epu32(a0_o, b0_dbl_o) };
    let prod1_dbl_e = unsafe { _mm256_mul_epu32(a1_e, b1_dbl_e) };
    let prod1_dbl_o = unsafe { _mm256_mul_epu32(a1_o, b1_dbl_o) };

    let prod_dbl_e: u32x16 = unsafe { transmute([prod0_dbl_e, prod1_dbl_e]) };
    let prod_dbl_o: u32x16 = unsafe { transmute([prod0_dbl_o, prod1_dbl_o]) };

    // The result of a multiplication holds a*b in as 64-bits.
    // Each 64b-bit word looks like this:
    //               1    31       31    1
    // prod_dbl_e - |0|prod_e_h|prod_e_l|0|
    // prod_dbl_o - |0|prod_o_h|prod_o_l|0|

    // Interleave the even words of prod_dbl_e with the even words of prod_dbl_o:
    let mut prod_lo = LoEvensInterleaveHiEvens::concat_swizzle(prod_dbl_e, prod_dbl_o);
    // prod_lo -    |prod_dbl_o_l|0|prod_dbl_e_l|0|
    // Divide by 2:
    prod_lo >>= 1;
    // prod_lo -    |0|prod_o_l|0|prod_e_l|

    // Interleave the odd words of prod_dbl_e with the odd words of prod_dbl_o:
    let prod_hi = LoOddsInterleaveHiOdds::concat_swizzle(prod_dbl_e, prod_dbl_o);
    // prod_hi -    |0|prod_o_h|0|prod_e_h|

    unsafe { PackedBaseField::from_simd(prod_lo) + PackedBaseField::from_simd(prod_hi) }
}

/// Returns `a * b`. Should only be used in the absence of a platform specific implementation.
fn _mul_simd(a: PackedBaseField, b: PackedBaseField) -> PackedBaseField {
    const MASK_EVENS: Simd<u64, { N_LANES / 2 }> = Simd::from_array([0xFFFFFFFF; { N_LANES / 2 }]);

    // Set up a word s.t. the lower half of each 64-bit word has the even 32-bit words of
    // the first operand.
    let a_e = unsafe { transmute::<_, Simd<u64, { N_LANES / 2 }>>(a.0) & MASK_EVENS };
    // Set up a word s.t. the lower half of each 64-bit word has the odd 32-bit words of
    // the first operand.
    let a_o = unsafe { transmute::<_, Simd<u64, { N_LANES / 2 }>>(a) >> 32 };

    // Double the second operand.
    let b_dbl = b.0 << 1;
    let b_dbl_e = unsafe { transmute::<_, Simd<u64, { N_LANES / 2 }>>(b_dbl) & MASK_EVENS };
    let b_dbl_o = unsafe { transmute::<_, Simd<u64, { N_LANES / 2 }>>(b_dbl) >> 32 };

    // To compute prod = a * b start by multiplying
    // a_e/o by b_dbl_e/o.
    let prod_e_dbl = a_e * b_dbl_e;
    let prod_o_dbl = a_o * b_dbl_o;

    // The result of a multiplication holds a*b in as 64-bits.
    // Each 64b-bit word looks like this:
    //               1    31       31    1
    // prod_e_dbl - |0|prod_e_h|prod_e_l|0|
    // prod_o_dbl - |0|prod_o_h|prod_o_l|0|

    // Interleave the even words of prod_e_dbl with the even words of prod_o_dbl:
    // let prod_lows = _mm512_permutex2var_epi32(prod_e_dbl, EVENS_INTERLEAVE_EVENS,
    // prod_o_dbl);
    // prod_ls -    |prod_o_l|0|prod_e_l|0|
    let mut prod_lows = LoEvensInterleaveHiEvens::concat_swizzle(
        unsafe { transmute::<_, Simd<u32, N_LANES>>(prod_e_dbl) },
        unsafe { transmute::<_, Simd<u32, N_LANES>>(prod_o_dbl) },
    );
    // Divide by 2:
    prod_lows >>= 1;
    // prod_ls -    |0|prod_o_l|0|prod_e_l|

    // Interleave the odd words of prod_e_dbl with the odd words of prod_o_dbl:
    let prod_highs = LoOddsInterleaveHiOdds::concat_swizzle(
        unsafe { transmute::<_, Simd<u32, N_LANES>>(prod_e_dbl) },
        unsafe { transmute::<_, Simd<u32, N_LANES>>(prod_o_dbl) },
    );

    // prod_hs -    |0|prod_o_h|0|prod_e_h|
    PackedBaseField(prod_lows) + PackedBaseField(prod_highs)
}

#[cfg(test)]
mod tests {
    use std::array;

    use aligned::{Aligned, A64};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::PackedBaseField;
    use crate::core::fields::m31::BaseField;

    #[test]
    fn addition_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedBaseField::from_array(lhs);
        let packed_rhs = PackedBaseField::from_array(rhs);

        let res = packed_lhs + packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] + rhs[i]));
    }

    #[test]
    fn subtraction_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedBaseField::from_array(lhs);
        let packed_rhs = PackedBaseField::from_array(rhs);

        let res = packed_lhs - packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] - rhs[i]));
    }

    #[test]
    fn multiplication_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedBaseField::from_array(lhs);
        let packed_rhs = PackedBaseField::from_array(rhs);

        let res = packed_lhs * packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] * rhs[i]));
    }

    #[test]
    fn negation_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let values = rng.gen();
        let packed_values = PackedBaseField::from_array(values);

        let res = -packed_values;

        assert_eq!(res.to_array(), array::from_fn(|i| -values[i]));
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
