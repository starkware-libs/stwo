use core::arch::x86_64::{
    __m256i, __m512i, _mm256_loadu_si256, _mm256_storeu_si256, _mm512_add_epi32, _mm512_add_epi64,
    _mm512_and_epi64, _mm512_cvtepi64_epi32, _mm512_cvtepu32_epi64, _mm512_min_epu32,
    _mm512_mul_epu32, _mm512_srli_epi64, _mm512_sub_epi32, _mm512_sub_epi64, _mm512_set1_epi32,
};
use std::arch::x86_64::{ _mm512_and_si512, _mm512_castps_si512, _mm512_castsi512_ps, _mm512_mask_blend_epi32, _mm512_movehdup_ps, _mm512_slli_epi64};
use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use bytemuck::{AnyBitPattern, NoUninit, Zeroable};
use num_traits::One;

use super::m31::{M31, MODULUS_BITS, P};
pub const K_BLOCK_SIZE: usize = 8;
pub const M512P: __m512i = unsafe { core::mem::transmute([P as u64; K_BLOCK_SIZE]) };
pub const M512ONE: __m512i = unsafe { core::mem::transmute([1u64; K_BLOCK_SIZE]) };

#[derive(Copy, Clone, Debug)]
pub struct M31AVX512(pub(crate)__m512i);

unsafe impl AnyBitPattern for M31AVX512 {}

unsafe impl Zeroable for M31AVX512 {
    fn zeroed() -> Self{
        unsafe { core::mem::zeroed() }
    }
}

unsafe impl NoUninit for M31AVX512 {}

const P_BRODCAST: __m512i = unsafe { core::mem::transmute([P; 16]) };
impl M31AVX512 {
    /// Given x1,...,x\[K_BLOCK_SIZE\] values, each in [0, 2*\[P\]), packed in
    /// x, returns packed xi % \[P\].
    /// If xi == 2*\[P\], then it reduces to \[P\].
    /// Note that this function can be used for both reduced and unreduced
    /// representations. [0, 2*\[P\]) -> [0, \[P\]), [0, 2*\[P\]] -> [0,
    /// \[P\]].
    #[inline(always)]
    fn partial_reduce(x: __m512i) -> Self {
        unsafe {
            let x_minus_p = _mm512_sub_epi32(x, M512P);
            Self(_mm512_min_epu32(x, x_minus_p))
        }
    }

    /// Given x1,...,x\[K_BLOCK_SIZE\] values, each in [0, \[P\]^2), packed in
    /// x, returns packed xi % \[P\].
    /// If xi == \[P\]^2, then it reduces to \[P\].
    /// Note that this function can be used for both reduced and unreduced
    /// representations. [0, \[P\]^2) -> [0, \[P\]), [0, \[P\]^2] -> [0,
    /// \[P\]].
    #[inline(always)]
    pub fn reduce(x: __m512i) -> Self {
        unsafe {
            let x_plus_one: __m512i = _mm512_add_epi64(x, M512ONE);

            // z_i = x_i // P (integer division).
            let z: __m512i = _mm512_srli_epi64(
                _mm512_add_epi64(_mm512_srli_epi64(x, MODULUS_BITS), x_plus_one),
                MODULUS_BITS,
            );
            let result: __m512i = _mm512_add_epi64(x, z);
            Self(_mm512_and_epi64(result, M512P))
        }
    }

    pub fn from_slice(v: &[M31]) -> M31AVX512 {
        unsafe {
            Self(_mm512_cvtepu32_epi64(_mm256_loadu_si256(
                v.as_ptr() as *const __m256i
            )))
        }
    }

    pub fn from_m512_unchecked(x: __m512i) -> Self {
        Self(x)
    }

    pub fn to_vec(self) -> Vec<M31> {
        unsafe {
            let mut v = Vec::with_capacity(K_BLOCK_SIZE);
            _mm256_storeu_si256(
                v.as_mut_ptr() as *mut __m256i,
                _mm512_cvtepi64_epi32(self.0),
            );
            v.set_len(K_BLOCK_SIZE);
            v
        }
    }

    fn square(&self) -> Self {
        (*self) * (*self)
    }

    fn pow(&self, exp: u128) -> Self {
        let mut res = Self::one();
        let mut base = *self;
        let mut exp = exp;
        while exp > 0 {
            if exp & 1 == 1 {
                res *= base;
            }
            base = base.square();
            exp >>= 1;
        }
        res
    }

    pub fn inverse(&self) -> Self {
        self.pow(P as u128 - 2)
    }

    fn movehdup_epi32(x: __m512i) -> __m512i {
        // The instruction is only available in the floating-point flavor; this distinction is only for
        // historical reasons and no longer matters. We cast to floats, duplicate, and cast back.
        unsafe {
            _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)))
        }
    }
}

impl One for M31AVX512 {
    fn one() -> Self {
        Self(unsafe { _mm512_set1_epi32(1) })
    }
}

impl Display for M31AVX512 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = self.to_vec();
        for elem in v.iter() {
            write!(f, "{} ", elem)?;
        }
        Ok(())
    }
}

impl Add for M31AVX512 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self::partial_reduce(_mm512_add_epi64(self.0, rhs.0)) }
    }
}

impl AddAssign for M31AVX512 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for M31AVX512 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            // vpmuludq only reads the bottom 32 bits of every 64-bit quadword.
            // The even indices are already in the bottom 32 bits of a quadword, so we can leave them.
            let lhs_evn = self.0;
            let rhs_evn = rhs.0;
            // Right shift by 31 is equivalent to moving the high 32 bits down to the low 32, and then
            // doubling it. So these are the odd indices in lhs, but doubled.
            let lhs_odd_dbl = _mm512_srli_epi64::<31>(self.0);
            // Copy the high 32 bits in each quadword of rhs down to the low 32.
            let rhs_odd = Self::movehdup_epi32(rhs.0);
    
            // Multiply odd indices; since lhs_odd_dbl is doubled, these products are also doubled.
            // prod_odd_dbl.quadword[i] = 2 * lsh.doubleword[2 * i + 1] * rhs.doubleword[2 * i + 1]
            let prod_odd_dbl = _mm512_mul_epu32(rhs_odd, lhs_odd_dbl);
            // Multiply even indices.
            // prod_evn.quadword[i] = lsh.doubleword[2 * i] * rhs.doubleword[2 * i]
            let prod_evn = _mm512_mul_epu32(rhs_evn, lhs_evn);
    
            // We now need to extract the low 31 bits and the high 31 bits of each 62 bit product and
            // prepare to add them.
            // Put the low 31 bits of the product (recall that it is shifted left by 1) in an odd
            // doubleword. (Notice that the high 31 bits are already in an odd doubleword in
            // prod_odd_dbl.) We will still need to clear the sign bit, hence we mark it _dirty.
            let prod_odd_lo_dirty = _mm512_slli_epi64::<31>(prod_odd_dbl);
            // Put the high 31 bits in an even doubleword, again noting that in prod_evn the even
            // doublewords contain the low 31 bits (with a dirty sign bit).
            let prod_evn_hi = _mm512_srli_epi64::<31>(prod_evn);
    
            // Put all the low halves of all the products into one vector. Take the even values from
            // prod_evn and odd values from prod_odd_lo_dirty. Note that the sign bits still need
            // clearing.
            let prod_lo_dirty = _mm512_mask_blend_epi32(0b101010100101010,prod_evn, prod_odd_lo_dirty);
            // Now put all the high halves into one vector. The even values come from prod_evn_hi and
            // the odd values come from prod_odd_dbl.
            let prod_hi = _mm512_mask_blend_epi32(0b101010100101010,prod_evn_hi, prod_odd_dbl);
            // Clear the most significant bit.
            let prod_lo = _mm512_and_si512(prod_lo_dirty, P_BRODCAST);
    
            // Standard addition of two 31-bit values.
            M31AVX512(prod_lo) + M31AVX512(prod_hi)
        }
    }
}

impl MulAssign for M31AVX512 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for M31AVX512 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { Self::partial_reduce(_mm512_sub_epi64(M512P, self.0)) }
    }
}

impl Sub for M31AVX512 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe {
            let a_minus_b = _mm512_sub_epi32(self.0, rhs.0);
            Self(_mm512_min_epu32(
                a_minus_b,
                _mm512_add_epi32(a_minus_b, M512P),
            ))
        }
    }
}


impl SubAssign for M31AVX512 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

#[cfg(test)]
mod tests {
    use core::arch::x86_64::_mm512_loadu_epi64;

    use rand::Rng;

    use super::{K_BLOCK_SIZE, M31AVX512};
    use crate::core::fields::m31::{M31, P};
    use crate::core::fields::Field;
    use crate::m31;

    /// Tests field operations where field elements are in reduced form.
    #[test]
    fn test_avx512_basic_ops() {
        if !crate::platform::avx512_detected() {
            return;
        }

        let values = [0, 1, 2, 10, (P - 1) / 2, (P + 1) / 2, P - 2, P - 1]
            .map(M31::from_u32_unchecked)
            .to_vec();
        let avx_values = M31AVX512::from_slice(&values);

        assert_eq!(
            (avx_values + avx_values).to_vec(),
            values.iter().map(|x| x.double()).collect::<Vec<_>>()
        );
        assert_eq!(
            (avx_values * avx_values).to_vec(),
            values.iter().map(|x| x.square()).collect::<Vec<_>>()
        );
        assert_eq!(
            (-avx_values).to_vec(),
            values.iter().map(|x| -*x).collect::<Vec<_>>()
        );
    }

    /// Tests that reduce functions are correct.
    #[test]
    fn test_reduce() {
        if !crate::platform::avx512_detected() {
            return;
        }
        let mut rng = rand::thread_rng();

        let const_values = [0, 1, (P + 1) / 2, P - 1, P, P + 1, 2 * P - 1, 2 * P];
        let avx_const_values =
            M31AVX512::from_slice(const_values.map(M31::from_u32_unchecked).as_ref());

        // Tests partial reduce.
        assert_eq!(
            M31AVX512::partial_reduce(avx_const_values.0).to_vec(),
            const_values
                .iter()
                .map(|x| m31!(if *x == 2 * P { P } else { x % P }))
                .collect::<Vec<_>>()
        );

        // Generate random values in [0, P^2).
        let rand_values = (0..K_BLOCK_SIZE)
            .map(|_x| rng.gen::<u64>() % (P as u64).pow(2))
            .collect::<Vec<u64>>();
        let avx_rand_values = M31AVX512::from_m512_unchecked(unsafe {
            _mm512_loadu_epi64(rand_values.as_ptr() as *const i64)
        });

        // Tests reduce.
        assert_eq!(
            M31AVX512::reduce(avx_const_values.0).to_vec(),
            const_values.iter().map(|x| m31!(x % P)).collect::<Vec<_>>()
        );

        assert_eq!(
            M31AVX512::reduce(avx_rand_values.0).to_vec(),
            rand_values
                .iter()
                .map(|x| m31!((x % P as u64) as u32))
                .collect::<Vec<_>>()
        );
    }
}
