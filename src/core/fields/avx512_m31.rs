use core::arch::x86_64::{
    __m256i, __m512i, _mm256_loadu_si256, _mm256_storeu_si256, _mm512_add_epi32, _mm512_add_epi64,
    _mm512_add_epi64, _mm512_add_epi64, _mm512_and_epi64, _mm512_cvtepi64_epi32,
    _mm512_cvtepu32_epi64, _mm512_loadu_epi64, _mm512_min_epu32, _mm512_mul_epu32,
    _mm512_srli_epi64, _mm512_srli_epi64, _mm512_sub_epi32, _mm512_sub_epi64,
};
use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::m31::{K_BITS, M31, P};
pub const K_BLOCK_SIZE: usize = 8;
pub const M512P: __m512i = unsafe { core::mem::transmute([P as u64; K_BLOCK_SIZE]) };
pub const M512ONE: __m512i = unsafe { core::mem::transmute([1u64; K_BLOCK_SIZE]) };

#[derive(Copy, Clone, Debug)]
pub struct M31AVX512(__m512i);

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
    fn reduce(x: __m512i) -> Self {
        unsafe {
            let x_plus_one: __m512i = _mm512_add_epi64(x, M512ONE);

            // z_i = x_i // P (integer division).
            let z: __m512i = _mm512_srli_epi64(
                _mm512_add_epi64(_mm512_srli_epi64(x, K_BITS), x_plus_one),
                K_BITS,
            );
            let result: __m512i = _mm512_add_epi64(x, z);
            Self(_mm512_and_epi64(result, M512P))
        }
    }

    pub fn from_vec(v: &Vec<M31>) -> M31AVX512 {
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
        unsafe { Self::reduce(_mm512_mul_epu32(self.0, rhs.0)) }
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

    /// Tests field operations where field elements are in reduced form.
    #[test]
    fn test_avx512_basic_ops() {
        if !crate::platform::avx512_detected() {
            return;
        }

        let values = [0, 1, 2, 10, (P - 1) / 2, (P + 1) / 2, P - 2, P - 1]
            .map(M31::from_u32_unchecked)
            .to_vec();
        let avx_values = M31AVX512::from_vec(&values);

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
            M31AVX512::from_vec(&const_values.map(M31::from_u32_unchecked).to_vec());

        // Tests partial reduce.
        assert_eq!(
            M31AVX512::partial_reduce(avx_const_values.0).to_vec(),
            const_values
                .iter()
                .map(|x| M31::from_u32_unchecked(if *x == 2 * P { P } else { x % P }))
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
            const_values
                .iter()
                .map(|x| M31::from_u32_unchecked(x % P))
                .collect::<Vec<_>>()
        );

        assert_eq!(
            M31AVX512::reduce(avx_rand_values.0).to_vec(),
            rand_values
                .iter()
                .map(|x| M31::from_u32_unchecked((x % P as u64) as u32))
                .collect::<Vec<_>>()
        );
    }
}
