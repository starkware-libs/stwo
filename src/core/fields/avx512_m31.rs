use core::arch::x86_64::*;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::MulAssign;

use super::m31::{K_BITS, M31, P};
pub const K_BLOCK_SIZE: usize = 8;
pub const M512P: __m512i = unsafe { core::mem::transmute([P as u64; K_BLOCK_SIZE]) };
pub const M512ONE: __m512i = unsafe { core::mem::transmute([1u64; K_BLOCK_SIZE]) };

#[derive(Copy, Clone, Debug)]
pub struct M31AVX512(__m512i);

impl M31AVX512 {
    #[inline(always)]
    /// Given x1,...,x\[K_BLOCK_SIZE\] values, each in [0, 2*P], packed in x, returns packed
    /// yi in [0,P] where xi % \[P\] = yi % \[P\].
    fn partial_reduce(x: __m512i) -> Self {
        unsafe {
            let shifted_x = _mm512_srli_epi64(x, K_BITS);
            Self(_mm512_and_si512(_mm512_add_epi64(x, shifted_x), M512P))
        }
    }

    #[inline(always)]
    /// Given x1,...,x\[K_BLOCK_SIZE\] values, each in [0, P^2], packed in x, returns packed
    /// yi in [0,P] where xi % \[P\] = yi % \[P\].
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

impl Add for M31AVX512 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self::partial_reduce(_mm512_add_epi64(self.0, rhs.0)) }
    }
}

impl AddAssign for M31AVX512 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for M31AVX512 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self::reduce(_mm512_mul_epu32(self.0, rhs.0)) }
    }
}

impl MulAssign for M31AVX512 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[test]
fn test_avx512_mul() {
    if !crate::platform::avx512_detected() {
        return;
    }

    use rand::Rng;
    let mut rng = rand::thread_rng();

    let values = (0..K_BLOCK_SIZE)
        .map(|_x| M31::from_u32_unchecked(rng.gen::<u32>() % P))
        .collect::<Vec<M31>>();
    let avx_values = M31AVX512::from_vec(&values);

    assert_eq!(
        (avx_values + avx_values).to_vec(),
        values.iter().map(|x| x.double()).collect::<Vec<_>>()
    );
    assert_eq!(
        (avx_values * avx_values).to_vec(),
        values.iter().map(|x| x.square()).collect::<Vec<_>>()
    );
}
