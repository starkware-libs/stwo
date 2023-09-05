use super::field::P;
use core::arch::x86_64::*;
use std::fmt::Display;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::DivAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;

pub const kModulusVec: __m512i = _mm512_set1_epi64(P as i64);
pub const kZero: __m512i = _mm512_set1_epi64(0);
pub const kOne: __m512i = _mm512_set1_epi64(1);
pub const kModulusBits: __m512i = _mm512_set1_epi64(31);

pub trait AVX512Field {
    const BlockSize: usize = 8;
    fn load_avx512(v: &Vec<i32>) -> Self;
    fn unload_avx512(self) -> Vec<i32>;
    fn reduce_avx512(a: &__m512i) -> Self;
}

pub struct M31Avx512(__m512i);
pub type AVXField = M31Avx512;

impl M31Avx512 {
    pub fn one() -> M31Avx512 {
        Self(kOne)
    }

    pub fn zero() -> M31Avx512 {
        Self(kZero)
    }

    fn add(self, rhs: Self) -> Self {
        let sum = _mm512_add_epi64(self.0, rhs.0);
        let shifted_sum = _mm512_srli_epi64(sum, 31);
        Self(_mm512_and_si512(
            _mm512_add_epi64(sum, shifted_sum),
            kModulusVec,
        ))
    }

    pub fn double(self) -> Self {
        self.add(self)
    }
}

impl AVX512Field for M31Avx512 {
    fn load_avx512(v: &Vec<i32>) -> Self {
        Self(_mm512_cvtepu32_epi64(_mm256_loadu_si256(
            v.as_ptr() as *const __m256i
        )))
    }

    fn unload_avx512(self) -> Vec<i32> {
        let mut v = Vec::<i32>::new();
        _mm512_storeu_si512(v.as_mut_ptr().cast::<i32>(), self.0);
        v
    }

    fn reduce_avx512(a: &__m512i) -> Self {
        let a_plus_one: __m512i = _mm512_add_epi64(*a, kOne);
        let z: __m512i = _mm512_srli_epi64(
            _mm512_add_epi64(_mm512_srli_epi64(*a, 31), a_plus_one),
            31,
        );
        let result: __m512i = _mm512_add_epi64(*a, z);
        Self(_mm512_and_epi64(result, kModulusVec))
    }
}

impl Add for M31Avx512 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl AddAssign for M31Avx512 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for M31Avx512 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::reduce_avx512(&_mm512_mul_epu32(self.0, rhs.0))
    }
}

impl MulAssign for M31Avx512 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for M31Avx512 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::reduce_avx512(&_mm512_sub_epi64(kModulusVec, self.0))
    }
}

impl Sub for M31Avx512 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.add(Self(_mm512_sub_epi64(kModulusVec, self.0)))
    }
}

impl SubAssign for M31Avx512 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
