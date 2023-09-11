use super::field::{M31, P};
use core::arch::x86_64::*;
use once_cell::sync::Lazy;
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

pub const kBlockSize: usize = 8;
// pub const kModulusVec: Lazy<__m512i> = unsafe { Lazy::new(|| _mm512_set1_epi64(P as i64)) };
// pub const kZero: Lazy<__m512i> = unsafe { Lazy::new(|| _mm512_set1_epi64(0)) };
// pub const kOne: Lazy<__m512i> = unsafe { Lazy::new(|| _mm512_set1_epi64(1)) };
// pub const kModulusBits: Lazy<__m512i> = unsafe { Lazy::new(|| _mm512_set1_epi64(31)) };

#[derive(Copy, Clone)]
pub struct Consts {
    p: __m512i,
    one: __m512i,
}

impl Consts {
    pub fn new() -> Self {
        Self {
            p: unsafe { _mm512_set1_epi64(P as i64) },
            one: unsafe { _mm512_set1_epi64(1) },
        }
    }
}

pub trait Operations {
    fn add(self, cn: &Consts, rhs: Self) -> Self;
    fn mul(self, cn: &Consts, rhs: Self) -> Self;
    fn neg(self, cn: &Consts) -> Self;
    fn sub(self, cn: &Consts, rhs: Self) -> Self;
    fn inv(self, cn: &Consts) -> Self;
    fn double(self, cn: &Consts) -> Self;
}

#[derive(Copy, Clone, Debug)]
pub struct M31Avx512(__m512i);
pub type AVXField = M31Avx512;

impl M31Avx512 {
    pub unsafe fn add(self, cn: &Consts, rhs: Self) -> Self {
        let sum = _mm512_add_epi64(self.0, rhs.0);
        let shifted_sum = _mm512_srli_epi64(sum, 31);
        Self(_mm512_and_si512(_mm512_add_epi64(sum, shifted_sum), cn.p))
    }

    pub fn load_avx512(v: &Vec<M31>) -> Self {
        unsafe {
            Self(_mm512_cvtepu32_epi64(_mm256_loadu_si256(
                v.as_ptr() as *const __m256i
            )))
        }
    }

    pub fn unload_avx512(self) -> Vec<M31> {
        let mut v = Vec::<M31>::new();
        unsafe { _mm512_storeu_si512(v.as_mut_ptr().cast::<i32>(), self.0) };
        v
    }

    fn reduce_avx512(cn: &Consts, a: __m512i) -> Self {
        unsafe {
            let a_plus_one: __m512i = _mm512_add_epi64(a, cn.one);
            let z: __m512i =
                _mm512_srli_epi64(_mm512_add_epi64(_mm512_srli_epi64(a, 31), a_plus_one), 31);
            let result: __m512i = _mm512_add_epi64(a, z);
            Self(_mm512_and_epi64(result, cn.p))
        }
    }
}

impl Operations for M31Avx512 {
    fn add(self, cn: &Consts, rhs: Self) -> Self {
        unsafe { self.add(cn, rhs) }
    }

    fn mul(self, cn: &Consts, rhs: Self) -> Self {
        unsafe { Self::partial_reduce_avx512(&cn, _mm512_mul_epu32(self.0, rhs.0)) }
    }

    fn neg(self, cn: &Consts) -> Self {
        unsafe { Self::reduce_avx512(&cn, _mm512_sub_epi64(cn.p, self.0)) }
    }

    fn sub(self, cn: &Consts, rhs: Self) -> Self {
        unsafe { self.add(cn, Self(_mm512_sub_epi64(cn.p, rhs.0))) }
    }

    fn inv(self, cn: &Consts) -> Self {
        self
    }

    fn double(self, cn: &Consts) -> Self {
        unsafe { self.add(cn, self) }
    }
}

// impl Add for M31Avx512 {
//     type Output = Self;

//     fn add(self, rhs: Self) -> Self::Output {
//         unsafe { self.add(rhs) }
//     }
// }

// impl AddAssign for M31Avx512 {
//     fn add_assign(&mut self, rhs: Self) {
//         *self = *self + rhs;
//     }
// }

// impl Mul for M31Avx512 {
//     type Output = Self;

//     fn mul(self, rhs: Self) -> Self::Output {
//         unsafe { Self::reduce_avx512(consts, &_mm512_mul_epu32(self.0, rhs.0)) }
//     }
// }

// impl MulAssign for M31Avx512 {
//     fn mul_assign(&mut self, rhs: Self) {
//         *self = *self * rhs;
//     }
// }

// impl Neg for M31Avx512 {
//     type Output = Self;

//     fn neg(self) -> Self::Output {
//         unsafe { Self::reduce_avx512(&_mm512_sub_epi64(*kModulusVec, self.0)) }
//     }
// }

// impl Sub for M31Avx512 {
//     type Output = Self;

//     fn sub(self, rhs: Self) -> Self::Output {
//         unsafe { self.add(Self(_mm512_sub_epi64(*kModulusVec, rhs.0))) }
//     }
// }

// impl SubAssign for M31Avx512 {
//     fn sub_assign(&mut self, rhs: Self) {
//         *self = *self - rhs;
//     }
// }
