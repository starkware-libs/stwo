#![feature(array_chunks)]

use std::ops::{Add, Mul, Sub};

use criterion::Criterion;
use stwo::core::fields::avx512_m31::M31AVX512;
use stwo::core::fields::m31::M31;

#[derive(Copy, Clone)]
struct QM31AVX512(pub [CM31AVX512; 2]);
impl QM31AVX512 {
    fn from_array(x: [M31; 16 * 4]) -> Self {
        unsafe { std::mem::transmute(x) }
    }
}
impl Add for QM31AVX512 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}
impl Mul for QM31AVX512 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let ac = self.0[0] * rhs.0[0];
        let bd = self.0[1] * rhs.0[1];
        let bd2 = bd + bd;
        let ac_p_bd = ac + bd;
        let m = (self.0[0] + self.0[1]) * (rhs.0[0] + rhs.0[1]) - ac_p_bd;
        // l = ac+bd*(1+2i) = ac+bd=2ibd
        let l = CM31AVX512([ac_p_bd.0[0] - bd2.0[1], ac_p_bd.0[1] + bd2.0[0]]);
        Self([l, m])
    }
}
#[derive(Copy, Clone)]
struct CM31AVX512(pub [M31AVX512; 2]);
impl Add for CM31AVX512 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}
impl Sub for CM31AVX512 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}
impl Mul for CM31AVX512 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let ac = self.0[0] * rhs.0[0];
        let bd = self.0[1] * rhs.0[1];
        let m = (self.0[0] + self.0[1]) * (rhs.0[0] + rhs.0[1]);
        Self([ac - bd, m - ac - bd])
    }
}

#[cfg(target_arch = "x86_64")]
pub fn gkr_bench(c: &mut criterion::Criterion) {
    use itertools::Itertools;

    const SIZE: usize = 1 << (28 - 4);
    let mut data = (0..SIZE as u32)
        .map(|i| QM31AVX512::from_array([M31::from_u32_unchecked(i); 16 * 4]))
        .collect_vec();

    c.bench_function("gkr", |b| {
        b.iter(|| {
            for x in data.array_chunks_mut::<8>() {
                for _ in 0..13 {
                    for y in &mut *x {
                        *y = *y * *y;
                    }
                }
                for _ in 0..22 {
                    for y in &mut *x {
                        *y = *y + *y;
                    }
                }
            }
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_group!(
    name=gkr;
    config = Criterion::default().sample_size(10);
    targets=gkr_bench);
criterion::criterion_main!(gkr);
