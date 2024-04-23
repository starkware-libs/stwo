#![feature(iter_array_chunks)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use stwo::core::backend::simd::column::BaseFieldVec;
use stwo::core::fields::m31::BaseField;

pub fn cpu_bit_rev(c: &mut Criterion) {
    use stwo::core::utils::bit_reverse;
    const SIZE: usize = 1 << 24;
    let mut data = (0..SIZE).map(BaseField::from).collect_vec();
    c.bench_function("cpu bit_rev 24bit", |b| {
        b.iter(|| bit_reverse(black_box(&mut data)))
    });
}

pub fn simd_bit_rev(c: &mut Criterion) {
    use stwo::core::backend::simd::bit_reverse::bit_reverse_m31;
    const SIZE: usize = 1 << 26;
    let mut data = (0..SIZE).map(BaseField::from).collect::<BaseFieldVec>();
    c.bench_function("simd bit_rev 26bit", |b| {
        b.iter(|| bit_reverse_m31(black_box(&mut data.data)))
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_bit_rev(c: &mut Criterion) {
    use stwo::core::backend::avx512::bit_reverse::bit_reverse_m31;
    const SIZE: usize = 1 << 26;
    if !stwo::platform::avx512_detected() {
        return;
    }
    let mut data = (0..SIZE).map(BaseField::from).collect::<BaseFieldVec>();
    c.bench_function("avx bit_rev 26bit", |b| {
        b.iter(|| bit_reverse_m31(black_box(&mut data.data)))
    });
}

#[cfg(target_arch = "x86_64")]
criterion_group!(
    name=bit_rev;
    config = Criterion::default().sample_size(10);
    targets=avx512_bit_rev, simd_bit_rev, cpu_bit_rev);
#[cfg(not(target_arch = "x86_64"))]
criterion_group!(
    name=bit_rev;
    config = Criterion::default().sample_size(10);
    targets=simd_bit_rev, cpu_bit_rev);
criterion_main!(bit_rev);
