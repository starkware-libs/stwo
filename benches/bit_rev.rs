#![feature(iter_array_chunks)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stwo::core::fields::m31::BaseField;
use stwo::core::utils::bit_reverse;

pub fn cpu_bit_reverse(c: &mut Criterion) {
    const SIZE: usize = 1 << 16;
    let mut data: Vec<_> = (0..SIZE).map(BaseField::from).collect();

    c.bench_function("cpu bit_reverse 24bit", |b| {
        b.iter(|| {
            bit_reverse(black_box(&mut data));
        })
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_bit_reverse(c: &mut criterion::Criterion) {
    use bytemuck::cast_slice_mut;
    use stwo::core::backend::avx512::bit_reverse::bit_reverse_m31;
    use stwo::core::backend::avx512::m31::PackedBaseField;
    use stwo::platform::avx512_detected;

    if !avx512_detected() {
        return;
    }

    const SIZE: usize = 1 << 26;
    let data: Vec<_> = (0..SIZE).map(BaseField::from).collect();
    let mut data: Vec<_> = data
        .into_iter()
        .array_chunks::<16>()
        .map(PackedBaseField::from_array)
        .collect();

    c.bench_function("avx bit_reverse 26bit", |b| {
        b.iter(|| {
            bit_reverse_m31(cast_slice_mut(&mut data[..]));
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion_group!(benches, avx512_bit_reverse, cpu_bit_reverse);
#[cfg(not(target_arch = "x86_64"))]
criterion_group!(benches, cpu_bit_reverse);
criterion_main!(benches);
