#![feature(iter_array_chunks)]

use criterion::Criterion;

#[cfg(target_arch = "x86_64")]
pub fn cpu_bit_rev(c: &mut criterion::Criterion) {
    use stwo_prover::core::fields::m31::BaseField;

    const SIZE: usize = 1 << 24;
    let mut data: Vec<_> = (0..SIZE as u32)
        .map(BaseField::from_u32_unchecked)
        .collect();

    c.bench_function("cpu bit_rev 24bit", |b| {
        b.iter(|| {
            stwo_prover::core::utils::bit_reverse(&mut data);
        })
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_bit_rev(c: &mut criterion::Criterion) {
    use bytemuck::cast_slice_mut;
    use stwo_prover::core::backend::avx512::bit_reverse::bit_reverse_m31;
    use stwo_prover::core::backend::avx512::m31::PackedBaseField;
    use stwo_prover::core::fields::m31::BaseField;
    use stwo_prover::platform;
    if !platform::avx512_detected() {
        return;
    }

    const SIZE: usize = 1 << 26;
    let data: Vec<_> = (0..SIZE as u32)
        .map(BaseField::from_u32_unchecked)
        .collect();
    let mut data: Vec<_> = data
        .into_iter()
        .array_chunks::<16>()
        .map(PackedBaseField::from_array)
        .collect();

    c.bench_function("avx bit_rev 26bit", |b| {
        b.iter(|| {
            bit_reverse_m31(cast_slice_mut(&mut data[..]));
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_group!(
    name=avx_bit_rev;
    config = Criterion::default().sample_size(10);
    targets=avx512_bit_rev, cpu_bit_rev);
criterion::criterion_main!(avx_bit_rev);
