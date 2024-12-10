#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;
use stwo_prover::core::fields::m31::BaseField;

pub fn cpu_bit_rev(c: &mut Criterion) {
    use stwo_prover::core::backend::cpu::bit_reverse;
    // TODO(andrew): Consider using same size for all.
    const SIZE: usize = 1 << 24;
    let data = (0..SIZE).map(BaseField::from).collect_vec();
    c.bench_function("cpu bit_rev 24bit", |b| {
        b.iter_batched(
            || data.clone(),
            |mut data| bit_reverse(&mut data),
            BatchSize::LargeInput,
        );
    });
}

pub fn simd_bit_rev(c: &mut Criterion) {
    use stwo_prover::core::backend::simd::bit_reverse::bit_reverse_m31;
    use stwo_prover::core::backend::simd::column::BaseColumn;
    const SIZE: usize = 1 << 26;
    let data = (0..SIZE).map(BaseField::from).collect::<BaseColumn>();
    c.bench_function("simd bit_rev 26bit", |b| {
        b.iter_batched(
            || data.data.clone(),
            |mut data| bit_reverse_m31(&mut data),
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = simd_bit_rev, cpu_bit_rev);
criterion_main!(bit_rev);
