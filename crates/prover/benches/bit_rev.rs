#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;
use stwo_prover::core::backend::gpu::GpuBackend;
use stwo_prover::core::backend::ColumnOps;
use stwo_prover::core::fields::m31::{BaseField, M31};

pub fn cpu_bit_rev(c: &mut Criterion) {
    use stwo_prover::core::utils::bit_reverse;
    // TODO(andrew): Consider using same size for all.
    const SIZE: usize = 1 << 28;
    let data = (0..SIZE).map(BaseField::from).collect_vec();
    c.bench_function("cpu bit_rev 28bit", |b| {
        b.iter_batched(
            || data.clone(),
            |mut data| bit_reverse(&mut data),
            BatchSize::LargeInput,
        );
    });
}

pub fn simd_bit_rev(c: &mut Criterion) {
    use stwo_prover::core::backend::simd::bit_reverse::bit_reverse_m31;
    use stwo_prover::core::backend::simd::column::BaseFieldVec;
    const SIZE: usize = 1 << 28;
    let data = (0..SIZE).map(BaseField::from).collect::<BaseFieldVec>();
    c.bench_function("simd bit_rev 28 bit", |b| {
        b.iter_batched(
            || data.data.clone(),
            |mut data| bit_reverse_m31(&mut data),
            BatchSize::LargeInput,
        );
    });
}

pub fn gpu_bit_rev(c: &mut Criterion) {
    use stwo_prover::core::backend::gpu::column::BaseFieldCudaColumn;
    const SIZE: usize = 1 << 28;
    let mut data = BaseFieldCudaColumn::from_vec((0..SIZE).map(BaseField::from).collect_vec());

    c.bench_function("gpu bit_rev 28 bit", |b| {
        b.iter(|| {
            <GpuBackend as ColumnOps<M31>>::bit_reverse_column(&mut data);
        })
    });
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = gpu_bit_rev, simd_bit_rev, cpu_bit_rev);
criterion_main!(bit_rev);
