use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_prover::core::backend::gpu::column::{BaseFieldCudaColumn, SecureFieldCudaColumn};
use stwo_prover::core::backend::gpu::GpuBackend;
use stwo_prover::core::backend::simd::column::{BaseFieldVec, SecureFieldVec};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::backend::CpuBackend;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::FieldOps;

pub fn cpu_batch_inverse(c: &mut Criterion) {
    // TODO(andrew): Consider using same size for all.
    const SIZE: usize = 1 << 24;
    let data = (1..(SIZE + 1)).map(BaseField::from).collect_vec();
    let res = vec![BaseField::default(); SIZE];
    c.bench_function("cpu batch_inverse 24bit", |b| {
        b.iter_batched(
            || res.clone(),
            |mut res| <CpuBackend as FieldOps<BaseField>>::batch_inverse(&data, &mut res),
            BatchSize::LargeInput,
        );
    });
}

pub fn simd_batch_inverse(c: &mut Criterion) {
    // TODO(andrew): Consider using same size for all.
    const SIZE: usize = 1 << 28;
    let data = (0..SIZE).map(BaseField::from).collect::<BaseFieldVec>();
    let res = data.clone();
    c.bench_function("simd batch_inverse 28bit", |b| {
        b.iter_batched(
            || res.clone(),
            |mut res| <SimdBackend as FieldOps<BaseField>>::batch_inverse(&data, &mut res),
            BatchSize::LargeInput,
        );
    });
}

pub fn gpu_batch_inverse(c: &mut Criterion) {
    // TODO(andrew): Consider using same size for all.
    const SIZE: usize = 1 << 28;
    let data = BaseFieldCudaColumn::from_vec((0..SIZE).map(BaseField::from).collect_vec());
    let res = data.clone();
    c.bench_function("gpu batch_inverse 28bit", |b| {
        b.iter_batched(
            || res.clone(),
            |mut res| <GpuBackend as FieldOps<BaseField>>::batch_inverse(&data, &mut res),
            BatchSize::LargeInput,
        );
    });
}

pub fn simd_batch_inverse_secure_field(c: &mut Criterion) {
    // TODO(andrew): Consider using same size for all.
    const SIZE: usize = 1 << 26;

    let mut rng = SmallRng::seed_from_u64(0);
    let data: SecureFieldVec = (0..SIZE)
        .map(|_| rng.gen())
        .collect::<Vec<SecureField>>()
        .into_iter()
        .collect();

    let res = data.clone();
    c.bench_function("simd batch_inverse secure field 28bit", |b| {
        b.iter_batched(
            || res.clone(),
            |mut res| <SimdBackend as FieldOps<SecureField>>::batch_inverse(&data, &mut res),
            BatchSize::LargeInput,
        );
    });
}

pub fn gpu_batch_inverse_secure_field(c: &mut Criterion) {
    // TODO(andrew): Consider using same size for all.
    const SIZE: usize = 1 << 26;

    let mut rng = SmallRng::seed_from_u64(0);
    let data = SecureFieldCudaColumn::from_vec((0..SIZE).map(|_| rng.gen()).collect());

    let res = data.clone();
    c.bench_function("gpu batch_inverse secure field 28bit", |b| {
        b.iter_batched(
            || res.clone(),
            |mut res| <GpuBackend as FieldOps<SecureField>>::batch_inverse(&data, &mut res),
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    name = batch_inverse;
    config = Criterion::default().sample_size(10);
    targets = cpu_batch_inverse, simd_batch_inverse, gpu_batch_inverse, simd_batch_inverse_secure_field, gpu_batch_inverse_secure_field);
criterion_main!(batch_inverse);
