#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use itertools::Itertools;
use stwo_prover::core::backend::gpu::column::BaseFieldCudaColumn;
use stwo_prover::core::backend::gpu::GpuBackend;
use stwo_prover::core::backend::simd::column::BaseFieldVec;

use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::circle::{CanonicCoset, PolyOps};

pub fn simd_interpolate(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolate");

    for log_size in 16..=28 {
        let coset = CanonicCoset::new(log_size);
        let values: BaseFieldVec = (0..coset.size()).map(BaseField::from).collect();
        let evaluations = SimdBackend::new_canonical_ordered(coset, values);
        let twiddle_tree = SimdBackend::precompute_twiddles(coset.half_coset());
        group.bench_function(BenchmarkId::new("simd interpolate", log_size), |b| {
            b.iter_batched(
                || evaluations.clone(),
                |evaluations| SimdBackend::interpolate(evaluations, &twiddle_tree),
                BatchSize::LargeInput,
            )
        });
    }
}

pub fn gpu_interpolate(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolate");

    for log_size in 16..=28 {
        let coset = CanonicCoset::new(log_size);
        let values = BaseFieldCudaColumn::from_vec((0..coset.size()).map(BaseField::from).collect_vec());
        let evaluations = GpuBackend::new_canonical_ordered(coset, values);
        let twiddle_tree = GpuBackend::precompute_twiddles(coset.half_coset());
        group.bench_function(BenchmarkId::new("gpu interpolate", log_size), |b| {
            b.iter_batched(
                || evaluations.clone(),
                |evaluations| GpuBackend::interpolate(evaluations, &twiddle_tree),
                BatchSize::LargeInput,
            )
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = simd_interpolate, gpu_interpolate);
criterion_main!(benches);
