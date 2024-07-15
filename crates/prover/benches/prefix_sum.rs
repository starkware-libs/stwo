#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::examples::xor::prefix_sum::exclusive_prefix_sum_simd;

pub fn simd_prefix_sum_bench(c: &mut Criterion) {
    const LOG_SIZE: u32 = 24;
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let vals = (0..1 << LOG_SIZE).map(BaseField::from).collect();
    let eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(domain, vals);
    c.bench_function(&format!("simd prefix_sum 2^{LOG_SIZE}"), |b| {
        b.iter_batched(
            || eval.clone(),
            exclusive_prefix_sum_simd,
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = simd_prefix_sum_bench);
criterion_main!(benches);
