use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use stwo_prover::core::backend::simd::column::BaseColumn;
use stwo_prover::core::backend::simd::prefix_sum::inclusive_prefix_sum;
use stwo_prover::core::fields::m31::BaseField;

pub fn simd_prefix_sum_bench(c: &mut Criterion) {
    const LOG_SIZE: u32 = 24;
    let evals: BaseColumn = (0..1 << LOG_SIZE).map(BaseField::from).collect();
    c.bench_function(&format!("simd prefix_sum 2^{LOG_SIZE}"), |b| {
        b.iter_batched(
            || evals.clone(),
            inclusive_prefix_sum,
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, simd_prefix_sum_bench);
criterion_main!(benches);
