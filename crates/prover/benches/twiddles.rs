use criterion::{criterion_group, criterion_main, Criterion};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::poly::circle::{CanonicCoset, PolyOps};

const LOG_SIZE: u32 = 26;
fn twiddles_benches(c: &mut Criterion) {
    c.bench_function(&format!("twiddles 2^{LOG_SIZE}"), |b| {
        b.iter(|| SimdBackend::precompute_twiddles(CanonicCoset::new(LOG_SIZE).coset()));
    });
}

criterion_group!(
        name = benches;
        config = Criterion::default().sample_size(10);
        targets = twiddles_benches);
criterion_main!(benches);
