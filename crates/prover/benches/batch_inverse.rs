use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_prover::core::backend::simd::m31::PackedM31;
use stwo_prover::core::fields::batch_inverse;

pub const N_ELEMENTS: usize = 1 << 18;

pub fn m31_batch_inverse_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<PackedM31> = (0..N_ELEMENTS)
        .map(|_| PackedM31::from_array(std::array::from_fn(|_| rng.gen())))
        .collect();

    c.bench_function("M31 batch inverse not batched", |b| {
        b.iter(|| {
            black_box(batch_inverse(&elements));
        })
    });

    c.bench_function("M31 batched", |b| {
        b.iter(|| {
            black_box(batch_inverse(&elements));
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = m31_batch_inverse_bench,);
criterion_main!(benches);
