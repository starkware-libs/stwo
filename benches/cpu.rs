use criterion::{criterion_group, criterion_main, Criterion};
use num_traits::One;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::backend::avx512::AVX512Backend;
use stwo::core::backend::CPUBackend;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::{Col, Column, FieldOps};

fn batch_inverse_bench(c: &mut Criterion) {
    const N_ELEMENTS: usize = 1 << 18;
    let mut rng = SmallRng::seed_from_u64(0);
    let elements: Vec<BaseField> = (0..N_ELEMENTS)
        .map(|_| {
            BaseField::from_u32_unchecked(
                rng.gen::<u32>(),

            )
        })
        .collect();

    let mut dst = vec![BaseField::one(); N_ELEMENTS];

    c.bench_function("batch_inverse_bench", |b| {
        b.iter(|| CPUBackend::batch_inverse(&elements, &mut dst));
    });
}

fn batch_inverse_avx512_bench(c: &mut Criterion) {
    const N_ELEMENTS: usize = 1 << 18;
    let col = Col::<AVX512Backend, BaseField>::from_iter((1..N_ELEMENTS + 1).map(BaseField::from));
    let mut dst = Col::<AVX512Backend, BaseField>::zeros(N_ELEMENTS);

    c.bench_function("batch_inverse_avx512_bench", |b| {
        b.iter(|| AVX512Backend::batch_inverse(&col, &mut dst));
    });
}

criterion_group!(
    cpu_backend_benches,
    batch_inverse_bench,
    batch_inverse_avx512_bench,
);

criterion_main!(cpu_backend_benches);
