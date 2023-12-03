use criterion::{black_box, criterion_group, criterion_main, Criterion};
use prover_research::core::fields::m31::BaseField;
use prover_research::core::fri::apply_drp;

fn fri_drp(c: &mut Criterion) {
    const N: usize = 1 << 18;
    let evals = (0..N as u32)
        .map(BaseField::from_u32_unchecked)
        .collect::<Vec<BaseField>>();
    let alpha = BaseField::from_u32_unchecked(12389);
    c.bench_function("drp", |b| {
        b.iter(|| {
            black_box(apply_drp(black_box(&evals), black_box(alpha)));
        })
    });
}

criterion_group!(benches, fri_drp);
criterion_main!(benches);
