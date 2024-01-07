use criterion::{black_box, criterion_group, criterion_main, Criterion};
use prover_research::core::fields::m31::BaseField;
use prover_research::core::fri::fold_line;
use prover_research::core::poly::line::LineEvaluation;

fn folding_benchmark(c: &mut Criterion) {
    const N: usize = 1 << 12;
    let evals = LineEvaluation::new(vec![BaseField::from_u32_unchecked(712837213); N]);
    let alpha = BaseField::from_u32_unchecked(12389);
    c.bench_function("fold_line", |b| {
        b.iter(|| {
            black_box(fold_line(black_box(&evals), black_box(alpha)));
        })
    });
}

criterion_group!(benches, folding_benchmark);
criterion_main!(benches);
