use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stwo::core::fields::m31::BaseField;
use stwo::core::fri::fold_line;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::poly::line::{LineDomain, LineEvaluation};

fn folding_benchmark(c: &mut Criterion) {
    const LOG_SIZE: u32 = 12;
    let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());
    let evals = LineEvaluation::new(
        domain,
        vec![BaseField::from_u32_unchecked(712837213); 1 << LOG_SIZE],
    );
    let alpha = BaseField::from_u32_unchecked(12389);
    c.bench_function("fold_line", |b| {
        b.iter(|| {
            black_box(fold_line(black_box(&evals), black_box(alpha)));
        })
    });
}

criterion_group!(benches, folding_benchmark);
criterion_main!(benches);
