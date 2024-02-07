use criterion::{black_box, criterion_group, criterion_main, Criterion};
use prover_research::core::fields::m31::BaseField;
// use prover_research::core::fields::qm31::ExtensionField;
use prover_research::core::fri::fold_line;
use prover_research::core::poly::circle::CanonicCoset;
use prover_research::core::poly::line::{LineDomain, LineEvaluation};

fn lookup_benchmark(c: &mut Criterion) {
    const LOG_SIZE: u32 = 12;
    let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE).coset());
    let evals = LineEvaluation::new(
        domain,
        vec![BaseField::from_u32_unchecked(712837213); 1 << LOG_SIZE],
    );
    let alpha = BaseField::from_u32_unchecked(12389);
    c.bench_function("grand_product", |b| {
        b.iter(|| {
            black_box(fold_line(black_box(&evals), black_box(alpha)));
        })
    });
}

// fn grand_product(col1: Vec<BaseField>, col2: Vec<BaseField>) -> Vec<ExtensionField> {

// }

criterion_group!(benches, lookup_benchmark);
criterion_main!(benches);
