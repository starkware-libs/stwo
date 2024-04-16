use criterion::{black_box, criterion_group, criterion_main, Criterion};
use prover::core::backend::CPUBackend;
use prover::core::fields::m31::BaseField;
use prover::core::fields::qm31::SecureField;
use prover::core::fields::secure_column::SecureColumn;
use prover::core::fri::FriOps;
use prover::core::poly::circle::{CanonicCoset, PolyOps};
use prover::core::poly::line::{LineDomain, LineEvaluation};

fn folding_benchmark(c: &mut Criterion) {
    const LOG_SIZE: u32 = 12;
    let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());
    let evals = LineEvaluation::new(
        domain,
        SecureColumn {
            columns: std::array::from_fn(|i| {
                vec![BaseField::from_u32_unchecked(i as u32); 1 << LOG_SIZE]
            }),
        },
    );
    let alpha = SecureField::from_u32_unchecked(2213980, 2213981, 2213982, 2213983);
    let twiddles = CPUBackend::precompute_twiddles(domain.coset());
    c.bench_function("fold_line", |b| {
        b.iter(|| {
            black_box(CPUBackend::fold_line(
                black_box(&evals),
                black_box(alpha),
                &twiddles,
            ));
        })
    });
}

criterion_group!(benches, folding_benchmark);
criterion_main!(benches);
