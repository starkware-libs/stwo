use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::poly::line::LineDomain;
use stwo::prover::backend::CpuBackend;
use stwo::prover::fri::FriOps;
use stwo::prover::line::LineEvaluation;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::secure_column::SecureColumnByCoords;

fn folding_benchmark(c: &mut Criterion) {
    const LOG_SIZE: u32 = 12;
    let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());
    let evals = LineEvaluation::new(
        domain,
        SecureColumnByCoords {
            columns: std::array::from_fn(|i| {
                vec![BaseField::from_u32_unchecked(i as u32); 1 << LOG_SIZE]
            }),
        },
    );
    let alpha = SecureField::from_u32_unchecked(2213980, 2213981, 2213982, 2213983);
    let twiddles = CpuBackend::precompute_twiddles(domain.coset());
    c.bench_function("fold_line", |b| {
        b.iter(|| {
            black_box(CpuBackend::fold_line(
                black_box(&evals),
                black_box(alpha),
                &twiddles,
            ));
        })
    });
}

criterion_group!(benches, folding_benchmark);
criterion_main!(benches);
