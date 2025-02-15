use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stwo_prover::core::backend::simd::column::BaseFieldVec;
use stwo_prover::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::backend::CpuBackend;
use stwo_prover::core::fields::m31::{BaseField, M31};
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;
use stwo_prover::core::fri::FriOps;
use stwo_prover::core::poly::circle::{CanonicCoset, PolyOps};
use stwo_prover::core::poly::line::{LineDomain, LineEvaluation};

fn folding_benchmark_simd(c: &mut Criterion) {
    const LOG_SIZE: u32 = 12;
    let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());

    let evals = LineEvaluation::new(
        domain,
        SecureColumn {
            columns: std::array::from_fn(|i| BaseFieldVec {
                data: vec![
                    PackedBaseField::broadcast(M31::from_u32_unchecked(i as u32));
                    1 << (LOG_SIZE - LOG_N_LANES)
                ],
                length: 1 << LOG_SIZE,
            }),
        },
    );

    let alpha = SecureField::from_u32_unchecked(2213980, 2213981, 2213982, 2213983);
    let twiddles = SimdBackend::precompute_twiddles(domain.coset());
    c.bench_function(&format!("simd fold_line 2^{LOG_SIZE}"), |b| {
        b.iter(|| {
            black_box(SimdBackend::fold_line(
                black_box(&evals),
                black_box(alpha),
                &twiddles,
            ));
        })
    });
}

fn folding_benchmark_cpu(c: &mut Criterion) {
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
    c.bench_function(&format!("cpu fold_line 2^{LOG_SIZE}"), |b| {
        b.iter(|| {
            black_box(CpuBackend::fold_line(
                black_box(&evals),
                black_box(alpha),
                &twiddles,
            ));
        })
    });
}

fn folding_benchmark(c: &mut Criterion) {
    folding_benchmark_simd(c);
    folding_benchmark_cpu(c);
}

criterion_group!(benches, folding_benchmark);
criterion_main!(benches);
