use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::circle::SECURE_FIELD_CIRCLE_GEN;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::quotients::{
    build_samples_with_randomness_and_periodicity, ColumnSampleBatch, PointSample,
};
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::pcs::quotient_ops::AccumulatedNumerators;
use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::QuotientOps;

#[allow(clippy::type_complexity)]
fn setup(
    trace_log_size: u32,
    log_blowup_factor: u32,
    n_cols: usize,
) -> (
    Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    Vec<ColumnSampleBatch>,
) {
    let mut rng = SmallRng::seed_from_u64(0);

    let eval_log_size = trace_log_size + log_blowup_factor;
    let eval_domain = CanonicCoset::new(eval_log_size).circle_domain();
    let twiddles = SimdBackend::precompute_twiddles(eval_domain.half_coset);

    let polys: Vec<CircleCoefficients<SimdBackend>> = (0..n_cols)
        .map(|_| {
            CircleCoefficients::new((0..1 << trace_log_size).map(|_| rng.gen::<M31>()).collect())
        })
        .collect();

    let columns: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> = polys
        .iter()
        .map(|poly| poly.evaluate_with_twiddles(eval_domain, &twiddles))
        .collect();

    let sample_points = [SECURE_FIELD_CIRCLE_GEN, SECURE_FIELD_CIRCLE_GEN.double()];

    let samples: Vec<Vec<PointSample>> = polys
        .iter()
        .map(|poly| {
            sample_points
                .iter()
                .map(|&point| PointSample {
                    point,
                    value: poly.eval_at_point(point),
                })
                .collect()
        })
        .collect();

    let random_coeff =
        SecureField::from_m31_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>())));

    let sample_batches = ColumnSampleBatch::new_vec(
        &build_samples_with_randomness_and_periodicity(
            &TreeVec(vec![samples]),
            vec![vec![eval_log_size; n_cols].into_iter()],
            eval_log_size,
            random_coeff,
        )
        .iter()
        .flatten()
        .collect_vec(),
    );
    (columns, sample_batches)
}

fn bench_accumulate_numerators(c: &mut Criterion) {
    let trace_log_size = 20;
    let log_blowup_factor = 1;
    let eval_log_size = trace_log_size + log_blowup_factor;
    let n_cols = 100;
    let (columns, sample_batches) = setup(trace_log_size, log_blowup_factor, n_cols);
    let col_refs: Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> =
        columns.iter().collect();

    c.bench_function(
        &format!("accumulate_numerators 2^{eval_log_size} x {n_cols} cols"),
        |b| {
            b.iter_batched(
                Vec::<AccumulatedNumerators<SimdBackend>>::new,
                |mut acc| {
                    SimdBackend::accumulate_numerators(
                        black_box(&col_refs),
                        black_box(&sample_batches),
                        black_box(&mut acc),
                    );
                    acc
                },
                BatchSize::LargeInput,
            );
        },
    );
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_accumulate_numerators
);
criterion_main!(benches);
