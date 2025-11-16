use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::circle::CirclePoint;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::CpuBackend;
use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;

const LOG_SIZE: u32 = 20;

fn bench_barycentric_eval_at_secure_point<B: PolyOps>(c: &mut Criterion, id: &str) {
    let poly = CircleCoefficients::new((0..1 << LOG_SIZE).map(BaseField::from).collect());
    let coset = CanonicCoset::new(LOG_SIZE);
    let evals = poly.evaluate(coset.circle_domain());
    let mut rng = SmallRng::seed_from_u64(0);
    let x = rng.gen();
    let y = rng.gen();
    let point = CirclePoint { x, y };
    let weights =
        CircleEvaluation::<B, BaseField, BitReversedOrder>::barycentric_weights(coset, point);
    c.bench_function(
        &format!("{id} barycentric_eval_at_secure_field_point 2^{LOG_SIZE}"),
        |b| {
            b.iter(|| B::barycentric_eval_at_point(black_box(&evals), black_box(&weights)));
        },
    );
}

fn bench_barycentric_eval_at_secure_point_weights_calculation<B: PolyOps>(
    c: &mut Criterion,
    id: &str,
) {
    let mut rng = SmallRng::seed_from_u64(0);
    let x = rng.gen();
    let y = rng.gen();
    let point = CirclePoint { x, y };
    let coset = CanonicCoset::new(LOG_SIZE);
    c.bench_function(
        &format!("{id} barycentric_eval_at_secure_point_weights_calculation 2^{LOG_SIZE}"),
        |b| {
            b.iter(|| {
                CircleEvaluation::<B, BaseField, BitReversedOrder>::barycentric_weights(
                    black_box(coset),
                    black_box(point),
                )
            });
        },
    );
}

fn barycentric_eval_at_secure_point_benches(c: &mut Criterion) {
    bench_barycentric_eval_at_secure_point::<SimdBackend>(c, "simd");
    bench_barycentric_eval_at_secure_point::<CpuBackend>(c, "cpu");
    bench_barycentric_eval_at_secure_point_weights_calculation::<SimdBackend>(c, "simd");
    bench_barycentric_eval_at_secure_point_weights_calculation::<CpuBackend>(c, "cpu");
}

criterion_group!(
        name = benches;
        config = Criterion::default().sample_size(10);
        targets = barycentric_eval_at_secure_point_benches);
criterion_main!(benches);
