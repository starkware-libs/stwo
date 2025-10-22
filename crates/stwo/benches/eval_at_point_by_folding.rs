use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::circle::CirclePoint;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::cpu::CpuBackend;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::{CirclePoly, PolyOps};

const LOG_SIZE: u32 = 20;

fn bench_eval_at_secure_point_by_folding<B: PolyOps>(c: &mut Criterion, id: &str) {
    let poly = CirclePoly::new((0..1 << LOG_SIZE).map(BaseField::from).collect());
    let twiddles =
        B::precompute_twiddles(CanonicCoset::new(LOG_SIZE + 1).circle_domain().half_coset);
    let evals = poly.evaluate(CanonicCoset::new(LOG_SIZE).circle_domain());
    let mut rng = SmallRng::seed_from_u64(0);
    let x = rng.gen();
    let y = rng.gen();
    let point = CirclePoint { x, y };
    c.bench_function(
        &format!("{id} eval_at_secure_field_point_by_folding 2^{LOG_SIZE}"),
        |b| {
            b.iter(|| {
                B::eval_at_point_by_folding(
                    black_box(&evals),
                    black_box(point),
                    black_box(&twiddles),
                )
            });
        },
    );
}

fn eval_at_secure_point_by_folding_benches(c: &mut Criterion) {
    bench_eval_at_secure_point_by_folding::<SimdBackend>(c, "simd");
    bench_eval_at_secure_point_by_folding::<CpuBackend>(c, "cpu");
}

criterion_group!(
        name = benches;
        config = Criterion::default().sample_size(10);
        targets = eval_at_secure_point_by_folding_benches);
criterion_main!(benches);
