use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_prover::core::backend::cpu::CpuBackend;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::circle::CirclePoint;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::circle::{CirclePoly, PolyOps};

const LOG_SIZE: u32 = 20;

fn bench_eval_at_secure_point<B: PolyOps>(c: &mut Criterion, id: &str) {
    let poly = CirclePoly::new((0..1 << LOG_SIZE).map(BaseField::from).collect());
    let mut rng = SmallRng::seed_from_u64(0);
    let x = rng.gen();
    let y = rng.gen();
    let point = CirclePoint { x, y };
    c.bench_function(
        &format!("{id} eval_at_secure_field_point 2^{LOG_SIZE}"),
        |b| {
            b.iter(|| B::eval_at_point(black_box(&poly), black_box(point)));
        },
    );
}

fn eval_at_secure_point_benches(c: &mut Criterion) {
    bench_eval_at_secure_point::<SimdBackend>(c, "simd");
    bench_eval_at_secure_point::<CpuBackend>(c, "cpu");
}

criterion_group!(
        name = benches;
        config = Criterion::default().sample_size(10);
        targets = eval_at_secure_point_benches);
criterion_main!(benches);
