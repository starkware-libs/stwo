use criterion::{black_box, Criterion};

#[cfg(target_arch = "x86_64")]
pub fn cpu_eval_at_secure_point(c: &mut criterion::Criterion) {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use stwo_prover::core::backend::CPUBackend;
    use stwo_prover::core::circle::CirclePoint;
    use stwo_prover::core::fields::m31::BaseField;
    use stwo_prover::core::fields::qm31::QM31;
    use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
    use stwo_prover::core::poly::NaturalOrder;
    let log_size = 20;
    let mut rng = SmallRng::seed_from_u64(0);

    let domain = CanonicCoset::new(log_size as u32).circle_domain();
    let evaluation = CircleEvaluation::<CPUBackend, _, NaturalOrder>::new(
        domain,
        (0..(1 << log_size))
            .map(BaseField::from_u32_unchecked)
            .collect(),
    );
    let poly = evaluation.bit_reverse().interpolate();
    let x = QM31::from_u32_unchecked(
        rng.gen::<u32>(),
        rng.gen::<u32>(),
        rng.gen::<u32>(),
        rng.gen::<u32>(),
    );
    let y = QM31::from_u32_unchecked(
        rng.gen::<u32>(),
        rng.gen::<u32>(),
        rng.gen::<u32>(),
        rng.gen::<u32>(),
    );

    let point = CirclePoint { x, y };
    c.bench_function("cpu eval_at_secure_field_point 2^20", |b| {
        b.iter(|| {
            black_box(<CPUBackend as PolyOps>::eval_at_point(&poly, point));
        })
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_eval_at_secure_point(c: &mut criterion::Criterion) {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use stwo_prover::core::backend::avx512::AVX512Backend;
    use stwo_prover::core::circle::CirclePoint;
    use stwo_prover::core::fields::m31::BaseField;
    use stwo_prover::core::fields::qm31::QM31;
    use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
    use stwo_prover::core::poly::NaturalOrder;
    let log_size = 20;
    let mut rng = SmallRng::seed_from_u64(0);

    let domain = CanonicCoset::new(log_size as u32).circle_domain();
    let evaluation = CircleEvaluation::<AVX512Backend, BaseField, NaturalOrder>::new(
        domain,
        (0..(1 << log_size))
            .map(BaseField::from_u32_unchecked)
            .collect(),
    );
    let poly = evaluation.bit_reverse().interpolate();
    let x: QM31 = rng.gen();
    let y: QM31 = rng.gen();

    let point = CirclePoint { x, y };
    c.bench_function("avx eval_at_secure_field_point 2^20", |b| {
        b.iter(|| {
            black_box(<AVX512Backend as PolyOps>::eval_at_point(&poly, point));
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_group!(
    name=secure_eval;
    config = Criterion::default().sample_size(10);
    targets=avx512_eval_at_secure_point, cpu_eval_at_secure_point);
criterion::criterion_main!(secure_eval);
