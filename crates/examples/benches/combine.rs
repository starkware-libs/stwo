use std::array;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_traits::Zero;
use stwo_constraint_framework::logup::LookupElements;
use stwo_prover::core::channel::Blake2sChannel;
use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::fields::qm31::QM31;
use stwo_prover::prover::backend::simd::m31::PackedM31;
use stwo_prover::prover::backend::simd::qm31::PackedQM31;

pub fn qm31_combine_bench(c: &mut Criterion) {
    let relation = LookupElements::<20>::draw(&mut Blake2sChannel::default());
    const LOG_SIZE: u32 = 16;
    let evals: Vec<[PackedQM31; 20]> = (0..1 << LOG_SIZE)
        .map(|j| {
            array::from_fn(|i| PackedQM31::broadcast(QM31::from_m31_array([M31::from(i + j); 4])))
        })
        .collect();
    c.bench_function(&format!("simd combine 2^{LOG_SIZE}"), |b| {
        b.iter(|| {
            for v in &evals {
                black_box(relation.combine::<PackedQM31, PackedQM31>(v));
            }
        });
    });

    let relation_alpha = PackedQM31::broadcast(relation.alpha);
    c.bench_function(&format!("simd horner 2^{LOG_SIZE}"), |b| {
        b.iter(|| {
            for v in &evals {
                black_box(horner_eval(v, relation_alpha));
            }
        });
    });
}

criterion_group!(benches, qm31_combine_bench);
criterion_main!(benches);

fn horner_eval(evals: &[PackedQM31], relation_alpha: PackedQM31) -> PackedQM31 {
    let mut acc = PackedQM31::zero();
    for eval in evals.iter().rev() {
        acc = acc * relation_alpha + *eval;
    }
    acc
}

#[test]
fn test_horner_eval() {
    let relation = LookupElements::<20>::draw(&mut Blake2sChannel::default());
    const LOG_SIZE: u32 = 16;
    let evals: Vec<[PackedQM31; 20]> = (0..1 << LOG_SIZE)
        .map(|j| {
            array::from_fn(|i| PackedQM31::broadcast(QM31::from_m31_array([M31::from(i + j); 4])))
        })
        .collect();
    let relation_alpha = PackedQM31::broadcast(relation.alpha);
    let result = horner_eval(&evals[0], relation_alpha);
    let result2 = relation.combine::<PackedQM31, PackedQM31>(&evals[0]);
    assert_eq!(result, result2);
}