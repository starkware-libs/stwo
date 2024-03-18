#![feature(iter_array_chunks)]

use criterion::Criterion;
use criterion::black_box;

#[cfg(target_arch = "x86_64")]
pub fn cpu_eval_at_point(c: &mut criterion::Criterion) {
    use stwo::core::{backend::CPUBackend, fields::m31::BaseField, poly::{circle::{CanonicCoset, CircleEvaluation, PolyOps}, NaturalOrder}};
    let log_size = 18;
    
    let domain = CanonicCoset::new(log_size as u32).circle_domain();
    let evaluation = CircleEvaluation::<CPUBackend, _, NaturalOrder>::new(
        domain,
        (0..(1 << log_size))
            .map(BaseField::from_u32_unchecked)
            .collect(),
    );
    let poly = evaluation.bit_reverse().interpolate();
    let point = domain.at(1 << (log_size - 1));
    c.bench_function("cpu eval_at_point", |b| {
        b.iter(|| {
            black_box(<CPUBackend as PolyOps>::eval_at_point(&poly, point));
        })
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_eval_at_point(c: &mut criterion::Criterion) {
    use stwo::{core::{backend::avx512::AVX512Backend, fields::m31::BaseField, poly::{circle::{CanonicCoset, CircleEvaluation, PolyOps}, NaturalOrder}}, platform};
    if !platform::avx512_detected() {
        return;
    }

    let log_size = 28;
    
    let domain = CanonicCoset::new(log_size as u32).circle_domain();
    let evaluation = CircleEvaluation::<AVX512Backend, _, NaturalOrder>::new(
        domain,
        (0..(1 << log_size))
            .map(BaseField::from_u32_unchecked)
            .collect(),
    );
    let poly = evaluation.bit_reverse().interpolate();
    let point = domain.at(1 << (log_size - 1));
    c.bench_function("avx eval_At_point", |b| {
        b.iter(|| {
            black_box(<AVX512Backend as PolyOps>::eval_at_basefield_point(&poly, point));
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_group!(
    name=avx_bit_rev;
    config = Criterion::default().sample_size(10);
    targets=avx512_eval_at_point, cpu_eval_at_point);
criterion::criterion_main!(avx_bit_rev);
