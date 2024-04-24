#![feature(iter_array_chunks)]

use criterion::{black_box, Criterion};
use itertools::Itertools;
use stwo_prover::core::backend::CPUBackend;
use stwo_prover::core::circle::SECURE_FIELD_CIRCLE_GEN;
use stwo_prover::core::commitment_scheme::quotients::{ColumnSampleBatch, QuotientOps};
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_verifier::core::fields::m31::BaseField;
use stwo_verifier::core::fields::qm31::SecureField;

pub fn cpu_quotients(c: &mut criterion::Criterion) {
    const LOG_SIZE: u32 = 16;
    const SIZE: usize = 1 << LOG_SIZE;
    const N_COLS: usize = 1 << 8;
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let cols = (0..N_COLS)
        .map(|_| {
            let values = (0..SIZE).map(BaseField::from).collect();
            CircleEvaluation::<CPUBackend, _, BitReversedOrder>::new(domain, values)
        })
        .collect_vec();
    let random_coeff = SecureField::from_u32_unchecked(0, 1, 2, 3);
    let a = SecureField::from_u32_unchecked(5, 6, 7, 8);
    let samples = vec![ColumnSampleBatch {
        point: SECURE_FIELD_CIRCLE_GEN,
        columns_and_values: (0..N_COLS).map(|i| (i, a)).collect(),
    }];

    let col_refs = &cols.iter().collect_vec();
    c.bench_function("cpu quotients 2^8 x 2^16", |b| {
        b.iter(|| {
            black_box(CPUBackend::accumulate_quotients(
                black_box(domain),
                black_box(col_refs),
                black_box(random_coeff),
                black_box(&samples),
            ))
        })
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_quotients(c: &mut criterion::Criterion) {
    use stwo_prover::core::backend::avx512::AVX512Backend;

    const LOG_SIZE: u32 = 20;
    const SIZE: usize = 1 << LOG_SIZE;
    const N_COLS: usize = 1 << 8;
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let cols = (0..N_COLS)
        .map(|_| {
            let values = (0..SIZE as u32)
                .map(BaseField::from_u32_unchecked)
                .collect();
            CircleEvaluation::<AVX512Backend, _, BitReversedOrder>::new(domain, values)
        })
        .collect_vec();
    let random_coeff = SecureField::from_m31_array(std::array::from_fn(BaseField::from));
    let a = SecureField::from_m31_array(std::array::from_fn(|i| BaseField::from(3 * i)));
    let samples = vec![ColumnSampleBatch {
        point: SECURE_FIELD_CIRCLE_GEN,
        columns_and_values: (0..N_COLS).map(|i| (i, a)).collect(),
    }];

    let col_refs = &cols.iter().collect_vec();
    c.bench_function("avx quotients 2^8 x 2^20", |b| {
        b.iter(|| {
            black_box(AVX512Backend::accumulate_quotients(
                black_box(domain),
                black_box(col_refs),
                black_box(random_coeff),
                black_box(&samples),
            ))
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_group!(
    name=quotients;
    config = Criterion::default().sample_size(10);
    targets=avx512_quotients, cpu_quotients);
#[cfg(not(target_arch = "x86_64"))]
criterion::criterion_group!(
    name=quotients;
    config = Criterion::default().sample_size(10);
    targets=cpu_quotients);
criterion::criterion_main!(quotients);
