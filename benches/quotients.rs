#![feature(iter_array_chunks)]

use criterion::Criterion;

#[cfg(target_arch = "x86_64")]
pub fn cpu_quotients(c: &mut criterion::Criterion) {
    use itertools::Itertools;
    use stwo::core::backend::CPUBackend;
    use stwo::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use stwo::core::commitment_scheme::quotients::{BatchedColumnOpenings, QuotientOps};
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use stwo::core::poly::BitReversedOrder;

    const LOG_SIZE: u32 = 16;
    const SIZE: usize = 1 << LOG_SIZE;
    const N_COLS: usize = 1 << 8;
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let cols = (0..N_COLS)
        .map(|_| {
            let values = (0..SIZE as u32)
                .map(BaseField::from_u32_unchecked)
                .collect();
            CircleEvaluation::<CPUBackend, _, BitReversedOrder>::new(domain, values)
        })
        .collect_vec();
    let random_coeff = SecureField::from_m31_array(std::array::from_fn(BaseField::from));
    let a = SecureField::from_m31_array(std::array::from_fn(|i| BaseField::from(3 * i)));
    let openings = vec![BatchedColumnOpenings {
        point: SECURE_FIELD_CIRCLE_GEN,
        column_indices_and_values: (0..N_COLS).map(|i| (i, a)).collect(),
    }];

    c.bench_function("cpu quotients 2^8 x 2^16", |b| {
        b.iter(|| {
            CPUBackend::accumulate_quotients(
                domain,
                &cols.iter().collect_vec(),
                random_coeff,
                &openings,
            )
        })
    });
}

#[cfg(target_arch = "x86_64")]
pub fn avx512_quotients(c: &mut criterion::Criterion) {
    use itertools::Itertools;
    use stwo::core::backend::avx512::AVX512Backend;
    use stwo::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use stwo::core::commitment_scheme::quotients::{BatchedColumnOpenings, QuotientOps};
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use stwo::core::poly::BitReversedOrder;

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
    let openings = vec![BatchedColumnOpenings {
        point: SECURE_FIELD_CIRCLE_GEN,
        column_indices_and_values: (0..N_COLS).map(|i| (i, a)).collect(),
    }];

    c.bench_function("avx quotients 2^8 x 2^20", |b| {
        b.iter(|| {
            AVX512Backend::accumulate_quotients(
                domain,
                &cols.iter().collect_vec(),
                random_coeff,
                &openings,
            )
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_group!(
    name=avx_quotients;
    config = Criterion::default().sample_size(10);
    targets=avx512_quotients, cpu_quotients);
criterion::criterion_main!(avx_quotients);
