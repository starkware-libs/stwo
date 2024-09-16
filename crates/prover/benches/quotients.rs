#![feature(iter_array_chunks)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use stwo_prover::core::backend::cpu::CpuBackend;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::circle::SECURE_FIELD_CIRCLE_GEN;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::BitReversedOrder;

// TODO(andrew): Check if deinterleave affects performance. Use optimized domain iteration if so.
fn bench_quotients<B: QuotientOps, const LOG_N_ROWS: u32, const LOG_N_COLS: u32>(
    c: &mut Criterion,
    id: &str,
) {
    let domain = CanonicCoset::new(LOG_N_ROWS).circle_domain();
    let values = (0..domain.size()).map(BaseField::from).collect();
    let col = CircleEvaluation::<B, BaseField, BitReversedOrder>::new(domain, values);
    let cols = (0..1 << LOG_N_COLS).map(|_| col.clone()).collect_vec();
    let col_refs = cols.iter().collect_vec();
    let random_coeff = SecureField::from_u32_unchecked(0, 1, 2, 3);
    let a = SecureField::from_u32_unchecked(5, 6, 7, 8);
    let samples = vec![ColumnSampleBatch {
        point: SECURE_FIELD_CIRCLE_GEN,
        columns_and_values: (0..1 << LOG_N_COLS).map(|i| (i, a)).collect(),
    }];
    c.bench_function(
        &format!("{id} quotients 2^{LOG_N_COLS} x 2^{LOG_N_ROWS}"),
        |b| {
            b.iter_with_large_drop(|| {
                B::accumulate_quotients(
                    black_box(domain),
                    black_box(&col_refs),
                    black_box(random_coeff),
                    black_box(&samples),
                    1,
                )
            })
        },
    );
}

fn quotients_benches(c: &mut Criterion) {
    bench_quotients::<SimdBackend, 20, 8>(c, "simd");
    bench_quotients::<CpuBackend, 16, 8>(c, "cpu");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = quotients_benches);
criterion_main!(benches);
