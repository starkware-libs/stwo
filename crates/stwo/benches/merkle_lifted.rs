#![feature(iter_array_chunks)]
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use itertools::Itertools;
use num_traits::Zero;
use stwo::core::fields::m31::{BaseField, N_BYTES_FELT};
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Col;
use stwo::prover::vcs_lifted::ops::MerkleOpsLifted;

const LOG_N_ROWS: u32 = 16;

const LOG_N_COLS: u32 = 8;

fn bench_blake2s_merkle<B: MerkleOpsLifted<Blake2sMerkleHasher>>(c: &mut Criterion, id: &str) {
    let col: Col<B, BaseField> = (0..1 << LOG_N_ROWS).map(|_| BaseField::zero()).collect();
    let cols = (0..1 << LOG_N_COLS).map(|_| col.clone()).collect_vec();
    let col_refs = cols.iter().collect_vec();
    let mut group = c.benchmark_group("merkle throughput");
    let n_elements = 1 << (LOG_N_COLS + LOG_N_ROWS);
    group.throughput(Throughput::Elements(n_elements));
    group.throughput(Throughput::Bytes(N_BYTES_FELT as u64 * n_elements));
    group.bench_function(format!("{id} merkle lifted"), |b| {
        b.iter_with_large_drop(|| B::commit_on_first_layer(&col_refs))
    });
}

fn blake2s_merkle_benches_lifted(c: &mut Criterion) {
    bench_blake2s_merkle::<SimdBackend>(c, "simd");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = blake2s_merkle_benches_lifted);
criterion_main!(benches);
